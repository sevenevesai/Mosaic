//! Greedy mesh building for chunks
//!
//! Implements the greedy meshing algorithm to combine adjacent faces
//! with the same texture into larger quads, dramatically reducing vertex count.
//! Supports LOD (Level of Detail) for distant chunks.

use std::sync::Arc;

use super::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32, BLOCK_SCALE};

/// Level of detail for chunk meshing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    /// Full detail - every block
    Full = 1,
    /// Half detail - every 2nd block
    Half = 2,
    /// Quarter detail - every 4th block
    Quarter = 4,
}

/// Packed vertex for chunk rendering (24 bytes vs 36 bytes = 33% smaller)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkVertex {
    pub position: [f32; 3],      // 12 bytes - need full precision for world coords
    pub normal_packed: u32,      // 4 bytes - packed i8x4 normalized (-1/0/1 per axis)
    pub uv: [u16; 2],            // 4 bytes - u16 gives 65536 precision, plenty for atlas
    pub ao_tile_packed: u32,     // 4 bytes - [ao, tile_u, tile_v, 0] for tiling support
}

impl ChunkVertex {
    /// Pack a normal vector into a u32 (only 6 possible normals for cube faces)
    pub fn pack_normal(normal: [f32; 3]) -> u32 {
        let nx = ((normal[0] * 127.0) as i8) as u8;
        let ny = ((normal[1] * 127.0) as i8) as u8;
        let nz = ((normal[2] * 127.0) as i8) as u8;
        u32::from_le_bytes([nx, ny, nz, 0])
    }

    /// Pack UV coordinates (0.0-1.0) into u16 pair
    pub fn pack_uv(uv: [f32; 2]) -> [u16; 2] {
        [(uv[0] * 65535.0) as u16, (uv[1] * 65535.0) as u16]
    }

    /// Pack AO value (0.0-1.0) and tile counts into u32
    /// Layout: [ao (0-255), tile_u (1-16), tile_v (1-16), 0]
    pub fn pack_ao_tile(ao: f32, tile_u: u8, tile_v: u8) -> u32 {
        let ao_byte = (ao * 255.0) as u8;
        u32::from_le_bytes([ao_byte, tile_u, tile_v, 0])
    }

    /// Create vertex with tiling info (for greedy meshing)
    pub fn new_tiled(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32, tile_u: u8, tile_v: u8) -> Self {
        Self {
            position,
            normal_packed: Self::pack_normal(normal),
            uv: Self::pack_uv(uv),
            ao_tile_packed: Self::pack_ao_tile(ao, tile_u, tile_v),
        }
    }

    /// Create vertex without tiling (single block face)
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32) -> Self {
        Self::new_tiled(position, normal, uv, ao, 1, 1)
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ChunkVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // position
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Sint8x4, // normal (will normalize in shader)
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm16x2, // uv (auto-normalized to 0-1)
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Unorm8x4, // ao (uses first component)
                },
            ],
        }
    }
}

/// GPU buffers for a chunk mesh
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

/// Mesh data for a chunk
pub struct ChunkMesh {
    pub pos: ChunkPos,
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
    pub gpu_mesh: Option<GpuMesh>,
    pub lod: LodLevel,
}

impl ChunkMesh {
    /// Upload mesh to GPU
    pub fn upload(&mut self, device: &wgpu::Device) {
        use wgpu::util::DeviceExt;

        if self.vertices.is_empty() {
            self.gpu_mesh = None;
            return;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.gpu_mesh = Some(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: self.indices.len() as u32,
        });
    }
}

/// Face directions for meshing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    X, // Left/Right faces
    Y, // Bottom/Top faces
    Z, // Back/Front faces
}

/// Information about a face for greedy meshing
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceInfo {
    block: Block,
    texture_idx: u32,
}

/// Build mesh for a chunk using greedy meshing
pub fn build_chunk_mesh(
    pos: ChunkPos,
    chunk: &Arc<Chunk>,
    neighbors: &[Option<Arc<Chunk>>; 6], // -X, +X, -Y, +Y, -Z, +Z
) -> ChunkMesh {
    // Greedy meshing: ~2.5x triangle reduction, ~2x FPS improvement
    build_chunk_mesh_lod(pos, chunk, neighbors, LodLevel::Full)
}

/// Simple face-by-face meshing (more triangles but guaranteed correct)
fn build_chunk_mesh_simple(
    pos: ChunkPos,
    chunk: &Arc<Chunk>,
    neighbors: &[Option<Arc<Chunk>>; 6],
) -> ChunkMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let world_origin = pos.world_origin();

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = chunk.get_block(x, y, z);
                if !block.is_solid() {
                    continue;
                }

                let bx = world_origin.x + x as f32 * BLOCK_SCALE;
                let by = world_origin.y + y as f32 * BLOCK_SCALE;
                let bz = world_origin.z + z as f32 * BLOCK_SCALE;

                let tex_indices = block.texture_indices();

                // Top face (+Y)
                if is_face_visible(chunk, neighbors, (x as i32, y as i32 + 1, z as i32)) {
                    add_face_top(&mut vertices, &mut indices, bx, by, bz, tex_indices[0]);
                }
                // Bottom face (-Y)
                if is_face_visible(chunk, neighbors, (x as i32, y as i32 - 1, z as i32)) {
                    add_face_bottom(&mut vertices, &mut indices, bx, by, bz, tex_indices[1]);
                }
                // Front face (+Z)
                if is_face_visible(chunk, neighbors, (x as i32, y as i32, z as i32 + 1)) {
                    add_face_front(&mut vertices, &mut indices, bx, by, bz, tex_indices[2]);
                }
                // Back face (-Z)
                if is_face_visible(chunk, neighbors, (x as i32, y as i32, z as i32 - 1)) {
                    add_face_back(&mut vertices, &mut indices, bx, by, bz, tex_indices[3]);
                }
                // Left face (-X)
                if is_face_visible(chunk, neighbors, (x as i32 - 1, y as i32, z as i32)) {
                    add_face_left(&mut vertices, &mut indices, bx, by, bz, tex_indices[4]);
                }
                // Right face (+X)
                if is_face_visible(chunk, neighbors, (x as i32 + 1, y as i32, z as i32)) {
                    add_face_right(&mut vertices, &mut indices, bx, by, bz, tex_indices[5]);
                }
            }
        }
    }

    ChunkMesh {
        pos,
        vertices,
        indices,
        gpu_mesh: None,
        lod: LodLevel::Full,
    }
}

fn get_tile_uvs(tex_idx: u32) -> ([f32; 2], [f32; 2], [f32; 2], [f32; 2]) {
    let tiles_per_row = 4u32; // 4x4 grid of 256x256 tiles
    let tile_size = 1.0 / tiles_per_row as f32;
    let u0 = (tex_idx % tiles_per_row) as f32 * tile_size;
    let v0 = (tex_idx / tiles_per_row) as f32 * tile_size;
    let u1 = u0 + tile_size;
    let v1 = v0 + tile_size;
    // Return: bottom-left, bottom-right, top-right, top-left
    ([u0, v1], [u1, v1], [u1, v0], [u0, v0])
}

fn add_quad(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    positions: [[f32; 3]; 4],
    normal: [f32; 3],
    uvs: [[f32; 2]; 4],
) {
    let base = vertices.len() as u32;
    for idx in 0..4 {
        vertices.push(ChunkVertex::new(positions[idx], normal, uvs[idx], 1.0));
    }
    // Vertices are in order: 0-1-2-3 (bottom-left, bottom-right, top-right, top-left)
    // Two CCW triangles when viewed from outside: 0-2-1 and 0-3-2
    indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

// All faces use CCW vertex order when viewed from outside the block
// Order: bottom-left -> bottom-right -> top-right -> top-left

// Top face (+Y normal) - camera above looking down
fn add_face_top(v: &mut Vec<ChunkVertex>, i: &mut Vec<u32>, x: f32, y: f32, z: f32, tex: u32) {
    let (uv0, uv1, uv2, uv3) = get_tile_uvs(tex);
    let s = BLOCK_SCALE;
    // Looking down from +Y: +X is right, +Z is "down" on screen
    add_quad(v, i,
        [
            [x,     y + s, z],       // 0: back-left (screen bottom-left)
            [x + s, y + s, z],       // 1: back-right (screen bottom-right)
            [x + s, y + s, z + s],   // 2: front-right (screen top-right)
            [x,     y + s, z + s],   // 3: front-left (screen top-left)
        ],
        [0.0, 1.0, 0.0],
        [uv0, uv1, uv2, uv3],
    );
}

// Bottom face (-Y normal) - camera below looking up
fn add_face_bottom(v: &mut Vec<ChunkVertex>, i: &mut Vec<u32>, x: f32, y: f32, z: f32, tex: u32) {
    let (uv0, uv1, uv2, uv3) = get_tile_uvs(tex);
    let s = BLOCK_SCALE;
    // Looking up from -Y: +X is right, +Z is "up" on screen
    add_quad(v, i,
        [
            [x,     y, z + s],   // 0: front-left (screen bottom-left)
            [x + s, y, z + s],   // 1: front-right (screen bottom-right)
            [x + s, y, z],       // 2: back-right (screen top-right)
            [x,     y, z],       // 3: back-left (screen top-left)
        ],
        [0.0, -1.0, 0.0],
        [uv0, uv1, uv2, uv3],
    );
}

// Front face (+Z normal) - camera in front looking at -Z
fn add_face_front(v: &mut Vec<ChunkVertex>, i: &mut Vec<u32>, x: f32, y: f32, z: f32, tex: u32) {
    let (uv0, uv1, uv2, uv3) = get_tile_uvs(tex);
    let s = BLOCK_SCALE;
    // Looking at face from +Z: +X is left (mirrored), +Y is up
    add_quad(v, i,
        [
            [x + s, y,     z + s],   // 0: bottom-right of block (screen bottom-left)
            [x,     y,     z + s],   // 1: bottom-left of block (screen bottom-right)
            [x,     y + s, z + s],   // 2: top-left of block (screen top-right)
            [x + s, y + s, z + s],   // 3: top-right of block (screen top-left)
        ],
        [0.0, 0.0, 1.0],
        [uv0, uv1, uv2, uv3],
    );
}

// Back face (-Z normal) - camera behind looking at +Z
fn add_face_back(v: &mut Vec<ChunkVertex>, i: &mut Vec<u32>, x: f32, y: f32, z: f32, tex: u32) {
    let (uv0, uv1, uv2, uv3) = get_tile_uvs(tex);
    let s = BLOCK_SCALE;
    // Looking at face from -Z: +X is right, +Y is up
    add_quad(v, i,
        [
            [x,     y,     z],   // 0: bottom-left (screen bottom-left)
            [x + s, y,     z],   // 1: bottom-right (screen bottom-right)
            [x + s, y + s, z],   // 2: top-right (screen top-right)
            [x,     y + s, z],   // 3: top-left (screen top-left)
        ],
        [0.0, 0.0, -1.0],
        [uv0, uv1, uv2, uv3],
    );
}

// Left face (-X normal) - camera to the left looking at +X
fn add_face_left(v: &mut Vec<ChunkVertex>, i: &mut Vec<u32>, x: f32, y: f32, z: f32, tex: u32) {
    let (uv0, uv1, uv2, uv3) = get_tile_uvs(tex);
    let s = BLOCK_SCALE;
    // Looking at face from -X: +Z is right, +Y is up
    add_quad(v, i,
        [
            [x, y,     z],       // 0: bottom-back
            [x, y + s, z],       // 1: top-back
            [x, y + s, z + s],   // 2: top-front
            [x, y,     z + s],   // 3: bottom-front
        ],
        [-1.0, 0.0, 0.0],
        [uv0, uv3, uv2, uv1],  // Reordered to match vertex order
    );
}

// Right face (+X normal) - camera to the right looking at -X
fn add_face_right(v: &mut Vec<ChunkVertex>, i: &mut Vec<u32>, x: f32, y: f32, z: f32, tex: u32) {
    let (uv0, uv1, uv2, uv3) = get_tile_uvs(tex);
    let s = BLOCK_SCALE;
    // Looking at face from +X: +Z is left (mirrored), +Y is up
    add_quad(v, i,
        [
            [x + s, y,     z + s],   // 0: bottom-front
            [x + s, y + s, z + s],   // 1: top-front
            [x + s, y + s, z],       // 2: top-back
            [x + s, y,     z],       // 3: bottom-back
        ],
        [1.0, 0.0, 0.0],
        [uv0, uv3, uv2, uv1],  // Reordered to match vertex order
    );
}

/// Build mesh for a chunk with specified LOD level
pub fn build_chunk_mesh_lod(
    pos: ChunkPos,
    chunk: &Arc<Chunk>,
    neighbors: &[Option<Arc<Chunk>>; 6],
    lod: LodLevel,
) -> ChunkMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let world_origin = pos.world_origin();
    let step = lod as usize;

    // Process each axis
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        // Process both directions (negative and positive facing)
        for positive in [false, true] {
            if step == 1 {
                // Use optimized binary meshing for full detail
                binary_greedy_mesh_axis(
                    chunk,
                    neighbors,
                    axis,
                    positive,
                    world_origin,
                    &mut vertices,
                    &mut indices,
                );
            } else {
                // Use LOD-aware meshing for reduced detail
                greedy_mesh_axis_lod(
                    chunk,
                    neighbors,
                    axis,
                    positive,
                    world_origin,
                    step,
                    &mut vertices,
                    &mut indices,
                );
            }
        }
    }

    ChunkMesh {
        pos,
        vertices,
        indices,
        gpu_mesh: None,
        lod,
    }
}

/// Binary greedy meshing for full detail (step=1)
/// Uses bitwise operations for fast face culling, then texture-aware greedy merge
fn binary_greedy_mesh_axis(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    axis: Axis,
    positive: bool,
    world_origin: glam::Vec3,
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
) {
    // Get neighbor chunk for this face direction
    let neighbor_idx = match (axis, positive) {
        (Axis::X, false) => 0, // -X
        (Axis::X, true) => 1,  // +X
        (Axis::Y, false) => 2, // -Y
        (Axis::Y, true) => 3,  // +Y
        (Axis::Z, false) => 4, // -Z
        (Axis::Z, true) => 5,  // +Z
    };
    let neighbor_chunk = &neighbors[neighbor_idx];

    // Create a 2D mask for texture-aware greedy merge
    let mut mask = [[None::<FaceInfo>; CHUNK_SIZE]; CHUNK_SIZE];

    // Process each slice perpendicular to the axis
    for d in 0..CHUNK_SIZE {
        // Clear mask
        for row in &mut mask {
            row.fill(None);
        }

        // Build binary visibility mask using bitwise operations
        // Each row is a u16 where bit i = 1 means visible face at position i
        let mut visible_mask = [0u16; CHUNK_SIZE];

        // Build occupancy for current slice
        let mut current_occupancy = [0u16; CHUNK_SIZE];
        for v in 0..CHUNK_SIZE {
            for u in 0..CHUNK_SIZE {
                let (x, y, z) = match axis {
                    Axis::X => (d, v, u),
                    Axis::Y => (u, d, v),
                    Axis::Z => (u, v, d),
                };
                if chunk.get_block(x, y, z).is_solid() {
                    current_occupancy[v] |= 1 << u;
                }
            }
        }

        // Build occupancy for neighbor slice (d+1 for positive, d-1 for negative)
        let mut neighbor_occupancy = [0u16; CHUNK_SIZE];

        if positive {
            // Check slice at d+1
            if d + 1 < CHUNK_SIZE {
                // Neighbor is in same chunk
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (d + 1, v, u),
                            Axis::Y => (u, d + 1, v),
                            Axis::Z => (u, v, d + 1),
                        };
                        if chunk.get_block(x, y, z).is_solid() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            } else if let Some(nc) = neighbor_chunk {
                // Neighbor is in adjacent chunk (first slice)
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (0, v, u),
                            Axis::Y => (u, 0, v),
                            Axis::Z => (u, v, 0),
                        };
                        if nc.get_block(x, y, z).is_solid() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            }
            // If no neighbor chunk, neighbor_occupancy stays 0 (all air = all faces visible)
        } else {
            // Check slice at d-1
            if d > 0 {
                // Neighbor is in same chunk
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (d - 1, v, u),
                            Axis::Y => (u, d - 1, v),
                            Axis::Z => (u, v, d - 1),
                        };
                        if chunk.get_block(x, y, z).is_solid() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            } else if let Some(nc) = neighbor_chunk {
                // Neighbor is in adjacent chunk (last slice)
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (CHUNK_SIZE - 1, v, u),
                            Axis::Y => (u, CHUNK_SIZE - 1, v),
                            Axis::Z => (u, v, CHUNK_SIZE - 1),
                        };
                        if nc.get_block(x, y, z).is_solid() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            }
            // If no neighbor chunk, neighbor_occupancy stays 0 (all air = all faces visible)
        }

        // Compute visible faces: solid current AND air neighbor
        // visible = current & !neighbor
        for v in 0..CHUNK_SIZE {
            visible_mask[v] = current_occupancy[v] & !neighbor_occupancy[v];
        }

        // Now build texture-aware mask from visible_mask
        for v in 0..CHUNK_SIZE {
            let mut row_mask = visible_mask[v];
            while row_mask != 0 {
                let u = row_mask.trailing_zeros() as usize;
                row_mask &= !(1 << u); // Clear this bit

                // Get block and texture info
                let (x, y, z) = match axis {
                    Axis::X => (d, v, u),
                    Axis::Y => (u, d, v),
                    Axis::Z => (u, v, d),
                };
                let block = chunk.get_block(x, y, z);
                let tex_indices = block.texture_indices();
                let texture_idx = match (axis, positive) {
                    (Axis::Y, true) => tex_indices[0],  // Top
                    (Axis::Y, false) => tex_indices[1], // Bottom
                    (Axis::Z, true) => tex_indices[2],  // Front
                    (Axis::Z, false) => tex_indices[3], // Back
                    (Axis::X, false) => tex_indices[4], // Left
                    (Axis::X, true) => tex_indices[5],  // Right
                };

                mask[v][u] = Some(FaceInfo { block, texture_idx });
            }
        }

        // Greedy merge (same as before)
        let mut processed = [[false; CHUNK_SIZE]; CHUNK_SIZE];

        for start_v in 0..CHUNK_SIZE {
            for start_u in 0..CHUNK_SIZE {
                if processed[start_v][start_u] {
                    continue;
                }

                let Some(face_info) = mask[start_v][start_u] else {
                    continue;
                };

                // Find width using binary search on visible_mask for same texture
                let mut width = 1;
                while start_u + width < CHUNK_SIZE
                    && !processed[start_v][start_u + width]
                    && mask[start_v][start_u + width] == Some(face_info)
                {
                    width += 1;
                }

                // Find height
                let mut height = 1;
                'height: while start_v + height < CHUNK_SIZE {
                    for u in start_u..start_u + width {
                        if processed[start_v + height][u]
                            || mask[start_v + height][u] != Some(face_info)
                        {
                            break 'height;
                        }
                    }
                    height += 1;
                }

                // Mark as processed
                for v in start_v..start_v + height {
                    for u in start_u..start_u + width {
                        processed[v][u] = true;
                    }
                }

                // Emit quad
                emit_greedy_quad(
                    vertices,
                    indices,
                    world_origin,
                    axis,
                    positive,
                    d,
                    start_u,
                    start_v,
                    width,
                    height,
                    face_info.texture_idx,
                );
            }
        }
    }
}

/// Greedy mesh one axis direction with LOD support
///
/// For LOD > 1, we scan ALL blocks in the LOD cell (all depth slices, not just the outer one)
/// to find any visible faces. This ensures we don't miss interior faces within a LOD cell.
fn greedy_mesh_axis_lod(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    axis: Axis,
    positive: bool,
    world_origin: glam::Vec3,
    step: usize,
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
) {
    let lod_size = CHUNK_SIZE / step;

    // Create a 2D mask for each slice along this axis (scaled by LOD)
    // Using fixed-size arrays for performance
    let mut mask = [[None::<FaceInfo>; CHUNK_SIZE]; CHUNK_SIZE];

    // Iterate through slices perpendicular to the axis
    // For LOD, we process 'step' slices at once, checking all blocks in the region
    for d_lod in 0..lod_size {
        let d_base = d_lod * step;

        // Clear mask for this slice
        for row in &mut mask[..lod_size] {
            row[..lod_size].fill(None);
        }

        // Fill mask by scanning ALL blocks in each LOD cell (all depth slices!)
        for v_lod in 0..lod_size {
            for u_lod in 0..lod_size {
                let u_base = u_lod * step;
                let v_base = v_lod * step;

                // Scan all blocks in this LOD cell to find if ANY has a visible face
                // We must check ALL depth slices within the cell, not just the outer one
                let mut found_face: Option<FaceInfo> = None;

                // Scan ALL depth slices in this LOD cell
                'outer: for dd in 0..step {
                    let d = d_base + dd;
                    if d >= CHUNK_SIZE {
                        continue;
                    }

                    // Scan all blocks in the u-v plane at this depth
                    for dv in 0..step {
                        for du in 0..step {
                            let u = u_base + du;
                            let v = v_base + dv;

                            if u >= CHUNK_SIZE || v >= CHUNK_SIZE {
                                continue;
                            }

                            // Convert (d, u, v) to (x, y, z) based on axis
                            let (x, y, z) = match axis {
                                Axis::X => (d, v, u),
                                Axis::Y => (u, d, v),
                                Axis::Z => (u, v, d),
                            };

                            let block = chunk.get_block(x, y, z);
                            if !block.is_solid() {
                                continue;
                            }

                            // Check if face should be visible (immediate neighbor in the face direction)
                            let neighbor_pos = match (axis, positive) {
                                (Axis::X, false) => (x as i32 - 1, y as i32, z as i32),
                                (Axis::X, true) => (x as i32 + 1, y as i32, z as i32),
                                (Axis::Y, false) => (x as i32, y as i32 - 1, z as i32),
                                (Axis::Y, true) => (x as i32, y as i32 + 1, z as i32),
                                (Axis::Z, false) => (x as i32, y as i32, z as i32 - 1),
                                (Axis::Z, true) => (x as i32, y as i32, z as i32 + 1),
                            };

                            if !is_face_visible(chunk, neighbors, neighbor_pos) {
                                continue;
                            }

                            // Found a visible face - use this block's texture
                            let tex_indices = block.texture_indices();
                            let texture_idx = match (axis, positive) {
                                (Axis::Y, true) => tex_indices[0],  // Top
                                (Axis::Y, false) => tex_indices[1], // Bottom
                                (Axis::Z, true) => tex_indices[2],  // Front
                                (Axis::Z, false) => tex_indices[3], // Back
                                (Axis::X, false) => tex_indices[4], // Left
                                (Axis::X, true) => tex_indices[5],  // Right
                            };

                            found_face = Some(FaceInfo { block, texture_idx });
                            break 'outer; // Found at least one visible face, that's enough for LOD
                        }
                    }
                }

                if let Some(face_info) = found_face {
                    mask[v_lod][u_lod] = Some(face_info);
                }
            }
        }

        // Greedy merge: find largest rectangles in the mask
        let mut processed = [[false; CHUNK_SIZE]; CHUNK_SIZE];

        for start_v in 0..lod_size {
            for start_u in 0..lod_size {
                if processed[start_v][start_u] {
                    continue;
                }

                let Some(face_info) = mask[start_v][start_u] else {
                    continue;
                };

                // Find width (extend in u direction)
                let mut width = 1;
                while start_u + width < lod_size
                    && !processed[start_v][start_u + width]
                    && mask[start_v][start_u + width] == Some(face_info)
                {
                    width += 1;
                }

                // Find height (extend in v direction)
                let mut height = 1;
                'height: while start_v + height < lod_size {
                    for u in start_u..start_u + width {
                        if processed[start_v + height][u]
                            || mask[start_v + height][u] != Some(face_info)
                        {
                            break 'height;
                        }
                    }
                    height += 1;
                }

                // Mark as processed
                for v in start_v..start_v + height {
                    for u in start_u..start_u + width {
                        processed[v][u] = true;
                    }
                }

                // Emit quad (scaled by LOD step)
                // For the slice position, use the outer face of the LOD cell
                let d_emit = if positive { d_base + step - 1 } else { d_base };
                emit_greedy_quad(
                    vertices,
                    indices,
                    world_origin,
                    axis,
                    positive,
                    d_emit,
                    start_u * step,
                    start_v * step,
                    width * step,
                    height * step,
                    face_info.texture_idx,
                );
            }
        }
    }
}

/// Check if face is visible for LOD (accounts for larger step sizes)
fn is_face_visible_lod(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    (nx, ny, nz): (i32, i32, i32),
) -> bool {
    // Clamp to valid range for checking
    let nx = nx.clamp(-1, CHUNK_SIZE_I32);
    let ny = ny.clamp(-1, CHUNK_SIZE_I32);
    let nz = nz.clamp(-1, CHUNK_SIZE_I32);

    is_face_visible(chunk, neighbors, (nx, ny, nz))
}

/// Check if a face should be visible (neighbor is transparent)
fn is_face_visible(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    (nx, ny, nz): (i32, i32, i32),
) -> bool {
    // Check if neighbor is in current chunk
    if Chunk::in_bounds(nx, ny, nz) {
        let neighbor_block = chunk.get_block(nx as usize, ny as usize, nz as usize);
        return neighbor_block.is_transparent();
    }

    // Check neighbor chunk
    let (neighbor_idx, local_x, local_y, local_z) = if nx < 0 {
        (0, CHUNK_SIZE - 1, ny as usize, nz as usize)
    } else if nx >= CHUNK_SIZE_I32 {
        (1, 0, ny as usize, nz as usize)
    } else if ny < 0 {
        (2, nx as usize, CHUNK_SIZE - 1, nz as usize)
    } else if ny >= CHUNK_SIZE_I32 {
        (3, nx as usize, 0, nz as usize)
    } else if nz < 0 {
        (4, nx as usize, ny as usize, CHUNK_SIZE - 1)
    } else {
        (5, nx as usize, ny as usize, 0)
    };

    if let Some(neighbor_chunk) = &neighbors[neighbor_idx] {
        let neighbor_block = neighbor_chunk.get_block(local_x, local_y, local_z);
        neighbor_block.is_transparent()
    } else {
        // No neighbor chunk loaded - render the face
        true
    }
}

/// Get whether a block at the given position is solid (for AO calculation)
/// Handles cross-chunk boundary lookups
fn is_block_solid_at(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    // Check if in current chunk
    if Chunk::in_bounds(x, y, z) {
        return chunk.get_block(x as usize, y as usize, z as usize).is_solid();
    }

    // Determine which neighbor chunk and local coordinates
    let (neighbor_idx, local_x, local_y, local_z) = if x < 0 {
        (0, (CHUNK_SIZE_I32 + x) as usize, y as usize, z as usize)
    } else if x >= CHUNK_SIZE_I32 {
        (1, (x - CHUNK_SIZE_I32) as usize, y as usize, z as usize)
    } else if y < 0 {
        (2, x as usize, (CHUNK_SIZE_I32 + y) as usize, z as usize)
    } else if y >= CHUNK_SIZE_I32 {
        (3, x as usize, (y - CHUNK_SIZE_I32) as usize, z as usize)
    } else if z < 0 {
        (4, x as usize, y as usize, (CHUNK_SIZE_I32 + z) as usize)
    } else {
        (5, x as usize, y as usize, (z - CHUNK_SIZE_I32) as usize)
    };

    // Bounds check for the local coordinates
    if local_x >= CHUNK_SIZE || local_y >= CHUNK_SIZE || local_z >= CHUNK_SIZE {
        return false; // Out of bounds, treat as air
    }

    if let Some(neighbor_chunk) = &neighbors[neighbor_idx] {
        neighbor_chunk.get_block(local_x, local_y, local_z).is_solid()
    } else {
        false // No neighbor chunk, treat as air
    }
}

/// Calculate ambient occlusion for a vertex corner
/// side1, side2, corner are the 3 blocks adjacent to this vertex that could occlude light
/// Returns 0.0 (fully occluded) to 1.0 (no occlusion)
fn calculate_ao(side1: bool, side2: bool, corner: bool) -> f32 {
    // If both sides are solid, corner is considered occluded (prevents light leak)
    let effective_corner = if side1 && side2 { true } else { corner };

    let occluders = (side1 as u8) + (side2 as u8) + (effective_corner as u8);

    // AO levels: 0 occluders = 1.0, 1 = 0.75, 2 = 0.5, 3 = 0.25
    match occluders {
        0 => 1.0,
        1 => 0.7,
        2 => 0.5,
        3 => 0.2,
        _ => 0.0,
    }
}

/// Calculate AO values for all 4 corners of a face
/// Returns [ao0, ao1, ao2, ao3] matching vertex order
fn calculate_face_ao(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    x: i32,
    y: i32,
    z: i32,
    axis: Axis,
    positive: bool,
) -> [f32; 4] {
    // For each face orientation, we need to check different neighbor positions
    // The face is on the positive or negative side of the axis
    // Each vertex corner needs to sample 3 neighbors: side1, side2, corner

    match (axis, positive) {
        // Top face (+Y) - vertices at y+1 plane
        (Axis::Y, true) => {
            let ny = y + 1;
            [
                // Vertex 0: back-left (x, y+1, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, ny, z - 1),     // side2: -Z
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z - 1), // corner
                ),
                // Vertex 1: back-right (x+1, y+1, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, ny, z - 1),     // side2: -Z
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z - 1), // corner
                ),
                // Vertex 2: front-right (x+1, y+1, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, ny, z + 1),     // side2: +Z
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z + 1), // corner
                ),
                // Vertex 3: front-left (x, y+1, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, ny, z + 1),     // side2: +Z
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z + 1), // corner
                ),
            ]
        }
        // Bottom face (-Y) - vertices at y plane
        (Axis::Y, false) => {
            let ny = y - 1;
            [
                // Vertex 0: front-left (x, y, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, ny, z + 1),     // side2: +Z
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z + 1), // corner
                ),
                // Vertex 1: front-right (x+1, y, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, ny, z + 1),     // side2: +Z
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z + 1), // corner
                ),
                // Vertex 2: back-right (x+1, y, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, ny, z - 1),     // side2: -Z
                    is_block_solid_at(chunk, neighbors, x + 1, ny, z - 1), // corner
                ),
                // Vertex 3: back-left (x, y, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, ny, z - 1),     // side2: -Z
                    is_block_solid_at(chunk, neighbors, x - 1, ny, z - 1), // corner
                ),
            ]
        }
        // Front face (+Z) - vertices at z+1 plane
        (Axis::Z, true) => {
            let nz = z + 1;
            [
                // Vertex 0: bottom-right (x+1, y, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, y, nz),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, y - 1, nz),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, x + 1, y - 1, nz), // corner
                ),
                // Vertex 1: bottom-left (x, y, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, y, nz),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, y - 1, nz),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, x - 1, y - 1, nz), // corner
                ),
                // Vertex 2: top-left (x, y+1, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, y, nz),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, y + 1, nz),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, x - 1, y + 1, nz), // corner
                ),
                // Vertex 3: top-right (x+1, y+1, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, y, nz),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, y + 1, nz),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, x + 1, y + 1, nz), // corner
                ),
            ]
        }
        // Back face (-Z) - vertices at z plane
        (Axis::Z, false) => {
            let nz = z - 1;
            [
                // Vertex 0: bottom-left (x, y, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, y, nz),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, y - 1, nz),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, x - 1, y - 1, nz), // corner
                ),
                // Vertex 1: bottom-right (x+1, y, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, y, nz),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, y - 1, nz),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, x + 1, y - 1, nz), // corner
                ),
                // Vertex 2: top-right (x+1, y+1, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x + 1, y, nz),     // side1: +X
                    is_block_solid_at(chunk, neighbors, x, y + 1, nz),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, x + 1, y + 1, nz), // corner
                ),
                // Vertex 3: top-left (x, y+1, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, x - 1, y, nz),     // side1: -X
                    is_block_solid_at(chunk, neighbors, x, y + 1, nz),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, x - 1, y + 1, nz), // corner
                ),
            ]
        }
        // Left face (-X) - vertices at x plane
        (Axis::X, false) => {
            let nx = x - 1;
            [
                // Vertex 0: bottom-back (x, y, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z - 1),     // side1: -Z
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z - 1), // corner
                ),
                // Vertex 1: top-back (x, y+1, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z - 1),     // side1: -Z
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z - 1), // corner
                ),
                // Vertex 2: top-front (x, y+1, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z + 1),     // side1: +Z
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z + 1), // corner
                ),
                // Vertex 3: bottom-front (x, y, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z + 1),     // side1: +Z
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z + 1), // corner
                ),
            ]
        }
        // Right face (+X) - vertices at x+1 plane
        (Axis::X, true) => {
            let nx = x + 1;
            [
                // Vertex 0: bottom-front (x+1, y, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z + 1),     // side1: +Z
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z + 1), // corner
                ),
                // Vertex 1: top-front (x+1, y+1, z+1)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z + 1),     // side1: +Z
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z + 1), // corner
                ),
                // Vertex 2: top-back (x+1, y+1, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z - 1),     // side1: -Z
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z),     // side2: +Y
                    is_block_solid_at(chunk, neighbors, nx, y + 1, z - 1), // corner
                ),
                // Vertex 3: bottom-back (x+1, y, z)
                calculate_ao(
                    is_block_solid_at(chunk, neighbors, nx, y, z - 1),     // side1: -Z
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z),     // side2: -Y
                    is_block_solid_at(chunk, neighbors, nx, y - 1, z - 1), // corner
                ),
            ]
        }
    }
}

/// Emit a greedy quad (merged faces) - matches vertex ordering from simple meshing
fn emit_greedy_quad(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    world_origin: glam::Vec3,
    axis: Axis,
    positive: bool,
    slice: usize,
    start_u: usize,
    start_v: usize,
    width: usize,
    height: usize,
    texture_idx: u32,
) {
    let base_index = vertices.len() as u32;

    // Get UVs from texture atlas
    let tiles_per_row = 4u32; // 4x4 grid of 256x256 tiles
    let tile_size = 1.0 / tiles_per_row as f32;
    let uv_u0 = (texture_idx % tiles_per_row) as f32 * tile_size;
    let uv_v0 = (texture_idx / tiles_per_row) as f32 * tile_size;
    let uv_u1 = uv_u0 + tile_size;
    let uv_v1 = uv_v0 + tile_size;
    // UV corners: bottom-left, bottom-right, top-right, top-left
    let uvs = [[uv_u0, uv_v1], [uv_u1, uv_v1], [uv_u1, uv_v0], [uv_u0, uv_v0]];

    // Tile counts for texture tiling (in blocks)
    // Width/height need to be mapped to tile_u/tile_v based on axis orientation
    let (tile_u, tile_v) = match axis {
        Axis::X => (width as u8, height as u8),  // Left/Right: width=Z, height=Y
        Axis::Y => (width as u8, height as u8),  // Top/Bottom: width=X, height=Z
        Axis::Z => (width as u8, height as u8),  // Front/Back: width=X, height=Y
    };

    let ao = 1.0;
    let s = BLOCK_SCALE;
    let ox = world_origin.x;
    let oy = world_origin.y;
    let oz = world_origin.z;

    // Small epsilon to expand faces and prevent sub-pixel gaps at edges
    // This makes adjacent faces slightly overlap, eliminating white pixel artifacts
    // The overlap is tiny (< 1mm in world units) and invisible to the player
    let e = 0.001;

    // Convert slice/u/v to world coordinates based on axis
    // Match vertex ordering exactly from simple meshing add_face_* functions
    // Each face is expanded by epsilon in its plane to overlap with adjacent faces
    let (positions, normal, uv_order): ([[f32; 3]; 4], [f32; 3], [usize; 4]) = match (axis, positive) {
        // Top face (+Y) - from add_face_top, expand in X and Z
        (Axis::Y, true) => {
            let x0 = ox + start_u as f32 * s - e;
            let x1 = ox + (start_u + width) as f32 * s + e;
            let y = oy + (slice + 1) as f32 * s;
            let z0 = oz + start_v as f32 * s - e;
            let z1 = oz + (start_v + height) as f32 * s + e;
            ([
                [x0, y, z0], // back-left
                [x1, y, z0], // back-right
                [x1, y, z1], // front-right
                [x0, y, z1], // front-left
            ], [0.0, 1.0, 0.0], [0, 1, 2, 3])
        }
        // Bottom face (-Y) - from add_face_bottom, expand in X and Z
        (Axis::Y, false) => {
            let x0 = ox + start_u as f32 * s - e;
            let x1 = ox + (start_u + width) as f32 * s + e;
            let y = oy + slice as f32 * s;
            let z0 = oz + start_v as f32 * s - e;
            let z1 = oz + (start_v + height) as f32 * s + e;
            ([
                [x0, y, z1], // front-left
                [x1, y, z1], // front-right
                [x1, y, z0], // back-right
                [x0, y, z0], // back-left
            ], [0.0, -1.0, 0.0], [0, 1, 2, 3])
        }
        // Front face (+Z) - from add_face_front, expand in X and Y
        (Axis::Z, true) => {
            let x0 = ox + start_u as f32 * s - e;
            let x1 = ox + (start_u + width) as f32 * s + e;
            let y0 = oy + start_v as f32 * s - e;
            let y1 = oy + (start_v + height) as f32 * s + e;
            let z = oz + (slice + 1) as f32 * s;
            ([
                [x1, y0, z], // bottom-right
                [x0, y0, z], // bottom-left
                [x0, y1, z], // top-left
                [x1, y1, z], // top-right
            ], [0.0, 0.0, 1.0], [0, 1, 2, 3])
        }
        // Back face (-Z) - from add_face_back, expand in X and Y
        (Axis::Z, false) => {
            let x0 = ox + start_u as f32 * s - e;
            let x1 = ox + (start_u + width) as f32 * s + e;
            let y0 = oy + start_v as f32 * s - e;
            let y1 = oy + (start_v + height) as f32 * s + e;
            let z = oz + slice as f32 * s;
            ([
                [x0, y0, z], // bottom-left
                [x1, y0, z], // bottom-right
                [x1, y1, z], // top-right
                [x0, y1, z], // top-left
            ], [0.0, 0.0, -1.0], [0, 1, 2, 3])
        }
        // Left face (-X) - from add_face_left, expand in Y and Z
        (Axis::X, false) => {
            let x = ox + slice as f32 * s;
            let y0 = oy + start_v as f32 * s - e;
            let y1 = oy + (start_v + height) as f32 * s + e;
            let z0 = oz + start_u as f32 * s - e;
            let z1 = oz + (start_u + width) as f32 * s + e;
            ([
                [x, y0, z0], // bottom-back
                [x, y1, z0], // top-back
                [x, y1, z1], // top-front
                [x, y0, z1], // bottom-front
            ], [-1.0, 0.0, 0.0], [0, 3, 2, 1]) // UV reorder like simple meshing
        }
        // Right face (+X) - from add_face_right, expand in Y and Z
        (Axis::X, true) => {
            let x = ox + (slice + 1) as f32 * s;
            let y0 = oy + start_v as f32 * s - e;
            let y1 = oy + (start_v + height) as f32 * s + e;
            let z0 = oz + start_u as f32 * s - e;
            let z1 = oz + (start_u + width) as f32 * s + e;
            ([
                [x, y0, z1], // bottom-front
                [x, y1, z1], // top-front
                [x, y1, z0], // top-back
                [x, y0, z0], // bottom-back
            ], [1.0, 0.0, 0.0], [0, 3, 2, 1]) // UV reorder like simple meshing
        }
    };

    // Add vertices with tiling info
    for i in 0..4 {
        vertices.push(ChunkVertex::new_tiled(positions[i], normal, uvs[uv_order[i]], ao, tile_u, tile_v));
    }

    // Same index pattern as add_quad: 0-2-1 and 0-3-2
    indices.extend_from_slice(&[
        base_index, base_index + 2, base_index + 1,
        base_index, base_index + 3, base_index + 2,
    ]);
}
