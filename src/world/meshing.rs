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

/// Vertex for chunk rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub ao: f32,
}

impl ChunkVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ChunkVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
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

/// Build mesh for a chunk using simple face-by-face meshing (no greedy)
pub fn build_chunk_mesh(
    pos: ChunkPos,
    chunk: &Arc<Chunk>,
    neighbors: &[Option<Arc<Chunk>>; 6], // -X, +X, -Y, +Y, -Z, +Z
) -> ChunkMesh {
    // Use simple meshing for now until greedy is debugged
    build_chunk_mesh_simple(pos, chunk, neighbors)
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
    let tiles_per_row = 16u32;
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
        vertices.push(ChunkVertex {
            position: positions[idx],
            normal,
            uv: uvs[idx],
            ao: 1.0,
        });
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

    ChunkMesh {
        pos,
        vertices,
        indices,
        gpu_mesh: None,
        lod,
    }
}

/// Greedy mesh one axis direction with LOD support
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
    for d_lod in 0..lod_size {
        let d = d_lod * step;

        // Clear mask for this slice
        for row in &mut mask[..lod_size] {
            row[..lod_size].fill(None);
        }

        // Fill mask with visible faces (sampled at LOD intervals)
        for v_lod in 0..lod_size {
            for u_lod in 0..lod_size {
                let u = u_lod * step;
                let v = v_lod * step;

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

                // Check if face should be visible
                let neighbor_offset = step as i32;
                let neighbor_pos = match (axis, positive) {
                    (Axis::X, false) => (x as i32 - neighbor_offset, y as i32, z as i32),
                    (Axis::X, true) => (x as i32 + neighbor_offset, y as i32, z as i32),
                    (Axis::Y, false) => (x as i32, y as i32 - neighbor_offset, z as i32),
                    (Axis::Y, true) => (x as i32, y as i32 + neighbor_offset, z as i32),
                    (Axis::Z, false) => (x as i32, y as i32, z as i32 - neighbor_offset),
                    (Axis::Z, true) => (x as i32, y as i32, z as i32 + neighbor_offset),
                };

                if !is_face_visible_lod(chunk, neighbors, neighbor_pos) {
                    continue;
                }

                // Get texture index for this face
                let tex_indices = block.texture_indices();
                let texture_idx = match (axis, positive) {
                    (Axis::Y, true) => tex_indices[0],  // Top
                    (Axis::Y, false) => tex_indices[1], // Bottom
                    (Axis::Z, true) => tex_indices[2],  // Front
                    (Axis::Z, false) => tex_indices[3], // Back
                    (Axis::X, false) => tex_indices[4], // Left
                    (Axis::X, true) => tex_indices[5],  // Right
                };

                mask[v_lod][u_lod] = Some(FaceInfo { block, texture_idx });
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
                emit_greedy_quad(
                    vertices,
                    indices,
                    world_origin,
                    axis,
                    positive,
                    d,
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

/// Emit a greedy quad (merged faces)
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

    // Calculate UV coordinates from texture atlas
    // For greedy meshing, we keep UVs within the single tile
    // The texture will stretch across the merged quad
    let tiles_per_row = 16;
    let tile_size = 1.0 / tiles_per_row as f32;
    let u_min = (texture_idx % tiles_per_row) as f32 * tile_size;
    let v_min = (texture_idx / tiles_per_row) as f32 * tile_size;
    let u_max = u_min + tile_size;
    let v_max = v_min + tile_size;

    // Normal for this face
    let normal = match (axis, positive) {
        (Axis::X, false) => [-1.0, 0.0, 0.0],
        (Axis::X, true) => [1.0, 0.0, 0.0],
        (Axis::Y, false) => [0.0, -1.0, 0.0],
        (Axis::Y, true) => [0.0, 1.0, 0.0],
        (Axis::Z, false) => [0.0, 0.0, -1.0],
        (Axis::Z, true) => [0.0, 0.0, 1.0],
    };

    // Calculate world positions of quad corners (scaled by BLOCK_SCALE)
    let d = slice as f32 * BLOCK_SCALE + if positive { BLOCK_SCALE } else { 0.0 };
    let u0 = start_u as f32 * BLOCK_SCALE;
    let v0 = start_v as f32 * BLOCK_SCALE;
    let u1 = (start_u + width) as f32 * BLOCK_SCALE;
    let v1 = (start_v + height) as f32 * BLOCK_SCALE;

    // Convert (d, u, v) coordinates to world (x, y, z)
    let (p00, p01, p10, p11) = match axis {
        Axis::X => (
            [d, v0, u0], // (d, v, u) -> (x, y, z)
            [d, v1, u0],
            [d, v0, u1],
            [d, v1, u1],
        ),
        Axis::Y => (
            [u0, d, v0],
            [u0, d, v1],
            [u1, d, v0],
            [u1, d, v1],
        ),
        Axis::Z => (
            [u0, v0, d],
            [u0, v1, d],
            [u1, v0, d],
            [u1, v1, d],
        ),
    };

    // Add world origin offset
    let offset = |p: [f32; 3]| -> [f32; 3] {
        [
            p[0] + world_origin.x,
            p[1] + world_origin.y,
            p[2] + world_origin.z,
        ]
    };

    // AO values (simplified - use 1.0 for greedy quads)
    let ao = 1.0;

    // Add vertices with proper UV mapping
    // UVs stay within the tile - texture stretches across merged quad
    vertices.push(ChunkVertex {
        position: offset(p00),
        normal,
        uv: [u_min, v_max],
        ao,
    });
    vertices.push(ChunkVertex {
        position: offset(p01),
        normal,
        uv: [u_min, v_min],
        ao,
    });
    vertices.push(ChunkVertex {
        position: offset(p10),
        normal,
        uv: [u_max, v_max],
        ao,
    });
    vertices.push(ChunkVertex {
        position: offset(p11),
        normal,
        uv: [u_max, v_min],
        ao,
    });

    // Add indices with correct winding order (CCW when viewed from outside)
    if positive {
        // Positive facing - CCW when looking in negative direction
        indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 3,
            base_index,
            base_index + 3,
            base_index + 2,
        ]);
    } else {
        // Negative facing - CCW when looking in positive direction
        indices.extend_from_slice(&[
            base_index,
            base_index + 3,
            base_index + 1,
            base_index,
            base_index + 2,
            base_index + 3,
        ]);
    }
}
