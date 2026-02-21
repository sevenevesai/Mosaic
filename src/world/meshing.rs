//! Binary Greedy Mesh Builder
//!
//! Implements a pure binary greedy meshing algorithm using bitwise operations
//! to merge adjacent faces into larger quads, dramatically reducing vertex count.
//!
//! ## Algorithm Overview
//!
//! For each axis (X, Y, Z) and direction (positive, negative):
//!   For each slice perpendicular to the axis:
//!     1. Build occupancy bitmasks (u16 per row, 16 rows = 256 bits total)
//!     2. Compute visibility via bitwise AND-NOT: `visible = current & !neighbor`
//!     3. Group visible faces by texture index into per-texture bitmasks
//!     4. Binary greedy merge each texture's mask:
//!        - Find horizontal runs using `trailing_zeros()` and bit manipulation
//!        - Extend vertically by comparing row bitmasks with single AND operation
//!        - Clear processed faces from mask with bitwise operations
//!
//! ## Performance
//!
//! Pure binary operations process 16 blocks per CPU instruction vs scalar loops.
//! Expected improvement: 5-10x faster mesh generation compared to scalar approach.
//!
//! ## References
//!
//! - Binary Greedy Meshing: https://github.com/cgerikj/binary-greedy-meshing
//! - Bit manipulation: trailing_zeros (TZCNT), population count (POPCNT)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::Instant;

use super::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32, BLOCK_SCALE};

// ═══════════════════════════════════════════════════════════════════════════
// MESHING PERFORMANCE INSTRUMENTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Global mesh statistics - thread-safe atomic counters
pub struct MeshStats {
    /// Total mesh generation time in nanoseconds
    pub total_time_ns: AtomicU64,
    /// Number of chunks meshed
    pub chunks_meshed: AtomicU32,
    /// Total quads generated
    pub total_quads: AtomicU64,
    /// Total vertices generated
    pub total_vertices: AtomicU64,
}

impl MeshStats {
    pub const fn new() -> Self {
        Self {
            total_time_ns: AtomicU64::new(0),
            chunks_meshed: AtomicU32::new(0),
            total_quads: AtomicU64::new(0),
            total_vertices: AtomicU64::new(0),
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.total_time_ns.store(0, Ordering::Relaxed);
        self.chunks_meshed.store(0, Ordering::Relaxed);
        self.total_quads.store(0, Ordering::Relaxed);
        self.total_vertices.store(0, Ordering::Relaxed);
    }

    /// Get average mesh time per chunk in microseconds
    pub fn avg_mesh_time_us(&self) -> f64 {
        let chunks = self.chunks_meshed.load(Ordering::Relaxed);
        if chunks == 0 {
            return 0.0;
        }
        let total_ns = self.total_time_ns.load(Ordering::Relaxed);
        (total_ns as f64 / chunks as f64) / 1000.0
    }

    /// Get total mesh time in milliseconds
    pub fn total_mesh_time_ms(&self) -> f64 {
        self.total_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Get average quads per chunk
    pub fn avg_quads_per_chunk(&self) -> f64 {
        let chunks = self.chunks_meshed.load(Ordering::Relaxed);
        if chunks == 0 {
            return 0.0;
        }
        self.total_quads.load(Ordering::Relaxed) as f64 / chunks as f64
    }

    /// Log current statistics
    pub fn log_stats(&self) {
        let chunks = self.chunks_meshed.load(Ordering::Relaxed);
        let total_ms = self.total_mesh_time_ms();
        let avg_us = self.avg_mesh_time_us();
        let quads = self.total_quads.load(Ordering::Relaxed);
        let verts = self.total_vertices.load(Ordering::Relaxed);

        tracing::info!(
            "MESH STATS: {} chunks in {:.2}ms (avg {:.1}μs/chunk) | {} quads, {} verts",
            chunks, total_ms, avg_us, quads, verts
        );
    }

    /// Log and reset - for periodic reporting
    pub fn log_and_reset(&self) {
        self.log_stats();
        self.reset();
    }
}

/// Global mesh statistics instance
pub static MESH_STATS: MeshStats = MeshStats::new();

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
    /// The 4th byte is used for flags: bit 0 = is_foliage (for wind animation)
    pub fn pack_normal(normal: [f32; 3]) -> u32 {
        let nx = ((normal[0] * 127.0) as i8) as u8;
        let ny = ((normal[1] * 127.0) as i8) as u8;
        let nz = ((normal[2] * 127.0) as i8) as u8;
        u32::from_le_bytes([nx, ny, nz, 0])
    }

    /// Pack a normal vector with foliage flag for wind animation
    pub fn pack_normal_with_flags(normal: [f32; 3], is_foliage: bool) -> u32 {
        let nx = ((normal[0] * 127.0) as i8) as u8;
        let ny = ((normal[1] * 127.0) as i8) as u8;
        let nz = ((normal[2] * 127.0) as i8) as u8;
        let flags = if is_foliage { 1u8 } else { 0u8 };
        u32::from_le_bytes([nx, ny, nz, flags])
    }

    /// Pack a normal vector with all flags
    /// Flags byte layout: bit 0 = is_foliage (wind animation), bit 1 = is_emissive (always bright)
    pub fn pack_normal_all_flags(normal: [f32; 3], is_foliage: bool, is_emissive: bool) -> u32 {
        let nx = ((normal[0] * 127.0) as i8) as u8;
        let ny = ((normal[1] * 127.0) as i8) as u8;
        let nz = ((normal[2] * 127.0) as i8) as u8;
        let mut flags = 0u8;
        if is_foliage { flags |= 1; }
        if is_emissive { flags |= 2; }
        u32::from_le_bytes([nx, ny, nz, flags])
    }

    /// Pack UV coordinates (0.0-1.0) into u16 pair
    pub fn pack_uv(uv: [f32; 2]) -> [u16; 2] {
        [(uv[0] * 65535.0) as u16, (uv[1] * 65535.0) as u16]
    }

    /// Pack AO value (0.0-1.0), tile counts, and light level into u32
    /// Layout: [ao (0-255), tile_u (1-16), tile_v (1-16), light (0-255)]
    /// Light is stored as 0-255 where 0=dark, 255=full brightness (scaled from 0-15)
    pub fn pack_ao_tile(ao: f32, tile_u: u8, tile_v: u8) -> u32 {
        let ao_byte = (ao * 255.0) as u8;
        u32::from_le_bytes([ao_byte, tile_u, tile_v, 0])
    }

    /// Pack AO, tile counts, and light level into u32
    /// Light level is 0-15, scaled to 0-255 for shader
    pub fn pack_ao_tile_light(ao: f32, tile_u: u8, tile_v: u8, light: u8) -> u32 {
        let ao_byte = (ao * 255.0) as u8;
        // Scale light from 0-15 to 0-255 (multiply by 17 gives 0,17,34,...,255)
        let light_byte = light.min(15) * 17;
        u32::from_le_bytes([ao_byte, tile_u, tile_v, light_byte])
    }

    /// Create vertex with tiling and light info (for greedy meshing with lighting)
    pub fn new_tiled_lit(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32, tile_u: u8, tile_v: u8, light: u8) -> Self {
        Self {
            position,
            normal_packed: Self::pack_normal(normal),
            uv: Self::pack_uv(uv),
            ao_tile_packed: Self::pack_ao_tile_light(ao, tile_u, tile_v, light),
        }
    }

    /// Create vertex with tiling, light, and foliage flag (for leaves with wind animation)
    pub fn new_tiled_lit_foliage(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32, tile_u: u8, tile_v: u8, light: u8, is_foliage: bool) -> Self {
        Self {
            position,
            normal_packed: Self::pack_normal_with_flags(normal, is_foliage),
            uv: Self::pack_uv(uv),
            ao_tile_packed: Self::pack_ao_tile_light(ao, tile_u, tile_v, light),
        }
    }

    /// Create vertex with tiling, light, and emissive flag (for torches/glowing blocks visible at distance)
    pub fn new_tiled_lit_emissive(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32, tile_u: u8, tile_v: u8, light: u8, is_emissive: bool) -> Self {
        Self {
            position,
            normal_packed: Self::pack_normal_all_flags(normal, false, is_emissive),
            uv: Self::pack_uv(uv),
            ao_tile_packed: Self::pack_ao_tile_light(ao, tile_u, tile_v, light),
        }
    }

    /// Create vertex with tiling info (for greedy meshing)
    pub fn new_tiled(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32, tile_u: u8, tile_v: u8) -> Self {
        Self::new_tiled_lit(position, normal, uv, ao, tile_u, tile_v, 0)
    }

    /// Create vertex without tiling (single block face)
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32) -> Self {
        Self::new_tiled(position, normal, uv, ao, 1, 1)
    }

    /// Create vertex with light but no tiling
    pub fn new_lit(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], ao: f32, light: u8) -> Self {
        Self::new_tiled_lit(position, normal, uv, ao, 1, 1, light)
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
    /// When this mesh was created (for drop animation)
    pub created_at: std::time::Instant,
    /// Average color of visible surface blocks for impostor rendering [R, G, B, A]
    /// Alpha represents occupancy (0 = empty, 255 = fully dense)
    pub average_color: [u8; 4],
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

/// Information about a face for greedy meshing (used by LOD path)
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceInfo {
    block: Block,
    texture_idx: u32,
}

/// Convert (axis, positive) to face index for texture lookup
/// Returns: 0=top(+Y), 1=bottom(-Y), 2=front(+Z), 3=back(-Z), 4=left(-X), 5=right(+X)
#[inline]
const fn face_index(axis: Axis, positive: bool) -> usize {
    match (axis, positive) {
        (Axis::Y, true) => 0,  // Top
        (Axis::Y, false) => 1, // Bottom
        (Axis::Z, true) => 2,  // Front
        (Axis::Z, false) => 3, // Back
        (Axis::X, false) => 4, // Left
        (Axis::X, true) => 5,  // Right
    }
}

/// Maximum number of texture indices (8x8 atlas = 64 textures)
const MAX_TEXTURES: usize = 256; // Support up to 256 texture types (16x16 atlas)

/// Sample the combined light level at a block position
/// Checks neighbor chunks if position is outside current chunk bounds
fn sample_light_at(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    x: i32,
    y: i32,
    z: i32,
) -> u8 {
    // Check if position is within current chunk
    if x >= 0 && x < CHUNK_SIZE_I32 && y >= 0 && y < CHUNK_SIZE_I32 && z >= 0 && z < CHUNK_SIZE_I32 {
        return chunk.get_combined_light(x as usize, y as usize, z as usize);
    }

    // Position is in a neighbor chunk
    let (neighbor_idx, local_x, local_y, local_z) = if x < 0 {
        (0, (CHUNK_SIZE_I32 - 1) as usize, y as usize, z as usize) // -X neighbor
    } else if x >= CHUNK_SIZE_I32 {
        (1, 0usize, y as usize, z as usize) // +X neighbor
    } else if y < 0 {
        (2, x as usize, (CHUNK_SIZE_I32 - 1) as usize, z as usize) // -Y neighbor
    } else if y >= CHUNK_SIZE_I32 {
        (3, x as usize, 0usize, z as usize) // +Y neighbor
    } else if z < 0 {
        (4, x as usize, y as usize, (CHUNK_SIZE_I32 - 1) as usize) // -Z neighbor
    } else {
        (5, x as usize, y as usize, 0usize) // +Z neighbor
    };

    // Bounds check the local coordinates before accessing
    if local_x >= CHUNK_SIZE || local_y >= CHUNK_SIZE || local_z >= CHUNK_SIZE {
        return 0;
    }

    if let Some(neighbor) = &neighbors[neighbor_idx] {
        neighbor.get_combined_light(local_x, local_y, local_z)
    } else {
        // No neighbor loaded, assume full sky light (outdoor)
        15
    }
}

/// Get the light level for a face at the given position
/// Samples light from the air block adjacent to the face
fn get_face_light(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    x: usize,
    y: usize,
    z: usize,
    axis: Axis,
    positive: bool,
) -> u8 {
    // Calculate the position of the air block adjacent to this face
    let (ax, ay, az) = match (axis, positive) {
        (Axis::X, true) => (x as i32 + 1, y as i32, z as i32),
        (Axis::X, false) => (x as i32 - 1, y as i32, z as i32),
        (Axis::Y, true) => (x as i32, y as i32 + 1, z as i32),
        (Axis::Y, false) => (x as i32, y as i32 - 1, z as i32),
        (Axis::Z, true) => (x as i32, y as i32, z as i32 + 1),
        (Axis::Z, false) => (x as i32, y as i32, z as i32 - 1),
    };

    sample_light_at(chunk, neighbors, ax, ay, az)
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
/// Kept as fallback/reference implementation
#[allow(dead_code)]
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
                if !block.is_solid() || block.is_prop() {
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

    // Compute average color for impostor rendering
    let average_color = compute_chunk_average_color(chunk);

    ChunkMesh {
        pos,
        vertices,
        indices,
        gpu_mesh: None,
        lod: LodLevel::Full,
        created_at: Instant::now(),
        average_color,
    }
}

#[allow(dead_code)]
fn get_tile_uvs(tex_idx: u32) -> ([f32; 2], [f32; 2], [f32; 2], [f32; 2]) {
    let tiles_per_row = crate::constants::TILES_PER_ROW;
    let tile_size = 1.0 / tiles_per_row as f32;
    let inset = 0.001;
    let u0 = (tex_idx % tiles_per_row) as f32 * tile_size + inset;
    let v0 = (tex_idx / tiles_per_row) as f32 * tile_size + inset;
    let u1 = u0 + tile_size - inset * 2.0;
    let v1 = v0 + tile_size - inset * 2.0;
    // Return: bottom-left, bottom-right, top-right, top-left
    ([u0, v1], [u1, v1], [u1, v0], [u0, v0])
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Add a quad with emissive flag for light sources visible at any distance
fn add_quad_emissive(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    positions: [[f32; 3]; 4],
    normal: [f32; 3],
    uvs: [[f32; 2]; 4],
    light: u8,
) {
    let base = vertices.len() as u32;
    for idx in 0..4 {
        vertices.push(ChunkVertex::new_tiled_lit_emissive(
            positions[idx], normal, uvs[idx], 1.0, 1, 1, light, true
        ));
    }
    // Vertices are in order: 0-1-2-3
    // Two CCW triangles when viewed from outside: 0-2-1 and 0-3-2
    indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

/// Generate foliage faces using greedy meshing
/// Merges adjacent same-type foliage faces into larger quads for better performance
/// Since wind animation is disabled, we can safely merge foliage faces
fn add_foliage_blocks(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    world_origin: glam::Vec3,
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
) {
    // Process all 6 face directions using greedy meshing
    for &(axis, positive) in &[
        (Axis::Y, true),   // Top (+Y)
        (Axis::Y, false),  // Bottom (-Y)
        (Axis::Z, true),   // Front (+Z)
        (Axis::Z, false),  // Back (-Z)
        (Axis::X, false),  // Left (-X)
        (Axis::X, true),   // Right (+X)
    ] {
        greedy_mesh_foliage_axis(chunk, neighbors, axis, positive, world_origin, vertices, indices);
    }
}

/// Greedy mesh foliage blocks for one axis direction
/// Similar to binary_greedy_mesh_axis but for foliage with solid-neighbor culling
fn greedy_mesh_foliage_axis(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    axis: Axis,
    positive: bool,
    world_origin: glam::Vec3,
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
) {
    // Determine which neighbor chunk to check for face culling
    let neighbor_idx = match (axis, positive) {
        (Axis::X, false) => 0, // -X neighbor
        (Axis::X, true) => 1,  // +X neighbor
        (Axis::Y, false) => 2, // -Y neighbor
        (Axis::Y, true) => 3,  // +Y neighbor
        (Axis::Z, false) => 4, // -Z neighbor
        (Axis::Z, true) => 5,  // +Z neighbor
    };
    let neighbor_chunk = &neighbors[neighbor_idx];

    // Process each 2D slice perpendicular to the axis
    for d in 0..CHUNK_SIZE {
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: Build foliage occupancy bitmasks
        // Each row is a u16 where bit i = 1 means foliage block at position i
        // ═══════════════════════════════════════════════════════════════════

        let mut foliage_occupancy = [0u16; CHUNK_SIZE];
        let mut block_types = [[Block::Air; CHUNK_SIZE]; CHUNK_SIZE]; // Cache for texture lookup

        for v in 0..CHUNK_SIZE {
            for u in 0..CHUNK_SIZE {
                let (x, y, z) = match axis {
                    Axis::X => (d, v, u),
                    Axis::Y => (u, d, v),
                    Axis::Z => (u, v, d),
                };
                let block = chunk.get_block(x, y, z);
                if block.is_foliage() {
                    foliage_occupancy[v] |= 1 << u;
                    block_types[v][u] = block;
                }
            }
        }

        // Early exit: Skip this slice if no foliage blocks
        let slice_has_foliage = foliage_occupancy.iter().any(|&row| row != 0);
        if !slice_has_foliage {
            continue;
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: Build neighbor solid occupancy (blocks that occlude faces)
        // ═══════════════════════════════════════════════════════════════════

        let mut solid_neighbor_occupancy = [0u16; CHUNK_SIZE];

        if positive {
            // Positive direction: check slice at d+1
            if d + 1 < CHUNK_SIZE {
                // Neighbor slice is within this chunk
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (d + 1, v, u),
                            Axis::Y => (u, d + 1, v),
                            Axis::Z => (u, v, d + 1),
                        };
                        let nb = chunk.get_block(x, y, z);
                        // Any solid block (including foliage) occludes the face
                        if nb.is_solid() {
                            solid_neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            } else if let Some(nc) = neighbor_chunk {
                // Neighbor slice is in adjacent chunk (slice 0)
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (0, v, u),
                            Axis::Y => (u, 0, v),
                            Axis::Z => (u, v, 0),
                        };
                        let nb = nc.get_block(x, y, z);
                        if nb.is_solid() {
                            solid_neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            }
            // No neighbor chunk = all air = all faces visible (occupancy stays 0)
        } else {
            // Negative direction: check slice at d-1
            if d > 0 {
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (d - 1, v, u),
                            Axis::Y => (u, d - 1, v),
                            Axis::Z => (u, v, d - 1),
                        };
                        let nb = chunk.get_block(x, y, z);
                        if nb.is_solid() {
                            solid_neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            } else if let Some(nc) = neighbor_chunk {
                // Neighbor slice is in adjacent chunk (last slice)
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (CHUNK_SIZE - 1, v, u),
                            Axis::Y => (u, CHUNK_SIZE - 1, v),
                            Axis::Z => (u, v, CHUNK_SIZE - 1),
                        };
                        let nb = nc.get_block(x, y, z);
                        if nb.is_solid() {
                            solid_neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: Compute visibility via bitwise AND-NOT
        // A foliage face is visible when: foliage AND neighbor is NOT solid
        // visible = foliage & !solid_neighbor
        // ═══════════════════════════════════════════════════════════════════

        let mut visible_mask = [0u16; CHUNK_SIZE];
        for v in 0..CHUNK_SIZE {
            visible_mask[v] = foliage_occupancy[v] & !solid_neighbor_occupancy[v];
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: Group visible faces by texture index
        // ═══════════════════════════════════════════════════════════════════

        let mut texture_masks = [[0u16; CHUNK_SIZE]; MAX_TEXTURES];
        let mut textures_present_list: Vec<usize> = Vec::new();

        for v in 0..CHUNK_SIZE {
            let mut row_bits = visible_mask[v];
            while row_bits != 0 {
                let u = row_bits.trailing_zeros() as usize;
                row_bits &= row_bits - 1; // Clear lowest bit

                // Get texture index for this foliage block
                let tex_idx = block_types[v][u].texture_indices()[0] as usize;

                if texture_masks[tex_idx][v] == 0 && !textures_present_list.contains(&tex_idx) {
                    textures_present_list.push(tex_idx);
                }
                texture_masks[tex_idx][v] |= 1 << u;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 5: Binary Greedy Merge for each texture
        // ═══════════════════════════════════════════════════════════════════

        for &tex_idx in &textures_present_list {
            let tex_mask = &mut texture_masks[tex_idx];

            for start_v in 0..CHUNK_SIZE {
                while tex_mask[start_v] != 0 {
                    // Step 1: Find horizontal run start
                    let start_u = tex_mask[start_v].trailing_zeros() as usize;

                    // Step 2: Find horizontal run length
                    let shifted = tex_mask[start_v] >> start_u;
                    let width = (!shifted).trailing_zeros() as usize;
                    let width = width.min(CHUNK_SIZE - start_u);

                    // Step 3: Clear this run from current row
                    let run_mask: u16 = (((1u32 << width) - 1) << start_u) as u16;
                    tex_mask[start_v] &= !run_mask;

                    // Step 4: Extend vertically
                    let mut height = 1usize;
                    for v in (start_v + 1)..CHUNK_SIZE {
                        if (tex_mask[v] & run_mask) == run_mask {
                            tex_mask[v] &= !run_mask;
                            height += 1;
                        } else {
                            break;
                        }
                    }

                    // Step 5: Sample light at quad corners
                    let corners = [
                        (start_u, start_v),
                        (start_u + width - 1, start_v),
                        (start_u, start_v + height - 1),
                        (start_u + width - 1, start_v + height - 1),
                    ];
                    let light = corners.iter().map(|&(u, v)| {
                        let (bx, by, bz) = match axis {
                            Axis::X => (d, v, u),
                            Axis::Y => (u, d, v),
                            Axis::Z => (u, v, d),
                        };
                        get_face_light(chunk, neighbors, bx, by, bz, axis, positive)
                    }).max().unwrap_or(0);

                    // Calculate AO for the 4 corners
                    let ao_values = calculate_greedy_ao(
                        chunk, neighbors, axis, positive,
                        d, start_u, start_v, width, height,
                    );

                    // Emit greedy quad with foliage flag
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
                        tex_idx as u32,
                        light,
                        ao_values,
                        true, // is_foliage = true
                    );
                }
            }
        }
    }
}

/// Generate cube faces for emissive blocks (torches, uranium) with all 6 faces rendered
/// Emissive blocks are visible at any distance regardless of lighting
fn add_emissive_blocks(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    world_origin: glam::Vec3,
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
) {
    let s = BLOCK_SCALE;

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = chunk.get_block(x, y, z);
                if !block.is_emissive() {
                    continue;
                }

                let bx = world_origin.x + x as f32 * s;
                let by = world_origin.y + y as f32 * s;
                let bz = world_origin.z + z as f32 * s;

                // Get texture UVs with half-texel inset
                let tex_idx = block.texture_indices()[0];
                let tiles_per_row = crate::constants::TILES_PER_ROW;
                let tile_size = 1.0 / tiles_per_row as f32;
                let inset = 0.001;
                let u0 = (tex_idx % tiles_per_row) as f32 * tile_size + inset;
                let v0 = (tex_idx / tiles_per_row) as f32 * tile_size + inset;
                let u1 = u0 + tile_size - inset * 2.0;
                let v1 = v0 + tile_size - inset * 2.0;
                let uv0 = [u0, v1]; // bottom-left
                let uv1 = [u1, v1]; // bottom-right
                let uv2 = [u1, v0]; // top-right
                let uv3 = [u0, v0]; // top-left

                // Get block light (emissive blocks are their own light source)
                let emission = block.light_emission();

                // Top face (+Y)
                {
                    let light = get_face_light(chunk, neighbors, x, y, z, Axis::Y, true).max(emission);
                    add_quad_emissive(vertices, indices,
                        [
                            [bx,     by + s, bz],
                            [bx + s, by + s, bz],
                            [bx + s, by + s, bz + s],
                            [bx,     by + s, bz + s],
                        ],
                        [0.0, 1.0, 0.0],
                        [uv0, uv1, uv2, uv3],
                        light,
                    );
                }

                // Bottom face (-Y)
                {
                    let light = get_face_light(chunk, neighbors, x, y, z, Axis::Y, false).max(emission);
                    add_quad_emissive(vertices, indices,
                        [
                            [bx,     by, bz + s],
                            [bx + s, by, bz + s],
                            [bx + s, by, bz],
                            [bx,     by, bz],
                        ],
                        [0.0, -1.0, 0.0],
                        [uv0, uv1, uv2, uv3],
                        light,
                    );
                }

                // Front face (+Z)
                {
                    let light = get_face_light(chunk, neighbors, x, y, z, Axis::Z, true).max(emission);
                    add_quad_emissive(vertices, indices,
                        [
                            [bx + s, by,     bz + s],
                            [bx,     by,     bz + s],
                            [bx,     by + s, bz + s],
                            [bx + s, by + s, bz + s],
                        ],
                        [0.0, 0.0, 1.0],
                        [uv0, uv1, uv2, uv3],
                        light,
                    );
                }

                // Back face (-Z)
                {
                    let light = get_face_light(chunk, neighbors, x, y, z, Axis::Z, false).max(emission);
                    add_quad_emissive(vertices, indices,
                        [
                            [bx,     by,     bz],
                            [bx + s, by,     bz],
                            [bx + s, by + s, bz],
                            [bx,     by + s, bz],
                        ],
                        [0.0, 0.0, -1.0],
                        [uv0, uv1, uv2, uv3],
                        light,
                    );
                }

                // Left face (-X)
                {
                    let light = get_face_light(chunk, neighbors, x, y, z, Axis::X, false).max(emission);
                    add_quad_emissive(vertices, indices,
                        [
                            [bx, by,     bz],
                            [bx, by + s, bz],
                            [bx, by + s, bz + s],
                            [bx, by,     bz + s],
                        ],
                        [-1.0, 0.0, 0.0],
                        [uv0, uv3, uv2, uv1],  // Reordered for left face
                        light,
                    );
                }

                // Right face (+X)
                {
                    let light = get_face_light(chunk, neighbors, x, y, z, Axis::X, true).max(emission);
                    add_quad_emissive(vertices, indices,
                        [
                            [bx + s, by,     bz + s],
                            [bx + s, by + s, bz + s],
                            [bx + s, by + s, bz],
                            [bx + s, by,     bz],
                        ],
                        [1.0, 0.0, 0.0],
                        [uv0, uv3, uv2, uv1],  // Reordered for right face
                        light,
                    );
                }
            }
        }
    }
}

/// Build mesh for a chunk with specified LOD level
pub fn build_chunk_mesh_lod(
    pos: ChunkPos,
    chunk: &Arc<Chunk>,
    neighbors: &[Option<Arc<Chunk>>; 6],
    lod: LodLevel,
) -> ChunkMesh {
    let start_time = Instant::now();

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let world_origin = pos.world_origin();
    let step = lod as usize;

    // ═══════════════════════════════════════════════════════════════════════════
    // EARLY EXIT: Skip main meshing for chunks with no solid blocks
    // The binary greedy mesher only processes solid non-foliage blocks, so if
    // solid_count == 0 we can skip that expensive loop entirely.
    // Foliage and emissive handlers still run (they check their own conditions).
    // ═══════════════════════════════════════════════════════════════════════════
    let has_solid_blocks = chunk.solid_count() > 0;

    if has_solid_blocks {
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
    }

    // Add foliage blocks (leaves) separately - they render all 6 faces without neighbor culling
    // Only at full detail - at LOD levels, leaves are skipped entirely for performance
    if step == 1 {
        add_foliage_blocks(chunk, neighbors, world_origin, &mut vertices, &mut indices);
    }

    // Add emissive blocks (torches, uranium) separately - they need to be visible at any distance
    // Added at ALL LOD levels so light sources are visible from far away
    add_emissive_blocks(chunk, neighbors, world_origin, &mut vertices, &mut indices);

    // Compute average color for impostor rendering
    let average_color = compute_chunk_average_color(chunk);

    // Record timing and statistics
    let elapsed_ns = start_time.elapsed().as_nanos() as u64;
    let num_quads = (indices.len() / 6) as u64; // 6 indices per quad (2 triangles)
    let num_verts = vertices.len() as u64;

    MESH_STATS.total_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
    MESH_STATS.chunks_meshed.fetch_add(1, Ordering::Relaxed);
    MESH_STATS.total_quads.fetch_add(num_quads, Ordering::Relaxed);
    MESH_STATS.total_vertices.fetch_add(num_verts, Ordering::Relaxed);

    ChunkMesh {
        pos,
        vertices,
        indices,
        gpu_mesh: None,
        lod,
        created_at: Instant::now(),
        average_color,
    }
}

/// Compute average color for a chunk by sampling surface blocks
/// Returns [R, G, B, A] where A represents occupancy (solid block density)
fn compute_chunk_average_color(chunk: &Chunk) -> [u8; 4] {
    let mut color_sum: [u32; 3] = [0, 0, 0];
    let mut solid_count: u32 = 0;
    let mut surface_count: u32 = 0;

    // Sample blocks, weighting surface blocks more heavily
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = chunk.get_block(x, y, z);
                if block.is_solid() && !matches!(block, Block::Air | Block::Water) {
                    solid_count += 1;

                    // Check if this is a surface block (has at least one air/transparent neighbor)
                    let is_surface =
                        (x == 0 || !chunk.get_block(x - 1, y, z).is_solid()) ||
                        (x == CHUNK_SIZE - 1 || !chunk.get_block(x + 1, y, z).is_solid()) ||
                        (y == 0 || !chunk.get_block(x, y - 1, z).is_solid()) ||
                        (y == CHUNK_SIZE - 1 || !chunk.get_block(x, y + 1, z).is_solid()) ||
                        (z == 0 || !chunk.get_block(x, y, z - 1).is_solid()) ||
                        (z == CHUNK_SIZE - 1 || !chunk.get_block(x, y, z + 1).is_solid());

                    if is_surface {
                        let color = block.dominant_color();
                        // Weight surface blocks 3x for more accurate visual representation
                        color_sum[0] += color[0] as u32 * 3;
                        color_sum[1] += color[1] as u32 * 3;
                        color_sum[2] += color[2] as u32 * 3;
                        surface_count += 3;
                    } else {
                        // Include interior blocks with lower weight for underground chunks
                        let color = block.dominant_color();
                        color_sum[0] += color[0] as u32;
                        color_sum[1] += color[1] as u32;
                        color_sum[2] += color[2] as u32;
                        surface_count += 1;
                    }
                }
            }
        }
    }

    if surface_count == 0 {
        // Empty chunk - return transparent
        return [0, 0, 0, 0];
    }

    // Calculate average color
    let avg_r = (color_sum[0] / surface_count).min(255) as u8;
    let avg_g = (color_sum[1] / surface_count).min(255) as u8;
    let avg_b = (color_sum[2] / surface_count).min(255) as u8;

    // Alpha represents occupancy: ratio of solid blocks to total volume
    // Scale to 0-255 range, with minimum of 32 for any non-empty chunk
    let total_blocks = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as u32;
    let occupancy = ((solid_count * 255) / total_blocks).max(32).min(255) as u8;

    [avg_r, avg_g, avg_b, occupancy]
}

/// Pure Binary Greedy Meshing for Full Detail
///
/// This implementation uses bitwise operations throughout the entire pipeline:
/// 1. Occupancy masks built with bit-setting operations
/// 2. Visibility computed via `current & !neighbor` (single op per row)
/// 3. Faces grouped into per-texture bitmasks
/// 4. Greedy merge using `trailing_zeros()` for run extraction
/// 5. Vertical extension via bitmask comparison (single AND per row)
///
/// The scalar "processed" array and nested loops are eliminated entirely.
/// All 16 blocks in a row are processed with single CPU instructions.
fn binary_greedy_mesh_axis(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    axis: Axis,
    positive: bool,
    world_origin: glam::Vec3,
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
) {
    // Determine which neighbor chunk to check for face culling
    let neighbor_idx = match (axis, positive) {
        (Axis::X, false) => 0, // -X neighbor
        (Axis::X, true) => 1,  // +X neighbor
        (Axis::Y, false) => 2, // -Y neighbor
        (Axis::Y, true) => 3,  // +Y neighbor
        (Axis::Z, false) => 4, // -Z neighbor
        (Axis::Z, true) => 5,  // +Z neighbor
    };
    let neighbor_chunk = &neighbors[neighbor_idx];
    let face_idx = face_index(axis, positive);

    // Process each 2D slice perpendicular to the axis
    for d in 0..CHUNK_SIZE {
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: Build occupancy bitmasks
        // Each row is a u16 where bit i = 1 means solid block at position i
        // ═══════════════════════════════════════════════════════════════════

        let mut current_occupancy = [0u16; CHUNK_SIZE];
        let mut block_types = [[Block::Air; CHUNK_SIZE]; CHUNK_SIZE]; // Cache for texture lookup

        for v in 0..CHUNK_SIZE {
            for u in 0..CHUNK_SIZE {
                let (x, y, z) = match axis {
                    Axis::X => (d, v, u),
                    Axis::Y => (u, d, v),
                    Axis::Z => (u, v, d),
                };
                let block = chunk.get_block(x, y, z);
                // Skip cross-billboard, foliage, and prop blocks - they're rendered separately
                if block.is_solid() && !block.is_cross_billboard() && !block.is_foliage() && !block.is_prop() {
                    current_occupancy[v] |= 1 << u;
                    block_types[v][u] = block;
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // EARLY EXIT: Skip this slice if no solid blocks (all occupancy is 0)
        // This avoids building neighbor occupancy and visibility masks
        // ═══════════════════════════════════════════════════════════════════
        let slice_has_blocks = current_occupancy.iter().any(|&row| row != 0);
        if !slice_has_blocks {
            continue;
        }

        // Build neighbor occupancy (blocks that would occlude our faces)
        let mut neighbor_occupancy = [0u16; CHUNK_SIZE];

        if positive {
            // Positive direction: check slice at d+1
            if d + 1 < CHUNK_SIZE {
                // Neighbor slice is within this chunk
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (d + 1, v, u),
                            Axis::Y => (u, d + 1, v),
                            Axis::Z => (u, v, d + 1),
                        };
                        let nb = chunk.get_block(x, y, z);
                        // Foliage and props don't occlude - we want to see faces behind them
                        if nb.is_solid() && !nb.is_foliage() && !nb.is_prop() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            } else if let Some(nc) = neighbor_chunk {
                // Neighbor slice is in adjacent chunk (slice 0)
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (0, v, u),
                            Axis::Y => (u, 0, v),
                            Axis::Z => (u, v, 0),
                        };
                        let nb = nc.get_block(x, y, z);
                        if nb.is_solid() && !nb.is_foliage() && !nb.is_prop() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            }
            // No neighbor chunk = all air = all faces visible (occupancy stays 0)
        } else {
            // Negative direction: check slice at d-1
            if d > 0 {
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (d - 1, v, u),
                            Axis::Y => (u, d - 1, v),
                            Axis::Z => (u, v, d - 1),
                        };
                        let nb = chunk.get_block(x, y, z);
                        if nb.is_solid() && !nb.is_foliage() && !nb.is_prop() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            } else if let Some(nc) = neighbor_chunk {
                // Neighbor slice is in adjacent chunk (last slice)
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let (x, y, z) = match axis {
                            Axis::X => (CHUNK_SIZE - 1, v, u),
                            Axis::Y => (u, CHUNK_SIZE - 1, v),
                            Axis::Z => (u, v, CHUNK_SIZE - 1),
                        };
                        let nb = nc.get_block(x, y, z);
                        if nb.is_solid() && !nb.is_foliage() && !nb.is_prop() {
                            neighbor_occupancy[v] |= 1 << u;
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: Compute visibility via bitwise AND-NOT
        // A face is visible when: current is solid AND neighbor is air
        // visible = current & !neighbor (processes 16 blocks in one op)
        // ═══════════════════════════════════════════════════════════════════

        let mut visible_mask = [0u16; CHUNK_SIZE];
        for v in 0..CHUNK_SIZE {
            visible_mask[v] = current_occupancy[v] & !neighbor_occupancy[v];
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: Group visible faces by texture index
        // Creates per-texture bitmasks for binary greedy merge
        // ═══════════════════════════════════════════════════════════════════

        // Per-texture visibility masks: texture_masks[tex_idx][row] = bitmask
        let mut texture_masks = [[0u16; CHUNK_SIZE]; MAX_TEXTURES];
        let mut textures_present: Vec<usize> = Vec::new(); // Tracks which texture indices have faces

        for v in 0..CHUNK_SIZE {
            let mut row_bits = visible_mask[v];
            while row_bits != 0 {
                // Extract lowest set bit position using trailing_zeros (TZCNT instruction)
                let u = row_bits.trailing_zeros() as usize;

                // Clear this bit: x & (x-1) clears lowest set bit (faster than x & !(1<<u))
                row_bits &= row_bits - 1;

                // Get texture index for this block's face
                let tex_idx = block_types[v][u].face_texture(face_idx) as usize;

                // Set bit in that texture's mask
                if texture_masks[tex_idx][v] == 0 {
                    // First bit for this texture in this row — check if texture is new
                    if !textures_present.contains(&tex_idx) {
                        textures_present.push(tex_idx);
                    }
                }
                texture_masks[tex_idx][v] |= 1 << u;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3.5: Compute AO boundary mask
        // For each visible face, check if it has any AO occluders (blocks
        // adjacent above the face). Blocks WITH occluders can't merge with
        // blocks WITHOUT — this prevents AO gradient smearing across large
        // greedy quads. Uses same is_block_solid_at() as calculate_ao().
        // ═══════════════════════════════════════════════════════════════════

        let mut ao_occluded_mask = [0u16; CHUNK_SIZE];
        for v in 0..CHUNK_SIZE {
            let mut row_bits = visible_mask[v];
            while row_bits != 0 {
                let u = row_bits.trailing_zeros() as usize;
                row_bits &= row_bits - 1;

                let (x, y, z) = match axis {
                    Axis::X => (d as i32, v as i32, u as i32),
                    Axis::Y => (u as i32, d as i32, v as i32),
                    Axis::Z => (u as i32, v as i32, d as i32),
                };

                // Check for AO occluders on the face above (along the positive face normal).
                // For +Y face at (x,y,z): check y+1 neighbors.
                // For -Y face: check y-1 neighbors. Etc.
                // We check the 4 edge-adjacent blocks (not corners) for a fast approximation.
                let (n0, n1, n2, n3) = match (axis, positive) {
                    (Axis::Y, true) => (
                        is_block_solid_at(chunk, neighbors, x - 1, y + 1, z),
                        is_block_solid_at(chunk, neighbors, x + 1, y + 1, z),
                        is_block_solid_at(chunk, neighbors, x, y + 1, z - 1),
                        is_block_solid_at(chunk, neighbors, x, y + 1, z + 1),
                    ),
                    (Axis::Y, false) => (
                        is_block_solid_at(chunk, neighbors, x - 1, y - 1, z),
                        is_block_solid_at(chunk, neighbors, x + 1, y - 1, z),
                        is_block_solid_at(chunk, neighbors, x, y - 1, z - 1),
                        is_block_solid_at(chunk, neighbors, x, y - 1, z + 1),
                    ),
                    (Axis::X, true) => (
                        is_block_solid_at(chunk, neighbors, x + 1, y - 1, z),
                        is_block_solid_at(chunk, neighbors, x + 1, y + 1, z),
                        is_block_solid_at(chunk, neighbors, x + 1, y, z - 1),
                        is_block_solid_at(chunk, neighbors, x + 1, y, z + 1),
                    ),
                    (Axis::X, false) => (
                        is_block_solid_at(chunk, neighbors, x - 1, y - 1, z),
                        is_block_solid_at(chunk, neighbors, x - 1, y + 1, z),
                        is_block_solid_at(chunk, neighbors, x - 1, y, z - 1),
                        is_block_solid_at(chunk, neighbors, x - 1, y, z + 1),
                    ),
                    (Axis::Z, true) => (
                        is_block_solid_at(chunk, neighbors, x - 1, y, z + 1),
                        is_block_solid_at(chunk, neighbors, x + 1, y, z + 1),
                        is_block_solid_at(chunk, neighbors, x, y - 1, z + 1),
                        is_block_solid_at(chunk, neighbors, x, y + 1, z + 1),
                    ),
                    (Axis::Z, false) => (
                        is_block_solid_at(chunk, neighbors, x - 1, y, z - 1),
                        is_block_solid_at(chunk, neighbors, x + 1, y, z - 1),
                        is_block_solid_at(chunk, neighbors, x, y - 1, z - 1),
                        is_block_solid_at(chunk, neighbors, x, y + 1, z - 1),
                    ),
                };

                if n0 || n1 || n2 || n3 {
                    ao_occluded_mask[v] |= 1 << u;
                }
            }
        }

        // Split texture masks by AO class: "clean" (no occluders) vs "occluded".
        // This prevents the greedy merger from merging across AO boundaries,
        // eliminating the gradient smearing that causes dark blobs on terrain.
        let mut texture_masks_occluded = [[0u16; CHUNK_SIZE]; MAX_TEXTURES];
        let mut textures_present_occluded: Vec<usize> = Vec::new();
        for &tex_idx in &textures_present {
            for v in 0..CHUNK_SIZE {
                let occluded_bits = texture_masks[tex_idx][v] & ao_occluded_mask[v];
                if occluded_bits != 0 {
                    texture_masks_occluded[tex_idx][v] = occluded_bits;
                    if !textures_present_occluded.contains(&tex_idx) {
                        textures_present_occluded.push(tex_idx);
                    }
                }
                texture_masks[tex_idx][v] &= !ao_occluded_mask[v]; // Clean only
            }
        }
        // Recompute textures_present for clean masks (some textures may now be empty)
        textures_present.retain(|&tex_idx| {
            (0..CHUNK_SIZE).any(|v| texture_masks[tex_idx][v] != 0)
        });

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: Binary Greedy Merge for each texture
        //
        // For each row:
        //   While row has set bits:
        //     1. Find first set bit (start_u) via trailing_zeros
        //     2. Find run length: shift right, invert, trailing_zeros gives consecutive 1s
        //     3. Create run_mask and clear from current row
        //     4. Extend down: check if subsequent rows have same run via (row & mask) == mask
        //     5. Emit quad
        //
        // All operations work on 16 bits simultaneously.
        // ═══════════════════════════════════════════════════════════════════

        // --- Pass 1: Clean faces (no AO occluders) ---
        for tex_idx in textures_present {
            let tex_mask = &mut texture_masks[tex_idx];

            // Process all faces for this texture using binary operations
            for start_v in 0..CHUNK_SIZE {
                while tex_mask[start_v] != 0 {
                    // ─────────────────────────────────────────────────────────
                    // Step 1: Find horizontal run start position
                    // trailing_zeros returns index of first set bit (LSB)
                    // ─────────────────────────────────────────────────────────
                    let start_u = tex_mask[start_v].trailing_zeros() as usize;

                    // ─────────────────────────────────────────────────────────
                    // Step 2: Find horizontal run length (consecutive 1s)
                    //
                    // Algorithm:
                    //   shifted = mask >> start_u  (align run to bit 0)
                    //   inverted = !shifted        (1s become 0s, 0s become 1s)
                    //   width = trailing_zeros(inverted)  (count leading 1s of original)
                    //
                    // Example: mask = 0b0111100, start_u = 2
                    //   shifted  = 0b0001111
                    //   inverted = 0b1110000
                    //   trailing_zeros = 4 (the run length)
                    // ─────────────────────────────────────────────────────────
                    let shifted = tex_mask[start_v] >> start_u;
                    let width = (!shifted).trailing_zeros() as usize;

                    // Clamp width to chunk bounds (handles case where run extends to bit 15)
                    let width = width.min(CHUNK_SIZE - start_u);

                    // ─────────────────────────────────────────────────────────
                    // Step 3: Create bitmask for this run and clear from row
                    //
                    // run_mask = ((1 << width) - 1) << start_u
                    // Example: width=4, start_u=2 → run_mask = 0b0111100
                    // ─────────────────────────────────────────────────────────
                    let run_mask: u16 = (((1u32 << width) - 1) << start_u) as u16;
                    tex_mask[start_v] &= !run_mask; // Clear processed bits

                    // ─────────────────────────────────────────────────────────
                    // Step 4: Extend vertically using bitmask comparison
                    //
                    // Check if subsequent rows have the COMPLETE run:
                    //   (row & run_mask) == run_mask
                    //
                    // This single AND+CMP checks all 'width' positions at once.
                    // If true, clear those bits and increment height.
                    // ─────────────────────────────────────────────────────────
                    let mut height = 1usize;
                    for v in (start_v + 1)..CHUNK_SIZE {
                        if (tex_mask[v] & run_mask) == run_mask {
                            // This row has the complete run - consume it
                            tex_mask[v] &= !run_mask;
                            height += 1;
                        } else {
                            // Run broken - stop vertical extension
                            break;
                        }
                    }

                    // ─────────────────────────────────────────────────────────
                    // Step 5: Sample light at quad corners and emit
                    // ─────────────────────────────────────────────────────────
                    let corners = [
                        (start_u, start_v),
                        (start_u + width - 1, start_v),
                        (start_u, start_v + height - 1),
                        (start_u + width - 1, start_v + height - 1),
                    ];
                    let light = corners.iter().map(|&(u, v)| {
                        let (bx, by, bz) = match axis {
                            Axis::X => (d, v, u),
                            Axis::Y => (u, d, v),
                            Axis::Z => (u, v, d),
                        };
                        get_face_light(chunk, neighbors, bx, by, bz, axis, positive)
                    }).max().unwrap_or(0);

                    // Calculate AO for the 4 corners of the merged quad
                    let ao_values = calculate_greedy_ao(
                        chunk, neighbors, axis, positive,
                        d, start_u, start_v, width, height,
                    );

                    // Leaves (texture index 9) are foliage - enable wind animation
                    let is_foliage = tex_idx == 9;

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
                        tex_idx as u32,
                        light,
                        ao_values,
                        is_foliage,
                    );
                }
            }
        }

        // --- Pass 2: Occluded faces (have AO occluders) ---
        // Same greedy merge algorithm, applied to the occluded sub-masks
        for tex_idx in textures_present_occluded {
            let tex_mask = &mut texture_masks_occluded[tex_idx];

            for start_v in 0..CHUNK_SIZE {
                while tex_mask[start_v] != 0 {
                    let start_u = tex_mask[start_v].trailing_zeros() as usize;
                    let shifted = tex_mask[start_v] >> start_u;
                    let width = ((!shifted).trailing_zeros() as usize).min(CHUNK_SIZE - start_u);
                    let run_mask: u16 = (((1u32 << width) - 1) << start_u) as u16;
                    tex_mask[start_v] &= !run_mask;

                    let mut height = 1usize;
                    for v in (start_v + 1)..CHUNK_SIZE {
                        if (tex_mask[v] & run_mask) == run_mask {
                            tex_mask[v] &= !run_mask;
                            height += 1;
                        } else {
                            break;
                        }
                    }

                    let corners = [
                        (start_u, start_v),
                        (start_u + width - 1, start_v),
                        (start_u, start_v + height - 1),
                        (start_u + width - 1, start_v + height - 1),
                    ];
                    let light = corners.iter().map(|&(u, v)| {
                        let (bx, by, bz) = match axis {
                            Axis::X => (d, v, u),
                            Axis::Y => (u, d, v),
                            Axis::Z => (u, v, d),
                        };
                        get_face_light(chunk, neighbors, bx, by, bz, axis, positive)
                    }).max().unwrap_or(0);

                    let ao_values = calculate_greedy_ao(
                        chunk, neighbors, axis, positive,
                        d, start_u, start_v, width, height,
                    );

                    let is_foliage = tex_idx == 9;

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
                        tex_idx as u32,
                        light,
                        ao_values,
                        is_foliage,
                    );
                }
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
                            // Skip non-solid, cross-billboard, foliage, and prop blocks (rendered separately)
                            if !block.is_solid() || block.is_cross_billboard() || block.is_foliage() || block.is_prop() {
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

                // Sample light from all 4 corners of the merged quad and take the max
                // Coordinates are scaled by LOD step
                let corners = [
                    (start_u * step, start_v * step),
                    ((start_u + width - 1) * step, start_v * step),
                    (start_u * step, (start_v + height - 1) * step),
                    ((start_u + width - 1) * step, (start_v + height - 1) * step),
                ];
                let light = corners.iter().map(|&(u, v)| {
                    let (bx, by, bz) = match axis {
                        Axis::X => (d_emit, v, u),
                        Axis::Y => (u, d_emit, v),
                        Axis::Z => (u, v, d_emit),
                    };
                    get_face_light(chunk, neighbors, bx, by, bz, axis, positive)
                }).max().unwrap_or(0);

                // Calculate AO for LOD quads (scaled positions)
                let ao_values = calculate_greedy_ao(
                    chunk, neighbors, axis, positive,
                    d_emit, start_u * step, start_v * step, width * step, height * step,
                );

                // Leaves (texture index 9) are foliage - enable wind animation
                let is_foliage = face_info.texture_idx == 9;

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
                    light,
                    ao_values,
                    is_foliage,
                );
            }
        }
    }
}

/// Check if face is visible for LOD (accounts for larger step sizes)
#[allow(dead_code)]
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

/// Check if a face should be visible (neighbor is transparent or prop)
fn is_face_visible(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    (nx, ny, nz): (i32, i32, i32),
) -> bool {
    // Check if neighbor is in current chunk
    if Chunk::in_bounds(nx, ny, nz) {
        let neighbor_block = chunk.get_block(nx as usize, ny as usize, nz as usize);
        // Props don't occlude faces (they're rendered separately and aren't full blocks)
        return neighbor_block.is_transparent() || neighbor_block.is_prop();
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
        neighbor_block.is_transparent() || neighbor_block.is_prop()
    } else {
        // No neighbor chunk loaded - render the face
        true
    }
}

/// Get whether a block at the given position is solid (for AO calculation)
/// Handles cross-chunk boundary lookups
#[allow(dead_code)]
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
///
/// Uses a smooth cubic falloff for professional-quality AO transitions
fn calculate_ao(side1: bool, side2: bool, corner: bool) -> f32 {
    // If both sides are solid, corner is considered occluded (prevents light leak)
    let effective_corner = if side1 && side2 { true } else { corner };

    let occluders = (side1 as u8) + (side2 as u8) + (effective_corner as u8);

    // Per-vertex AO provides tight contact shadows at block corners.
    // Moderate values — stacks with voxel AO + shadows, so don't go too dark.
    let ao_table: [f32; 4] = [
        1.0,    // 0 occluders: fully exposed
        0.88,   // 1 occluder: subtle edge hint
        0.72,   // 2 occluders: visible crease
        0.55,   // 3 occluders: inner corner
    ];

    ao_table[occluders as usize]
}

/// Calculate AO for 4 corners of a greedy-meshed quad
/// Uses the same vertex ordering as emit_greedy_quad
/// All 6 face directions are now properly handled for professional-quality AO
fn calculate_greedy_ao(
    chunk: &Chunk,
    neighbors: &[Option<Arc<Chunk>>; 6],
    axis: Axis,
    positive: bool,
    slice: usize,
    start_u: usize,
    start_v: usize,
    width: usize,
    height: usize,
) -> [f32; 4] {
    // For greedy quads, sample AO at the 4 corner block positions
    // The coordinate mapping depends on axis and direction
    let d = slice as i32;

    // Corner positions in (u, v) space - matches emit_greedy_quad vertex order
    let corners_uv = [
        (start_u as i32, start_v as i32),                           // corner 0
        (start_u as i32 + width as i32 - 1, start_v as i32),       // corner 1
        (start_u as i32 + width as i32 - 1, start_v as i32 + height as i32 - 1), // corner 2
        (start_u as i32, start_v as i32 + height as i32 - 1),      // corner 3
    ];

    let mut ao_values = [1.0f32; 4];

    for (i, &(u, v)) in corners_uv.iter().enumerate() {
        // Convert (u, v, d) to (x, y, z) based on axis
        let (x, y, z) = match axis {
            Axis::X => (d, v, u),
            Axis::Y => (u, d, v),
            Axis::Z => (u, v, d),
        };

        // For each corner, check the 3 adjacent blocks that could occlude
        // The adjacency depends on which corner and which face direction
        // All 6 face directions are now fully implemented
        let (side1, side2, corner) = match (axis, positive, i) {
            // ═══════════════════════════════════════════════════════════
            // TOP FACE (+Y) - looking down at the face from above
            // Corners: 0=back-left, 1=back-right, 2=front-right, 3=front-left
            // ═══════════════════════════════════════════════════════════
            (Axis::Y, true, 0) => ( // back-left corner
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x, y + 1, z - 1),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z - 1),
            ),
            (Axis::Y, true, 1) => ( // back-right corner
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x, y + 1, z - 1),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z - 1),
            ),
            (Axis::Y, true, 2) => ( // front-right corner
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x, y + 1, z + 1),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z + 1),
            ),
            (Axis::Y, true, 3) => ( // front-left corner
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x, y + 1, z + 1),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z + 1),
            ),

            // ═══════════════════════════════════════════════════════════
            // BOTTOM FACE (-Y) - looking up at the face from below
            // Corners: 0=front-left, 1=front-right, 2=back-right, 3=back-left
            // ═══════════════════════════════════════════════════════════
            (Axis::Y, false, 0) => ( // front-left
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x, y - 1, z + 1),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z + 1),
            ),
            (Axis::Y, false, 1) => ( // front-right
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x, y - 1, z + 1),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z + 1),
            ),
            (Axis::Y, false, 2) => ( // back-right
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x, y - 1, z - 1),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z - 1),
            ),
            (Axis::Y, false, 3) => ( // back-left
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x, y - 1, z - 1),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z - 1),
            ),

            // ═══════════════════════════════════════════════════════════
            // FRONT FACE (+Z) - looking at the face from +Z direction
            // Corners: 0=bottom-right, 1=bottom-left, 2=top-left, 3=top-right
            // ═══════════════════════════════════════════════════════════
            (Axis::Z, true, 0) => ( // bottom-right
                is_block_solid_at(chunk, neighbors, x + 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x, y - 1, z + 1),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z + 1),
            ),
            (Axis::Z, true, 1) => ( // bottom-left
                is_block_solid_at(chunk, neighbors, x - 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x, y - 1, z + 1),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z + 1),
            ),
            (Axis::Z, true, 2) => ( // top-left
                is_block_solid_at(chunk, neighbors, x - 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x, y + 1, z + 1),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z + 1),
            ),
            (Axis::Z, true, 3) => ( // top-right
                is_block_solid_at(chunk, neighbors, x + 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x, y + 1, z + 1),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z + 1),
            ),

            // ═══════════════════════════════════════════════════════════
            // BACK FACE (-Z) - looking at the face from -Z direction
            // Corners: 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
            // ═══════════════════════════════════════════════════════════
            (Axis::Z, false, 0) => ( // bottom-left
                is_block_solid_at(chunk, neighbors, x - 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x, y - 1, z - 1),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z - 1),
            ),
            (Axis::Z, false, 1) => ( // bottom-right
                is_block_solid_at(chunk, neighbors, x + 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x, y - 1, z - 1),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z - 1),
            ),
            (Axis::Z, false, 2) => ( // top-right
                is_block_solid_at(chunk, neighbors, x + 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x, y + 1, z - 1),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z - 1),
            ),
            (Axis::Z, false, 3) => ( // top-left
                is_block_solid_at(chunk, neighbors, x - 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x, y + 1, z - 1),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z - 1),
            ),

            // ═══════════════════════════════════════════════════════════
            // LEFT FACE (-X) - looking at the face from -X direction
            // Corners: 0=bottom-back, 1=top-back, 2=top-front, 3=bottom-front
            // ═══════════════════════════════════════════════════════════
            (Axis::X, false, 0) => ( // bottom-back
                is_block_solid_at(chunk, neighbors, x - 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z - 1),
            ),
            (Axis::X, false, 1) => ( // top-back
                is_block_solid_at(chunk, neighbors, x - 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z - 1),
            ),
            (Axis::X, false, 2) => ( // top-front
                is_block_solid_at(chunk, neighbors, x - 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x - 1, y + 1, z + 1),
            ),
            (Axis::X, false, 3) => ( // bottom-front
                is_block_solid_at(chunk, neighbors, x - 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x - 1, y - 1, z + 1),
            ),

            // ═══════════════════════════════════════════════════════════
            // RIGHT FACE (+X) - looking at the face from +X direction
            // Corners: 0=bottom-front, 1=top-front, 2=top-back, 3=bottom-back
            // ═══════════════════════════════════════════════════════════
            (Axis::X, true, 0) => ( // bottom-front
                is_block_solid_at(chunk, neighbors, x + 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z + 1),
            ),
            (Axis::X, true, 1) => ( // top-front
                is_block_solid_at(chunk, neighbors, x + 1, y, z + 1),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z + 1),
            ),
            (Axis::X, true, 2) => ( // top-back
                is_block_solid_at(chunk, neighbors, x + 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z),
                is_block_solid_at(chunk, neighbors, x + 1, y + 1, z - 1),
            ),
            (Axis::X, true, 3) => ( // bottom-back
                is_block_solid_at(chunk, neighbors, x + 1, y, z - 1),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z),
                is_block_solid_at(chunk, neighbors, x + 1, y - 1, z - 1),
            ),

            // Fallback (should never reach)
            _ => (false, false, false),
        };

        ao_values[i] = calculate_ao(side1, side2, corner);
    }

    ao_values
}

/// Calculate AO values for all 4 corners of a face
/// Returns [ao0, ao1, ao2, ao3] matching vertex order
#[allow(dead_code)]
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
    light_level: u8,
    ao_values: [f32; 4],  // AO for each of the 4 corners
    is_foliage: bool,     // True for leaves - enables wind animation in shader
) {
    let base_index = vertices.len() as u32;

    // Get UVs from texture atlas with half-texel inset to prevent bleeding
    let tiles_per_row = crate::constants::TILES_PER_ROW;
    let tile_size = 1.0 / tiles_per_row as f32;
    let inset = 0.001;
    let uv_u0 = (texture_idx % tiles_per_row) as f32 * tile_size + inset;
    let uv_v0 = (texture_idx / tiles_per_row) as f32 * tile_size + inset;
    let uv_u1 = uv_u0 + tile_size - inset * 2.0;
    let uv_v1 = uv_v0 + tile_size - inset * 2.0;
    // UV corners: bottom-left, bottom-right, top-right, top-left
    let uvs = [[uv_u0, uv_v1], [uv_u1, uv_v1], [uv_u1, uv_v0], [uv_u0, uv_v0]];

    // Tile counts for texture tiling (in blocks)
    // Width/height need to be mapped to tile_u/tile_v based on axis orientation
    let (tile_u, tile_v) = match axis {
        Axis::X => (width as u8, height as u8),  // Left/Right: width=Z, height=Y
        Axis::Y => (width as u8, height as u8),  // Top/Bottom: width=X, height=Z
        Axis::Z => (width as u8, height as u8),  // Front/Back: width=X, height=Y
    };
    let s = BLOCK_SCALE;
    let ox = world_origin.x;
    let oy = world_origin.y;
    let oz = world_origin.z;

    // Epsilon to expand faces slightly to prevent sub-pixel gaps
    // Must be small enough to avoid visible z-fighting between adjacent faces
    // 0.002 (0.2% of a block) covers gaps without causing visible overlap artifacts
    let e = 0.002;

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

    // Add vertices with tiling, light, per-corner AO, and foliage flag
    for i in 0..4 {
        vertices.push(ChunkVertex::new_tiled_lit_foliage(positions[i], normal, uvs[uv_order[i]], ao_values[i], tile_u, tile_v, light_level, is_foliage));
    }

    // Same index pattern as add_quad: 0-2-1 and 0-3-2
    indices.extend_from_slice(&[
        base_index, base_index + 2, base_index + 1,
        base_index, base_index + 3, base_index + 2,
    ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Create a chunk with a single block at the given position
    fn single_block_chunk(x: usize, y: usize, z: usize, block: Block) -> Arc<Chunk> {
        let mut chunk = Chunk::new();
        chunk.set_block(x, y, z, block);
        Arc::new(chunk)
    }

    /// Create a chunk filled with a single block type
    fn solid_chunk(block: Block) -> Arc<Chunk> {
        Arc::new(Chunk::new_solid(block))
    }

    /// Empty neighbor array (all air)
    fn no_neighbors() -> [Option<Arc<Chunk>>; 6] {
        [None, None, None, None, None, None]
    }

    // ============ Single Block Meshing ============

    #[test]
    fn test_single_block_generates_six_faces() {
        let chunk = single_block_chunk(8, 8, 8, Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Single exposed block should have 6 faces = 6 quads = 24 vertices, 36 indices
        assert_eq!(mesh.vertices.len(), 24, "Single block should have 24 vertices (6 faces × 4 verts)");
        assert_eq!(mesh.indices.len(), 36, "Single block should have 36 indices (6 faces × 6 indices)");
    }

    #[test]
    fn test_single_block_vertex_positions() {
        let chunk = single_block_chunk(0, 0, 0, Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Collect all unique vertex positions
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();

        // A 1x1x1 block at origin should have vertices within [0,0,0] to [BLOCK_SCALE, BLOCK_SCALE, BLOCK_SCALE]
        for pos in &positions {
            assert!(pos[0] >= -0.01 && pos[0] <= BLOCK_SCALE + 0.01, "X position out of range: {}", pos[0]);
            assert!(pos[1] >= -0.01 && pos[1] <= BLOCK_SCALE + 0.01, "Y position out of range: {}", pos[1]);
            assert!(pos[2] >= -0.01 && pos[2] <= BLOCK_SCALE + 0.01, "Z position out of range: {}", pos[2]);
        }
    }

    #[test]
    fn test_single_block_normals_are_unit_length() {
        let chunk = single_block_chunk(8, 8, 8, Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        for vertex in &mesh.vertices {
            // Unpack normal
            let bytes = vertex.normal_packed.to_le_bytes();
            let nx = (bytes[0] as i8) as f32 / 127.0;
            let ny = (bytes[1] as i8) as f32 / 127.0;
            let nz = (bytes[2] as i8) as f32 / 127.0;

            // Normal should be axis-aligned (exactly one component is ±1)
            let sum = nx.abs() + ny.abs() + nz.abs();
            assert!((sum - 1.0).abs() < 0.1, "Normal should be unit axis-aligned, got ({}, {}, {})", nx, ny, nz);
        }
    }

    // ============ Face Culling ============

    #[test]
    fn test_internal_faces_culled() {
        // Two adjacent blocks should share a face that gets culled
        // With greedy meshing, the visible faces will be merged:
        // - Top (2 blocks → 1 merged quad)
        // - Bottom (2 blocks → 1 merged quad)
        // - Front (2 blocks → 1 merged quad)
        // - Back (2 blocks → 1 merged quad)
        // - Left (1 block → 1 quad)
        // - Right (1 block → 1 quad)
        // = 6 quads total (greedy merging reduces from 10 theoretical faces)
        let mut chunk = Chunk::new();
        chunk.set_block(8, 8, 8, Block::Stone);
        chunk.set_block(9, 8, 8, Block::Stone); // Adjacent in +X
        let chunk = Arc::new(chunk);

        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Greedy meshing merges the 4 shared-edge faces, plus 2 end faces = 6 quads
        assert_eq!(mesh.vertices.len(), 24, "Two adjacent blocks should have 24 vertices (6 merged quads)");

        // Verify the shared internal faces are actually culled by checking no +X or -X faces at boundary
        // The +X face of block at (8,8,8) should NOT exist
        // The -X face of block at (9,8,8) should NOT exist
        let world_x_boundary = 9.0 * BLOCK_SCALE; // Position between the two blocks

        for vertex in &mesh.vertices {
            let bytes = vertex.normal_packed.to_le_bytes();
            let nx = (bytes[0] as i8) as f32;

            // If facing +X, position should be at x=10 (right side of right block)
            if nx > 0.0 {
                assert!(
                    vertex.position[0] > world_x_boundary,
                    "+X face should only be at right edge, got x={}",
                    vertex.position[0]
                );
            }
            // If facing -X, position should be at x=8 (left side of left block)
            if nx < 0.0 {
                assert!(
                    vertex.position[0] < world_x_boundary,
                    "-X face should only be at left edge, got x={}",
                    vertex.position[0]
                );
            }
        }
    }

    #[test]
    fn test_solid_chunk_only_surfaces() {
        let chunk = solid_chunk(Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Solid chunk: only the 6 outer surfaces should be meshed
        // Each surface is 16×16 blocks, but greedy meshing merges them
        // Without neighbors, all 6 faces are visible
        // Should generate exactly 6 quads (one per face, fully merged)
        let num_quads = mesh.indices.len() / 6;
        assert_eq!(num_quads, 6, "Solid chunk should have exactly 6 quads (greedy merged surfaces)");
    }

    #[test]
    fn test_neighbor_chunk_culls_boundary_faces() {
        let chunk = solid_chunk(Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };

        // Add a solid neighbor in +X direction
        let mut neighbors = no_neighbors();
        neighbors[1] = Some(solid_chunk(Block::Stone)); // +X neighbor

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // With +X neighbor, the +X face should be culled
        // Should have 5 quads instead of 6
        let num_quads = mesh.indices.len() / 6;
        assert_eq!(num_quads, 5, "Solid chunk with +X neighbor should have 5 quads");
    }

    // ============ Greedy Meshing ============

    #[test]
    fn test_greedy_merges_same_texture() {
        // Create a 2×2 floor of same block type
        let mut chunk = Chunk::new();
        chunk.set_block(0, 0, 0, Block::Stone);
        chunk.set_block(1, 0, 0, Block::Stone);
        chunk.set_block(0, 0, 1, Block::Stone);
        chunk.set_block(1, 0, 1, Block::Stone);
        let chunk = Arc::new(chunk);

        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // The top face (4 blocks) should merge into 1 quad
        // Count quads with +Y normal
        let mut top_face_count = 0;
        for i in (0..mesh.indices.len()).step_by(6) {
            let v0 = &mesh.vertices[mesh.indices[i] as usize];
            let bytes = v0.normal_packed.to_le_bytes();
            let ny = (bytes[1] as i8) as f32;
            if ny > 0.0 {
                top_face_count += 1;
            }
        }

        assert_eq!(top_face_count, 1, "2×2 floor top face should merge into 1 quad");
    }

    #[test]
    fn test_greedy_separates_different_textures() {
        // Create adjacent blocks with different textures
        let mut chunk = Chunk::new();
        chunk.set_block(0, 0, 0, Block::Stone);
        chunk.set_block(1, 0, 0, Block::Dirt); // Different texture
        let chunk = Arc::new(chunk);

        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Two different block types can't merge - should be separate quads
        // Count top face quads
        let mut top_face_count = 0;
        for i in (0..mesh.indices.len()).step_by(6) {
            let v0 = &mesh.vertices[mesh.indices[i] as usize];
            let bytes = v0.normal_packed.to_le_bytes();
            let ny = (bytes[1] as i8) as f32;
            if ny > 0.0 {
                top_face_count += 1;
            }
        }

        assert_eq!(top_face_count, 2, "Different textures should produce separate quads");
    }

    // ============ Winding Order ============

    #[test]
    fn test_winding_order_ccw() {
        let chunk = single_block_chunk(8, 8, 8, Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // For each triangle, compute (v1-v0) × (v2-v0) and verify it points in same direction as normal
        for i in (0..mesh.indices.len()).step_by(3) {
            let i0 = mesh.indices[i] as usize;
            let i1 = mesh.indices[i + 1] as usize;
            let i2 = mesh.indices[i + 2] as usize;

            let v0 = Vec3::from(mesh.vertices[i0].position);
            let v1 = Vec3::from(mesh.vertices[i1].position);
            let v2 = Vec3::from(mesh.vertices[i2].position);

            // Cross product gives normal direction
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let computed_normal = edge1.cross(edge2);

            // Get stored normal
            let bytes = mesh.vertices[i0].normal_packed.to_le_bytes();
            let stored_normal = Vec3::new(
                (bytes[0] as i8) as f32 / 127.0,
                (bytes[1] as i8) as f32 / 127.0,
                (bytes[2] as i8) as f32 / 127.0,
            );

            // Dot product should be positive (same direction)
            let dot = computed_normal.dot(stored_normal);
            assert!(dot > 0.0, "Triangle winding should produce CCW normal, got dot={}", dot);
        }
    }

    // ============ LOD Tests ============

    #[test]
    fn test_lod_half_reduces_detail() {
        let chunk = solid_chunk(Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh_full = build_chunk_mesh_lod(pos, &chunk, &neighbors, LodLevel::Full);
        let mesh_half = build_chunk_mesh_lod(pos, &chunk, &neighbors, LodLevel::Half);

        // LOD Half should produce same or fewer quads (merged into larger blocks)
        let quads_full = mesh_full.indices.len() / 6;
        let quads_half = mesh_half.indices.len() / 6;

        assert!(quads_half <= quads_full, "LOD Half should have ≤ quads than Full");
        assert_eq!(mesh_half.lod, LodLevel::Half);
    }

    #[test]
    fn test_lod_quarter_reduces_detail_more() {
        let chunk = solid_chunk(Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh_full = build_chunk_mesh_lod(pos, &chunk, &neighbors, LodLevel::Full);
        let mesh_quarter = build_chunk_mesh_lod(pos, &chunk, &neighbors, LodLevel::Quarter);

        let quads_full = mesh_full.indices.len() / 6;
        let quads_quarter = mesh_quarter.indices.len() / 6;

        assert!(quads_quarter <= quads_full, "LOD Quarter should have ≤ quads than Full");
        assert_eq!(mesh_quarter.lod, LodLevel::Quarter);
    }

    // ============ Light Sampling ============

    #[test]
    fn test_light_value_in_vertices() {
        let mut chunk = Chunk::new();
        chunk.set_block(8, 8, 8, Block::Stone);
        chunk.set_block_light(9, 8, 8, 14); // Light adjacent to +X face
        let chunk = Arc::new(chunk);

        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Find vertices with +X normal (facing the lit area)
        let mut found_lit_face = false;
        for vertex in &mesh.vertices {
            let bytes = vertex.normal_packed.to_le_bytes();
            let nx = (bytes[0] as i8) as f32;

            if nx > 0.0 {
                // This is the +X face, should have light value 14
                // Light is packed in ao_tile_packed byte 3
                let ao_bytes = vertex.ao_tile_packed.to_le_bytes();
                let light = ao_bytes[3];
                // Light 14 scaled to 0-255 range = 14/15 * 255 ≈ 238
                assert!(light > 200, "Light value should be high on lit face, got {}", light);
                found_lit_face = true;
            }
        }

        assert!(found_lit_face, "Should find vertices with +X normal");
    }

    // ============ Edge Cases ============

    #[test]
    fn test_empty_chunk_produces_no_mesh() {
        let chunk = Arc::new(Chunk::new());
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        assert!(mesh.vertices.is_empty(), "Empty chunk should have no vertices");
        assert!(mesh.indices.is_empty(), "Empty chunk should have no indices");
    }

    #[test]
    fn test_transparent_blocks_not_meshed() {
        let chunk = single_block_chunk(8, 8, 8, Block::Air);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        assert!(mesh.vertices.is_empty(), "Air block should produce no mesh");
    }

    #[test]
    fn test_boundary_block_meshed_correctly() {
        // Block at chunk boundary (0,0,0)
        let chunk = single_block_chunk(0, 0, 0, Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Should still have 6 faces
        assert_eq!(mesh.vertices.len(), 24, "Boundary block should have 24 vertices");
    }

    #[test]
    fn test_corner_block_meshed_correctly() {
        // Block at chunk corner (15,15,15)
        let chunk = single_block_chunk(15, 15, 15, Block::Stone);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        let neighbors = no_neighbors();

        let mesh = build_chunk_mesh(pos, &chunk, &neighbors);

        // Should still have 6 faces
        assert_eq!(mesh.vertices.len(), 24, "Corner block should have 24 vertices");
    }

    // ============ Vertex Format ============

    #[test]
    fn test_vertex_size_is_24_bytes() {
        assert_eq!(std::mem::size_of::<ChunkVertex>(), 24, "ChunkVertex should be 24 bytes");
    }

    #[test]
    fn test_vertex_packing_roundtrip() {
        let normal = [0.0, 1.0, 0.0];
        let packed = ChunkVertex::pack_normal(normal);
        let bytes = packed.to_le_bytes();

        let nx = (bytes[0] as i8) as f32 / 127.0;
        let ny = (bytes[1] as i8) as f32 / 127.0;
        let nz = (bytes[2] as i8) as f32 / 127.0;

        assert!((nx - normal[0]).abs() < 0.01);
        assert!((ny - normal[1]).abs() < 0.01);
        assert!((nz - normal[2]).abs() < 0.01);
    }

    #[test]
    fn test_uv_packing() {
        let uv = [0.5, 0.75];
        let packed = ChunkVertex::pack_uv(uv);

        let u = packed[0] as f32 / 65535.0;
        let v = packed[1] as f32 / 65535.0;

        assert!((u - uv[0]).abs() < 0.001);
        assert!((v - uv[1]).abs() < 0.001);
    }
}
