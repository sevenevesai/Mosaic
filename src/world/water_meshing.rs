//! Water Mesh Builder
//!
//! Generates smooth water surface meshes from chunk heightfield data.
//! Unlike block meshing, water uses a continuous heightfield approach
//! with smooth vertex interpolation for natural-looking waves.

use std::sync::Arc;
use std::time::Instant;

use super::chunk::{Chunk, ChunkPos, WaterColumn};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32, BLOCK_SCALE};

/// Depth of the skirt geometry below water surface at water boundaries
/// Must be greater than maximum wave displacement to prevent gaps
const WATER_SKIRT_DEPTH: f32 = 1.5;

/// Water vertex format (24 bytes - matches ChunkVertex size)
///
/// Layout:
/// - position: World-space position of vertex
/// - normal_packed: Surface normal (computed in shader from waves)
/// - uv_data: [foam_factor, water_depth, flow_angle_lo, flow_angle_hi]
/// - light_data: [block_light, sky_light, reserved, reserved]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterVertex {
    pub position: [f32; 3],      // 12 bytes - world position
    pub normal_packed: u32,      // 4 bytes - packed normal (up for water surface)
    pub uv_data: u32,            // 4 bytes - [foam, depth, flow_lo, flow_hi]
    pub light_data: u32,         // 4 bytes - [block_light, sky_light, 0, 0]
}

impl WaterVertex {
    /// Pack a normal vector into u32
    #[inline]
    pub fn pack_normal(normal: [f32; 3]) -> u32 {
        let nx = ((normal[0] * 127.0) as i8) as u8;
        let ny = ((normal[1] * 127.0) as i8) as u8;
        let nz = ((normal[2] * 127.0) as i8) as u8;
        u32::from_le_bytes([nx, ny, nz, 0])
    }

    /// Pack water-specific UV data
    /// - foam: 0.0-1.0 foam intensity
    /// - depth: water depth in blocks (clamped to 0-255)
    /// - flow_angle: flow direction in radians (0-2π mapped to 0-65535)
    #[inline]
    pub fn pack_uv_data(foam: f32, depth: f32, flow_angle: f32) -> u32 {
        let foam_byte = (foam.clamp(0.0, 1.0) * 255.0) as u8;
        let depth_byte = (depth.clamp(0.0, 255.0)) as u8;
        // Pack flow angle into 16 bits (0-2π → 0-65535)
        let flow_u16 = ((flow_angle / (2.0 * std::f32::consts::PI)) * 65535.0) as u16;
        let flow_bytes = flow_u16.to_le_bytes();
        u32::from_le_bytes([foam_byte, depth_byte, flow_bytes[0], flow_bytes[1]])
    }

    /// Pack light data
    #[inline]
    pub fn pack_light_data(block_light: u8, sky_light: u8) -> u32 {
        // Scale from 0-15 to 0-255
        let block_byte = block_light.min(15) * 17;
        let sky_byte = sky_light.min(15) * 17;
        u32::from_le_bytes([block_byte, sky_byte, 0, 0])
    }

    /// Create a water vertex
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        foam: f32,
        depth: f32,
        flow_angle: f32,
        block_light: u8,
        sky_light: u8,
    ) -> Self {
        Self {
            position,
            normal_packed: Self::pack_normal(normal),
            uv_data: Self::pack_uv_data(foam, depth, flow_angle),
            light_data: Self::pack_light_data(block_light, sky_light),
        }
    }

    /// Vertex buffer layout for wgpu
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WaterVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal (packed)
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
                // UV data (foam, depth, flow)
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
                // Light data
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// GPU mesh data for water
pub struct GpuWaterMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

/// Water mesh for a chunk
pub struct WaterMesh {
    pub pos: ChunkPos,
    pub vertices: Vec<WaterVertex>,
    pub indices: Vec<u32>,
    pub gpu_mesh: Option<GpuWaterMesh>,
    /// Time when mesh was created (for fade-in animation)
    pub created_at: Instant,
}

impl WaterMesh {
    pub fn new(pos: ChunkPos, vertices: Vec<WaterVertex>, indices: Vec<u32>) -> Self {
        Self {
            pos,
            vertices,
            indices,
            gpu_mesh: None,
            created_at: Instant::now(),
        }
    }

    /// Check if this mesh has any geometry
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Get vertex count
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get triangle count
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Upload mesh to GPU
    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.gpu_mesh = Some(GpuWaterMesh {
            vertex_buffer,
            index_buffer,
            index_count: self.indices.len() as u32,
        });
    }
}

/// Build water mesh for a chunk
///
/// Generates a smooth heightfield mesh from the chunk's water data.
/// The mesh tessellates the water surface with vertices at each grid point,
/// creating a smooth surface that can be animated with waves.
pub fn build_water_mesh(
    pos: ChunkPos,
    chunk: &Arc<Chunk>,
    neighbors: &[Option<Arc<Chunk>>; 6],
) -> WaterMesh {
    let water_data = match chunk.water_data() {
        Some(data) if data.has_water() => data,
        _ => return WaterMesh::new(pos, Vec::new(), Vec::new()),
    };

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let world_origin = pos.world_origin();

    // Neighbor indices: [X-1, X+1, Y-1, Y+1, Z-1, Z+1]
    let neighbor_x_neg = neighbors[0].as_ref().and_then(|c| c.water_data());
    let neighbor_x_pos = neighbors[1].as_ref().and_then(|c| c.water_data());
    let neighbor_z_neg = neighbors[4].as_ref().and_then(|c| c.water_data());
    let neighbor_z_pos = neighbors[5].as_ref().and_then(|c| c.water_data());

    // Generate mesh grid - one vertex per water column corner
    // We create a (CHUNK_SIZE+1) x (CHUNK_SIZE+1) grid of vertices
    // to cover all the water surface area properly

    // Collect water heights and foam at vertex grid intersections
    // This ensures adjacent cells share the same interpolated values at corners
    let mut vertex_heights: [[Option<f32>; CHUNK_SIZE + 1]; CHUNK_SIZE + 1] = [[None; CHUNK_SIZE + 1]; CHUNK_SIZE + 1];
    let mut vertex_foam: [[f32; CHUNK_SIZE + 1]; CHUNK_SIZE + 1] = [[0.0; CHUNK_SIZE + 1]; CHUNK_SIZE + 1];

    // Helper to sample water column from current chunk or neighbors
    let sample_column = |cx: i32, cz: i32| -> Option<&WaterColumn> {
        if cx >= 0 && cx < CHUNK_SIZE_I32 && cz >= 0 && cz < CHUNK_SIZE_I32 {
            // Inside current chunk
            let col = water_data.get_column(cx as usize, cz as usize);
            if col.has_water { Some(col) } else { None }
        } else if cx < 0 && cz >= 0 && cz < CHUNK_SIZE_I32 {
            // X-1 neighbor
            neighbor_x_neg.and_then(|n| {
                let col = n.get_column((CHUNK_SIZE_I32 + cx) as usize, cz as usize);
                if col.has_water { Some(col) } else { None }
            })
        } else if cx >= CHUNK_SIZE_I32 && cz >= 0 && cz < CHUNK_SIZE_I32 {
            // X+1 neighbor
            neighbor_x_pos.and_then(|n| {
                let col = n.get_column((cx - CHUNK_SIZE_I32) as usize, cz as usize);
                if col.has_water { Some(col) } else { None }
            })
        } else if cz < 0 && cx >= 0 && cx < CHUNK_SIZE_I32 {
            // Z-1 neighbor
            neighbor_z_neg.and_then(|n| {
                let col = n.get_column(cx as usize, (CHUNK_SIZE_I32 + cz) as usize);
                if col.has_water { Some(col) } else { None }
            })
        } else if cz >= CHUNK_SIZE_I32 && cx >= 0 && cx < CHUNK_SIZE_I32 {
            // Z+1 neighbor
            neighbor_z_pos.and_then(|n| {
                let col = n.get_column(cx as usize, (cz - CHUNK_SIZE_I32) as usize);
                if col.has_water { Some(col) } else { None }
            })
        } else {
            // Corner neighbors (would need diagonal chunks, skip for now)
            None
        }
    };

    // Sample heights and foam at grid intersections
    // By averaging at corners, adjacent cells will have identical values for shared vertices
    for z in 0..=CHUNK_SIZE {
        for x in 0..=CHUNK_SIZE {
            // Sample from the four adjacent columns (may cross chunk boundaries)
            let mut has_water = false;
            let mut total_height = 0.0;
            let mut total_foam = 0.0;
            let mut count = 0;

            // Check the up to 4 columns that share this corner
            for dz in [-1i32, 0] {
                for dx in [-1i32, 0] {
                    let cx = x as i32 + dx;
                    let cz = z as i32 + dz;

                    if let Some(col) = sample_column(cx, cz) {
                        has_water = true;
                        total_height += col.surface_y;
                        total_foam += col.foam;
                        count += 1;
                    }
                }
            }

            if has_water && count > 0 {
                vertex_heights[z][x] = Some(total_height / count as f32);
                vertex_foam[z][x] = total_foam / count as f32;
            }
        }
    }

    // Generate triangles for each cell that has water
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let col = water_data.get_column(x, z);
            if !col.has_water {
                continue;
            }

            // Check if this is a boundary cell that needs neighbor data
            // If neighbor isn't available, skip this cell to avoid seams
            let needs_x_pos = x == CHUNK_SIZE - 1;
            let needs_z_pos = z == CHUNK_SIZE - 1;

            if needs_x_pos && neighbor_x_pos.is_none() {
                // Skip right-edge cells when X+ neighbor isn't available
                continue;
            }
            if needs_z_pos && neighbor_z_pos.is_none() {
                // Skip bottom-edge cells when Z+ neighbor isn't available
                continue;
            }

            // Get heights for the 4 corners of this cell
            let h00 = vertex_heights[z][x];
            let h10 = vertex_heights[z][x + 1];
            let h01 = vertex_heights[z + 1][x];
            let h11 = vertex_heights[z + 1][x + 1];

            // Count how many corners have valid heights
            let valid_heights: Vec<f32> = [h00, h10, h01, h11]
                .iter()
                .filter_map(|h| *h)
                .collect();

            // Need at least one valid height to render this cell
            if valid_heights.is_empty() {
                continue;
            }

            // Use average of valid heights to fill in missing corners
            // This is for shoreline transitions (where water meets land),
            // NOT chunk boundaries (which are skipped above)
            let avg_height = valid_heights.iter().sum::<f32>() / valid_heights.len() as f32;
            let h00 = h00.unwrap_or(avg_height);
            let h10 = h10.unwrap_or(avg_height);
            let h01 = h01.unwrap_or(avg_height);
            let h11 = h11.unwrap_or(avg_height);

            // Get foam for corners - small boost for actual shoreline edges only
            let missing_corners = [vertex_heights[z][x], vertex_heights[z][x + 1],
                           vertex_heights[z + 1][x], vertex_heights[z + 1][x + 1]]
                .iter().filter(|h| h.is_none()).count();
            // Only boost if multiple corners missing (true shoreline, not just chunk edge)
            let edge_boost = if missing_corners >= 2 { 0.35 } else { 0.0 };

            let f00 = (vertex_foam[z][x] + edge_boost).min(1.0);
            let f10 = (vertex_foam[z][x + 1] + edge_boost).min(1.0);
            let f01 = (vertex_foam[z + 1][x] + edge_boost).min(1.0);
            let f11 = (vertex_foam[z + 1][x + 1] + edge_boost).min(1.0);

            // Calculate world positions
            let wx = world_origin.x + (x as f32) * BLOCK_SCALE;
            let wz = world_origin.z + (z as f32) * BLOCK_SCALE;

            // Calculate actual water depth by finding terrain below water surface
            let water_y = col.surface_y;
            let depth = calculate_water_depth(chunk, x, z, water_y);

            // Water surface always gets full sky light (it's at the top)
            // This prevents chunk-boundary lighting artifacts
            let block_light: u8 = 0;
            let sky_light: u8 = 15;

            // Use uniform flow direction for visual waves to prevent banding artifacts
            // Varying flow angles cause interference patterns when interpolated across quads
            // The flow simulation still runs - could be used for foam/particles later
            let uniform_flow = 0.0f32;

            // Normal pointing up (waves will modify this in shader)
            let normal = [0.0, 1.0, 0.0];

            // Create 4 vertices for this cell
            let base_idx = vertices.len() as u32;

            // Vertex 0: (x, z)
            vertices.push(WaterVertex::new(
                [wx, h00, wz],
                normal,
                f00, depth, uniform_flow,
                block_light, sky_light,
            ));

            // Vertex 1: (x+1, z)
            vertices.push(WaterVertex::new(
                [wx + BLOCK_SCALE, h10, wz],
                normal,
                f10, depth, uniform_flow,
                block_light, sky_light,
            ));

            // Vertex 2: (x, z+1)
            vertices.push(WaterVertex::new(
                [wx, h01, wz + BLOCK_SCALE],
                normal,
                f01, depth, uniform_flow,
                block_light, sky_light,
            ));

            // Vertex 3: (x+1, z+1)
            vertices.push(WaterVertex::new(
                [wx + BLOCK_SCALE, h11, wz + BLOCK_SCALE],
                normal,
                f11, depth, uniform_flow,
                block_light, sky_light,
            ));

            // Two triangles (CCW winding when viewed from above)
            // Triangle 1: 0-2-1
            indices.push(base_idx);
            indices.push(base_idx + 2);
            indices.push(base_idx + 1);

            // Triangle 2: 1-2-3
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);
            indices.push(base_idx + 3);

            // Add skirt geometry - vertical walls extending below the surface
            // This hides gaps created by wave displacement
            let skirt_bottom = water_y - WATER_SKIRT_DEPTH;

            // Helper to add a skirt quad (vertical wall)
            // Takes 2 top vertices (already in buffer) and creates 2 bottom vertices + triangles
            let add_skirt_quad = |vertices: &mut Vec<WaterVertex>, indices: &mut Vec<u32>,
                                  top_pos1: [f32; 3], top_pos2: [f32; 3],
                                  normal: [f32; 3], foam: f32, depth: f32,
                                  flow_angle: f32, block_light: u8, sky_light: u8| {
                let skirt_idx = vertices.len() as u32;

                // Bottom vertices (at skirt depth)
                let bottom_pos1 = [top_pos1[0], skirt_bottom, top_pos1[2]];
                let bottom_pos2 = [top_pos2[0], skirt_bottom, top_pos2[2]];

                // Add 4 vertices for this skirt face (top-left, top-right, bottom-left, bottom-right)
                vertices.push(WaterVertex::new(top_pos1, normal, foam, depth, flow_angle, block_light, sky_light));
                vertices.push(WaterVertex::new(top_pos2, normal, foam, depth, flow_angle, block_light, sky_light));
                vertices.push(WaterVertex::new(bottom_pos1, normal, foam * 0.3, depth + WATER_SKIRT_DEPTH, flow_angle, block_light, sky_light));
                vertices.push(WaterVertex::new(bottom_pos2, normal, foam * 0.3, depth + WATER_SKIRT_DEPTH, flow_angle, block_light, sky_light));

                // Two triangles for the quad (CCW winding)
                indices.push(skirt_idx);      // top-left
                indices.push(skirt_idx + 2);  // bottom-left
                indices.push(skirt_idx + 1);  // top-right

                indices.push(skirt_idx + 1);  // top-right
                indices.push(skirt_idx + 2);  // bottom-left
                indices.push(skirt_idx + 3);  // bottom-right
            };

            // Get top vertex positions from what we just created
            let top_v0 = [wx, h00, wz];
            let top_v1 = [wx + BLOCK_SCALE, h10, wz];
            let top_v2 = [wx, h01, wz + BLOCK_SCALE];
            let top_v3 = [wx + BLOCK_SCALE, h11, wz + BLOCK_SCALE];

            // Check adjacent water for each edge
            // -X edge (left side): vertices 0 and 2
            let has_water_neg_x = if x > 0 {
                water_data.get_column(x - 1, z).has_water
            } else {
                neighbor_x_neg.map_or(false, |n| n.get_column(CHUNK_SIZE - 1, z).has_water)
            };

            // +X edge (right side): vertices 1 and 3
            let has_water_pos_x = if x < CHUNK_SIZE - 1 {
                water_data.get_column(x + 1, z).has_water
            } else {
                neighbor_x_pos.map_or(false, |n| n.get_column(0, z).has_water)
            };

            // -Z edge (front side): vertices 0 and 1
            let has_water_neg_z = if z > 0 {
                water_data.get_column(x, z - 1).has_water
            } else {
                neighbor_z_neg.map_or(false, |n| n.get_column(x, CHUNK_SIZE - 1).has_water)
            };

            // +Z edge (back side): vertices 2 and 3
            let has_water_pos_z = if z < CHUNK_SIZE - 1 {
                water_data.get_column(x, z + 1).has_water
            } else {
                neighbor_z_pos.map_or(false, |n| n.get_column(x, 0).has_water)
            };

            // Use upward normal for skirts so they render identically to surface
            let skirt_normal = [0.0, 1.0, 0.0];

            // Only add boundary skirts (where there's no adjacent water)
            // No internal skirts needed - keeps underwater view clean

            // -X edge (vertices 0 and 2)
            if !has_water_neg_x {
                add_skirt_quad(&mut vertices, &mut indices,
                    top_v2, top_v0,
                    skirt_normal, (f01 + f00) / 2.0, depth, uniform_flow, block_light, sky_light);
            }

            // +X edge (vertices 1 and 3)
            if !has_water_pos_x {
                add_skirt_quad(&mut vertices, &mut indices,
                    top_v1, top_v3,
                    skirt_normal, (f10 + f11) / 2.0, depth, uniform_flow, block_light, sky_light);
            }

            // -Z edge (vertices 0 and 1)
            if !has_water_neg_z {
                add_skirt_quad(&mut vertices, &mut indices,
                    top_v0, top_v1,
                    skirt_normal, (f00 + f10) / 2.0, depth, uniform_flow, block_light, sky_light);
            }

            // +Z edge (vertices 2 and 3)
            if !has_water_pos_z {
                add_skirt_quad(&mut vertices, &mut indices,
                    top_v3, top_v2,
                    skirt_normal, (f11 + f01) / 2.0, depth, uniform_flow, block_light, sky_light);
            }
        }
    }

    WaterMesh::new(pos, vertices, indices)
}

/// Calculate water depth at a column by finding terrain below
fn calculate_water_depth(chunk: &Chunk, x: usize, z: usize, water_surface_y: f32) -> f32 {
    // Search downward from water surface to find solid terrain
    for y in (0..CHUNK_SIZE).rev() {
        let block = chunk.get_block(x, y, z);
        if block.is_solid() {
            // Found terrain - depth is water surface minus terrain
            let terrain_y = y as f32 * BLOCK_SCALE;
            return (water_surface_y - terrain_y).max(0.0);
        }
    }

    // No terrain found in this chunk - assume deep water
    // Use chunk bottom as reference
    water_surface_y.max(8.0)
}

/// Check if a chunk has water that needs rendering
pub fn chunk_has_water(chunk: &Chunk) -> bool {
    chunk.has_water()
}
