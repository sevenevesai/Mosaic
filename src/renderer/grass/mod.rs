//! Procedural Grass Rendering System
//!
//! GPU-driven grass using tile-based placement, compute shader culling,
//! and vertex-pulling blade generation. Designed for efficiency:
//! - 16 bytes per tile (vs 32+ bytes per blade)
//! - Compute shader frustum + distance culling
//! - Bezier curve blades generated in vertex shader
//! - Single MDI draw call for all grass

use std::collections::HashMap;
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::world::ChunkPos;
use crate::world::generation::biome::BiomeId;

// ============================================================================
// QUALITY PRESETS
// ============================================================================

/// Grass rendering quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum GrassQuality {
    Off = 0,
    Low = 1,
    #[default]
    Medium = 2,
    High = 3,
    Ultra = 4,
}

impl GrassQuality {
    /// Blades per tile (4x4 block area) - bunched together
    pub fn blades_per_tile(&self) -> u32 {
        match self {
            GrassQuality::Off => 0,
            GrassQuality::Low => 6,
            GrassQuality::Medium => 12,
            GrassQuality::High => 18,
            GrassQuality::Ultra => 24,
        }
    }

    /// Segments per blade (more = smoother curve)
    pub fn blade_segments(&self) -> u32 {
        match self {
            GrassQuality::Off => 0,
            GrassQuality::Low => 2,
            GrassQuality::Medium => 3,
            GrassQuality::High => 4,
            GrassQuality::Ultra => 5,
        }
    }

    /// Distance (in chunks) for full detail rendering
    pub fn full_detail_distance(&self) -> f32 {
        match self {
            GrassQuality::Off => 0.0,
            GrassQuality::Low => 4.0,
            GrassQuality::Medium => 6.0,
            GrassQuality::High => 8.0,
            GrassQuality::Ultra => 12.0,
        }
    }

    /// Distance (in chunks) for low LOD rendering
    pub fn low_lod_distance(&self) -> f32 {
        match self {
            GrassQuality::Off => 0.0,
            GrassQuality::Low => 6.0,
            GrassQuality::Medium => 10.0,
            GrassQuality::High => 14.0,
            GrassQuality::Ultra => 20.0,
        }
    }

    /// Distance (in chunks) beyond which grass is culled
    pub fn cull_distance(&self) -> f32 {
        match self {
            GrassQuality::Off => 0.0,
            GrassQuality::Low => 8.0,
            GrassQuality::Medium => 12.0,
            GrassQuality::High => 16.0,
            GrassQuality::Ultra => 24.0,
        }
    }

    /// Maximum tiles that can be rendered
    pub fn max_tiles(&self) -> u32 {
        match self {
            GrassQuality::Off => 0,
            GrassQuality::Low => 8_000,
            GrassQuality::Medium => 24_000,
            GrassQuality::High => 48_000,
            GrassQuality::Ultra => 96_000,
        }
    }

    /// Wind strength multiplier
    pub fn wind_strength(&self) -> f32 {
        match self {
            GrassQuality::Off => 0.0,
            GrassQuality::Low => 0.5,
            GrassQuality::Medium => 0.7,
            GrassQuality::High => 1.0,
            GrassQuality::Ultra => 1.0,
        }
    }

    /// Cycle to next quality level
    pub fn cycle_next(&self) -> Self {
        match self {
            GrassQuality::Off => GrassQuality::Low,
            GrassQuality::Low => GrassQuality::Medium,
            GrassQuality::Medium => GrassQuality::High,
            GrassQuality::High => GrassQuality::Ultra,
            GrassQuality::Ultra => GrassQuality::Off,
        }
    }

    /// Cycle to previous quality level
    pub fn cycle_prev(&self) -> Self {
        match self {
            GrassQuality::Off => GrassQuality::Ultra,
            GrassQuality::Low => GrassQuality::Off,
            GrassQuality::Medium => GrassQuality::Low,
            GrassQuality::High => GrassQuality::Medium,
            GrassQuality::Ultra => GrassQuality::High,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            GrassQuality::Off => "OFF",
            GrassQuality::Low => "LOW",
            GrassQuality::Medium => "MEDIUM",
            GrassQuality::High => "HIGH",
            GrassQuality::Ultra => "ULTRA",
        }
    }

    /// Convert from index (for settings)
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => GrassQuality::Off,
            1 => GrassQuality::Low,
            2 => GrassQuality::Medium,
            3 => GrassQuality::High,
            4 => GrassQuality::Ultra,
            _ => GrassQuality::Medium,
        }
    }

    /// Convert to index (for settings)
    pub fn to_index(&self) -> usize {
        *self as usize
    }
}

// ============================================================================
// GPU DATA STRUCTURES
// ============================================================================

/// A grass tile representing a 4x4 block area that can contain grass.
/// 16 bytes - designed for efficient GPU upload and minimal memory.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassTile {
    /// World position XZ of tile center
    pub position: [f32; 2],
    /// Packed data: blade_count (8 bits) | height_var (8 bits) | seed (16 bits)
    pub density_seed: u32,
    /// Packed data: biome_id (8 bits) | surface_y as fixed-point (24 bits)
    pub biome_height: u32,
}

impl GrassTile {
    /// Create a new grass tile
    pub fn new(x: f32, z: f32, y: f32, blade_count: u8, biome: BiomeId, seed: u16) -> Self {
        // Pack density_seed: blade_count | height_var | seed
        let height_var = ((seed & 0xFF) as u8).wrapping_mul(37) % 64; // 0-63 variation
        let density_seed = (blade_count as u32)
            | ((height_var as u32) << 8)
            | ((seed as u32) << 16);

        // Pack biome_height: biome_id | y as 24-bit fixed point (8.16 format)
        // Y range 0-255 with 16 bits fractional
        let y_fixed = ((y * 256.0) as u32).min(0xFFFFFF);
        let biome_height = (biome as u32) | (y_fixed << 8);

        Self {
            position: [x, z],
            density_seed,
            biome_height,
        }
    }

    /// Unpack blade count
    pub fn blade_count(&self) -> u8 {
        (self.density_seed & 0xFF) as u8
    }

    /// Unpack height variation (0-63)
    pub fn height_variation(&self) -> u8 {
        ((self.density_seed >> 8) & 0xFF) as u8
    }

    /// Unpack seed
    pub fn seed(&self) -> u16 {
        (self.density_seed >> 16) as u16
    }

    /// Unpack biome ID
    pub fn biome(&self) -> BiomeId {
        BiomeId::from_u8((self.biome_height & 0xFF) as u8)
    }

    /// Unpack surface Y coordinate
    pub fn surface_y(&self) -> f32 {
        ((self.biome_height >> 8) & 0xFFFFFF) as f32 / 256.0
    }
}

/// Per-frame uniforms for grass rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassUniforms {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Wind: direction.xy (normalized), strength, time
    pub wind: [f32; 4],
    /// Camera position (xyz) + padding
    pub camera_pos: [f32; 4],
    /// LOD distances: full_detail, low_lod, cull, chunk_size
    pub lod_distances: [f32; 4],
    /// Quality params: blades_per_tile, blade_segments, max_tiles, padding
    pub quality: [u32; 4],
}

impl Default for GrassUniforms {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            wind: [1.0, 0.3, 0.7, 0.0], // Default wind direction + strength + time
            camera_pos: [0.0; 4],
            lod_distances: [96.0, 160.0, 192.0, 16.0], // In blocks (6, 10, 12 chunks)
            quality: [4, 3, 24000, 0], // Medium preset
        }
    }
}

/// Biome grass color palette entry
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BiomeGrassColor {
    /// Base color at blade root (RGBA)
    pub base: [f32; 4],
    /// Tip color at blade tip (RGBA)
    pub tip: [f32; 4],
}

impl BiomeGrassColor {
    pub fn new(base: [f32; 3], tip: [f32; 3]) -> Self {
        Self {
            base: [base[0], base[1], base[2], 1.0],
            tip: [tip[0], tip[1], tip[2], 1.0],
        }
    }
}

/// Complete biome color palette (8 biomes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BiomePalette {
    pub colors: [BiomeGrassColor; 8],
}

impl Default for BiomePalette {
    fn default() -> Self {
        // Base color: #212900 = RGB(33, 41, 0) = (0.129, 0.161, 0.0)
        Self {
            colors: [
                // Grasslands - dark olive matching terrain side walls
                BiomeGrassColor::new(
                    [0.129, 0.161, 0.0],   // #212900 base
                    [0.18, 0.22, 0.02],    // Slightly lighter tip
                ),
                // Cavernous Highlands - similar dark olive
                BiomeGrassColor::new(
                    [0.14, 0.17, 0.02],    // Slight variation
                    [0.19, 0.23, 0.04],
                ),
                // Jungle - darker green
                BiomeGrassColor::new(
                    [0.10, 0.14, 0.0],
                    [0.15, 0.20, 0.02],
                ),
                // Tundra - tan/brown dead grass
                BiomeGrassColor::new(
                    [0.25, 0.22, 0.10],
                    [0.32, 0.28, 0.14],
                ),
                // Swamp - murky dark
                BiomeGrassColor::new(
                    [0.11, 0.14, 0.02],
                    [0.16, 0.20, 0.04],
                ),
                // Desert - sandy sparse grass
                BiomeGrassColor::new(
                    [0.35, 0.30, 0.15],
                    [0.42, 0.36, 0.18],
                ),
                // Mountains - gray-green alpine
                BiomeGrassColor::new(
                    [0.15, 0.18, 0.08],
                    [0.22, 0.26, 0.12],
                ),
                // Ocean (unused)
                BiomeGrassColor::new(
                    [0.129, 0.161, 0.0],
                    [0.18, 0.22, 0.02],
                ),
            ],
        }
    }
}

/// Frustum planes for culling (6 planes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrustumPlanes {
    pub planes: [[f32; 4]; 6],
}

/// Indirect draw command (matches wgpu::DrawIndexedIndirect)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

// ============================================================================
// CONSTANTS
// ============================================================================

/// Size of a grass tile in blocks (4x4 area)
pub const TILE_SIZE: f32 = 4.0;

/// Chunk size in blocks
pub const CHUNK_SIZE: f32 = 16.0;

/// Tiles per chunk (4x4 grid of tiles per chunk = 16 tiles)
pub const TILES_PER_CHUNK: usize = 16;

/// Maximum vertices per blade (5 segments * 2 verts = 10, plus 2 for end caps)
pub const MAX_VERTS_PER_BLADE: u32 = 12;

/// Vertices per blade at each quality level
pub fn verts_per_blade(segments: u32) -> u32 {
    segments * 2
}

/// Indices per blade (triangle strip)
pub fn indices_per_blade(segments: u32) -> u32 {
    if segments < 2 { return 0; }
    (segments - 1) * 6 // 2 triangles per segment, 3 indices each
}

// ============================================================================
// GRASS TILE BUFFER
// ============================================================================

/// Manages grass tiles per chunk, similar to ChunkPool
pub struct GrassTileBuffer {
    /// GPU buffer holding all grass tiles
    pub tile_buffer: wgpu::Buffer,
    /// Map of chunk position to (start_index, count) in tile buffer
    chunk_tiles: HashMap<ChunkPos, (u32, u32)>,
    /// Free ranges in the buffer (start, count)
    free_ranges: Vec<(u32, u32)>,
    /// Next available index
    next_index: u32,
    /// Maximum tiles
    max_tiles: u32,
    /// Current total tile count
    tile_count: u32,
    /// Dirty flag for GPU updates
    dirty: bool,
}

impl GrassTileBuffer {
    pub fn new(device: &wgpu::Device, max_tiles: u32) -> Self {
        let buffer_size = max_tiles as u64 * std::mem::size_of::<GrassTile>() as u64;

        let tile_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grass Tile Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        tracing::info!(
            "Created grass tile buffer: {:.2} KB for {} tiles",
            buffer_size as f64 / 1024.0,
            max_tiles
        );

        Self {
            tile_buffer,
            chunk_tiles: HashMap::new(),
            free_ranges: Vec::new(),
            next_index: 0,
            max_tiles,
            tile_count: 0,
            dirty: true,
        }
    }

    /// Upload tiles for a chunk
    pub fn upload_chunk(&mut self, queue: &wgpu::Queue, pos: ChunkPos, tiles: &[GrassTile]) {
        if tiles.is_empty() {
            self.remove_chunk(pos);
            return;
        }

        let count = tiles.len() as u32;

        // Remove existing tiles for this chunk if any
        if let Some((old_start, old_count)) = self.chunk_tiles.remove(&pos) {
            self.free_ranges.push((old_start, old_count));
            self.tile_count -= old_count;
        }

        // Try to find a free range that fits
        let start = if let Some(idx) = self.free_ranges.iter().position(|(_, c)| *c >= count) {
            let (range_start, range_count) = self.free_ranges.remove(idx);
            if range_count > count {
                // Put remainder back
                self.free_ranges.push((range_start + count, range_count - count));
            }
            range_start
        } else if self.next_index + count <= self.max_tiles {
            let start = self.next_index;
            self.next_index += count;
            start
        } else {
            tracing::warn!("Grass tile buffer full, skipping chunk {:?}", pos);
            return;
        };

        // Upload tile data
        let offset = start as u64 * std::mem::size_of::<GrassTile>() as u64;
        queue.write_buffer(&self.tile_buffer, offset, bytemuck::cast_slice(tiles));

        self.chunk_tiles.insert(pos, (start, count));
        self.tile_count += count;
        self.dirty = true;
    }

    /// Remove tiles for a chunk
    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        if let Some((start, count)) = self.chunk_tiles.remove(&pos) {
            self.free_ranges.push((start, count));
            self.tile_count -= count;
            self.dirty = true;
        }
    }

    /// Check if chunk has tiles
    pub fn has_chunk(&self, pos: &ChunkPos) -> bool {
        self.chunk_tiles.contains_key(pos)
    }

    /// Get total tile count
    pub fn tile_count(&self) -> u32 {
        self.tile_count
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunk_tiles.len()
    }

    /// Check if dirty
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear dirty flag
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Remove chunks not in the visible set
    pub fn retain_chunks(&mut self, visible: &[ChunkPos]) {
        let visible_set: std::collections::HashSet<_> = visible.iter().collect();
        let to_remove: Vec<_> = self.chunk_tiles.keys()
            .filter(|pos| !visible_set.contains(pos))
            .copied()
            .collect();

        for pos in to_remove {
            self.remove_chunk(pos);
        }
    }
}

// ============================================================================
// GRASS SYSTEM (Main Interface)
// ============================================================================

/// Grass blade vertex for the blade mesh
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassBladeVertex {
    /// Position along blade (local space)
    pub position: [f32; 3],
    /// Normalized height along blade (0 = base, 1 = tip) for wind bending
    pub height_factor: f32,
}

/// Main grass rendering system
pub struct GrassSystem {
    /// Current quality setting
    pub quality: GrassQuality,
    /// Tile buffer
    pub tile_buffer: GrassTileBuffer,
    /// Uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Biome palette buffer
    palette_buffer: wgpu::Buffer,
    /// Current uniforms
    uniforms: GrassUniforms,
    /// Wind time accumulator
    wind_time: f32,
    /// Bind group layout for grass rendering
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    /// Blade mesh vertex buffer
    blade_vertex_buffer: wgpu::Buffer,
    /// Blade mesh index buffer
    blade_index_buffer: wgpu::Buffer,
    /// Number of indices in blade mesh
    blade_index_count: u32,
    /// Number of vertices per blade
    verts_per_blade: u32,
}

impl GrassSystem {
    pub fn new(
        device: &wgpu::Device,
        quality: GrassQuality,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let max_tiles = quality.max_tiles().max(24_000); // At least medium capacity

        let tile_buffer = GrassTileBuffer::new(device, max_tiles);

        // Create uniform buffer
        let uniforms = GrassUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grass Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create palette buffer
        let palette = BiomePalette::default();
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grass Biome Palette"),
            contents: bytemuck::bytes_of(&palette),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grass Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile buffer (storage, read-only in vertex shader)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Biome palette
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grass Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_buffer.tile_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: palette_buffer.as_entire_binding(),
                },
            ],
        });

        // Create blade mesh (a simple quad strip for one blade)
        let segments = 4u32; // Fixed segments for now, quality affects blade count not segments
        let (blade_vertices, blade_indices) = Self::generate_blade_mesh(segments);
        let verts_per_blade = blade_vertices.len() as u32;

        let blade_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grass Blade Vertex Buffer"),
            contents: bytemuck::cast_slice(&blade_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let blade_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grass Blade Index Buffer"),
            contents: bytemuck::cast_slice(&blade_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let blade_index_count = blade_indices.len() as u32;

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grass Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grass_shader.wgsl").into()),
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grass Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grass Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_grass",
                buffers: &[], // No vertex buffers - we use vertex pulling
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_grass",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Grass is two-sided
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        tracing::info!(
            "GrassSystem initialized: quality={:?}, max_tiles={}, blade_verts={}, blade_indices={}",
            quality, max_tiles, verts_per_blade, blade_index_count
        );

        Self {
            quality,
            tile_buffer,
            uniform_buffer,
            palette_buffer,
            uniforms,
            wind_time: 0.0,
            bind_group_layout,
            bind_group,
            render_pipeline,
            blade_vertex_buffer,
            blade_index_buffer,
            blade_index_count,
            verts_per_blade,
        }
    }

    /// Generate a blade mesh as a curved quad strip
    fn generate_blade_mesh(segments: u32) -> (Vec<GrassBladeVertex>, Vec<u16>) {
        let mut vertices = Vec::with_capacity((segments * 2) as usize);
        let mut indices = Vec::with_capacity(((segments - 1) * 6) as usize);

        let blade_height = 0.5f32;
        let base_width = 0.06f32;
        let curve_amount = 0.15f32;

        for i in 0..segments {
            let t = i as f32 / (segments - 1) as f32;
            let y = t * blade_height;

            // Bezier curve for blade shape (quadratic)
            let curve_offset = curve_amount * t * t; // Parabolic curve

            // Width tapers toward tip
            let width = base_width * (1.0 - t * 0.8);

            // Left vertex
            vertices.push(GrassBladeVertex {
                position: [curve_offset - width * 0.5, y, 0.0],
                height_factor: t,
            });

            // Right vertex
            vertices.push(GrassBladeVertex {
                position: [curve_offset + width * 0.5, y, 0.0],
                height_factor: t,
            });
        }

        // Generate indices for triangle strip converted to triangle list
        for i in 0..(segments - 1) {
            let base = i * 2;
            // First triangle
            indices.push(base as u16);
            indices.push((base + 2) as u16);
            indices.push((base + 1) as u16);
            // Second triangle
            indices.push((base + 1) as u16);
            indices.push((base + 2) as u16);
            indices.push((base + 3) as u16);
        }

        (vertices, indices)
    }

    /// Update uniforms for the current frame
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        camera_pos: Vec3,
        delta_time: f32,
    ) {
        if self.quality == GrassQuality::Off {
            return;
        }

        // Update wind time
        self.wind_time += delta_time;

        // Update uniforms
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, 0.0];
        self.uniforms.wind[3] = self.wind_time;

        // Update quality-dependent params
        self.uniforms.lod_distances = [
            self.quality.full_detail_distance() * CHUNK_SIZE,
            self.quality.low_lod_distance() * CHUNK_SIZE,
            self.quality.cull_distance() * CHUNK_SIZE,
            CHUNK_SIZE,
        ];
        self.uniforms.quality = [
            self.quality.blades_per_tile(),
            self.quality.blade_segments(),
            self.quality.max_tiles(),
            0,
        ];
        self.uniforms.wind[2] = self.quality.wind_strength();

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    /// Set wind direction (normalized XZ)
    pub fn set_wind_direction(&mut self, dir_x: f32, dir_z: f32) {
        let len = (dir_x * dir_x + dir_z * dir_z).sqrt();
        if len > 0.001 {
            self.uniforms.wind[0] = dir_x / len;
            self.uniforms.wind[1] = dir_z / len;
        }
    }

    /// Get current quality level
    pub fn quality(&self) -> GrassQuality {
        self.quality
    }

    /// Set quality level
    pub fn set_quality(&mut self, quality: GrassQuality) {
        self.quality = quality;
        tracing::info!("Grass quality set to {:?}", quality);
    }

    /// Get bind group layout for pipeline creation
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get bind group for rendering
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Get current tile count
    pub fn tile_count(&self) -> u32 {
        self.tile_buffer.tile_count()
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.tile_buffer.chunk_count()
    }

    /// Render grass
    ///
    /// Call this during the opaque rendering pass, after terrain but before transparent objects.
    /// Uses instancing: each instance is a blade, derived from tile data in the shader.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.quality == GrassQuality::Off {
            return;
        }

        let tile_count = self.tile_buffer.tile_count();
        if tile_count == 0 {
            return;
        }

        // Calculate total instances (blades)
        // Each tile spawns blades_per_tile blades
        let blades_per_tile = self.quality.blades_per_tile();
        let total_blades = tile_count * blades_per_tile;

        if total_blades == 0 {
            return;
        }

        // Calculate vertices per blade (segments * 2)
        let segments = self.quality.blade_segments();
        let verts_per_blade = segments * 2;
        let _indices_per_blade = (segments - 1) * 6;

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        // Draw using vertex pulling - no vertex buffer needed
        // instance_id encodes: tile_index * blades_per_tile + blade_index
        // vertex_id encodes position within blade
        render_pass.draw(0..verts_per_blade, 0..total_blades);
    }

    /// Get render statistics
    pub fn get_stats(&self) -> GrassStats {
        let tile_count = self.tile_buffer.tile_count();
        let blades_per_tile = self.quality.blades_per_tile();
        let total_blades = tile_count * blades_per_tile;
        let segments = self.quality.blade_segments();
        let triangles_per_blade = (segments.saturating_sub(1)) * 2;
        let total_triangles = total_blades * triangles_per_blade;

        GrassStats {
            tile_count,
            blade_count: total_blades,
            triangle_count: total_triangles,
            chunk_count: self.tile_buffer.chunk_count() as u32,
        }
    }
}

/// Statistics about grass rendering
#[derive(Debug, Clone, Copy)]
pub struct GrassStats {
    pub tile_count: u32,
    pub blade_count: u32,
    pub triangle_count: u32,
    pub chunk_count: u32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grass_tile_packing() {
        let tile = GrassTile::new(100.5, 200.5, 45.25, 6, BiomeId::Grasslands, 12345);

        assert!((tile.position[0] - 100.5).abs() < 0.001);
        assert!((tile.position[1] - 200.5).abs() < 0.001);
        assert_eq!(tile.blade_count(), 6);
        assert_eq!(tile.seed(), 12345);
        assert_eq!(tile.biome(), BiomeId::Grasslands);
        assert!((tile.surface_y() - 45.25).abs() < 0.01);
    }

    #[test]
    fn test_quality_presets() {
        assert_eq!(GrassQuality::Off.blades_per_tile(), 0);
        assert_eq!(GrassQuality::Medium.blades_per_tile(), 4);
        assert_eq!(GrassQuality::Ultra.blades_per_tile(), 10);

        assert_eq!(GrassQuality::Medium.blade_segments(), 3);
        assert_eq!(GrassQuality::High.blade_segments(), 4);
    }

    #[test]
    fn test_quality_cycling() {
        let q = GrassQuality::Medium;
        assert_eq!(q.cycle_next(), GrassQuality::High);
        assert_eq!(q.cycle_prev(), GrassQuality::Low);

        assert_eq!(GrassQuality::Ultra.cycle_next(), GrassQuality::Off);
        assert_eq!(GrassQuality::Off.cycle_prev(), GrassQuality::Ultra);
    }

    #[test]
    fn test_biome_palette_size() {
        // Ensure palette is correctly sized for GPU upload
        assert_eq!(std::mem::size_of::<BiomePalette>(), 256);
        assert_eq!(std::mem::size_of::<BiomeGrassColor>(), 32);
    }

    #[test]
    fn test_grass_tile_size() {
        // Critical: tiles must be exactly 16 bytes
        assert_eq!(std::mem::size_of::<GrassTile>(), 16);
    }

    #[test]
    fn test_uniforms_size() {
        // Uniforms should be a multiple of 16 for alignment
        let size = std::mem::size_of::<GrassUniforms>();
        assert!(size % 16 == 0, "Uniforms size {} not aligned to 16", size);
    }
}
