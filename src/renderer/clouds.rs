//! 3D Voxel Cloud Rendering System
//!
//! Renders clouds as actual 3D voxel geometry, similar to terrain but at cloud altitude.
//! Uses greedy meshing for efficiency and wind animation for movement.
//!
//! Features:
//! - 3D volumetric cloud shapes using noise generation
//! - Greedy meshing for efficient quad merging (typically 2-4k quads)
//! - 4x4x4 block scale per cloud voxel for chunky voxel aesthetic
//! - Wind animation moves entire cloud layer
//! - Soft white translucent rendering
//! - Quality presets control voxel scale and update rate

use wgpu::util::DeviceExt;
use glam::{Vec3, Mat4};

/// Cloud voxel scale - each cloud voxel is this many terrain blocks
pub const CLOUD_VOXEL_SCALE: i32 = 4;

/// Cloud layer base altitude (world Y coordinate)
pub const CLOUD_BASE_Y: f32 = 115.0;

/// Cloud layer thickness in cloud voxels
pub const CLOUD_THICKNESS: i32 = 4;

/// Cloud grid size in cloud voxels (covers MORE than render distance for smooth edges)
/// At scale 4, 160 voxels = 640 blocks = 40 chunks (extends beyond view)
pub const CLOUD_GRID_SIZE: i32 = 160;

/// Maximum vertices for cloud mesh (generous for large grid)
pub const MAX_CLOUD_VERTICES: usize = 65536;

/// Maximum indices for cloud mesh
pub const MAX_CLOUD_INDICES: usize = 131072;

/// Cloud rendering quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CloudQuality {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl CloudQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            CloudQuality::Off => "OFF",
            CloudQuality::Low => "LOW",
            CloudQuality::Medium => "MEDIUM",
            CloudQuality::High => "HIGH",
            CloudQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> CloudQuality {
        match self {
            CloudQuality::Off => CloudQuality::Low,
            CloudQuality::Low => CloudQuality::Medium,
            CloudQuality::Medium => CloudQuality::High,
            CloudQuality::High => CloudQuality::Ultra,
            CloudQuality::Ultra => CloudQuality::Off,
        }
    }

    pub fn prev(&self) -> CloudQuality {
        match self {
            CloudQuality::Off => CloudQuality::Ultra,
            CloudQuality::Low => CloudQuality::Off,
            CloudQuality::Medium => CloudQuality::Low,
            CloudQuality::High => CloudQuality::Medium,
            CloudQuality::Ultra => CloudQuality::High,
        }
    }

    pub fn from_str(s: &str) -> CloudQuality {
        match s.to_uppercase().as_str() {
            "OFF" => CloudQuality::Off,
            "LOW" => CloudQuality::Low,
            "MEDIUM" => CloudQuality::Medium,
            "HIGH" => CloudQuality::High,
            "ULTRA" => CloudQuality::Ultra,
            _ => CloudQuality::default(),
        }
    }

    /// Voxel scale multiplier (higher = chunkier clouds, fewer voxels)
    pub fn voxel_scale(&self) -> i32 {
        match self {
            CloudQuality::Off => 8,
            CloudQuality::Low => 8,
            CloudQuality::Medium => 4,
            CloudQuality::High => 4,
            CloudQuality::Ultra => 2,
        }
    }

    /// How often to regenerate mesh (seconds)
    pub fn update_interval(&self) -> f32 {
        match self {
            CloudQuality::Off => 999.0,
            CloudQuality::Low => 2.0,
            CloudQuality::Medium => 1.0,
            CloudQuality::High => 0.5,
            CloudQuality::Ultra => 0.25,
        }
    }

    /// Whether to cast shadows on terrain
    pub fn terrain_shadows(&self) -> bool {
        match self {
            CloudQuality::Off | CloudQuality::Low => false,
            CloudQuality::Medium | CloudQuality::High | CloudQuality::Ultra => true,
        }
    }
}

/// Simple vertex for cloud rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl CloudVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CloudVertex>() as wgpu::BufferAddress,
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
            ],
        }
    }
}

/// Uniform buffer for cloud rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],          // xyz = position, w = time
    pub sun_direction: [f32; 4],       // xyz = direction, w = intensity
    pub sun_color: [f32; 4],           // rgb = color
    pub ambient_color: [f32; 4],       // rgb = ambient
    pub wind_offset: [f32; 4],         // xy = wind offset in world units, z = base_y, w = alpha
    pub cloud_color: [f32; 4],         // rgb = cloud color, a = density
}

impl Default for CloudUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
            sun_direction: [0.0, 1.0, 0.0, 1.0],
            sun_color: [1.0, 0.98, 0.92, 1.0],
            ambient_color: [0.5, 0.55, 0.6, 1.0],
            wind_offset: [0.0, 0.0, CLOUD_BASE_Y, 0.7],
            cloud_color: [1.0, 1.0, 1.0, 0.85],
        }
    }
}

/// 3D voxel grid for cloud data
pub struct CloudGrid {
    /// 3D array of cloud voxels (true = cloud, false = air)
    /// Indexed as [y][z][x] for cache-friendly XZ plane iteration
    data: Vec<bool>,
    /// Grid dimensions
    size_x: i32,
    size_y: i32,
    size_z: i32,
    /// World offset (where grid origin maps to in world space)
    world_offset_x: f32,
    world_offset_z: f32,
    /// Voxel scale (blocks per cloud voxel)
    voxel_scale: i32,
}

impl CloudGrid {
    pub fn new(size_x: i32, size_y: i32, size_z: i32, voxel_scale: i32) -> Self {
        let total = (size_x * size_y * size_z) as usize;
        Self {
            data: vec![false; total],
            size_x,
            size_y,
            size_z,
            world_offset_x: 0.0,
            world_offset_z: 0.0,
            voxel_scale,
        }
    }

    #[inline]
    fn index(&self, x: i32, y: i32, z: i32) -> usize {
        (y * self.size_z * self.size_x + z * self.size_x + x) as usize
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> bool {
        if x < 0 || x >= self.size_x || y < 0 || y >= self.size_y || z < 0 || z >= self.size_z {
            return false;
        }
        self.data[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: i32, y: i32, z: i32, value: bool) {
        if x >= 0 && x < self.size_x && y >= 0 && y < self.size_y && z >= 0 && z < self.size_z {
            let idx = self.index(x, y, z);
            self.data[idx] = value;
        }
    }

    /// Generate clouds using cellular noise for guaranteed separation + detail
    pub fn generate(&mut self, center_x: f32, center_z: f32, coverage: f32, seed: u64) {
        // Update world offset to center grid on position
        let half_size_x = (self.size_x * self.voxel_scale) as f32 / 2.0;
        let half_size_z = (self.size_z * self.voxel_scale) as f32 / 2.0;
        self.world_offset_x = center_x - half_size_x;
        self.world_offset_z = center_z - half_size_z;

        // Coverage controls how large each cloud cell grows
        // 0% = tiny clouds, 100% = large clouds (but ALWAYS with gaps between cells)
        let cell_fill = 0.15 + coverage.clamp(0.0, 1.0) * 0.55; // 0.15 to 0.70

        // Clear grid
        self.data.fill(false);

        let seed_f = (seed % 10000) as f32;

        // Cloud cell size in world blocks - determines cloud spacing
        // Larger cells = bigger, more spaced clouds
        let cell_size = 80.0; // ~80 block spacing between cloud centers

        // Generate using cellular approach - each cell can have ONE cloud
        for y in 0..self.size_y {
            for z in 0..self.size_z {
                for x in 0..self.size_x {
                    // World position for this voxel (in blocks)
                    let wx = self.world_offset_x + (x * self.voxel_scale) as f32;
                    let wz = self.world_offset_z + (z * self.voxel_scale) as f32;
                    let wy = y as f32 * self.voxel_scale as f32;

                    // === CELLULAR CLOUD PLACEMENT ===
                    // Find which cell this voxel belongs to
                    let cell_x = (wx / cell_size).floor();
                    let cell_z = (wz / cell_size).floor();

                    // Get cell's cloud center (jittered from cell center)
                    let cell_hash = self.hash_2d(cell_x as i32, cell_z as i32, seed_f);
                    let jitter_x = (cell_hash * 0.6 + 0.2) * cell_size; // 20-80% into cell
                    let jitter_z = (self.hash_2d(cell_x as i32 + 73, cell_z as i32 + 37, seed_f) * 0.6 + 0.2) * cell_size;

                    let cloud_center_x = cell_x * cell_size + jitter_x;
                    let cloud_center_z = cell_z * cell_size + jitter_z;

                    // Distance from this voxel to cloud center (in XZ plane)
                    let dx = wx - cloud_center_x;
                    let dz = wz - cloud_center_z;
                    let dist_xz = (dx * dx + dz * dz).sqrt();

                    // Cloud radius for this cell (varies per cell)
                    let cell_size_var = self.hash_2d(cell_x as i32 + 100, cell_z as i32 + 100, seed_f);
                    let base_radius = cell_size * cell_fill * (0.6 + cell_size_var * 0.8); // 60-140% of base

                    // === HEIGHT FALLOFF ===
                    // Clouds are dome-shaped: radius shrinks at top and bottom
                    let height_norm = y as f32 / (self.size_y - 1).max(1) as f32;
                    let height_factor = 1.0 - (2.0 * height_norm - 1.0).powi(2);
                    let radius_at_height = base_radius * (0.3 + height_factor * 0.7);

                    // === EDGE NOISE ===
                    // Add noise to the edge for organic shapes
                    let edge_noise = self.fbm_noise_3d(wx, wy * 0.5, wz, seed_f + 200.0, 0.04, 3);
                    let noisy_radius = radius_at_height * (0.7 + edge_noise * 0.6);

                    // === DENSITY CHECK ===
                    // Inside cloud if distance < noisy radius
                    if dist_xz < noisy_radius {
                        // Additional density falloff from center (softer edges)
                        let density = 1.0 - (dist_xz / noisy_radius).powi(2);

                        // Small detail noise for internal variation
                        let detail = self.noise_3d(wx * 0.05 + seed_f, wy * 0.03, wz * 0.05);

                        // Cloud exists if density is high enough
                        if density + detail * 0.3 > 0.2 {
                            self.set(x, y, z, true);
                        }
                    }
                }
            }
        }
    }

    /// Hash function for cell jitter (2D)
    fn hash_2d(&self, x: i32, z: i32, seed: f32) -> f32 {
        let n = x.wrapping_mul(374761393)
            .wrapping_add(z.wrapping_mul(668265263))
            .wrapping_add((seed * 1000.0) as i32);
        let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        (n as u32 as f32) / (u32::MAX as f32)
    }

    /// Billow noise - creates ridges that naturally separate into distinct formations
    /// Valleys become cloud cores, ridges become boundaries between clouds
    fn billow_noise_3d(&self, x: f32, y: f32, z: f32, seed: f32, base_freq: f32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = base_freq;
        let mut total_amp = 0.0;

        // 3 octaves of billow noise
        for i in 0..3 {
            // Flatten in Y for horizontal cloud layers
            let freq_y = frequency * 0.3;

            let n = self.noise_3d(
                x * frequency + seed + i as f32 * 73.7,
                y * freq_y,
                z * frequency + seed * 0.7 + i as f32 * 31.3,
            );

            // Billow transform: creates ridges from noise
            // abs(n * 2 - 1) creates V-shaped valleys
            // 1 - abs() inverts so peaks become cloud centers
            let billowed = 1.0 - (n * 2.0 - 1.0).abs();

            value += billowed * amplitude;
            total_amp += amplitude;

            amplitude *= 0.5;
            frequency *= 2.1; // Slightly offset from 2x to avoid patterns
        }

        value / total_amp
    }

    /// FBM noise for detail/edge variation
    fn fbm_noise_3d(&self, x: f32, y: f32, z: f32, seed: f32, base_freq: f32, octaves: u32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = base_freq;
        let mut total_amp = 0.0;

        for i in 0..octaves {
            let freq_y = frequency * 0.4; // Flatten

            let n = self.noise_3d(
                x * frequency + seed + i as f32 * 127.1,
                y * freq_y,
                z * frequency + seed * 0.6 + i as f32 * 89.3,
            );

            value += n * amplitude;
            total_amp += amplitude;

            amplitude *= 0.5;
            frequency *= 2.0;
        }

        value / total_amp
    }

    /// Simple 3D value noise
    fn noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        // Smooth interpolation
        let u = xf * xf * (3.0 - 2.0 * xf);
        let v = yf * yf * (3.0 - 2.0 * yf);
        let w = zf * zf * (3.0 - 2.0 * zf);

        // Hash corners
        let n000 = self.hash_3d(xi, yi, zi);
        let n001 = self.hash_3d(xi, yi, zi + 1);
        let n010 = self.hash_3d(xi, yi + 1, zi);
        let n011 = self.hash_3d(xi, yi + 1, zi + 1);
        let n100 = self.hash_3d(xi + 1, yi, zi);
        let n101 = self.hash_3d(xi + 1, yi, zi + 1);
        let n110 = self.hash_3d(xi + 1, yi + 1, zi);
        let n111 = self.hash_3d(xi + 1, yi + 1, zi + 1);

        // Trilinear interpolation
        let nx00 = n000 + u * (n100 - n000);
        let nx01 = n001 + u * (n101 - n001);
        let nx10 = n010 + u * (n110 - n010);
        let nx11 = n011 + u * (n111 - n011);

        let nxy0 = nx00 + v * (nx10 - nx00);
        let nxy1 = nx01 + v * (nx11 - nx01);

        nxy0 + w * (nxy1 - nxy0)
    }

    /// Hash function for noise
    fn hash_3d(&self, x: i32, y: i32, z: i32) -> f32 {
        let n = x.wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(z.wrapping_mul(1274126177));
        let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        (n as u32 as f32) / (u32::MAX as f32)
    }
}

/// Cloud mesh generated from voxel grid
pub struct CloudMesh {
    pub vertices: Vec<CloudVertex>,
    pub indices: Vec<u32>,
}

impl CloudMesh {
    /// Generate mesh from cloud grid using greedy meshing
    pub fn from_grid(grid: &CloudGrid, base_y: f32) -> Self {
        let mut vertices = Vec::with_capacity(MAX_CLOUD_VERTICES);
        let mut indices = Vec::with_capacity(MAX_CLOUD_INDICES);

        let scale = grid.voxel_scale as f32;

        // Process each face direction
        // Top (+Y)
        Self::mesh_axis(grid, base_y, scale, 1, true, &mut vertices, &mut indices);
        // Bottom (-Y)
        Self::mesh_axis(grid, base_y, scale, 1, false, &mut vertices, &mut indices);
        // Front (+Z)
        Self::mesh_axis(grid, base_y, scale, 2, true, &mut vertices, &mut indices);
        // Back (-Z)
        Self::mesh_axis(grid, base_y, scale, 2, false, &mut vertices, &mut indices);
        // Right (+X)
        Self::mesh_axis(grid, base_y, scale, 0, true, &mut vertices, &mut indices);
        // Left (-X)
        Self::mesh_axis(grid, base_y, scale, 0, false, &mut vertices, &mut indices);

        Self { vertices, indices }
    }

    /// Greedy mesh one axis direction
    fn mesh_axis(
        grid: &CloudGrid,
        base_y: f32,
        scale: f32,
        axis: usize, // 0=X, 1=Y, 2=Z
        positive: bool,
        vertices: &mut Vec<CloudVertex>,
        indices: &mut Vec<u32>,
    ) {
        let (size_d, size_u, size_v) = match axis {
            0 => (grid.size_x, grid.size_z, grid.size_y), // X axis: iterate X, mesh in ZY
            1 => (grid.size_y, grid.size_x, grid.size_z), // Y axis: iterate Y, mesh in XZ
            _ => (grid.size_z, grid.size_x, grid.size_y), // Z axis: iterate Z, mesh in XY
        };

        let normal: [f32; 3] = match (axis, positive) {
            (0, true) => [1.0, 0.0, 0.0],
            (0, false) => [-1.0, 0.0, 0.0],
            (1, true) => [0.0, 1.0, 0.0],
            (1, false) => [0.0, -1.0, 0.0],
            (_, true) => [0.0, 0.0, 1.0],
            (_, false) => [0.0, 0.0, -1.0],
        };

        // For each slice perpendicular to axis
        for d in 0..size_d {
            // Build visibility mask for this slice
            let mut mask = vec![false; (size_u * size_v) as usize];

            for v in 0..size_v {
                for u in 0..size_u {
                    // Get voxel coordinates
                    let (x, y, z) = match axis {
                        0 => (d, v, u),
                        1 => (u, d, v),
                        _ => (u, v, d),
                    };

                    // Check if this voxel is solid
                    if !grid.get(x, y, z) {
                        continue;
                    }

                    // Check neighbor in face direction
                    let (nx, ny, nz) = match axis {
                        0 => (if positive { d + 1 } else { d - 1 }, v, u),
                        1 => (u, if positive { d + 1 } else { d - 1 }, v),
                        _ => (u, v, if positive { d + 1 } else { d - 1 }),
                    };

                    // Face is visible if neighbor is air (or out of bounds)
                    if !grid.get(nx, ny, nz) {
                        mask[(v * size_u + u) as usize] = true;
                    }
                }
            }

            // Greedy merge the mask into quads
            let mut processed = vec![false; (size_u * size_v) as usize];

            for v in 0..size_v {
                for u in 0..size_u {
                    let idx = (v * size_u + u) as usize;
                    if !mask[idx] || processed[idx] {
                        continue;
                    }

                    // Find width of run
                    let mut width = 1;
                    while u + width < size_u {
                        let next_idx = (v * size_u + u + width) as usize;
                        if !mask[next_idx] || processed[next_idx] {
                            break;
                        }
                        width += 1;
                    }

                    // Find height of run
                    let mut height = 1;
                    'outer: while v + height < size_v {
                        for wu in 0..width {
                            let check_idx = ((v + height) * size_u + u + wu) as usize;
                            if !mask[check_idx] || processed[check_idx] {
                                break 'outer;
                            }
                        }
                        height += 1;
                    }

                    // Mark as processed
                    for hv in 0..height {
                        for wu in 0..width {
                            let mark_idx = ((v + hv) * size_u + u + wu) as usize;
                            processed[mark_idx] = true;
                        }
                    }

                    // Emit quad
                    Self::emit_quad(
                        grid, base_y, scale, axis, positive, d, u, v, width, height, normal,
                        vertices, indices,
                    );
                }
            }
        }
    }

    /// Emit a single quad
    fn emit_quad(
        grid: &CloudGrid,
        base_y: f32,
        scale: f32,
        axis: usize,
        positive: bool,
        d: i32,
        u: i32,
        v: i32,
        width: i32,
        height: i32,
        normal: [f32; 3],
        vertices: &mut Vec<CloudVertex>,
        indices: &mut Vec<u32>,
    ) {
        // Calculate world positions for quad corners
        // d is the slice index, u/v are the face coordinates
        let d_offset = if positive { d + 1 } else { d };

        let base_vertex = vertices.len() as u32;

        // Generate 4 corners based on axis
        let corners: [(f32, f32, f32); 4] = match axis {
            0 => {
                // X axis: d=X, u=Z, v=Y
                let x = d_offset as f32 * scale + grid.world_offset_x;
                [
                    (x, v as f32 * scale + base_y, u as f32 * scale + grid.world_offset_z),
                    (x, v as f32 * scale + base_y, (u + width) as f32 * scale + grid.world_offset_z),
                    (x, (v + height) as f32 * scale + base_y, (u + width) as f32 * scale + grid.world_offset_z),
                    (x, (v + height) as f32 * scale + base_y, u as f32 * scale + grid.world_offset_z),
                ]
            }
            1 => {
                // Y axis: d=Y, u=X, v=Z
                let y = d_offset as f32 * scale + base_y;
                [
                    (u as f32 * scale + grid.world_offset_x, y, v as f32 * scale + grid.world_offset_z),
                    ((u + width) as f32 * scale + grid.world_offset_x, y, v as f32 * scale + grid.world_offset_z),
                    ((u + width) as f32 * scale + grid.world_offset_x, y, (v + height) as f32 * scale + grid.world_offset_z),
                    (u as f32 * scale + grid.world_offset_x, y, (v + height) as f32 * scale + grid.world_offset_z),
                ]
            }
            _ => {
                // Z axis: d=Z, u=X, v=Y
                let z = d_offset as f32 * scale + grid.world_offset_z;
                [
                    (u as f32 * scale + grid.world_offset_x, v as f32 * scale + base_y, z),
                    ((u + width) as f32 * scale + grid.world_offset_x, v as f32 * scale + base_y, z),
                    ((u + width) as f32 * scale + grid.world_offset_x, (v + height) as f32 * scale + base_y, z),
                    (u as f32 * scale + grid.world_offset_x, (v + height) as f32 * scale + base_y, z),
                ]
            }
        };

        for (x, y, z) in corners {
            vertices.push(CloudVertex {
                position: [x, y, z],
                normal,
            });
        }

        // Add indices for two triangles (CCW winding)
        if positive {
            indices.extend_from_slice(&[
                base_vertex, base_vertex + 2, base_vertex + 1,
                base_vertex, base_vertex + 3, base_vertex + 2,
            ]);
        } else {
            indices.extend_from_slice(&[
                base_vertex, base_vertex + 1, base_vertex + 2,
                base_vertex, base_vertex + 2, base_vertex + 3,
            ]);
        }
    }
}

/// Cloud rendering system
pub struct Clouds {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,

    // Cloud state
    grid: CloudGrid,
    quality: CloudQuality,
    uniforms: CloudUniforms,

    // Mesh state - clouds are generated at fixed world grid positions
    index_count: u32,
    grid_origin_x: f32,  // Fixed world X where grid starts
    grid_origin_z: f32,  // Fixed world Z where grid starts
    wind_offset_x: f32,  // Accumulated wind offset
    wind_offset_z: f32,
    needs_regenerate: bool,

    // Public settings
    pub coverage: f32,
    pub speed: f32,
    pub shadows_enabled: bool,
}

impl Clouds {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniforms = CloudUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloud Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create vertex/index buffers (will be filled when mesh is generated)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Vertex Buffer"),
            size: (MAX_CLOUD_VERTICES * std::mem::size_of::<CloudVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Index Buffer"),
            size: (MAX_CLOUD_INDICES * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cloud Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Shader"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cloud Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[CloudVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Render both sides for soft look
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth (translucent)
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let quality = CloudQuality::default();
        let voxel_scale = quality.voxel_scale();
        let grid_size = CLOUD_GRID_SIZE / (voxel_scale / CLOUD_VOXEL_SCALE).max(1);

        Self {
            pipeline,
            uniform_buffer,
            vertex_buffer,
            index_buffer,
            bind_group_layout,
            bind_group,
            grid: CloudGrid::new(grid_size, CLOUD_THICKNESS, grid_size, voxel_scale),
            quality,
            uniforms,
            index_count: 0,
            grid_origin_x: 0.0,
            grid_origin_z: 0.0,
            wind_offset_x: 0.0,
            wind_offset_z: 0.0,
            needs_regenerate: true,
            coverage: 0.5,
            speed: 0.3,
            shadows_enabled: true,
        }
    }

    /// Update cloud mesh if needed
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        _time: f32,
        delta_time: f32,
        view_proj: Mat4,
        sun_direction: Vec3,
        sun_intensity: f32,
        sun_color: Vec3,
        ambient: Vec3,
    ) {
        if self.quality == CloudQuality::Off {
            return;
        }

        // Wind movement - accumulate offset
        let wind_dir_x = 1.0_f32;
        let wind_dir_z = 0.3_f32;
        let wind_speed = self.speed * 8.0; // Blocks per second at full speed
        self.wind_offset_x += wind_dir_x * wind_speed * delta_time;
        self.wind_offset_z += wind_dir_z * wind_speed * delta_time;

        // Grid world size
        let voxel_scale = self.quality.voxel_scale() as f32;
        let grid_world_size = self.grid.size_x as f32 * voxel_scale;

        // The grid is large (640 blocks at scale 4), so we only regenerate when
        // the player has moved a significant distance from the grid center
        // This prevents the "hard flip" - grid is always larger than visible area
        let regen_distance = grid_world_size * 0.25; // Regenerate when 25% off-center

        let dx = camera_pos.x - self.grid_origin_x;
        let dz = camera_pos.z - self.grid_origin_z;
        let dist_from_center = (dx * dx + dz * dz).sqrt();

        let player_moved_far = dist_from_center > regen_distance;

        if self.needs_regenerate || player_moved_far {
            // Center grid on player
            self.grid_origin_x = camera_pos.x;
            self.grid_origin_z = camera_pos.z;
            self.regenerate_mesh(queue, camera_pos.x, camera_pos.z);
            self.needs_regenerate = false;
        }

        // Wind offset doesn't cause regeneration - it just moves the mesh
        // The mesh is large enough that wind can blow for a long time
        // before clouds exit the visible area

        // Update uniforms
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, 0.0];
        self.uniforms.sun_direction = [sun_direction.x, sun_direction.y, sun_direction.z, sun_intensity];
        self.uniforms.sun_color = [sun_color.x, sun_color.y, sun_color.z, 1.0];
        self.uniforms.ambient_color = [ambient.x, ambient.y, ambient.z, 1.0];
        self.uniforms.wind_offset = [
            self.wind_offset_x,
            self.wind_offset_z,
            CLOUD_BASE_Y,
            0.93, // Alpha
        ];

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    /// Regenerate cloud mesh centered on given position
    fn regenerate_mesh(&mut self, queue: &wgpu::Queue, center_x: f32, center_z: f32) {
        // Generate cloud voxels
        self.grid.generate(center_x, center_z, self.coverage, 12345);

        // Generate mesh
        let mesh = CloudMesh::from_grid(&self.grid, CLOUD_BASE_Y);

        // Upload to GPU
        if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
            self.index_count = mesh.indices.len() as u32;
        } else {
            self.index_count = 0;
        }

        tracing::debug!(
            "Cloud mesh: {} verts, {} indices ({} quads)",
            mesh.vertices.len(),
            mesh.indices.len(),
            mesh.indices.len() / 6
        );
    }

    /// Render clouds
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.quality == CloudQuality::Off || self.index_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    // === Getters and Setters ===

    pub fn quality(&self) -> CloudQuality {
        self.quality
    }

    pub fn set_quality(&mut self, quality: CloudQuality) {
        if quality != self.quality {
            self.quality = quality;
            // Recreate grid with new voxel scale
            let voxel_scale = quality.voxel_scale();
            let grid_size = CLOUD_GRID_SIZE / (voxel_scale / CLOUD_VOXEL_SCALE).max(1);
            self.grid = CloudGrid::new(grid_size, CLOUD_THICKNESS, grid_size, voxel_scale);
            self.needs_regenerate = true;
        }
    }

    pub fn cycle_quality(&mut self) {
        self.set_quality(self.quality.next());
    }

    pub fn cycle_quality_prev(&mut self) {
        self.set_quality(self.quality.prev());
    }

    pub fn set_coverage(&mut self, coverage: f32) {
        let new_coverage = coverage.clamp(0.0, 1.0);
        if (new_coverage - self.coverage).abs() > 0.01 {
            self.coverage = new_coverage;
            self.needs_regenerate = true;
        }
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.0, 1.0);
    }

    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.shadows_enabled = enabled;
    }

    /// Get current coverage value
    pub fn get_coverage(&self) -> f32 {
        self.coverage
    }

    /// Get current speed value
    pub fn get_speed(&self) -> f32 {
        self.speed
    }

    /// Get cloud density at world XZ position (for terrain shadows)
    /// Returns 0.0-1.0 where 1.0 = full cloud coverage
    pub fn get_shadow_at(&self, world_x: f32, world_z: f32) -> f32 {
        if self.quality == CloudQuality::Off || !self.shadows_enabled {
            return 0.0;
        }

        // Account for wind offset
        let sample_x = world_x - self.wind_offset_x;
        let sample_z = world_z - self.wind_offset_z;

        // Convert to grid coordinates
        let grid_x = ((sample_x - self.grid.world_offset_x) / self.grid.voxel_scale as f32) as i32;
        let grid_z = ((sample_z - self.grid.world_offset_z) / self.grid.voxel_scale as f32) as i32;

        // Check if any cloud voxel is present in this column
        for y in 0..self.grid.size_y {
            if self.grid.get(grid_x, y, grid_z) {
                return 0.3; // Partial shadow
            }
        }

        0.0
    }

    /// Get wind offset for shadow calculation in chunk shader
    pub fn get_wind_offset(&self) -> (f32, f32) {
        (self.wind_offset_x, self.wind_offset_z)
    }
}

// =============================================================================
// CLOUD SHADER - 3D Voxel Geometry
// =============================================================================

const CLOUD_SHADER: &str = r#"
struct CloudUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,          // xyz = position, w = time
    sun_direction: vec4<f32>,       // xyz = direction, w = intensity
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    wind_offset: vec4<f32>,         // xy = offset, z = base_y, w = alpha
    cloud_color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: CloudUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Apply wind offset to position
    var world_pos = in.position;
    world_pos.x += u.wind_offset.x;
    world_pos.z += u.wind_offset.y;

    out.clip_position = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_position = world_pos;
    out.normal = in.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(u.sun_direction.xyz);
    let sun_intensity = u.sun_direction.w;

    // === FACE-BASED LIGHTING ===
    let ndotl = dot(in.normal, sun_dir);

    // Face classification
    let is_top = max(in.normal.y, 0.0);
    let is_bottom = max(-in.normal.y, 0.0);
    let is_side = 1.0 - abs(in.normal.y);

    // Sun contribution - stronger effect
    let sun_factor = max(ndotl, 0.0) * sun_intensity;

    // === COLOR PALETTE - More contrast ===
    // Bright pure white for sun-lit tops
    let cloud_bright = vec3<f32>(1.0, 1.0, 1.0);
    // Light gray-blue for shaded areas
    let cloud_mid = vec3<f32>(0.82, 0.85, 0.90);
    // Darker blue-gray for undersides - more contrast
    let cloud_dark = vec3<f32>(0.50, 0.55, 0.65);

    // === COMPUTE BASE BRIGHTNESS ===
    var brightness = 0.0;

    // Top faces: very bright
    brightness += is_top * (0.90 + sun_factor * 0.10);

    // Side faces: medium brightness, sun angle matters more
    brightness += is_side * (0.55 + sun_factor * 0.35);

    // Bottom faces: dark, minimal sun
    brightness += is_bottom * (0.25 + sun_factor * 0.08);

    brightness = clamp(brightness, 0.0, 1.0);

    // === COLOR INTERPOLATION ===
    var color: vec3<f32>;
    if (brightness < 0.4) {
        // Dark range: deep shadow to mid
        let t = brightness / 0.4;
        color = mix(cloud_dark, cloud_mid, t);
    } else if (brightness < 0.7) {
        // Mid range: mid to light
        let t = (brightness - 0.4) / 0.3;
        color = mix(cloud_mid, cloud_bright, t * 0.5);
    } else {
        // Bright range: approaching white
        let t = (brightness - 0.7) / 0.3;
        color = mix(cloud_mid, cloud_bright, 0.5 + t * 0.5);
    }

    // === TIME OF DAY EFFECTS ===
    let sun_height = sun_dir.y;

    // Sunset/sunrise: warm golden-pink tint
    let sunset_factor = smoothstep(-0.1, 0.2, sun_height) * (1.0 - smoothstep(0.2, 0.5, sun_height));
    let sunset_color = vec3<f32>(1.0, 0.9, 0.75); // Warm gold
    let sunset_shadow = vec3<f32>(0.9, 0.7, 0.8);  // Pink-purple shadows

    // Apply sunset - brighter parts get gold, darker get pink
    if (sunset_factor > 0.01) {
        let sunset_mix = mix(sunset_shadow, sunset_color, brightness);
        color = mix(color, color * sunset_mix, sunset_factor * 0.5);
    }

    // Night: cool blue tint
    let night_factor = smoothstep(0.05, -0.15, sun_height);
    let night_tint = vec3<f32>(0.65, 0.70, 0.85);
    color = mix(color, color * night_tint, night_factor * 0.4);

    // Sun color influence during day
    color = mix(color, color * u.sun_color.rgb, sun_intensity * 0.15);

    // === DISTANCE FADE ===
    // Extended range for clouds visible far into distance
    let dist = length(u.camera_pos.xyz - in.world_position);
    let fade_start = 450.0;  // Start fading at 450 blocks
    let fade_end = 600.0;    // Fully faded at 600 blocks
    let distance_alpha = 1.0 - smoothstep(fade_start, fade_end, dist);

    // Final alpha
    let alpha = u.wind_offset.w * distance_alpha;

    return vec4<f32>(color, alpha);
}
"#;
