//! Ray-Marched Water Rendering System
//!
//! Renders water as ray-plane intersections rather than per-chunk meshes.
//! This eliminates chunk-boundary seams, camera jitter, and mesh generation overhead.
//!
//! Features:
//! - Multiple water levels (sea, underground lakes, lava pools)
//! - Procedural wave animation (no vertex displacement)
//! - Fresnel-based reflections with skybox fallback
//! - Screen-space reflections (SSR) for nearby geometry
//! - Depth-based refraction with chromatic aberration
//! - Shore foam at water-geometry intersections
//!
//! Pipeline:
//! 1. Fullscreen pass after opaque geometry
//! 2. For each pixel, intersect ray with water planes
//! 3. Depth test against scene to determine visibility
//! 4. Apply all effects in fragment shader

use wgpu::util::DeviceExt;

/// Maximum number of distinct water levels supported
pub const MAX_WATER_LEVELS: usize = 16;

/// Size of the water presence texture (covers WATER_PRESENCE_SIZE x WATER_PRESENCE_SIZE blocks)
/// This texture is centered on the camera and indicates where water actually exists
pub const WATER_PRESENCE_SIZE: u32 = 512;

/// Water presence texture format: RGBA8
/// - R: Water level (0 = no water, 36 = level 1, ..., 255 = level 7)
/// - G: Flow direction X (-1,0,+1 mapped to 0,127,255)
/// - B: Flow direction Z (-1,0,+1 mapped to 0,127,255)
/// - A: Flags (bit 0 = is_source, bit 1 = is_flowing)
pub const WATER_PRESENCE_BYTES_PER_TEXEL: u32 = 4;

/// Water rendering quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaterQuality {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl WaterQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            WaterQuality::Off => "OFF",
            WaterQuality::Low => "LOW",
            WaterQuality::Medium => "MEDIUM",
            WaterQuality::High => "HIGH",
            WaterQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> WaterQuality {
        match self {
            WaterQuality::Off => WaterQuality::Low,
            WaterQuality::Low => WaterQuality::Medium,
            WaterQuality::Medium => WaterQuality::High,
            WaterQuality::High => WaterQuality::Ultra,
            WaterQuality::Ultra => WaterQuality::Off,
        }
    }

    /// SSR ray march step count
    pub fn ssr_steps(&self) -> u32 {
        match self {
            WaterQuality::Off => 0,
            WaterQuality::Low => 8,
            WaterQuality::Medium => 16,
            WaterQuality::High => 24,
            WaterQuality::Ultra => 32,
        }
    }

    /// Wave noise octaves
    pub fn wave_octaves(&self) -> u32 {
        match self {
            WaterQuality::Off => 1,
            WaterQuality::Low => 2,
            WaterQuality::Medium => 3,
            WaterQuality::High => 3,
            WaterQuality::Ultra => 4,
        }
    }

    /// Refraction distortion scale
    pub fn refraction_distortion(&self) -> f32 {
        match self {
            WaterQuality::Off => 0.0,
            WaterQuality::Low => 0.02,
            WaterQuality::Medium => 0.04,
            WaterQuality::High => 0.06,
            WaterQuality::Ultra => 0.08,
        }
    }

    /// Chromatic aberration strength
    pub fn chromatic_aberration(&self) -> f32 {
        match self {
            WaterQuality::Off | WaterQuality::Low | WaterQuality::Medium => 0.0,
            WaterQuality::High => 0.003,
            WaterQuality::Ultra => 0.006,
        }
    }
}

/// Water parameters for a single water level
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterLevel {
    /// Y height of this water surface
    pub height: f32,
    /// Water type: 0=water, 1=lava (affects color/effects)
    pub water_type: f32,
    /// Reserved for future use
    pub _padding: [f32; 2],
}

impl Default for WaterLevel {
    fn default() -> Self {
        Self {
            height: -1000.0, // Inactive by default
            water_type: 0.0,
            _padding: [0.0; 2],
        }
    }
}

/// Uniform buffer for water rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniforms {
    // Camera data (duplicated for standalone water pass)
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],       // xyz = position, w = time

    // Sun/sky data
    pub sun_direction: [f32; 4],    // xyz = direction, w = intensity
    pub sun_color: [f32; 4],        // rgb = color, a = unused
    pub ambient_color: [f32; 4],    // rgb = ambient, a = brightness

    // Water visual parameters
    pub shallow_color: [f32; 4],    // rgb = color, a = alpha
    pub deep_color: [f32; 4],       // rgb = color, a = alpha

    // Wave parameters
    pub wave_speed: f32,
    pub wave_strength: f32,
    pub fresnel_power: f32,
    pub foam_threshold: f32,

    // Refraction parameters
    pub refraction_distortion: f32,
    pub chromatic_aberration: f32,
    pub refraction_depth_fade: f32,

    // SSR parameters
    pub ssr_steps: u32,
    pub ssr_max_distance: f32,
    pub ssr_thickness: f32,

    // Water level count and quality
    pub water_level_count: u32,
    pub wave_octaves: u32,

    // Screen dimensions for UV calculations
    pub screen_size: [f32; 2],

    // Water presence texture center (world XZ coordinates)
    pub presence_center: [f32; 2],

    // Water presence texture size in blocks
    pub presence_size: f32,

    // Padding to match WGSL struct layout:
    // - WGSL vec3<f32> has 16-byte alignment, causing implicit padding
    // - Total padding needed: 28 bytes (to get from offset 420 to 448 for array)
    pub _padding: [f32; 7],

    // Water levels array (up to MAX_WATER_LEVELS)
    pub water_levels: [WaterLevel; MAX_WATER_LEVELS],
}

impl Default for WaterUniforms {
    fn default() -> Self {
        let mut levels = [WaterLevel::default(); MAX_WATER_LEVELS];
        // Default: single sea-level water at y=32
        levels[0] = WaterLevel {
            height: 32.0,
            water_type: 0.0,
            _padding: [0.0; 2],
        };

        Self {
            view_proj: [[0.0; 4]; 4],
            inv_view_proj: [[0.0; 4]; 4],
            inv_view: [[0.0; 4]; 4],
            inv_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 4],
            sun_direction: [0.0, 1.0, 0.0, 1.0],
            sun_color: [1.0, 0.95, 0.9, 1.0],
            ambient_color: [0.3, 0.35, 0.4, 1.0],
            shallow_color: [0.15, 0.45, 0.50, 0.6],
            deep_color: [0.02, 0.12, 0.20, 0.9],
            wave_speed: 0.4,
            wave_strength: 0.08,
            fresnel_power: 5.0,
            foam_threshold: 0.6,
            refraction_distortion: 0.04,
            chromatic_aberration: 0.0,
            refraction_depth_fade: 12.0,
            ssr_steps: 16,
            ssr_max_distance: 50.0,
            ssr_thickness: 0.5,
            water_level_count: 1,
            wave_octaves: 3,
            screen_size: [1920.0, 1080.0],
            presence_center: [0.0, 0.0],
            presence_size: WATER_PRESENCE_SIZE as f32,
            _padding: [0.0; 7],
            water_levels: levels,
        }
    }
}

/// Water column data for presence texture updates
#[derive(Clone, Copy, Default)]
pub struct WaterColumnInfo {
    pub has_water: bool,
    pub level: u8,        // 0-7
    pub flow_dir_x: i8,   // -1, 0, +1
    pub flow_dir_z: i8,   // -1, 0, +1
    pub is_source: bool,
    pub is_flowing: bool,
}

/// Water rendering system
pub struct Water {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    quality: WaterQuality,
    uniforms: WaterUniforms,
    /// Water presence texture - indicates where water actually exists (R8 format)
    /// Each texel = 255 if water exists at that XZ position, 0 if no water
    presence_texture: wgpu::Texture,
    presence_texture_view: wgpu::TextureView,
    /// Center of the presence texture in world coordinates (updated when camera moves)
    presence_center: [f32; 2],
    /// CPU-side buffer for updating presence texture
    presence_data: Vec<u8>,
    /// Whether the presence texture has been initialized (ensures first frame always updates)
    presence_initialized: bool,
}

impl Water {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout for water-specific resources
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Bind Group Layout"),
            entries: &[
                // binding 0: Water uniforms
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
                // binding 1: Depth texture (for occlusion and SSR)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: Scene texture (for refraction)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 4: Water presence texture (indicates where water exists)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Create water presence texture (RGBA8Unorm - 4 bytes per texel)
        // R = water level, G = flow_dir_x, B = flow_dir_z, A = flags
        let presence_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Presence Texture"),
            size: wgpu::Extent3d {
                width: WATER_PRESENCE_SIZE,
                height: WATER_PRESENCE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let presence_texture_view = presence_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Initialize presence data to all zeros (no water) - RGBA format
        let presence_data = vec![0u8; (WATER_PRESENCE_SIZE * WATER_PRESENCE_SIZE * WATER_PRESENCE_BYTES_PER_TEXEL) as usize];

        // Create uniform buffer
        let uniforms = WaterUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(WATER_SHADER.into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline (fullscreen triangle, no vertex buffer)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffer - fullscreen triangle from vertex index
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
                ..Default::default()
            },
            // No depth write - water is transparent and composited over scene
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group: None,
            sampler,
            quality: WaterQuality::default(),
            uniforms,
            presence_texture,
            presence_texture_view,
            presence_center: [0.0, 0.0],
            presence_data,
            presence_initialized: false,
        }
    }

    /// Update bind group with current textures
    pub fn update_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_texture: &wgpu::TextureView,
        scene_texture: &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(scene_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.presence_texture_view),
                },
            ],
        }));
    }

    /// Update uniforms from camera and lighting state
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        inv_view: glam::Mat4,
        inv_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        time: f32,
        sun_direction: glam::Vec3,
        sun_intensity: f32,
        sun_color: glam::Vec3,
        ambient_color: glam::Vec3,
        brightness: f32,
        screen_width: f32,
        screen_height: f32,
    ) {
        let inv_view_proj = (view_proj).inverse();

        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.inv_view_proj = inv_view_proj.to_cols_array_2d();
        self.uniforms.inv_view = inv_view.to_cols_array_2d();
        self.uniforms.inv_proj = inv_proj.to_cols_array_2d();
        self.uniforms.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, time];
        self.uniforms.sun_direction = [sun_direction.x, sun_direction.y, sun_direction.z, sun_intensity];
        self.uniforms.sun_color = [sun_color.x, sun_color.y, sun_color.z, 1.0];
        self.uniforms.ambient_color = [ambient_color.x, ambient_color.y, ambient_color.z, brightness];
        self.uniforms.screen_size = [screen_width, screen_height];

        // Update presence texture center from camera position
        self.uniforms.presence_center = self.presence_center;
        self.uniforms.presence_size = WATER_PRESENCE_SIZE as f32;

        // Apply quality settings
        self.uniforms.ssr_steps = self.quality.ssr_steps();
        self.uniforms.wave_octaves = self.quality.wave_octaves();
        self.uniforms.refraction_distortion = self.quality.refraction_distortion();
        self.uniforms.chromatic_aberration = self.quality.chromatic_aberration();

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    /// Set water levels (for multiple water bodies at different heights)
    pub fn set_water_levels(&mut self, levels: &[(f32, f32)]) {
        // Clear all levels
        for level in &mut self.uniforms.water_levels {
            level.height = -1000.0;
            level.water_type = 0.0;
        }

        // Set active levels
        let count = levels.len().min(MAX_WATER_LEVELS);
        for (i, (height, water_type)) in levels.iter().take(count).enumerate() {
            self.uniforms.water_levels[i] = WaterLevel {
                height: *height,
                water_type: *water_type,
                _padding: [0.0; 2],
            };
        }
        self.uniforms.water_level_count = count as u32;
    }

    /// Get quality setting
    pub fn quality(&self) -> WaterQuality {
        self.quality
    }

    /// Set quality setting
    pub fn set_quality(&mut self, quality: WaterQuality) {
        self.quality = quality;
    }

    /// Cycle through quality presets
    pub fn cycle_quality(&mut self) {
        self.quality = self.quality.next();
    }

    /// Clear the water presence texture (call before updating with chunk data)
    pub fn clear_presence(&mut self) {
        self.presence_data.fill(0);
    }

    /// Update water presence from chunk water data
    ///
    /// Call this method for each chunk that has water, passing the chunk's water columns.
    /// The presence texture is centered on camera_pos and covers WATER_PRESENCE_SIZE blocks.
    ///
    /// # Arguments
    /// * `chunk_world_x` - World X coordinate of chunk origin (chunk_pos.x * 16)
    /// * `chunk_world_z` - World Z coordinate of chunk origin (chunk_pos.z * 16)
    /// * `water_columns` - 16x16 array indicating if each column has water (indexed as [z][x])
    pub fn update_presence_from_chunk(
        &mut self,
        chunk_world_x: i32,
        chunk_world_z: i32,
        water_columns: &[[bool; 16]; 16],
    ) {
        let half_size = (WATER_PRESENCE_SIZE / 2) as f32;
        let center_x = self.presence_center[0];
        let center_z = self.presence_center[1];

        for lz in 0..16 {
            for lx in 0..16 {
                if water_columns[lz][lx] {
                    // World position of this water column
                    let world_x = chunk_world_x + lx as i32;
                    let world_z = chunk_world_z + lz as i32;

                    // Convert to texture coordinates (centered on camera)
                    let tex_x = (world_x as f32 - center_x + half_size) as i32;
                    let tex_z = (world_z as f32 - center_z + half_size) as i32;

                    // Check bounds
                    if tex_x >= 0 && tex_x < WATER_PRESENCE_SIZE as i32
                        && tex_z >= 0 && tex_z < WATER_PRESENCE_SIZE as i32
                    {
                        // RGBA format: R=level, G=flow_x, B=flow_z, A=flags
                        let idx = ((tex_z as u32 * WATER_PRESENCE_SIZE + tex_x as u32) * WATER_PRESENCE_BYTES_PER_TEXEL) as usize;
                        self.presence_data[idx] = 255;     // R: full level (legacy compat)
                        self.presence_data[idx + 1] = 127; // G: neutral flow X
                        self.presence_data[idx + 2] = 127; // B: neutral flow Z
                        self.presence_data[idx + 3] = 1;   // A: is_source flag
                    }
                }
            }
        }
    }

    /// Update water presence from chunk water data with full column info
    ///
    /// This version includes water level and flow direction for visual effects.
    pub fn update_presence_from_chunk_detailed(
        &mut self,
        chunk_world_x: i32,
        chunk_world_z: i32,
        water_columns: &[[WaterColumnInfo; 16]; 16],
    ) {
        let half_size = (WATER_PRESENCE_SIZE / 2) as f32;
        let center_x = self.presence_center[0];
        let center_z = self.presence_center[1];

        for lz in 0..16 {
            for lx in 0..16 {
                let col = &water_columns[lz][lx];
                if col.has_water && col.level > 0 {
                    // World position of this water column
                    let world_x = chunk_world_x + lx as i32;
                    let world_z = chunk_world_z + lz as i32;

                    // Convert to texture coordinates (centered on camera)
                    let tex_x = (world_x as f32 - center_x + half_size) as i32;
                    let tex_z = (world_z as f32 - center_z + half_size) as i32;

                    // Check bounds
                    if tex_x >= 0 && tex_x < WATER_PRESENCE_SIZE as i32
                        && tex_z >= 0 && tex_z < WATER_PRESENCE_SIZE as i32
                    {
                        let idx = ((tex_z as u32 * WATER_PRESENCE_SIZE + tex_x as u32) * WATER_PRESENCE_BYTES_PER_TEXEL) as usize;

                        // R: Water level (0-7 mapped to 0-255)
                        self.presence_data[idx] = (col.level as u32 * 255 / 7).min(255) as u8;

                        // G: Flow direction X (-1,0,+1 mapped to 0,127,255)
                        self.presence_data[idx + 1] = match col.flow_dir_x {
                            -1 => 0,
                            0 => 127,
                            1 => 255,
                            _ => 127,
                        };

                        // B: Flow direction Z (-1,0,+1 mapped to 0,127,255)
                        self.presence_data[idx + 2] = match col.flow_dir_z {
                            -1 => 0,
                            0 => 127,
                            1 => 255,
                            _ => 127,
                        };

                        // A: Flags (bit 0 = is_source, bit 1 = is_flowing)
                        let mut flags = 0u8;
                        if col.is_source { flags |= 1; }
                        if col.is_flowing { flags |= 2; }
                        self.presence_data[idx + 3] = flags;
                    }
                }
            }
        }
    }

    /// Set the center of the water presence texture (call when camera moves significantly)
    /// Returns true if the center changed (texture needs re-uploading)
    pub fn set_presence_center(&mut self, x: f32, z: f32) -> bool {
        // First call always triggers update
        if !self.presence_initialized {
            self.presence_center = [x, z];
            self.presence_initialized = true;
            return true;
        }

        // Only update if moved more than 32 blocks (avoids constant re-uploading)
        let dx = (x - self.presence_center[0]).abs();
        let dz = (z - self.presence_center[1]).abs();
        if dx > 32.0 || dz > 32.0 {
            self.presence_center = [x, z];
            true
        } else {
            false
        }
    }

    /// Upload the presence texture to the GPU
    pub fn upload_presence(&self, queue: &wgpu::Queue) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.presence_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.presence_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(WATER_PRESENCE_SIZE * WATER_PRESENCE_BYTES_PER_TEXEL),
                rows_per_image: Some(WATER_PRESENCE_SIZE),
            },
            wgpu::Extent3d {
                width: WATER_PRESENCE_SIZE,
                height: WATER_PRESENCE_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Render water (fullscreen pass)
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.quality == WaterQuality::Off {
            return;
        }

        if let Some(ref bind_group) = self.bind_group {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, bind_group, &[]);
            // Draw fullscreen triangle (3 vertices, no buffer)
            render_pass.draw(0..3, 0..1);
        }
    }
}

// Water shader - ray-marched infinite planes with SSR
const WATER_SHADER: &str = r#"
// Maximum water levels (must match Rust constant)
const MAX_WATER_LEVELS: u32 = 16u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Water level data
struct WaterLevel {
    height: f32,
    water_type: f32,  // 0 = water, 1 = lava
    _padding: vec2<f32>,
}

// Main uniform buffer
struct WaterUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,       // xyz = position, w = time

    sun_direction: vec4<f32>,    // xyz = direction, w = intensity
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,    // rgb = ambient, a = brightness

    shallow_color: vec4<f32>,
    deep_color: vec4<f32>,

    wave_speed: f32,
    wave_strength: f32,
    fresnel_power: f32,
    foam_threshold: f32,

    refraction_distortion: f32,
    chromatic_aberration: f32,
    refraction_depth_fade: f32,

    ssr_steps: u32,
    ssr_max_distance: f32,
    ssr_thickness: f32,

    water_level_count: u32,
    wave_octaves: u32,

    screen_size: vec2<f32>,

    // Water presence texture parameters
    presence_center: vec2<f32>,  // World XZ of texture center
    presence_size: f32,          // Texture size in blocks

    // Explicit padding to match Rust struct layout (7 f32s = 28 bytes)
    // This ensures array starts at same offset in both Rust and WGSL
    _pad0: f32, _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32, _pad6: f32,

    // Note: array access in WGSL requires fixed size
    water_levels: array<WaterLevel, 16>,
}

// Camera uniform (shared with main render)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> water: WaterUniforms;
@group(1) @binding(1) var depth_texture: texture_depth_2d;
@group(1) @binding(2) var scene_texture: texture_2d<f32>;
@group(1) @binding(3) var tex_sampler: sampler;
@group(1) @binding(4) var presence_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle from vertex index
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Water presence info from texture sampling
struct WaterPresenceInfo {
    exists: f32,       // 0.0 = no water, 1.0 = has water
    level: f32,        // 0.0-1.0 water level (0=empty, 1=full)
    flow_dir: vec2<f32>, // Flow direction (-1 to +1 for X and Z)
    is_source: bool,   // Is this a source block?
    is_flowing: bool,  // Is this actively flowing?
}

// Sample water presence info at a world XZ position
fn sample_water_presence(world_x: f32, world_z: f32) -> WaterPresenceInfo {
    var info: WaterPresenceInfo;
    info.exists = 0.0;
    info.level = 0.0;
    info.flow_dir = vec2<f32>(0.0, 0.0);
    info.is_source = false;
    info.is_flowing = false;

    let half_size = water.presence_size * 0.5;
    let tex_x = (world_x - water.presence_center.x + half_size) / water.presence_size;
    let tex_z = (world_z - water.presence_center.y + half_size) / water.presence_size;

    // Check bounds
    if (tex_x < 0.0 || tex_x > 1.0 || tex_z < 0.0 || tex_z > 1.0) {
        return info;
    }

    // Sample presence texture (RGBA format)
    // R = water level, G = flow_dir_x, B = flow_dir_z, A = flags
    let sample = textureSampleLevel(presence_texture, tex_sampler, vec2<f32>(tex_x, tex_z), 0.0);

    // R channel: water level (0-1)
    info.level = sample.r;
    info.exists = select(0.0, 1.0, sample.r > 0.01);

    // G channel: flow direction X (0=−1, 0.5=0, 1=+1)
    info.flow_dir.x = (sample.g - 0.5) * 2.0;

    // B channel: flow direction Z (0=−1, 0.5=0, 1=+1)
    info.flow_dir.y = (sample.b - 0.5) * 2.0;

    // A channel: flags
    let flags = u32(sample.a * 255.0);
    info.is_source = (flags & 1u) != 0u;
    info.is_flowing = (flags & 2u) != 0u;

    return info;
}

// Legacy function for compatibility - just checks if water exists
fn check_water_presence(world_x: f32, world_z: f32) -> f32 {
    return sample_water_presence(world_x, world_z).exists;
}

// Hash function for noise
fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth noise
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

// Calculate animated wave normal with flow direction influence
fn get_wave_normal_with_flow(pos: vec2<f32>, time: f32, flow_dir: vec2<f32>, is_flowing: bool) -> vec3<f32> {
    let wave_speed = water.wave_speed * 0.5;

    // Stronger waves when flowing, gentler for still water
    let flow_strength = length(flow_dir);
    let wave_strength = water.wave_strength * select(0.3, 0.6, is_flowing);

    var dx = 0.0;
    var dz = 0.0;

    // Flow direction influence - waves move in flow direction
    let flow_offset = flow_dir * time * wave_speed * 2.0;

    // Layer 1: Large slow swells (follow flow direction)
    let p1 = pos * 0.015 + flow_offset * 0.3 + vec2<f32>(time * wave_speed * 0.2, time * wave_speed * 0.15);
    dx += sin(p1.x * 1.5 + p1.y * 1.0) * 0.4;
    dz += sin(p1.y * 1.5 - p1.x * 0.8) * 0.4;

    // Layer 2: Medium ripples (stronger influence from flow)
    let p2 = pos * 0.04 + flow_offset * 0.5 + vec2<f32>(time * wave_speed * 0.4, -time * wave_speed * 0.25);
    dx += sin(p2.x * 2.0 + p2.y * 1.8) * 0.35;
    dz += sin(p2.y * 2.0 - p2.x * 1.5) * 0.35;

    // Add directional bias based on flow
    if (flow_strength > 0.1) {
        dx += flow_dir.x * 0.3 * wave_strength;
        dz += flow_dir.y * 0.3 * wave_strength;
    }

    // Layer 3: Small detail (if quality allows)
    if (water.wave_octaves >= 3u) {
        let p3 = pos * 0.08 + flow_offset * 0.7 + vec2<f32>(-time * wave_speed * 0.3, time * wave_speed * 0.4);
        dx += sin(p3.x * 3.0 + p3.y * 2.5) * 0.2;
        dz += sin(p3.y * 2.8 - p3.x * 3.0) * 0.2;
    }

    // Layer 4: Fine detail (Ultra only)
    if (water.wave_octaves >= 4u) {
        let p4 = pos * 0.12 + vec2<f32>(time * wave_speed * 0.5, time * wave_speed * 0.35);
        dx += sin(p4.x * 4.0 + p4.y * 3.5) * 0.1;
        dz += sin(p4.y * 3.5 - p4.x * 4.0) * 0.1;
    }

    return normalize(vec3<f32>(-dx * wave_strength, 1.0, -dz * wave_strength));
}

// Calculate animated wave normal - subtle, calm water (legacy, no flow)
fn get_wave_normal(pos: vec2<f32>, time: f32) -> vec3<f32> {
    return get_wave_normal_with_flow(pos, time, vec2<f32>(0.0, 0.0), false);
}

// Legacy wave calculation kept for reference
fn get_wave_normal_simple(pos: vec2<f32>, time: f32) -> vec3<f32> {
    let wave_speed = water.wave_speed * 0.5; // Slower waves
    let wave_strength = water.wave_strength * 0.4; // Much gentler

    var dx = 0.0;
    var dz = 0.0;

    // Layer 1: Large slow swells (very gentle)
    let p1 = pos * 0.015 + vec2<f32>(time * wave_speed * 0.3, time * wave_speed * 0.2);
    dx += sin(p1.x * 1.5 + p1.y * 1.0) * 0.4;
    dz += sin(p1.y * 1.5 - p1.x * 0.8) * 0.4;

    // Layer 2: Medium ripples
    let p2 = pos * 0.04 + vec2<f32>(time * wave_speed * 0.5, -time * wave_speed * 0.3);
    dx += sin(p2.x * 2.0 + p2.y * 1.8) * 0.3;
    dz += sin(p2.y * 2.0 - p2.x * 1.5) * 0.3;

    // Layer 3: Small detail (if quality allows)
    if (water.wave_octaves >= 3u) {
        let p3 = pos * 0.08 + vec2<f32>(-time * wave_speed * 0.4, time * wave_speed * 0.5);
        dx += sin(p3.x * 3.0 + p3.y * 2.5) * 0.2;
        dz += sin(p3.y * 2.8 - p3.x * 3.0) * 0.2;
    }

    // Layer 4: Fine detail (Ultra only)
    if (water.wave_octaves >= 4u) {
        let p4 = pos * 0.12 + vec2<f32>(time * wave_speed * 0.6, time * wave_speed * 0.4);
        dx += sin(p4.x * 4.0 + p4.y * 3.5) * 0.1;
        dz += sin(p4.y * 3.5 - p4.x * 4.0) * 0.1;
    }

    return normalize(vec3<f32>(-dx * wave_strength, 1.0, -dz * wave_strength));
}

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos = water.inv_view_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Compute procedural sky color for reflection
fn compute_sky_color(dir: vec3<f32>) -> vec3<f32> {
    let brightness = water.ambient_color.a;
    let sun_dir = normalize(water.sun_direction.xyz);
    let sun_intensity = water.sun_direction.w;

    // Sky gradient based on height
    let sky_factor = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    // Base sky colors (warm palette matching the engine's style)
    let horizon_color = vec3<f32>(0.7, 0.75, 0.85) * brightness;
    let zenith_color = vec3<f32>(0.4, 0.55, 0.8) * brightness;
    let night_horizon = vec3<f32>(0.15, 0.12, 0.2) * brightness;
    let night_zenith = vec3<f32>(0.05, 0.05, 0.12) * brightness;

    // Blend day/night based on sun intensity
    let day_sky = mix(horizon_color, zenith_color, sky_factor);
    let night_sky = mix(night_horizon, night_zenith, sky_factor);
    var sky_color = mix(night_sky, day_sky, sun_intensity);

    // Add sun glow
    if (sun_intensity > 0.0) {
        let sun_dot = max(dot(dir, sun_dir), 0.0);
        let sun_glow = pow(sun_dot, 32.0) * sun_intensity * 0.3;
        sky_color += water.sun_color.rgb * sun_glow;
    }

    return sky_color;
}

// Schlick fresnel approximation
fn fresnel(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, water.fresnel_power);
}

// Screen-space reflection ray march
fn trace_ssr(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    if (water.ssr_steps == 0u) {
        return vec4<f32>(0.0);
    }

    let step_count = water.ssr_steps;
    let max_dist = water.ssr_max_distance;
    let thickness = water.ssr_thickness;

    let end_pos = origin + dir * max_dist;

    // Project start and end to screen space
    let start_clip = water.view_proj * vec4<f32>(origin, 1.0);
    let end_clip = water.view_proj * vec4<f32>(end_pos, 1.0);

    let start_ndc = start_clip.xyz / start_clip.w;
    let end_ndc = end_clip.xyz / end_clip.w;

    let start_uv = vec2<f32>(start_ndc.x * 0.5 + 0.5, 0.5 - start_ndc.y * 0.5);
    let end_uv = vec2<f32>(end_ndc.x * 0.5 + 0.5, 0.5 - end_ndc.y * 0.5);

    let delta_uv = (end_uv - start_uv) / f32(step_count);
    let delta_z = (end_ndc.z - start_ndc.z) / f32(step_count);

    var uv = start_uv;
    var ray_z = start_ndc.z;

    for (var i = 0u; i < step_count; i++) {
        uv += delta_uv;
        ray_z += delta_z;

        // Bounds check
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        // Sample depth buffer
        let sample_depth = textureSampleLevel(depth_texture, tex_sampler, uv, 0.0);

        // Check for intersection
        if (ray_z > sample_depth && ray_z - sample_depth < thickness) {
            // Binary search refinement
            var lo_uv = uv - delta_uv;
            var hi_uv = uv;
            var lo_z = ray_z - delta_z;
            var hi_z = ray_z;

            for (var j = 0u; j < 4u; j++) {
                let mid_uv = (lo_uv + hi_uv) * 0.5;
                let mid_z = (lo_z + hi_z) * 0.5;
                let mid_depth = textureSampleLevel(depth_texture, tex_sampler, mid_uv, 0.0);

                if (mid_z > mid_depth) {
                    hi_uv = mid_uv;
                    hi_z = mid_z;
                } else {
                    lo_uv = mid_uv;
                    lo_z = mid_z;
                }
            }

            let hit_uv = (lo_uv + hi_uv) * 0.5;

            // Smooth edge fade - wider margin for cleaner transitions
            let edge_dist = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
            let edge_fade = smoothstep(0.0, 0.15, edge_dist);

            // Distance fade - SSR fades out with distance
            let dist_factor = f32(i) / f32(step_count);
            let dist_fade = 1.0 - smoothstep(0.6, 1.0, dist_factor);

            // Sample scene at hit point
            let hit_color = textureSample(scene_texture, tex_sampler, hit_uv).rgb;
            return vec4<f32>(hit_color, edge_fade * dist_fade * 0.85);
        }
    }

    return vec4<f32>(0.0);
}

// DEBUG MODE:
// 0 = normal water
// 1 = magenta overlay (verify pass runs)
// 2 = diagnostic colors (red=depth fail, green=hit, blue=no intersection)
const DEBUG_MODE: u32 = 0u;

// Main fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // DEBUG: Show magenta overlay to verify shader runs
    if (DEBUG_MODE == 1u) {
        return vec4<f32>(1.0, 0.0, 1.0, 0.5);
    }

    let time = water.camera_pos.w;
    let cam_pos = water.camera_pos.xyz;

    // Reconstruct view ray
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let far_point = water.inv_view_proj * ndc;
    let ray_dir = normalize(far_point.xyz / far_point.w - cam_pos);

    // Sample scene depth
    let scene_depth = textureSampleLevel(depth_texture, tex_sampler, in.uv, 0.0);
    let scene_world_pos = reconstruct_world_pos(in.uv, scene_depth);

    // Find closest water level that intersects this ray
    var hit_t = 1e10;
    var hit_level = -1;
    var hit_height = 0.0;
    var hit_type = 0.0;
    var depth_test_failed = false;

    for (var i = 0u; i < water.water_level_count; i++) {
        let level = water.water_levels[i];
        if (level.height < -999.0) {
            continue; // Inactive level
        }

        // Ray-plane intersection: y = level.height
        // cam_pos.y + t * ray_dir.y = level.height
        // t = (level.height - cam_pos.y) / ray_dir.y

        if (abs(ray_dir.y) < 0.0001) {
            continue; // Ray parallel to water plane
        }

        let t = (level.height - cam_pos.y) / ray_dir.y;

        if (t > 0.0 && t < hit_t) {
            let hit_pos = cam_pos + ray_dir * t;

            // === PRESENCE TEST: Check if water actually exists at this XZ position ===
            // This is the primary test - only draw water where water blocks exist
            let presence = check_water_presence(hit_pos.x, hit_pos.z);
            if (presence < 0.5) {
                // No water at this location - skip
                depth_test_failed = true;
                continue;
            }

            // === DEPTH TEST 0: Underground cave/cavern check ===
            // If BOTH camera AND scene geometry are below the water level,
            // we're in an air-filled cave/cavern - don't render fake water plane
            // Use margin of 1.0 to allow underwater surface viewing when close to surface
            let cam_below_water = cam_pos.y < level.height - 1.0;
            let scene_below_water = scene_world_pos.y < level.height - 1.0;
            if (cam_below_water && scene_below_water) {
                depth_test_failed = true;
                continue;
            }

            // === DEPTH TEST 1: Terrain height check ===
            // If terrain is clearly above water, don't draw water on it
            // Use tighter threshold (0.1 instead of 0.5) for better precision
            if (scene_world_pos.y > level.height + 0.1) {
                depth_test_failed = true;
                continue;
            }

            // === DEPTH TEST 2: Distance comparison (world-space, avoids NDC precision banding) ===
            let water_dist = length(hit_pos - cam_pos);
            let scene_dist = length(scene_world_pos - cam_pos);

            // Water is visible if it's closer than scene geometry
            // Use larger bias (1%) to avoid precision artifacts
            let dist_bias = max(water_dist, scene_dist) * 0.01 + 0.5;
            if (water_dist < scene_dist + dist_bias) {
                hit_t = t;
                hit_level = i32(i);
                hit_height = level.height;
                hit_type = level.water_type;
            } else {
                depth_test_failed = true;
            }
        }
    }

    // DEBUG MODE 2: Diagnostic colors
    if (DEBUG_MODE == 2u) {
        if (hit_level >= 0) {
            // Hit and passed depth test - green
            return vec4<f32>(0.0, 1.0, 0.0, 0.7);
        } else if (depth_test_failed) {
            // Hit but depth test failed - red
            return vec4<f32>(1.0, 0.0, 0.0, 0.7);
        } else {
            // No intersection (ray parallel or behind camera) - discard to show scene
            discard;
        }
    }

    // No water hit - discard
    if (hit_level < 0) {
        discard;
    }

    // Calculate water surface position
    let water_pos = cam_pos + ray_dir * hit_t;
    let view_dir = -ray_dir;

    // Skip drawing water on vertical shore block faces
    // Only discard if scene geometry is VERY close to water level (within 0.3 blocks)
    // This catches vertical shore faces but not floors 1 block below water
    let scene_at_water_level = abs(scene_world_pos.y - hit_height) < 0.3;
    let looking_horizontal = abs(ray_dir.y) < 0.3; // Very steep horizontal angle
    if (scene_at_water_level && looking_horizontal) {
        discard;
    }

    // Sample water presence info for flow direction
    let water_info = sample_water_presence(water_pos.x, water_pos.z);

    // Calculate wave normal with flow direction influence
    let wave_normal = get_wave_normal_with_flow(water_pos.xz, time, water_info.flow_dir, water_info.is_flowing);

    // Distance-based normal fade (reduce waves at distance for stability)
    let dist = length(water_pos - cam_pos);
    let normal_fade = 1.0 - smoothstep(50.0, 200.0, dist);
    // Blend to flat normal at distance for stable reflections
    let final_normal = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), wave_normal, normal_fade * 0.7));

    // Calculate water depth (vertical distance to underwater geometry)
    let depth_diff = max(0.0, hit_height - scene_world_pos.y);

    // Fresnel - use mostly flat normal for stable reflections
    // Limit the "looking straight down = transparent" effect
    let flat_normal = vec3<f32>(0.0, 1.0, 0.0);
    let fresnel_normal = normalize(mix(flat_normal, final_normal, 0.3));
    let n_dot_v = max(dot(fresnel_normal, view_dir), 0.001);
    // Clamp n_dot_v to prevent extreme transparency when looking straight down
    let clamped_n_dot_v = min(n_dot_v, 0.85);
    let f = fresnel(clamped_n_dot_v, 0.04); // Higher F0 = more reflective baseline

    // Reflection direction
    let reflect_dir = reflect(-view_dir, final_normal);

    // SSR - DISABLED for now to debug banding issue
    // TODO: Re-enable with proper anti-banding solution
    var reflection_color = compute_sky_color(reflect_dir);

    // Only use SSR when looking at shallow angles (not down/up)
    // and add distance-based fade to reduce artifacts
    let view_vertical = abs(view_dir.y);
    if (view_vertical < 0.5 && water.ssr_steps > 0u) {
        let ssr_angle_fade = 1.0 - smoothstep(0.3, 0.5, view_vertical);
        let ssr_result = trace_ssr(water_pos + final_normal * 0.1, reflect_dir);
        reflection_color = mix(reflection_color, ssr_result.rgb, ssr_result.a * ssr_angle_fade * 0.7);
    }

    // Refraction (sample scene with distortion)
    var refracted_color = vec3<f32>(0.0);
    if (water.refraction_distortion > 0.001) {
        let distortion = final_normal.xz * water.refraction_distortion;
        let depth_fade = 1.0 - smoothstep(0.0, water.refraction_depth_fade, depth_diff);
        let offset = distortion * depth_fade;

        if (water.chromatic_aberration > 0.001) {
            // Chromatic aberration
            let r_uv = clamp(in.uv + offset * (1.0 + water.chromatic_aberration * 30.0), vec2<f32>(0.0), vec2<f32>(1.0));
            let g_uv = clamp(in.uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
            let b_uv = clamp(in.uv + offset * (1.0 - water.chromatic_aberration * 30.0), vec2<f32>(0.0), vec2<f32>(1.0));
            refracted_color = vec3<f32>(
                textureSample(scene_texture, tex_sampler, r_uv).r,
                textureSample(scene_texture, tex_sampler, g_uv).g,
                textureSample(scene_texture, tex_sampler, b_uv).b
            );
        } else {
            let refract_uv = clamp(in.uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
            refracted_color = textureSample(scene_texture, tex_sampler, refract_uv).rgb;
        }
    } else {
        refracted_color = textureSample(scene_texture, tex_sampler, in.uv).rgb;
    }

    // Depth-based water color
    let brightness = water.ambient_color.a;
    let shallow = water.shallow_color.rgb * brightness;
    let deep = water.deep_color.rgb * brightness;
    let max_depth = 24.0;
    let depth_factor = clamp(depth_diff / max_depth, 0.0, 1.0);
    let water_tint = mix(shallow, deep, depth_factor);

    // Blend refracted scene with water tint
    let underwater_visibility = exp(-depth_diff * 0.2);
    var water_color = mix(water_tint, refracted_color * 0.8 + water_tint * 0.2, underwater_visibility);

    // Final composition: blend water color with reflection based on Fresnel
    // Higher minimum reflection (35%) prevents flat/transparent look when looking down
    let reflection_strength = clamp(f * 1.2 + 0.25, 0.35, 0.90);
    var final_color = mix(water_color, reflection_color, reflection_strength);

    // Specular highlight from sun - subtle, focused
    let sun_dir = normalize(water.sun_direction.xyz);
    let sun_intensity = water.sun_direction.w;
    if (sun_dir.y > 0.0 && sun_intensity > 0.0) {
        let half_vec = normalize(sun_dir + view_dir);
        // Higher power = tighter, smaller highlight
        let spec = pow(max(dot(final_normal, half_vec), 0.0), 256.0);
        // Narrower threshold and reduced intensity
        let sun_spec = smoothstep(0.7, 0.85, spec) * sun_intensity * brightness;
        final_color += water.sun_color.rgb * sun_spec * 0.3;
    }

    // Shore foam
    let shore_foam_intensity = smoothstep(1.5, 0.0, depth_diff) * 0.4;
    let foam_noise = noise2(water_pos.xz * 0.3 + vec2<f32>(time * 0.1, time * 0.07));
    var foam = shore_foam_intensity * smoothstep(water.foam_threshold - 0.1, water.foam_threshold, foam_noise);

    // Flow foam - add extra foam where water is actively flowing
    if (water_info.is_flowing) {
        let flow_strength = length(water_info.flow_dir);
        // Animated foam patterns following flow direction
        let flow_offset = water_info.flow_dir * time * 0.5;
        let flow_foam_noise = noise2(water_pos.xz * 0.4 + flow_offset);
        let flow_foam = flow_strength * smoothstep(0.4, 0.7, flow_foam_noise) * 0.35;
        foam = max(foam, flow_foam);
    }

    // Extra foam at low water levels (visible flowing edges)
    let level_foam = (1.0 - water_info.level) * 0.2;
    foam = max(foam, level_foam * smoothstep(0.5, 0.7, foam_noise));

    let foam_color = vec3<f32>(0.9, 0.92, 0.95) * brightness;
    final_color = mix(final_color, foam_color, foam);

    // Alpha: water should always be visible, not see-through
    // Base opacity from water depth
    var alpha = mix(0.75, 0.95, depth_factor); // Higher base opacity
    alpha = mix(alpha, 1.0, foam);

    // Shore fade - only fade at very shallow edges (< 0.5 blocks)
    // 1-block-deep water should be fully visible
    let shore_fade = smoothstep(0.0, 0.5, depth_diff);

    // High minimum alpha - water should never be transparent/invisible
    let min_alpha = 0.75;
    let final_alpha = max(min_alpha, alpha) * shore_fade;

    // Ensure water is always at least somewhat visible (except at very edge)
    let final_alpha_clamped = max(final_alpha, shore_fade * 0.7);

    return vec4<f32>(final_color, final_alpha_clamped);
}
"#;
