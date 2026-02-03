//! Hybrid Morphological-Temporal Anti-Aliasing (HMTAA)
//!
//! A novel anti-aliasing system designed specifically for voxel engines:
//! - Depth-based edge detection (perfect for hard block boundaries)
//! - Morphological pattern matching for axis-aligned staircases
//! - Temporal accumulation with camera reprojection (no motion vectors needed)
//! - Adaptive sharpening to recover detail
//!
//! Pipeline:
//! 1. Edge detection: Find depth discontinuities and luma edges
//! 2. Blend pass: Apply morphological blending along detected edges
//! 3. Temporal pass: Accumulate with history using reprojection
//! 4. Sharpen pass: Recover detail with adaptive sharpening

/// AA quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AaQuality {
    Off,
    Low,       // FXAA-style fast blend, no temporal
    #[default]
    Medium,    // Morphological + light temporal
    High,      // Full morphological + temporal
    Ultra,     // Maximum quality + aggressive sharpening
}

impl AaQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            AaQuality::Off => "OFF",
            AaQuality::Low => "LOW",
            AaQuality::Medium => "MEDIUM",
            AaQuality::High => "HIGH",
            AaQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> AaQuality {
        match self {
            AaQuality::Off => AaQuality::Low,
            AaQuality::Low => AaQuality::Medium,
            AaQuality::Medium => AaQuality::High,
            AaQuality::High => AaQuality::Ultra,
            AaQuality::Ultra => AaQuality::Off,
        }
    }

    pub fn prev(&self) -> AaQuality {
        match self {
            AaQuality::Off => AaQuality::Ultra,
            AaQuality::Low => AaQuality::Off,
            AaQuality::Medium => AaQuality::Low,
            AaQuality::High => AaQuality::Medium,
            AaQuality::Ultra => AaQuality::High,
        }
    }

    /// Edge detection threshold (lower = more edges detected)
    pub fn edge_threshold(&self) -> f32 {
        match self {
            AaQuality::Off => 0.0,
            AaQuality::Low => 0.166,    // Conservative
            AaQuality::Medium => 0.125,
            AaQuality::High => 0.1,
            AaQuality::Ultra => 0.063,  // Aggressive
        }
    }

    /// Depth edge threshold (in NDC units, lower = more sensitive)
    pub fn depth_threshold(&self) -> f32 {
        match self {
            AaQuality::Off => 0.0,
            AaQuality::Low => 0.002,
            AaQuality::Medium => 0.001,
            AaQuality::High => 0.0008,
            AaQuality::Ultra => 0.0005,
        }
    }

    /// Temporal blend factor (higher = more history contribution)
    pub fn temporal_blend(&self) -> f32 {
        match self {
            AaQuality::Off => 0.0,
            AaQuality::Low => 0.0,      // No temporal in Low mode
            AaQuality::Medium => 0.75,
            AaQuality::High => 0.85,
            AaQuality::Ultra => 0.9,
        }
    }

    /// Sharpening strength
    pub fn sharpen_strength(&self) -> f32 {
        match self {
            AaQuality::Off => 0.0,
            AaQuality::Low => 0.2,
            AaQuality::Medium => 0.35,
            AaQuality::High => 0.45,
            AaQuality::Ultra => 0.6,
        }
    }

    /// Whether temporal accumulation is enabled
    pub fn temporal_enabled(&self) -> bool {
        matches!(self, AaQuality::Medium | AaQuality::High | AaQuality::Ultra)
    }
}

/// AA parameters (runtime adjustable)
#[derive(Debug, Clone, Copy)]
pub struct AaParams {
    /// Edge detection luma threshold (0.0 - 0.5)
    pub edge_threshold: f32,
    /// Depth discontinuity threshold (0.0 - 0.01)
    pub depth_threshold: f32,
    /// Subpixel quality - controls amount of subpixel aliasing removal (0.0 - 1.0)
    pub subpixel_quality: f32,
    /// Temporal blend factor (0.0 - 0.95)
    pub temporal_blend: f32,
    /// Sharpening strength (0.0 - 1.0)
    pub sharpen_strength: f32,
    /// Depth rejection threshold for temporal (0.0 - 0.1)
    pub depth_rejection: f32,
    /// Color rejection threshold for temporal (0.0 - 1.0)
    pub color_rejection: f32,
}

impl Default for AaParams {
    fn default() -> Self {
        Self {
            edge_threshold: 0.125,
            depth_threshold: 0.001,
            subpixel_quality: 0.75,
            temporal_blend: 0.85,
            sharpen_strength: 0.4,
            depth_rejection: 0.01,
            color_rejection: 0.25,
        }
    }
}

/// Uniform data for AA shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AaUniform {
    /// Inverse view-projection for reprojection
    pub inv_view_proj: [[f32; 4]; 4],
    /// Previous frame view-projection for reprojection
    pub prev_view_proj: [[f32; 4]; 4],
    /// Screen dimensions (xy), texel size (zw)
    pub screen_params: [f32; 4],
    /// edge_threshold (x), depth_threshold (y), subpixel_quality (z), temporal_blend (w)
    pub aa_params: [f32; 4],
    /// sharpen_strength (x), depth_rejection (y), color_rejection (z), frame_index (w)
    pub aa_params2: [f32; 4],
    /// Camera position for velocity rejection
    pub camera_pos: [f32; 4],
    /// Previous camera position
    pub prev_camera_pos: [f32; 4],
}

/// Anti-Aliasing post-processing system
pub struct AntiAlias {
    /// Quality preset
    pub quality: AaQuality,
    /// Runtime parameters
    pub params: AaParams,

    /// History texture (previous frame result)
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,

    /// Intermediate texture for multi-pass
    intermediate_texture: wgpu::Texture,
    intermediate_view: wgpu::TextureView,

    /// Samplers
    nearest_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,

    /// Bind group layouts
    aa_bind_group_layout: wgpu::BindGroupLayout,
    temporal_bind_group_layout: wgpu::BindGroupLayout,
    sharpen_bind_group_layout: wgpu::BindGroupLayout,

    /// Bind groups (created per-frame)
    aa_bind_group: Option<wgpu::BindGroup>,
    temporal_bind_group: Option<wgpu::BindGroup>,
    sharpen_bind_group: Option<wgpu::BindGroup>,

    /// Pipelines
    aa_pipeline: wgpu::RenderPipeline,
    temporal_pipeline: wgpu::RenderPipeline,
    sharpen_pipeline: wgpu::RenderPipeline,

    /// Uniform buffers
    uniform_buffer: wgpu::Buffer,

    /// Current dimensions
    width: u32,
    height: u32,

    /// Previous frame matrices for reprojection
    prev_view_proj: glam::Mat4,
    prev_camera_pos: glam::Vec3,

    /// Frame counter for jitter
    frame_index: u32,

    /// Surface format
    surface_format: wgpu::TextureFormat,
}

impl AntiAlias {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let quality = AaQuality::default();
        let params = AaParams::default();

        // Create history texture (stores previous frame for temporal)
        let history_texture = Self::create_color_texture(device, width, height, surface_format, "aa_history");
        let history_view = history_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create intermediate texture (for multi-pass)
        let intermediate_texture = Self::create_color_texture(device, width, height, surface_format, "aa_intermediate");
        let intermediate_view = intermediate_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Samplers
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aa_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aa_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group layouts
        let aa_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aa_bind_group_layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scene texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Scene sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let temporal_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aa_temporal_bind_group_layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Current frame (intermediate)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // History texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let sharpen_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aa_sharpen_bind_group_layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Source texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aa_uniform_buffer"),
            size: std::mem::size_of::<AaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create pipelines
        let aa_pipeline = Self::create_aa_pipeline(device, &aa_bind_group_layout, surface_format);
        let temporal_pipeline = Self::create_temporal_pipeline(device, &temporal_bind_group_layout, surface_format);
        let sharpen_pipeline = Self::create_sharpen_pipeline(device, &sharpen_bind_group_layout, surface_format);

        Self {
            quality,
            params,
            history_texture,
            history_view,
            intermediate_texture,
            intermediate_view,
            nearest_sampler,
            linear_sampler,
            aa_bind_group_layout,
            temporal_bind_group_layout,
            sharpen_bind_group_layout,
            aa_bind_group: None,
            temporal_bind_group: None,
            sharpen_bind_group: None,
            aa_pipeline,
            temporal_pipeline,
            sharpen_pipeline,
            uniform_buffer,
            width,
            height,
            prev_view_proj: glam::Mat4::IDENTITY,
            prev_camera_pos: glam::Vec3::ZERO,
            frame_index: 0,
            surface_format,
        }
    }

    fn create_color_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn create_aa_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aa_shader"),
            source: wgpu::ShaderSource::Wgsl(AA_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aa_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("aa_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    fn create_temporal_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aa_temporal_shader"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aa_temporal_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("aa_temporal_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    fn create_sharpen_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aa_sharpen_shader"),
            source: wgpu::ShaderSource::Wgsl(SHARPEN_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aa_sharpen_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("aa_sharpen_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    /// Create bind groups with current textures (called per frame)
    pub fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        scene_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        // AA pass bind group
        self.aa_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aa_bind_group"),
            layout: &self.aa_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        }));

        // Temporal pass bind group
        self.temporal_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aa_temporal_bind_group"),
            layout: &self.temporal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        }));

        // Sharpen pass bind group (reads from intermediate if no temporal, else needs updating)
        self.sharpen_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aa_sharpen_bind_group"),
            layout: &self.sharpen_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        }));
    }

    /// Update uniform buffers
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
    ) {
        let inv_view_proj = view_proj.inverse();

        let uniform = AaUniform {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            screen_params: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            aa_params: [
                self.quality.edge_threshold(),
                self.quality.depth_threshold(),
                self.params.subpixel_quality,
                self.quality.temporal_blend(),
            ],
            aa_params2: [
                self.quality.sharpen_strength(),
                self.params.depth_rejection,
                self.params.color_rejection,
                self.frame_index as f32,
            ],
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            prev_camera_pos: [self.prev_camera_pos.x, self.prev_camera_pos.y, self.prev_camera_pos.z, 0.0],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        // Store for next frame
        self.prev_view_proj = view_proj;
        self.prev_camera_pos = camera_pos;
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Render AA and apply to scene
    /// Returns the view that contains the final result
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        scene_texture: &wgpu::Texture,
    ) {
        if self.quality == AaQuality::Off {
            return;
        }

        let Some(aa_bind_group) = &self.aa_bind_group else {
            return;
        };

        // Pass 1: Morphological AA (scene -> intermediate)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aa_morphological_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.intermediate_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.aa_pipeline);
            pass.set_bind_group(0, aa_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Temporal accumulation (if enabled)
        if self.quality.temporal_enabled() {
            if let Some(temporal_bind_group) = &self.temporal_bind_group {
                // Render temporal pass to history (reusing scene as temp since we don't need it anymore)
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("aa_temporal_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: scene_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    pass.set_pipeline(&self.temporal_pipeline);
                    pass.set_bind_group(0, temporal_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Copy result back to intermediate for sharpen pass
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: scene_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &self.intermediate_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                );

                // Also update history
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: scene_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &self.history_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        // Pass 3: Adaptive sharpening (intermediate -> scene)
        if let Some(sharpen_bind_group) = &self.sharpen_bind_group {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aa_sharpen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.sharpen_pipeline);
            pass.set_bind_group(0, sharpen_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Resize textures when window size changes
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;

        // Recreate textures
        self.history_texture = Self::create_color_texture(device, width, height, self.surface_format, "aa_history");
        self.history_view = self.history_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.intermediate_texture = Self::create_color_texture(device, width, height, self.surface_format, "aa_intermediate");
        self.intermediate_view = self.intermediate_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Clear bind groups (need recreating)
        self.aa_bind_group = None;
        self.temporal_bind_group = None;
        self.sharpen_bind_group = None;

        // Reset temporal state on resize
        self.frame_index = 0;
    }

    /// Set quality preset
    pub fn set_quality(&mut self, quality: AaQuality) {
        if self.quality != quality {
            self.quality = quality;
            // Reset frame index when quality changes to avoid artifacts
            self.frame_index = 0;
        }
    }

    /// Cycle to next quality level
    pub fn cycle_quality(&mut self) -> AaQuality {
        self.set_quality(self.quality.next());
        self.quality
    }

    /// Cycle to previous quality level
    pub fn cycle_quality_prev(&mut self) -> AaQuality {
        self.set_quality(self.quality.prev());
        self.quality
    }

    /// Get current quality
    pub fn quality(&self) -> AaQuality {
        self.quality
    }

    /// Check if AA is enabled
    pub fn is_enabled(&self) -> bool {
        self.quality != AaQuality::Off
    }
}


// =============================================================================
// MORPHOLOGICAL AA SHADER
// =============================================================================
// Novel hybrid of FXAA edge detection with SMAA-style morphological blending
// Optimized for voxel engines with axis-aligned geometry

const AA_SHADER: &str = r#"
struct AaUniform {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    screen_params: vec4<f32>,    // width, height, 1/width, 1/height
    aa_params: vec4<f32>,        // edge_threshold, depth_threshold, subpixel_quality, temporal_blend
    aa_params2: vec4<f32>,       // sharpen_strength, depth_rejection, color_rejection, frame_index
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: AaUniform;
@group(0) @binding(1) var t_scene: texture_2d<f32>;
@group(0) @binding(2) var s_scene: sampler;
@group(0) @binding(3) var t_depth: texture_depth_2d;
@group(0) @binding(4) var s_depth: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Relative luminance (ITU-R BT.709)
fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Sample depth at offset
fn sample_depth(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(t_depth, s_depth, uv, 0.0);
}

// Check if there's a depth edge between two samples
fn is_depth_edge(d1: f32, d2: f32, threshold: f32) -> bool {
    // Convert from NDC depth to linear for better comparison
    // For voxel worlds, we care about sharp depth changes
    let diff = abs(d1 - d2);
    return diff > threshold;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = uniforms.screen_params.zw;
    let edge_threshold = uniforms.aa_params.x;
    let depth_threshold = uniforms.aa_params.y;
    let subpixel_quality = uniforms.aa_params.z;

    // Sample center and neighbors
    let center = textureSample(t_scene, s_scene, in.uv);
    let center_luma = luma(center.rgb);
    let center_depth = sample_depth(in.uv);

    // Skip sky pixels
    if (center_depth >= 0.9999) {
        return center;
    }

    // Sample 4-neighborhood for edge detection
    let n_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(0.0, -texel.y));
    let s_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(0.0, texel.y));
    let e_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(texel.x, 0.0));
    let w_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(-texel.x, 0.0));

    let n_luma = luma(n_color.rgb);
    let s_luma = luma(s_color.rgb);
    let e_luma = luma(e_color.rgb);
    let w_luma = luma(w_color.rgb);

    // Sample depths
    let n_depth = sample_depth(in.uv + vec2<f32>(0.0, -texel.y));
    let s_depth = sample_depth(in.uv + vec2<f32>(0.0, texel.y));
    let e_depth = sample_depth(in.uv + vec2<f32>(texel.x, 0.0));
    let w_depth = sample_depth(in.uv + vec2<f32>(-texel.x, 0.0));

    // Compute luma range
    let luma_min = min(min(min(n_luma, s_luma), min(e_luma, w_luma)), center_luma);
    let luma_max = max(max(max(n_luma, s_luma), max(e_luma, w_luma)), center_luma);
    let luma_range = luma_max - luma_min;

    // Check for luma edge
    let has_luma_edge = luma_range > max(edge_threshold * 0.5, edge_threshold * luma_max);

    // Check for depth edges (voxel-specific: hard depth discontinuities)
    let has_depth_edge_h = is_depth_edge(e_depth, w_depth, depth_threshold) ||
                          is_depth_edge(center_depth, e_depth, depth_threshold) ||
                          is_depth_edge(center_depth, w_depth, depth_threshold);
    let has_depth_edge_v = is_depth_edge(n_depth, s_depth, depth_threshold) ||
                          is_depth_edge(center_depth, n_depth, depth_threshold) ||
                          is_depth_edge(center_depth, s_depth, depth_threshold);
    let has_depth_edge = has_depth_edge_h || has_depth_edge_v;

    // Early out if no edge detected
    if (!has_luma_edge && !has_depth_edge) {
        return center;
    }

    // Sample diagonal neighbors for better edge detection
    let ne_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(texel.x, -texel.y));
    let nw_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(-texel.x, -texel.y));
    let se_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(texel.x, texel.y));
    let sw_color = textureSample(t_scene, s_scene, in.uv + vec2<f32>(-texel.x, texel.y));

    let ne_luma = luma(ne_color.rgb);
    let nw_luma = luma(nw_color.rgb);
    let se_luma = luma(se_color.rgb);
    let sw_luma = luma(sw_color.rgb);

    // Determine edge direction using Sobel-like gradients
    let edge_h = abs(n_luma + 2.0 * e_luma + ne_luma - s_luma - 2.0 * w_luma - sw_luma);
    let edge_v = abs(e_luma + 2.0 * n_luma + ne_luma - w_luma - 2.0 * s_luma - sw_luma);

    let is_horizontal = edge_h >= edge_v;

    // Get the two neighbors along the edge
    var luma_neg: f32;
    var luma_pos: f32;
    var color_neg: vec4<f32>;
    var color_pos: vec4<f32>;
    var depth_neg: f32;
    var depth_pos: f32;

    if (is_horizontal) {
        luma_neg = n_luma;
        luma_pos = s_luma;
        color_neg = n_color;
        color_pos = s_color;
        depth_neg = n_depth;
        depth_pos = s_depth;
    } else {
        luma_neg = w_luma;
        luma_pos = e_luma;
        color_neg = w_color;
        color_pos = e_color;
        depth_neg = w_depth;
        depth_pos = e_depth;
    }

    // Compute gradient
    let gradient_neg = abs(luma_neg - center_luma);
    let gradient_pos = abs(luma_pos - center_luma);

    // Determine which side is the edge
    let step_length = select(texel.x, texel.y, is_horizontal);
    var offset: f32;
    var blend_color: vec4<f32>;

    if (gradient_neg >= gradient_pos) {
        // Edge is on negative side
        offset = -step_length * 0.5;
        blend_color = color_neg;
    } else {
        // Edge is on positive side
        offset = step_length * 0.5;
        blend_color = color_pos;
    }

    // Compute subpixel blend factor
    // Use depth edge strength to enhance blending at block boundaries
    var blend: f32;

    if (has_depth_edge) {
        // For depth edges, use a fixed blend to smooth the hard transition
        blend = 0.5 * subpixel_quality;
    } else {
        // For color edges, compute blend based on local contrast
        let local_avg = (n_luma + s_luma + e_luma + w_luma) * 0.25;
        let avg_with_diag = (local_avg + ne_luma + nw_luma + se_luma + sw_luma) * 0.2;
        blend = saturate(abs(center_luma - avg_with_diag) / luma_range);
        blend = (blend * blend) * subpixel_quality;
    }

    // Voxel-specific: Limit blending to preserve texture detail
    blend = min(blend, 0.5);

    // Sample at subpixel offset for morphological blend
    var final_uv = in.uv;
    if (is_horizontal) {
        final_uv.y += offset * blend;
    } else {
        final_uv.x += offset * blend;
    }

    let blended = textureSample(t_scene, s_scene, final_uv);

    // Mix between center and blend based on computed factor
    return mix(center, blended, blend);
}
"#;


// =============================================================================
// TEMPORAL ACCUMULATION SHADER
// =============================================================================
// Uses camera reprojection for temporal stability without motion vectors

const TEMPORAL_SHADER: &str = r#"
struct AaUniform {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    screen_params: vec4<f32>,
    aa_params: vec4<f32>,
    aa_params2: vec4<f32>,
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: AaUniform;
@group(0) @binding(1) var t_current: texture_2d<f32>;
@group(0) @binding(2) var s_linear: sampler;
@group(0) @binding(3) var t_history: texture_2d<f32>;
@group(0) @binding(4) var t_depth: texture_depth_2d;
@group(0) @binding(5) var s_depth: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos = uniforms.inv_view_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Project world position to previous frame UV
fn reproject(world_pos: vec3<f32>) -> vec2<f32> {
    let prev_clip = uniforms.prev_view_proj * vec4<f32>(world_pos, 1.0);
    var prev_ndc = prev_clip.xy / prev_clip.w;
    prev_ndc = prev_ndc * 0.5 + 0.5;
    prev_ndc.y = 1.0 - prev_ndc.y;
    return prev_ndc;
}

// YCoCg color space conversion for better temporal stability
fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y  = 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b;
    let co = 0.5 * rgb.r - 0.5 * rgb.b;
    let cg = -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    return vec3<f32>(r, g, b);
}

// Clip history color to neighborhood AABB (variance clipping)
fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, center: vec3<f32>, history: vec3<f32>) -> vec3<f32> {
    let dir = history - center;
    let dir_abs = abs(dir);

    // Avoid division by zero
    if (max(max(dir_abs.x, dir_abs.y), dir_abs.z) < 0.0001) {
        return history;
    }

    let aabb_center = (aabb_min + aabb_max) * 0.5;
    let aabb_extent = (aabb_max - aabb_min) * 0.5;

    // Compute clip factor
    let clip_factor = min(min(
        select(1e10, aabb_extent.x / dir_abs.x, dir_abs.x > 0.0001),
        select(1e10, aabb_extent.y / dir_abs.y, dir_abs.y > 0.0001)),
        select(1e10, aabb_extent.z / dir_abs.z, dir_abs.z > 0.0001));

    let t = saturate(1.0 / max(clip_factor, 0.0001));
    return mix(history, center, t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let temporal_blend = uniforms.aa_params.w;
    let depth_rejection = uniforms.aa_params2.y;
    let color_rejection = uniforms.aa_params2.z;
    let frame_index = uniforms.aa_params2.w;
    let texel = uniforms.screen_params.zw;

    // Get current frame color
    let current = textureSample(t_current, s_linear, in.uv);
    let depth = textureSampleLevel(t_depth, s_depth, in.uv, 0.0);

    // Skip sky pixels - no temporal for sky
    if (depth >= 0.9999) {
        return current;
    }

    // Reconstruct world position and reproject to previous frame
    let world_pos = reconstruct_world_pos(in.uv, depth);
    let prev_uv = reproject(world_pos);

    // Check if reprojected UV is valid
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        return current;
    }

    // Sample history with bilinear filtering
    let history = textureSample(t_history, s_linear, prev_uv);

    // First frame or invalid history
    if (frame_index < 1.0) {
        return current;
    }

    // Build neighborhood AABB for variance clipping (3x3)
    var color_min = current.rgb;
    var color_max = current.rgb;
    var color_avg = vec3<f32>(0.0);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let sample_color = textureSample(t_current, s_linear, in.uv + offset).rgb;
            color_min = min(color_min, sample_color);
            color_max = max(color_max, sample_color);
            color_avg += sample_color;
        }
    }
    color_avg /= 9.0;

    // Expand AABB slightly for smoother result
    let aabb_margin = (color_max - color_min) * 0.25;
    color_min -= aabb_margin;
    color_max += aabb_margin;

    // Convert to YCoCg for better clipping
    let current_ycocg = rgb_to_ycocg(current.rgb);
    let history_ycocg = rgb_to_ycocg(history.rgb);
    let aabb_min_ycocg = rgb_to_ycocg(color_min);
    let aabb_max_ycocg = rgb_to_ycocg(color_max);
    let center_ycocg = rgb_to_ycocg(color_avg);

    // Clip history to neighborhood
    let clipped_ycocg = clip_aabb(aabb_min_ycocg, aabb_max_ycocg, center_ycocg, history_ycocg);
    let clipped_history = ycocg_to_rgb(clipped_ycocg);

    // Compute color difference for adaptive rejection
    let color_diff = length(current.rgb - clipped_history);
    let rejection_factor = smoothstep(0.0, color_rejection, color_diff);

    // Adjust blend based on color difference
    let adjusted_blend = temporal_blend * (1.0 - rejection_factor * 0.5);

    // Blend current with clipped history
    let result = mix(current.rgb, clipped_history, adjusted_blend);

    return vec4<f32>(result, current.a);
}
"#;


// =============================================================================
// ADAPTIVE SHARPENING SHADER
// =============================================================================
// Recovers detail lost to AA blending with edge-aware sharpening

const SHARPEN_SHADER: &str = r#"
struct AaUniform {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    screen_params: vec4<f32>,
    aa_params: vec4<f32>,
    aa_params2: vec4<f32>,
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: AaUniform;
@group(0) @binding(1) var t_source: texture_2d<f32>;
@group(0) @binding(2) var s_source: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sharpen_strength = uniforms.aa_params2.x;
    let texel = uniforms.screen_params.zw;

    // Sample center and cross neighbors
    let center = textureSample(t_source, s_source, in.uv);

    // If no sharpening needed, early out
    if (sharpen_strength < 0.001) {
        return center;
    }

    let n = textureSample(t_source, s_source, in.uv + vec2<f32>(0.0, -texel.y));
    let s = textureSample(t_source, s_source, in.uv + vec2<f32>(0.0, texel.y));
    let e = textureSample(t_source, s_source, in.uv + vec2<f32>(texel.x, 0.0));
    let w = textureSample(t_source, s_source, in.uv + vec2<f32>(-texel.x, 0.0));

    // Compute local average
    let avg = (n.rgb + s.rgb + e.rgb + w.rgb) * 0.25;

    // Compute local contrast
    let center_luma = luma(center.rgb);
    let avg_luma = luma(avg);
    let local_contrast = abs(center_luma - avg_luma);

    // Adaptive sharpening: sharpen more in low-contrast areas (where AA blurred)
    // Less sharpening in high-contrast areas (edges that should stay smooth)
    let adaptive_strength = sharpen_strength * (1.0 - smoothstep(0.0, 0.25, local_contrast));

    // CAS-style sharpening (Contrast Adaptive Sharpening)
    // Enhances detail while avoiding oversharpening at edges
    let min_luma = min(min(min(luma(n.rgb), luma(s.rgb)), min(luma(e.rgb), luma(w.rgb))), center_luma);
    let max_luma = max(max(max(luma(n.rgb), luma(s.rgb)), max(luma(e.rgb), luma(w.rgb))), center_luma);

    // Compute sharpening weight based on local contrast
    let d = max(0.0, 1.0 - max_luma + min_luma);
    let sharpening = adaptive_strength * d;

    // Apply unsharp mask with adaptive strength
    let diff = center.rgb - avg;
    var sharpened = center.rgb + diff * sharpening;

    // Clamp to prevent ringing artifacts
    let neighborhood_min = min(min(n.rgb, s.rgb), min(e.rgb, w.rgb));
    let neighborhood_max = max(max(n.rgb, s.rgb), max(e.rgb, w.rgb));

    // Allow slight overshoot for detail, but limit it
    let overshoot = 0.1 * sharpen_strength;
    sharpened = clamp(sharpened, neighborhood_min - overshoot, neighborhood_max + overshoot);

    return vec4<f32>(sharpened, center.a);
}
"#;
