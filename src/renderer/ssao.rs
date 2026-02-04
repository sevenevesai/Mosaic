//! Screen-Space Ambient Occlusion (SSAO)
//!
//! Implements hemisphere-sampled SSAO with bilateral blur for soft contact shadows.
//! Uses depth-based normal reconstruction since the engine uses forward rendering.
//!
//! Pipeline:
//! 1. SSAO pass: Sample hemisphere around each pixel, compare depths
//! 2. Blur pass (H): Horizontal bilateral blur preserving edges
//! 3. Blur pass (V): Vertical bilateral blur preserving edges
//! 4. Apply pass: Multiply AO onto scene ambient lighting

use wgpu::util::DeviceExt;

/// SSAO quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SsaoQuality {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl SsaoQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            SsaoQuality::Off => "OFF",
            SsaoQuality::Low => "LOW",
            SsaoQuality::Medium => "MEDIUM",
            SsaoQuality::High => "HIGH",
            SsaoQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> SsaoQuality {
        match self {
            SsaoQuality::Off => SsaoQuality::Low,
            SsaoQuality::Low => SsaoQuality::Medium,
            SsaoQuality::Medium => SsaoQuality::High,
            SsaoQuality::High => SsaoQuality::Ultra,
            SsaoQuality::Ultra => SsaoQuality::Off,
        }
    }

    /// Get sample count for this quality level
    pub fn sample_count(&self) -> u32 {
        match self {
            SsaoQuality::Off => 0,
            SsaoQuality::Low => 12,
            SsaoQuality::Medium => 24,
            SsaoQuality::High => 32,
            SsaoQuality::Ultra => 64,
        }
    }

    /// Get blur kernel size for this quality level
    pub fn blur_size(&self) -> u32 {
        match self {
            SsaoQuality::Off => 0,
            SsaoQuality::Low => 3,
            SsaoQuality::Medium => 7,
            SsaoQuality::High => 7,
            SsaoQuality::Ultra => 11,
        }
    }

    /// Whether to use half resolution
    pub fn half_resolution(&self) -> bool {
        match self {
            SsaoQuality::Off => true,
            SsaoQuality::Low => true,
            SsaoQuality::Medium => true,
            SsaoQuality::High => false,
            SsaoQuality::Ultra => false,
        }
    }

    /// Default radius for this quality level (in blocks)
    pub fn default_radius(&self) -> f32 {
        match self {
            SsaoQuality::Off => 0.0,
            SsaoQuality::Low => 1.5,
            SsaoQuality::Medium => 2.0,
            SsaoQuality::High => 2.5,
            SsaoQuality::Ultra => 3.0,
        }
    }
}

/// SSAO parameters (runtime adjustable)
#[derive(Debug, Clone, Copy)]
pub struct SsaoParams {
    /// World-space radius in blocks (0.5 - 6.0)
    pub radius: f32,
    /// Intensity multiplier (0.0 - 1.5)
    pub intensity: f32,
    /// Surface bias to prevent self-occlusion (0.001 - 0.1)
    pub bias: f32,
    /// Edge sharpness for bilateral blur (0.0 - 1.0)
    pub sharpness: f32,
    /// Distance where AO starts fading (in blocks)
    pub fade_start: f32,
    /// Distance where AO is fully gone (in blocks)
    pub fade_end: f32,
}

impl Default for SsaoParams {
    fn default() -> Self {
        Self {
            radius: 2.0,
            intensity: 0.55,
            bias: 0.025,
            sharpness: 0.5,
            fade_start: 60.0,
            fade_end: 100.0,
        }
    }
}

/// Uniform data for SSAO shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoUniform {
    /// Inverse projection matrix for view-space reconstruction
    pub inv_proj: [[f32; 4]; 4],
    /// Projection matrix for reprojection
    pub proj: [[f32; 4]; 4],
    /// Screen dimensions (xy), near plane (z), far plane (w)
    pub screen_params: [f32; 4],
    /// Radius (x), intensity (y), bias (z), sample_count (w)
    pub ssao_params: [f32; 4],
    /// Fade start (x), fade end (y), sharpness (z), unused (w)
    pub fade_params: [f32; 4],
    /// AO texture dimensions (xy), frame count for temporal noise (z), unused (w)
    pub ao_params: [f32; 4],
}

/// Uniform data for blur shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurUniform {
    /// Direction: (1,0) for horizontal, (0,1) for vertical
    pub direction: [f32; 2],
    /// 1.0 / texture_width, 1.0 / texture_height (AO texture)
    pub texel_size: [f32; 2],
    /// Sharpness for edge preservation
    pub sharpness: f32,
    /// Blur kernel radius (in texels)
    pub blur_radius: f32,
    /// Depth texture texel size (may differ from AO if half-res)
    pub depth_texel_size: [f32; 2],
}

/// SSAO post-processing system
pub struct Ssao {
    /// Quality preset
    pub quality: SsaoQuality,
    /// Runtime parameters
    pub params: SsaoParams,

    /// AO texture (single channel, half or full res)
    ao_texture: wgpu::Texture,
    ao_view: wgpu::TextureView,

    /// Blur ping-pong texture
    blur_texture: wgpu::Texture,
    blur_view: wgpu::TextureView,

    /// 4x4 noise texture for kernel rotation
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,

    /// Samplers
    nearest_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,

    /// Bind group layouts
    ssao_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    apply_bind_group_layout: wgpu::BindGroupLayout,

    /// SSAO bind group (created per-frame with depth texture)
    ssao_bind_group: Option<wgpu::BindGroup>,

    /// Blur bind groups (for H and V passes, created per-frame with depth texture)
    blur_h_bind_group: Option<wgpu::BindGroup>,
    blur_v_bind_group: Option<wgpu::BindGroup>,

    /// Apply bind group
    apply_bind_group: wgpu::BindGroup,

    /// Pipelines
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    apply_pipeline: wgpu::RenderPipeline,

    /// Uniform buffers
    ssao_uniform_buffer: wgpu::Buffer,
    blur_h_uniform_buffer: wgpu::Buffer,
    blur_v_uniform_buffer: wgpu::Buffer,

    /// Current dimensions (AO texture)
    width: u32,
    height: u32,
    /// Full screen dimensions (for depth sampling)
    screen_width: u32,
    screen_height: u32,
    /// Whether using half resolution
    half_res: bool,
    /// Frame counter for temporal noise variation
    frame_count: u32,
}

impl Ssao {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let quality = SsaoQuality::default();
        let params = SsaoParams::default();
        let half_res = quality.half_resolution();

        let (ao_width, ao_height) = if half_res {
            (width / 2, height / 2)
        } else {
            (width, height)
        };

        // Create AO texture (R8 is sufficient for single-channel AO)
        let ao_texture = Self::create_ao_texture(device, ao_width, ao_height, "ssao_ao");
        let ao_view = ao_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create blur texture (same size as AO)
        let blur_texture = Self::create_ao_texture(device, ao_width, ao_height, "ssao_blur");
        let blur_view = blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create 4x4 noise texture
        let (noise_texture, noise_view) = Self::create_noise_texture(device);

        // Samplers
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_noise_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Bind group layouts
        let ssao_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bind_group_layout"),
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
                // Depth texture
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
                // Depth sampler (non-filtering for depth)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Noise texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Noise sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_blur_bind_group_layout"),
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
                // AO texture
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
                // Depth texture (for edge detection)
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

        let apply_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_apply_bind_group_layout"),
            entries: &[
                // AO texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create uniform buffers
        let ssao_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_uniform_buffer"),
            size: std::mem::size_of::<SsaoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blur_h_uniform = BlurUniform {
            direction: [1.0, 0.0],
            texel_size: [1.0 / ao_width as f32, 1.0 / ao_height as f32],
            sharpness: params.sharpness,
            blur_radius: (quality.blur_size() / 2) as f32,
            depth_texel_size: [1.0 / width as f32, 1.0 / height as f32],
        };
        let blur_h_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_blur_h_uniform"),
            contents: bytemuck::cast_slice(&[blur_h_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blur_v_uniform = BlurUniform {
            direction: [0.0, 1.0],
            texel_size: [1.0 / ao_width as f32, 1.0 / ao_height as f32],
            sharpness: params.sharpness,
            blur_radius: (quality.blur_size() / 2) as f32,
            depth_texel_size: [1.0 / width as f32, 1.0 / height as f32],
        };
        let blur_v_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_blur_v_uniform"),
            contents: bytemuck::cast_slice(&[blur_v_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create pipelines
        let ssao_pipeline = Self::create_ssao_pipeline(device, &ssao_bind_group_layout);
        let blur_pipeline = Self::create_blur_pipeline(device, &blur_bind_group_layout);
        let apply_pipeline = Self::create_apply_pipeline(device, &apply_bind_group_layout, surface_format);

        // Blur bind groups will be created per-frame with the actual depth texture
        // (they need the depth texture for edge-aware blurring)

        // Create apply bind group
        let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_apply_bind_group"),
            layout: &apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        Self {
            quality,
            params,
            ao_texture,
            ao_view,
            blur_texture,
            blur_view,
            noise_texture,
            noise_view,
            nearest_sampler,
            linear_sampler,
            noise_sampler,
            ssao_bind_group_layout,
            blur_bind_group_layout,
            apply_bind_group_layout,
            ssao_bind_group: None,
            blur_h_bind_group: None,
            blur_v_bind_group: None,
            apply_bind_group,
            ssao_pipeline,
            blur_pipeline,
            apply_pipeline,
            ssao_uniform_buffer,
            blur_h_uniform_buffer,
            blur_v_uniform_buffer,
            width: ao_width,
            height: ao_height,
            screen_width: width,
            screen_height: height,
            half_res,
            frame_count: 0,
        }
    }

    fn create_ao_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
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
            // Use R16Float for better precision (avoids banding)
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_noise_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        // Generate 4x4 random rotation vectors
        // Each texel stores a unit vector in the tangent plane (x, y, 0)
        // We store normalized (x, y) with z = 0 implied
        let mut noise_data = [0u8; 4 * 4 * 4]; // 4x4 pixels, RGBA8

        // Use a simple LCG for deterministic noise
        let mut seed: u32 = 0xDEADBEEF;
        for i in 0..16 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let angle = (seed as f32 / u32::MAX as f32) * std::f32::consts::TAU;

            // Store as normalized values in [0, 255] range
            // x = cos(angle), y = sin(angle), both in [-1, 1] -> [0, 255]
            let x = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
            let y = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;

            noise_data[i * 4] = x;
            noise_data[i * 4 + 1] = y;
            noise_data[i * 4 + 2] = 0;
            noise_data[i * 4 + 3] = 255;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_noise_texture"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Write noise data to texture (call after device creation with queue access)
    pub fn upload_noise_texture(&self, queue: &wgpu::Queue) {
        let mut noise_data = [0u8; 4 * 4 * 4];
        let mut seed: u32 = 0xDEADBEEF;
        for i in 0..16 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let angle = (seed as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            let x = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
            let y = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;
            noise_data[i * 4] = x;
            noise_data[i * 4 + 1] = y;
            noise_data[i * 4 + 2] = 0;
            noise_data[i * 4 + 3] = 255;
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
    }

    fn create_ssao_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_pipeline"),
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
                    format: wgpu::TextureFormat::R16Float,
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

    fn create_blur_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_blur_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_blur_pipeline"),
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
                    format: wgpu::TextureFormat::R16Float,
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

    fn create_apply_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_apply_shader"),
            source: wgpu::ShaderSource::Wgsl(APPLY_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_apply_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_apply_pipeline"),
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
                    // Multiplicative blending: result = dst * src
                    // src = AO value (ao, ao, ao, 1), dst = scene color
                    // With src_factor=Zero, dst_factor=Src: result = 0*src + src*dst = ao*scene
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::Src,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
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

    /// Create bind groups with actual depth texture (called per frame)
    pub fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
    ) {
        // SSAO bind group
        self.ssao_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bind_group"),
            layout: &self.ssao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ssao_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
            ],
        }));

        // Blur H bind group (AO -> blur)
        self.blur_h_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_blur_h_bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_h_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
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

        // Blur V bind group (blur -> AO)
        self.blur_v_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_blur_v_bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_v_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view),
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
    }

    /// Update uniform buffers
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        inv_proj: glam::Mat4,
        proj: glam::Mat4,
        near: f32,
        far: f32,
        screen_width: u32,
        screen_height: u32,
    ) {
        // Increment frame counter for temporal noise variation
        self.frame_count = self.frame_count.wrapping_add(1);

        let uniform = SsaoUniform {
            inv_proj: inv_proj.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            // Screen params uses FULL screen dimensions (for depth sampling)
            screen_params: [screen_width as f32, screen_height as f32, near, far],
            ssao_params: [
                self.params.radius,
                self.params.intensity,
                self.params.bias,
                self.quality.sample_count() as f32,
            ],
            fade_params: [
                self.params.fade_start,
                self.params.fade_end,
                self.params.sharpness,
                0.0,
            ],
            // AO params uses actual AO texture dimensions
            ao_params: [
                self.width as f32,
                self.height as f32,
                self.frame_count as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        // Depth texel size for proper edge detection when AO is half-res
        let depth_texel_size = [1.0 / screen_width as f32, 1.0 / screen_height as f32];

        // Update blur uniforms
        let blur_h = BlurUniform {
            direction: [1.0, 0.0],
            texel_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
            sharpness: self.params.sharpness,
            blur_radius: (self.quality.blur_size() / 2) as f32,
            depth_texel_size,
        };
        queue.write_buffer(&self.blur_h_uniform_buffer, 0, bytemuck::cast_slice(&[blur_h]));

        let blur_v = BlurUniform {
            direction: [0.0, 1.0],
            texel_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
            sharpness: self.params.sharpness,
            blur_radius: (self.quality.blur_size() / 2) as f32,
            depth_texel_size,
        };
        queue.write_buffer(&self.blur_v_uniform_buffer, 0, bytemuck::cast_slice(&[blur_v]));
    }

    /// Render SSAO and apply to scene
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
    ) {
        if self.quality == SsaoQuality::Off {
            return;
        }

        let Some(ssao_bind_group) = &self.ssao_bind_group else {
            return;
        };
        let Some(blur_h_bind_group) = &self.blur_h_bind_group else {
            return;
        };
        let Some(blur_v_bind_group) = &self.blur_v_bind_group else {
            return;
        };

        // Pass 1: SSAO sampling
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), // White = no occlusion
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, ssao_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Horizontal blur (AO -> blur texture)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur_h_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, blur_h_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3: Vertical blur (blur texture -> AO)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur_v_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, blur_v_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 4: Apply AO to scene (multiplicative blend)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_apply_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &self.apply_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Resize textures when window size changes
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Store full screen dimensions
        self.screen_width = width;
        self.screen_height = height;

        // Calculate AO texture dimensions (ensure minimum size of 1 and even values for half-res)
        let (ao_width, ao_height) = if self.half_res {
            // Use max to avoid zero-sized textures, and round to even to avoid sampling artifacts
            let w = (width / 2).max(1);
            let h = (height / 2).max(1);
            (w, h)
        } else {
            (width.max(1), height.max(1))
        };

        if ao_width == self.width && ao_height == self.height {
            return;
        }

        self.width = ao_width;
        self.height = ao_height;

        // Recreate textures
        self.ao_texture = Self::create_ao_texture(device, ao_width, ao_height, "ssao_ao");
        self.ao_view = self.ao_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.blur_texture = Self::create_ao_texture(device, ao_width, ao_height, "ssao_blur");
        self.blur_view = self.blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate apply bind group with new AO view
        self.apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_apply_bind_group"),
            layout: &self.apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        // Clear SSAO bind group (needs recreating with depth texture)
        self.ssao_bind_group = None;
    }

    /// Set quality preset
    pub fn set_quality(&mut self, device: &wgpu::Device, width: u32, height: u32, quality: SsaoQuality) {
        self.quality = quality;
        self.half_res = quality.half_resolution();
        self.params.radius = quality.default_radius();

        // Resize if half_res setting changed
        self.resize(device, width, height);
    }

    /// Cycle to next quality level
    pub fn cycle_quality(&mut self, device: &wgpu::Device, width: u32, height: u32) -> SsaoQuality {
        self.set_quality(device, width, height, self.quality.next());
        self.quality
    }

    /// Check if SSAO is enabled
    pub fn is_enabled(&self) -> bool {
        self.quality != SsaoQuality::Off
    }
}


// =============================================================================
// SSAO SAMPLING SHADER
// =============================================================================

const SSAO_SHADER: &str = r#"
struct SsaoUniform {
    inv_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
    screen_params: vec4<f32>,  // width, height, near, far (FULL screen for depth)
    ssao_params: vec4<f32>,    // radius, intensity, bias, sample_count
    fade_params: vec4<f32>,    // fade_start, fade_end, sharpness, unused
    ao_params: vec4<f32>,      // ao_width, ao_height, frame_count, unused
};

@group(0) @binding(0) var<uniform> uniforms: SsaoUniform;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_depth: sampler;
@group(0) @binding(3) var t_noise: texture_2d<f32>;
@group(0) @binding(4) var s_noise: sampler;

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

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_pos = uniforms.inv_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

// Reconstruct normal from depth using screen-space derivatives
fn reconstruct_normal(uv: vec2<f32>, view_pos: vec3<f32>) -> vec3<f32> {
    // Use FULL screen texel size for depth sampling
    let texel = vec2<f32>(1.0 / uniforms.screen_params.x, 1.0 / uniforms.screen_params.y);

    // Clamp UVs to avoid sampling outside screen (prevents edge artifacts)
    let uv_r = clamp(uv + vec2<f32>(texel.x, 0.0), vec2<f32>(0.001), vec2<f32>(0.999));
    let uv_l = clamp(uv - vec2<f32>(texel.x, 0.0), vec2<f32>(0.001), vec2<f32>(0.999));
    let uv_u = clamp(uv + vec2<f32>(0.0, texel.y), vec2<f32>(0.001), vec2<f32>(0.999));
    let uv_d = clamp(uv - vec2<f32>(0.0, texel.y), vec2<f32>(0.001), vec2<f32>(0.999));

    // Sample neighboring depths
    let depth_r = textureSampleLevel(t_depth, s_depth, uv_r, 0.0);
    let depth_l = textureSampleLevel(t_depth, s_depth, uv_l, 0.0);
    let depth_u = textureSampleLevel(t_depth, s_depth, uv_u, 0.0);
    let depth_d = textureSampleLevel(t_depth, s_depth, uv_d, 0.0);

    // Reconstruct neighboring positions
    let pos_r = reconstruct_position(uv_r, depth_r);
    let pos_l = reconstruct_position(uv_l, depth_l);
    let pos_u = reconstruct_position(uv_u, depth_u);
    let pos_d = reconstruct_position(uv_d, depth_d);

    // Use the smaller derivatives to avoid artifacts at depth discontinuities
    let dx_r = pos_r - view_pos;
    let dx_l = view_pos - pos_l;
    let dy_u = pos_u - view_pos;
    let dy_d = view_pos - pos_d;

    let dx = select(dx_r, dx_l, length(dx_l) < length(dx_r));
    let dy = select(dy_u, dy_d, length(dy_d) < length(dy_u));

    return normalize(cross(dy, dx));
}

// High-quality hash function for per-pixel randomness
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Generate hemisphere sample with cosine weighting
fn hemisphere_sample(i: u32, random_rot: vec2<f32>, extra_jitter: f32) -> vec3<f32> {
    let fi = f32(i);
    let sample_count = uniforms.ssao_params.w;

    // Spiral pattern with per-pixel rotation to break up patterns
    let angle = fi * 2.39996323 + random_rot.x * 6.28318 + extra_jitter * 0.5;
    let radius = sqrt((fi + 0.5 + extra_jitter * 0.3) / sample_count);

    let x = cos(angle) * radius;
    let y = sin(angle) * radius;
    let z = sqrt(max(0.0, 1.0 - x*x - y*y)); // Hemisphere (clamped to avoid sqrt of negative)

    // Scale radius to vary distance (closer samples more frequent)
    let scale = (fi + 1.0) / sample_count;
    let scaled_radius = mix(0.1, 1.0, scale * scale);

    return vec3<f32>(x, y, z) * scaled_radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let depth = textureSampleLevel(t_depth, s_depth, in.uv, 0.0);

    // Skip sky pixels
    if (depth >= 0.9999) {
        return 1.0;
    }

    let view_pos = reconstruct_position(in.uv, depth);

    // Distance fade
    let dist = length(view_pos);
    let fade_start = uniforms.fade_params.x;
    let fade_end = uniforms.fade_params.y;
    if (dist > fade_end) {
        return 1.0; // Early out for distant pixels
    }
    let dist_fade = 1.0 - smoothstep(fade_start, fade_end, dist);

    // Edge fade - reduce SSAO at screen edges to avoid normal reconstruction artifacts
    let edge_dist = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
    let edge_fade = smoothstep(0.0, 0.08, edge_dist);

    let fade = dist_fade * edge_fade;

    let normal = reconstruct_normal(in.uv, view_pos);

    // Use AO texture dimensions for noise scaling (not full screen)
    // This ensures noise tiles correctly at whatever resolution we're rendering at
    let ao_dims = uniforms.ao_params.xy;
    let frame = uniforms.ao_params.z;

    // Calculate pixel position in AO texture space
    let pixel_pos = in.uv * ao_dims;

    // Sample noise texture at correct scale for AO resolution
    // Each noise texel covers 4 AO pixels
    let noise_uv = pixel_pos / 4.0;
    let noise_base = textureSample(t_noise, s_noise, noise_uv).xy * 2.0 - 1.0;

    // Add per-pixel hash for extra variation to break up any remaining banding
    // Include frame count for temporal variation
    let pixel_hash = hash22(pixel_pos + vec2<f32>(frame * 0.618, frame * 0.381));
    let noise = noise_base + (pixel_hash * 2.0 - 1.0) * 0.3;

    // Extra jitter based on pixel position and frame
    let extra_jitter = hash21(pixel_pos + vec2<f32>(frame));

    // Build TBN matrix (tangent-bitangent-normal)
    let tangent_raw = vec3<f32>(noise.x, noise.y, 0.0);
    let tangent = normalize(tangent_raw - normal * dot(tangent_raw, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    // SSAO parameters
    let radius = uniforms.ssao_params.x;
    let intensity = uniforms.ssao_params.y;
    let bias = uniforms.ssao_params.z;
    let sample_count = u32(uniforms.ssao_params.w);

    var occlusion = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        // Get sample in tangent space and transform to view space
        let sample_ts = hemisphere_sample(i, noise, extra_jitter);
        let sample_vs = tbn * sample_ts;

        // Offset from fragment position
        let sample_pos = view_pos + sample_vs * radius;

        // Project to screen space
        let sample_clip = uniforms.proj * vec4<f32>(sample_pos, 1.0);
        var sample_uv = sample_clip.xy / sample_clip.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;

        // Skip samples outside screen
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        // Sample depth at projected position
        let sample_depth = textureSampleLevel(t_depth, s_depth, sample_uv, 0.0);
        let sample_view_pos = reconstruct_position(sample_uv, sample_depth);

        // Range check - only occlude if sample is close enough
        let range_check = smoothstep(0.0, 1.0, radius / max(abs(view_pos.z - sample_view_pos.z), 0.001));

        // Compare depths (with bias)
        let is_occluded = select(0.0, 1.0, sample_view_pos.z >= sample_pos.z + bias);

        occlusion += is_occluded * range_check;
    }

    occlusion = 1.0 - (occlusion / f32(sample_count)) * intensity;

    // Apply distance fade
    occlusion = mix(1.0, occlusion, fade);

    return clamp(occlusion, 0.0, 1.0);
}
"#;


// =============================================================================
// BILATERAL BLUR SHADER
// =============================================================================

const BLUR_SHADER: &str = r#"
struct BlurUniform {
    direction: vec2<f32>,
    texel_size: vec2<f32>,         // AO texture texel size
    sharpness: f32,
    blur_radius: f32,
    depth_texel_size: vec2<f32>,   // Depth texture texel size (may differ if half-res)
};

@group(0) @binding(0) var<uniform> uniforms: BlurUniform;
@group(0) @binding(1) var t_ao: texture_2d<f32>;
@group(0) @binding(2) var s_ao: sampler;
@group(0) @binding(3) var t_depth: texture_depth_2d;
@group(0) @binding(4) var s_depth: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let center_ao = textureSample(t_ao, s_ao, in.uv).r;
    let center_depth = textureSampleLevel(t_depth, s_depth, in.uv, 0.0);

    // Skip sky
    if (center_depth >= 0.9999) {
        return 1.0;
    }

    let blur_radius = i32(uniforms.blur_radius);
    let sharpness = uniforms.sharpness;

    // Depth-aware bilateral blur
    // Higher sharpness = more weight on depth similarity = sharper edges
    let depth_sigma = mix(0.001, 0.0001, sharpness);

    var sum = 0.0;
    var weight_sum = 0.0;

    for (var i = -blur_radius; i <= blur_radius; i++) {
        // Use AO texel size for AO texture sampling
        let ao_offset = vec2<f32>(f32(i)) * uniforms.direction * uniforms.texel_size;
        let sample_uv = clamp(in.uv + ao_offset, vec2<f32>(0.0), vec2<f32>(1.0));

        let sample_ao = textureSample(t_ao, s_ao, sample_uv).r;
        // Sample depth at the same UV (linear interpolation handles resolution difference)
        let sample_depth = textureSampleLevel(t_depth, s_depth, sample_uv, 0.0);

        // Spatial weight (Gaussian) - use blur_radius for sigma
        let sigma_sq = max(uniforms.blur_radius * uniforms.blur_radius, 1.0);
        let spatial_weight = exp(-f32(i * i) / (2.0 * sigma_sq));

        // Depth weight (bilateral)
        let depth_diff = abs(center_depth - sample_depth);
        let depth_weight = exp(-depth_diff * depth_diff / depth_sigma);

        let weight = spatial_weight * depth_weight;

        sum += sample_ao * weight;
        weight_sum += weight;
    }

    return sum / max(weight_sum, 0.0001);
}
"#;


// =============================================================================
// APPLY SHADER (MULTIPLICATIVE)
// =============================================================================

const APPLY_SHADER: &str = r#"
@group(0) @binding(0) var t_ao: texture_2d<f32>;
@group(0) @binding(1) var s_ao: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ao = textureSample(t_ao, s_ao, in.uv).r;

    // Output AO as gray value - blend state will multiply with scene
    return vec4<f32>(ao, ao, ao, 1.0);
}
"#;
