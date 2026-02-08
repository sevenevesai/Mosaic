//! Volumetric Light Shafts (God Rays)
//!
//! DDA-based world-space raymarching through the voxel volume for physically-motivated
//! in-scattering. Uses the same Amanatides & Woo traversal as the shadow system.
//!
//! Pipeline: Raymarch (half-res) -> Bilateral Blur (half-res) -> Composite (full-res additive)

/// Quality preset names
const QUALITY_NAMES: [&str; 5] = ["OFF", "LOW", "MEDIUM", "HIGH", "ULTRA"];

/// Volumetric lighting system with quality presets and bilateral blur
pub struct VolumetricLighting {
    // Half-resolution textures
    volumetric_texture: wgpu::Texture,
    volumetric_view: wgpu::TextureView,
    blur_texture: wgpu::Texture,
    blur_view: wgpu::TextureView,

    // Samplers
    depth_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,

    // Raymarch pass
    raymarch_bind_group_layout: wgpu::BindGroupLayout,
    raymarch_bind_group: Option<wgpu::BindGroup>,
    raymarch_pipeline: wgpu::RenderPipeline,

    // Bilateral blur pass
    blur_bind_group_layout: wgpu::BindGroupLayout,
    blur_uniform_buffer_h: wgpu::Buffer,
    blur_uniform_buffer_v: wgpu::Buffer,
    blur_bind_group_h: Option<wgpu::BindGroup>,
    blur_bind_group_v: Option<wgpu::BindGroup>,
    blur_pipeline: wgpu::RenderPipeline,

    // Composite pass
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,
    composite_pipeline: wgpu::RenderPipeline,

    // State
    width: u32,
    height: u32,
    quality: u32,
    frame_counter: u32,
}

/// Uniform data for volumetric raymarching
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricUniform {
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_direction: [f32; 4],  // xyz = direction, w = intensity
    pub sun_color: [f32; 4],     // rgb = color, a = density
    pub volume_center: [f32; 4], // xyz = center, w = volume_size (256)
    pub screen_params: [f32; 4], // x = width, y = height, z = quality (0-4), w = frame_counter
}

/// Uniform data for bilateral blur
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    /// xy = texel_size (1/width, 1/height), z = direction (0=H, 1=V), w = depth_threshold
    params: [f32; 4],
}

impl VolumetricLighting {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        voxel_bind_group_layout: &wgpu::BindGroupLayout,
        quality: u32,
    ) -> Self {
        let vol_width = (width / 2).max(1);
        let vol_height = (height / 2).max(1);

        // Create half-res textures
        let volumetric_texture = Self::create_half_res_texture(device, vol_width, vol_height, "volumetric_texture");
        let volumetric_view = volumetric_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blur_texture = Self::create_half_res_texture(device, vol_width, vol_height, "volumetric_blur_texture");
        let blur_view = blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Non-filtering sampler for depth texture
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetric_depth_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Linear sampler for compositing and blur
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetric_linear_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // === Raymarch pipeline ===
        let raymarch_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_raymarch_bgl"),
            entries: &[
                // Volumetric uniform
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
                // Depth sampler (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let raymarch_pipeline = Self::create_raymarch_pipeline(
            device,
            &raymarch_bind_group_layout,
            voxel_bind_group_layout,
        );

        // === Blur pipeline ===
        let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_blur_bgl"),
            entries: &[
                // Blur uniform
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
                // Input texture (volumetric or blur, depending on pass)
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
                // Linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Depth texture for bilateral weighting
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
                // Depth sampler (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let blur_pipeline = Self::create_blur_pipeline(device, &blur_bind_group_layout);

        // Create blur uniform buffers (H and V directions)
        use wgpu::util::DeviceExt;
        let blur_uniform_h = BlurUniform {
            params: [1.0 / vol_width as f32, 1.0 / vol_height as f32, 0.0, 2.0],
        };
        let blur_uniform_v = BlurUniform {
            params: [1.0 / vol_width as f32, 1.0 / vol_height as f32, 1.0, 2.0],
        };
        let blur_uniform_buffer_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("volumetric_blur_uniform_h"),
            contents: bytemuck::cast_slice(&[blur_uniform_h]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur_uniform_buffer_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("volumetric_blur_uniform_v"),
            contents: bytemuck::cast_slice(&[blur_uniform_v]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Composite pipeline ===
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_composite_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let composite_pipeline = Self::create_composite_pipeline(
            device,
            &composite_bind_group_layout,
            surface_format,
        );

        // Default composite reads from blur_texture (after blur pass)
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_composite_bg"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        Self {
            volumetric_texture,
            volumetric_view,
            blur_texture,
            blur_view,
            depth_sampler,
            linear_sampler,
            raymarch_bind_group_layout,
            raymarch_bind_group: None,
            raymarch_pipeline,
            blur_bind_group_layout,
            blur_uniform_buffer_h,
            blur_uniform_buffer_v,
            blur_bind_group_h: None,
            blur_bind_group_v: None,
            blur_pipeline,
            composite_bind_group_layout,
            composite_bind_group,
            composite_pipeline,
            width: vol_width,
            height: vol_height,
            quality,
            frame_counter: 0,
        }
    }

    fn create_half_res_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_raymarch_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        voxel_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_raymarch_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_RAYMARCH_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_raymarch_layout"),
            bind_group_layouts: &[bind_group_layout, voxel_bind_group_layout],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_raymarch_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
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
            label: Some("volumetric_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_BLUR_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_blur_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_blur_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
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

    fn create_composite_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_COMPOSITE_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_composite_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_composite_pipeline"),
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
                    // Additive blending: src + dst
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
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

    /// Create raymarch bind group with depth texture (called each frame)
    pub fn create_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) {
        self.raymarch_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_raymarch_bg"),
            layout: &self.raymarch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));

        // Create blur bind groups:
        // H pass: reads volumetric_texture, writes to blur_texture
        self.blur_bind_group_h = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_blur_bg_h"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_uniform_buffer_h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.volumetric_view),
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
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));

        // V pass: reads blur_texture, writes to volumetric_texture
        self.blur_bind_group_v = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_blur_bg_v"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_uniform_buffer_v.as_entire_binding(),
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
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));
    }

    /// Render volumetric light shafts (3 passes: raymarch -> blur -> composite)
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        voxel_bind_group: &wgpu::BindGroup,
    ) {
        if self.quality == 0 {
            return;
        }

        let Some(raymarch_bg) = &self.raymarch_bind_group else { return; };
        let Some(blur_bg_h) = &self.blur_bind_group_h else { return; };

        // Pass 1: Raymarch at half resolution
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_raymarch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.volumetric_view,
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
            pass.set_pipeline(&self.raymarch_pipeline);
            pass.set_bind_group(0, raymarch_bg, &[]);
            pass.set_bind_group(1, voxel_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Bilateral blur
        // For LOW quality: H blur only (volumetric -> blur)
        // For MEDIUM+: H + V blur (volumetric -> blur -> volumetric)
        {
            // Horizontal blur: volumetric_texture -> blur_texture
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_blur_h_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur_view,
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
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, blur_bg_h, &[]);
            pass.draw(0..3, 0..1);
        }

        // Vertical blur for MEDIUM+ (blur_texture -> volumetric_texture)
        if self.quality >= 2 {
            if let Some(blur_bg_v) = &self.blur_bind_group_v {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("volumetric_blur_v_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.volumetric_view,
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
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, blur_bg_v, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // For HIGH/ULTRA: second blur pass (volumetric -> blur -> volumetric again)
        if self.quality >= 3 {
            // H pass again
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("volumetric_blur_h2_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_view,
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
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, blur_bg_h, &[]);
                pass.draw(0..3, 0..1);
            }
            // V pass again
            if let Some(blur_bg_v) = &self.blur_bind_group_v {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("volumetric_blur_v2_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.volumetric_view,
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
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, blur_bg_v, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Pass 3: Composite onto scene (additive blend)
        // After blur chain: composite reads from the final output texture
        // LOW: final is blur_texture (H only), composite_bind_group points to blur_view
        // MEDIUM+: final is volumetric_texture (H+V), need to use the right bind group
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Resize textures on window resize
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let vol_width = (width / 2).max(1);
        let vol_height = (height / 2).max(1);

        if vol_width == self.width && vol_height == self.height {
            return;
        }

        self.width = vol_width;
        self.height = vol_height;

        // Recreate textures
        self.volumetric_texture = Self::create_half_res_texture(device, vol_width, vol_height, "volumetric_texture");
        self.volumetric_view = self.volumetric_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.blur_texture = Self::create_half_res_texture(device, vol_width, vol_height, "volumetric_blur_texture");
        self.blur_view = self.blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Update blur uniforms with new texel size
        // (will be written via queue in the next frame's create_bind_group)

        // Recreate composite bind group
        self.rebuild_composite_bind_group(device);

        // Clear bind groups (will be recreated in create_bind_group)
        self.raymarch_bind_group = None;
        self.blur_bind_group_h = None;
        self.blur_bind_group_v = None;
    }

    fn rebuild_composite_bind_group(&mut self, device: &wgpu::Device) {
        // Composite reads from the final blur output
        // LOW: reads blur_texture; MEDIUM+: reads volumetric_texture (after V pass writes back)
        let view = if self.quality <= 1 {
            &self.blur_view
        } else {
            &self.volumetric_view
        };
        self.composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_composite_bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
    }

    /// Update blur uniforms when resolution changes
    pub fn update_blur_uniforms(&self, queue: &wgpu::Queue) {
        let blur_h = BlurUniform {
            params: [1.0 / self.width as f32, 1.0 / self.height as f32, 0.0, 2.0],
        };
        let blur_v = BlurUniform {
            params: [1.0 / self.width as f32, 1.0 / self.height as f32, 1.0, 2.0],
        };
        queue.write_buffer(&self.blur_uniform_buffer_h, 0, bytemuck::cast_slice(&[blur_h]));
        queue.write_buffer(&self.blur_uniform_buffer_v, 0, bytemuck::cast_slice(&[blur_v]));
    }

    // === Quality API ===

    pub fn quality(&self) -> u32 {
        self.quality
    }

    pub fn quality_str(&self) -> &str {
        QUALITY_NAMES.get(self.quality as usize).unwrap_or(&"OFF")
    }

    pub fn quality_index(&self) -> usize {
        self.quality as usize
    }

    pub fn set_quality(&mut self, quality: u32, device: &wgpu::Device) {
        let old = self.quality;
        self.quality = quality.min(4);
        // If crossing the LOW/MEDIUM boundary, rebuild composite bind group
        // because the source texture changes
        if (old <= 1) != (self.quality <= 1) {
            self.rebuild_composite_bind_group(device);
        }
    }

    pub fn set_quality_index(&mut self, index: usize, device: &wgpu::Device) {
        self.set_quality(index as u32, device);
    }

    pub fn cycle_quality_next(&mut self, device: &wgpu::Device) {
        let next = (self.quality + 1) % 5;
        self.set_quality(next, device);
    }

    pub fn cycle_quality_prev(&mut self, device: &wgpu::Device) {
        let prev = if self.quality == 0 { 4 } else { self.quality - 1 };
        self.set_quality(prev, device);
    }

    pub fn is_enabled(&self) -> bool {
        self.quality > 0
    }

    pub fn frame_counter(&self) -> u32 {
        self.frame_counter
    }

    pub fn increment_frame(&mut self) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
    }
}

// =============================================================================
// VOLUMETRIC RAYMARCH SHADER - DDA-based world-space raymarching
// =============================================================================

const VOLUMETRIC_RAYMARCH_SHADER: &str = r#"
// Volumetric uniform
struct VolumetricUniform {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_direction: vec4<f32>,  // xyz = direction, w = intensity
    sun_color: vec4<f32>,      // rgb = color, a = density
    volume_center: vec4<f32>,  // xyz = center, w = volume_size (256)
    screen_params: vec4<f32>,  // x = width, y = height, z = quality (0-4), w = frame_counter
};

// Voxel volume uniform (from voxel_gi)
struct VoxelVolumeUniform {
    volume_center: vec4<f32>,
    volume_size: vec4<f32>,
    inv_volume_size: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    time_of_day: vec4<f32>,
    player_light: vec4<f32>,
    volume_origin: vec4<i32>,
    player_shadow_box: vec4<f32>,
};

@group(0) @binding(0) var<uniform> volumetric: VolumetricUniform;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_depth: sampler;

@group(1) @binding(0) var<uniform> voxel_volume: VoxelVolumeUniform;
@group(1) @binding(1) var t_voxel: texture_3d<f32>;
@group(1) @binding(2) var t_radiance: texture_3d<f32>;
@group(1) @binding(3) var s_voxel: sampler;

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

// ═══════════════════════════════════════════════════════════════════════════
// DDA SHADOW TEST - Adapted from pipeline.rs dda_trace_shadow
// Uses textureLoad with integer coords for exact per-voxel sampling.
// Returns 0.0 (shadow) or 1.0 (lit). Binary test with early exit.
// ═══════════════════════════════════════════════════════════════════════════

fn voxel_in_bounds(v: vec3<i32>) -> bool {
    return all(v >= vec3<i32>(0)) && all(v < vec3<i32>(256));
}

fn dda_shadow_test(world_pos: vec3<f32>, light_dir: vec3<f32>, max_steps: i32) -> f32 {
    let center = voxel_volume.volume_center.xyz;
    let local_start = world_pos - center + vec3<f32>(128.0);

    var voxel = vec3<i32>(floor(local_start));

    if (!voxel_in_bounds(voxel)) { return 1.0; }

    let step_sign = vec3<i32>(sign(light_dir));
    let abs_dir = max(abs(light_dir), vec3<f32>(1e-8));
    let t_delta = vec3<f32>(1.0) / abs_dir;

    let frac = local_start - vec3<f32>(floor(local_start));
    let dir_pos = vec3<bool>(light_dir.x >= 0.0, light_dir.y >= 0.0, light_dir.z >= 0.0);
    var t_max = select(frac * t_delta, (vec3<f32>(1.0) - frac) * t_delta, dir_pos);

    // Skip starting voxel to prevent self-shadowing (matches pipeline.rs pattern)
    {
        let sx = t_max.x <= t_max.y && t_max.x <= t_max.z;
        let sy = !(t_max.x <= t_max.y) && t_max.y <= t_max.z;
        let sz = !sx && !sy;
        let m = vec3<bool>(sx, sy, sz);
        voxel += select(vec3<i32>(0), step_sign, m);
        t_max += select(vec3<f32>(0.0), t_delta, m);
    }

    for (var i = 0; i < max_steps; i++) {
        if (!voxel_in_bounds(voxel)) { break; }

        // Early termination at volume edges
        let vd = abs(vec3<f32>(voxel) - vec3<f32>(127.5));
        let center_dist = max(vd.x, max(vd.y, vd.z));
        if (center_dist > 120.0) { break; }

        // Integer textureLoad: exact voxel value, no interpolation
        let voxel_data = textureLoad(t_voxel, voxel, 0);
        if (voxel_data.r > 0.5) {
            return 0.0;  // Solid hit = in shadow
        }

        // Branchless DDA step
        let x_le_y = t_max.x <= t_max.y;
        let x_le_z = t_max.x <= t_max.z;
        let y_le_z = t_max.y <= t_max.z;
        let step_x = x_le_y && x_le_z;
        let step_y = !x_le_y && y_le_z;
        let step_z = !step_x && !step_y;
        let mask = vec3<bool>(step_x, step_y, step_z);
        voxel += select(vec3<i32>(0), step_sign, mask);
        t_max += select(vec3<f32>(0.0), t_delta, mask);
    }

    return 1.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERLEAVED GRADIENT NOISE - Temporal dithering with golden angle rotation
// ═══════════════════════════════════════════════════════════════════════════

fn interleaved_gradient_noise(pixel_coords: vec2<f32>, frame: f32) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let frame_rot = frame * 5.588238;
    let rotated = pixel_coords + vec2<f32>(cos(frame_rot), sin(frame_rot)) * frame * 7.0;
    return fract(magic.z * fract(dot(rotated, magic.xy)));
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE FUNCTIONS - Dual-lobe Henyey-Greenstein for realistic scattering
// ═══════════════════════════════════════════════════════════════════════════

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(denom, 0.0001), 1.5));
}

fn phase_function(cos_theta: f32) -> f32 {
    let forward = henyey_greenstein(cos_theta, 0.76);   // Strong forward scattering
    let back = henyey_greenstein(cos_theta, -0.25);      // Mild backward fill
    return mix(forward, back, 0.15);                     // 85% forward, 15% back
}

// ═══════════════════════════════════════════════════════════════════════════
// QUALITY DECODE
// ═══════════════════════════════════════════════════════════════════════════

fn get_quality_params() -> vec4<f32> {
    let quality = i32(volumetric.screen_params.z);
    // (march_steps, shadow_steps, max_distance, _unused)
    if (quality <= 1) { return vec4<f32>(24.0, 16.0, 48.0,  0.0); }
    if (quality == 2) { return vec4<f32>(40.0, 32.0, 80.0,  0.0); }
    if (quality == 3) { return vec4<f32>(56.0, 48.0, 112.0, 0.0); }
    return vec4<f32>(72.0, 64.0, 128.0, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN FRAGMENT
// ═══════════════════════════════════════════════════════════════════════════

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let quality = i32(volumetric.screen_params.z);
    if (quality <= 0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Sample depth
    let depth = textureSampleLevel(t_depth, s_depth, in.uv, 0.0);

    // Sky pixels: still march but limit distance (god rays visible against sky)
    let is_sky = depth >= 0.9999;

    // Reconstruct world position from depth
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_h = volumetric.inv_view_proj * ndc;
    let world_pos = world_h.xyz / world_h.w;

    let camera_pos = volumetric.camera_pos.xyz;
    let ray_dir = normalize(world_pos - camera_pos);
    let raw_ray_length = length(world_pos - camera_pos);

    // Sun parameters
    let sun_dir = normalize(volumetric.sun_direction.xyz);
    let sun_intensity = volumetric.sun_direction.w;

    // Smooth fade at low sun intensity (sunset/sunrise transition)
    let intensity_fade = smoothstep(0.05, 0.3, sun_intensity);
    if (intensity_fade < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Quality-dependent parameters
    let params = get_quality_params();
    let num_steps = i32(params.x);
    let shadow_steps = i32(params.y);
    let max_distance = params.z;
    let density = volumetric.sun_color.a;

    // Clamp ray length
    let ray_length = select(min(raw_ray_length, max_distance), max_distance, is_sky);
    let step_size = ray_length / f32(num_steps);

    // Phase function (constant per pixel - view-sun angle doesn't change along ray)
    let cos_theta = dot(ray_dir, sun_dir);
    let phase = phase_function(cos_theta);

    // Interleaved gradient noise for temporal dithering
    let pixel_coords = in.uv * vec2<f32>(volumetric.screen_params.xy) * 0.5;
    let frame = volumetric.screen_params.w;
    let noise = interleaved_gradient_noise(pixel_coords, frame);

    // Start marching with dithered offset to break up banding
    var pos = camera_pos + ray_dir * (step_size * noise);

    // Accumulate in-scattering
    var accumulated = vec3<f32>(0.0);
    let volume_center = voxel_volume.volume_center.xyz;

    for (var i = 0; i < num_steps; i++) {
        pos += ray_dir * step_size;

        // Check if position is inside voxel volume
        let local = pos - volume_center + vec3<f32>(128.0);
        let voxel_coord = vec3<i32>(floor(local));
        if (!voxel_in_bounds(voxel_coord)) {
            continue;
        }

        // Skip solid voxels (inside geometry)
        let sample_data = textureLoad(t_voxel, voxel_coord, 0);
        if (sample_data.r > 0.3) {
            continue;
        }

        // DDA shadow test toward sun
        let shadow = dda_shadow_test(pos, sun_dir, shadow_steps);

        // In-scattering: phase * density * step * sun_visibility * sun_color * sun_intensity
        let scatter = phase * density * step_size * shadow;
        accumulated += volumetric.sun_color.rgb * scatter * sun_intensity;
    }

    // Apply intensity fade for day/night transition
    accumulated *= intensity_fade;

    // Soft tonemap to prevent blown-out highlights
    accumulated = accumulated / (accumulated + vec3<f32>(1.0));
    accumulated *= 1.5;  // Compensate for tonemap compression

    return vec4<f32>(accumulated, 1.0);
}
"#;

// =============================================================================
// BILATERAL BLUR SHADER - Depth-aware Gaussian blur at half resolution
// =============================================================================

const VOLUMETRIC_BLUR_SHADER: &str = r#"
struct BlurUniform {
    // xy = texel_size, z = direction (0=H, 1=V), w = depth_threshold
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> blur: BlurUniform;
@group(0) @binding(1) var t_input: texture_2d<f32>;
@group(0) @binding(2) var s_input: sampler;
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

fn linearize_depth(d: f32) -> f32 {
    let near = 0.1;
    let far = 1000.0;
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_color = textureSample(t_input, s_input, in.uv);
    let center_depth = linearize_depth(textureSampleLevel(t_depth, s_depth, in.uv, 0.0));

    // 9-tap Gaussian weights (sigma ~2.5)
    let w0 = 0.227027;
    let w1 = 0.194946;
    let w2 = 0.121622;
    let w3 = 0.054054;
    let w4 = 0.016216;

    let is_horizontal = blur.params.z < 0.5;
    let dir = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), is_horizontal);
    let texel = blur.params.xy;
    let depth_threshold = blur.params.w;

    var result = center_color * w0;
    var total_weight = w0;

    // Unrolled bilateral samples (WGSL requires constant indexing for arrays)
    // Tap 1
    {
        let offset = dir * texel * 1.0;
        let uv_pos = in.uv + offset;
        let uv_neg = in.uv - offset;
        let cp = textureSample(t_input, s_input, uv_pos);
        let cn = textureSample(t_input, s_input, uv_neg);
        let dp = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_pos, 0.0));
        let dn = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_neg, 0.0));
        let wp = w1 * step(abs(dp - center_depth), depth_threshold);
        let wn = w1 * step(abs(dn - center_depth), depth_threshold);
        result += cp * wp + cn * wn;
        total_weight += wp + wn;
    }
    // Tap 2
    {
        let offset = dir * texel * 2.0;
        let uv_pos = in.uv + offset;
        let uv_neg = in.uv - offset;
        let cp = textureSample(t_input, s_input, uv_pos);
        let cn = textureSample(t_input, s_input, uv_neg);
        let dp = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_pos, 0.0));
        let dn = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_neg, 0.0));
        let wp = w2 * step(abs(dp - center_depth), depth_threshold);
        let wn = w2 * step(abs(dn - center_depth), depth_threshold);
        result += cp * wp + cn * wn;
        total_weight += wp + wn;
    }
    // Tap 3
    {
        let offset = dir * texel * 3.0;
        let uv_pos = in.uv + offset;
        let uv_neg = in.uv - offset;
        let cp = textureSample(t_input, s_input, uv_pos);
        let cn = textureSample(t_input, s_input, uv_neg);
        let dp = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_pos, 0.0));
        let dn = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_neg, 0.0));
        let wp = w3 * step(abs(dp - center_depth), depth_threshold);
        let wn = w3 * step(abs(dn - center_depth), depth_threshold);
        result += cp * wp + cn * wn;
        total_weight += wp + wn;
    }
    // Tap 4
    {
        let offset = dir * texel * 4.0;
        let uv_pos = in.uv + offset;
        let uv_neg = in.uv - offset;
        let cp = textureSample(t_input, s_input, uv_pos);
        let cn = textureSample(t_input, s_input, uv_neg);
        let dp = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_pos, 0.0));
        let dn = linearize_depth(textureSampleLevel(t_depth, s_depth, uv_neg, 0.0));
        let wp = w4 * step(abs(dp - center_depth), depth_threshold);
        let wn = w4 * step(abs(dn - center_depth), depth_threshold);
        result += cp * wp + cn * wn;
        total_weight += wp + wn;
    }

    return result / total_weight;
}
"#;

// =============================================================================
// COMPOSITE SHADER - Simple additive blend onto scene
// =============================================================================

const VOLUMETRIC_COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0) var t_volumetric: texture_2d<f32>;
@group(0) @binding(1) var s_tex: sampler;

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
    let color = textureSample(t_volumetric, s_tex, in.uv).rgb;
    return vec4<f32>(color, 1.0);
}
"#;
