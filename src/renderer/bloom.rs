//! Bloom post-processing effect
//!
//! Extracts bright pixels and creates a glow effect around them.
//! Uses Kawase blur for efficient, high-quality blurring.

use wgpu::util::DeviceExt;

/// Number of blur iterations (more = softer, wider bloom)
const BLOOM_BLUR_PASSES: u32 = 5;

/// Bloom brightness threshold (pixels above this will glow)
const BLOOM_THRESHOLD: f32 = 0.8;

/// Bloom intensity multiplier
const BLOOM_INTENSITY: f32 = 0.5;

/// Bloom effect manager
pub struct Bloom {
    /// Texture for extraction pass (bright pixels)
    extract_texture: wgpu::Texture,
    extract_view: wgpu::TextureView,

    /// Ping-pong textures for blur passes
    blur_textures: [wgpu::Texture; 2],
    blur_views: [wgpu::TextureView; 2],

    /// Sampler for bloom textures
    sampler: wgpu::Sampler,

    /// Bind group layout for bloom passes
    bind_group_layout: wgpu::BindGroupLayout,

    /// Bind groups for each blur pass (ping-pong)
    blur_bind_groups: [wgpu::BindGroup; 2],

    /// Bind group for the extraction pass (reads from main scene)
    extract_bind_group: Option<wgpu::BindGroup>,

    /// Pipeline for extraction pass
    extract_pipeline: wgpu::RenderPipeline,

    /// Pipeline for blur passes
    blur_pipeline: wgpu::RenderPipeline,

    /// Pipeline for final composite
    composite_pipeline: wgpu::RenderPipeline,

    /// Bind group layout for composite pass (scene + bloom)
    composite_bind_group_layout: wgpu::BindGroupLayout,

    /// Current texture dimensions
    width: u32,
    height: u32,
}

impl Bloom {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        // Use half resolution for bloom (performance optimization)
        let bloom_width = width / 2;
        let bloom_height = height / 2;

        // Create bloom textures
        let extract_texture = Self::create_bloom_texture(device, bloom_width, bloom_height, "bloom_extract");
        let extract_view = extract_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blur_textures = [
            Self::create_bloom_texture(device, bloom_width, bloom_height, "bloom_blur_0"),
            Self::create_bloom_texture(device, bloom_width, bloom_height, "bloom_blur_1"),
        ];
        let blur_views = [
            blur_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            blur_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // Sampler for smooth bloom
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group layout for bloom passes (texture + sampler + uniforms)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Composite bind group layout (scene texture + bloom texture)
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_composite_bind_group_layout"),
            entries: &[
                // Scene texture
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
                // Bloom texture
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipelines
        let extract_pipeline = Self::create_extract_pipeline(device, &bind_group_layout);
        let blur_pipeline = Self::create_blur_pipeline(device, &bind_group_layout);
        let composite_pipeline = Self::create_composite_pipeline(device, &composite_bind_group_layout, surface_format);

        // Create blur bind groups (ping-pong)
        let blur_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom_blur_uniform"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0f32, bloom_width as f32, bloom_height as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blur_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_blur_bind_group_0"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&extract_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_blur_bind_group_1"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&blur_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            }),
        ];

        Self {
            extract_texture,
            extract_view,
            blur_textures,
            blur_views,
            sampler,
            bind_group_layout,
            blur_bind_groups,
            extract_bind_group: None,
            extract_pipeline,
            blur_pipeline,
            composite_pipeline,
            composite_bind_group_layout,
            width: bloom_width,
            height: bloom_height,
        }
    }

    fn create_bloom_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
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
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_extract_pipeline(device: &wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_extract_shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_EXTRACT_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_extract_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_extract_pipeline"),
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

    fn create_blur_pipeline(device: &wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_BLUR_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_blur_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_blur_pipeline"),
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
            label: Some("bloom_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_COMPOSITE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_composite_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_composite_pipeline"),
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

    /// Create bind group for extracting bright pixels from the scene
    pub fn create_extract_bind_group(&mut self, device: &wgpu::Device, scene_view: &wgpu::TextureView) {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom_extract_uniform"),
            contents: bytemuck::cast_slice(&[BLOOM_THRESHOLD, BLOOM_INTENSITY, self.width as f32, self.height as f32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        self.extract_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_extract_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Create bind group for the final composite pass
    pub fn create_composite_bind_group(&self, device: &wgpu::Device, scene_view: &wgpu::TextureView) -> wgpu::BindGroup {
        // Use the last blur result (blur_views[1] after odd number of passes, blur_views[0] after even)
        let bloom_view = if BLOOM_BLUR_PASSES % 2 == 0 {
            &self.blur_views[0]
        } else {
            &self.blur_views[1]
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_composite_bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Render the bloom effect
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
        device: &wgpu::Device,
    ) {
        // 1. Extract bright pixels
        if let Some(extract_bind_group) = &self.extract_bind_group {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_extract_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.extract_view,
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

            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, extract_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. Blur passes (ping-pong between two textures)
        // First pass: extract -> blur[0]
        // Second pass: blur[0] -> blur[1]
        // Third pass: blur[1] -> blur[0]
        // etc.
        for i in 0..BLOOM_BLUR_PASSES {
            let (src_bind_group, dst_view) = if i == 0 {
                // First pass reads from extract texture
                (&self.blur_bind_groups[0], &self.blur_views[0])
            } else if i % 2 == 1 {
                // Odd passes: blur[0] -> blur[1]
                (&self.blur_bind_groups[1], &self.blur_views[1])
            } else {
                // Even passes: blur[1] -> blur[0]
                // Need to create bind group for blur[1] -> blur[0]
                (&self.blur_bind_groups[0], &self.blur_views[0])
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("bloom_blur_pass_{}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst_view,
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
            pass.set_bind_group(0, src_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Composite: combine scene + bloom
        let composite_bind_group = self.create_composite_bind_group(device, scene_view);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Don't clear - we want to keep UI
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Resize bloom textures when window size changes
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let bloom_width = width / 2;
        let bloom_height = height / 2;

        if bloom_width == self.width && bloom_height == self.height {
            return;
        }

        self.width = bloom_width;
        self.height = bloom_height;

        // Recreate textures
        self.extract_texture = Self::create_bloom_texture(device, bloom_width, bloom_height, "bloom_extract");
        self.extract_view = self.extract_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.blur_textures = [
            Self::create_bloom_texture(device, bloom_width, bloom_height, "bloom_blur_0"),
            Self::create_bloom_texture(device, bloom_width, bloom_height, "bloom_blur_1"),
        ];
        self.blur_views = [
            self.blur_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            self.blur_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // Recreate blur bind groups
        let blur_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom_blur_uniform"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0f32, bloom_width as f32, bloom_height as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        self.blur_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_blur_bind_group_0"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.extract_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_blur_bind_group_1"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.blur_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            }),
        ];

        // Clear extract bind group (needs to be recreated with new scene view)
        self.extract_bind_group = None;
    }

    /// Get the extract texture view (for debugging)
    pub fn extract_view(&self) -> &wgpu::TextureView {
        &self.extract_view
    }
}

/// Bloom extraction shader - thresholds bright pixels
const BLOOM_EXTRACT_SHADER: &str = r#"
struct Uniforms {
    threshold: f32,
    intensity: f32,
    width: f32,
    height: f32,
};

@group(0) @binding(0)
var t_scene: texture_2d<f32>;
@group(0) @binding(1)
var s_scene: sampler;
@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_scene, s_scene, in.uv);

    // Calculate luminance
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Soft threshold with knee
    let knee = 0.1;
    let soft = luminance - uniforms.threshold + knee;
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    let contribution = soft_clamped * soft_clamped / (4.0 * knee + 0.0001);

    // Hard threshold fallback
    let hard = max(luminance - uniforms.threshold, 0.0);

    // Combine soft and hard threshold
    let bloom_factor = max(contribution, hard);

    // Apply bloom intensity
    let bloom_color = color.rgb * bloom_factor * uniforms.intensity;

    return vec4<f32>(bloom_color, 1.0);
}
"#;

/// Kawase blur shader - efficient blur algorithm
const BLOOM_BLUR_SHADER: &str = r#"
struct Uniforms {
    offset_x: f32,
    offset_y: f32,
    width: f32,
    height: f32,
};

@group(0) @binding(0)
var t_input: texture_2d<f32>;
@group(0) @binding(1)
var s_input: sampler;
@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

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
    let texel_size = vec2<f32>(1.0 / uniforms.width, 1.0 / uniforms.height);
    let offset = texel_size * 1.5;  // Kawase blur offset

    // Sample 4 corners + center (5-tap Kawase blur)
    var color = textureSample(t_input, s_input, in.uv) * 0.25;
    color += textureSample(t_input, s_input, in.uv + vec2<f32>(-offset.x, -offset.y)) * 0.1875;
    color += textureSample(t_input, s_input, in.uv + vec2<f32>( offset.x, -offset.y)) * 0.1875;
    color += textureSample(t_input, s_input, in.uv + vec2<f32>(-offset.x,  offset.y)) * 0.1875;
    color += textureSample(t_input, s_input, in.uv + vec2<f32>( offset.x,  offset.y)) * 0.1875;

    return color;
}
"#;

/// Bloom composite shader - combines scene with bloom
const BLOOM_COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0)
var t_scene: texture_2d<f32>;
@group(0) @binding(1)
var t_bloom: texture_2d<f32>;
@group(0) @binding(2)
var s_tex: sampler;

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
    let scene_color = textureSample(t_scene, s_tex, in.uv).rgb;
    let bloom_color = textureSample(t_bloom, s_tex, in.uv).rgb;

    // Additive blending with intensity control
    let bloom_intensity = 0.7;
    let final_color = scene_color + bloom_color * bloom_intensity;

    // Simple tone mapping to prevent over-exposure
    let mapped = final_color / (final_color + vec3<f32>(1.0));

    return vec4<f32>(mapped, 1.0);
}
"#;
