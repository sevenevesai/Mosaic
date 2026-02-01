//! Impostor Rendering for Distant Chunks
//!
//! Renders distant chunks as single screen-aligned colored quads instead of full geometry.
//! This dramatically reduces triangle count for chunks that are effectively <1 pixel per block.
//!
//! Pipeline:
//! 1. GPU culling partitions chunks into mesh vs impostor based on distance
//! 2. Impostor pass renders each distant chunk as a single colored quad
//! 3. Quad size based on chunk's screen-space size
//!
//! Based on Voxelot's impostor implementation.

use wgpu::util::DeviceExt;
use glam::Vec3;

use crate::world::ChunkPos;

/// Maximum number of impostors that can be rendered per frame
pub const MAX_IMPOSTORS: u32 = 4096;

/// Extended view range using impostors (additive to render distance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImpostorRange {
    /// No extended view - only render_distance chunks visible
    Off,
    /// +16 chunks of impostors beyond render distance
    Plus16,
    #[default]
    /// +24 chunks of impostors beyond render distance (default)
    Plus24,
    /// +32 chunks of impostors beyond render distance
    Plus32,
    /// +40 chunks of impostors beyond render distance
    Plus40,
}

impl ImpostorRange {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImpostorRange::Off => "OFF",
            ImpostorRange::Plus16 => "+16",
            ImpostorRange::Plus24 => "+24",
            ImpostorRange::Plus32 => "+32",
            ImpostorRange::Plus40 => "+40",
        }
    }

    pub fn next(&self) -> ImpostorRange {
        match self {
            ImpostorRange::Off => ImpostorRange::Plus16,
            ImpostorRange::Plus16 => ImpostorRange::Plus24,
            ImpostorRange::Plus24 => ImpostorRange::Plus32,
            ImpostorRange::Plus32 => ImpostorRange::Plus40,
            ImpostorRange::Plus40 => ImpostorRange::Off,
        }
    }

    /// Extra distance in chunks to render as impostors
    pub fn extra_chunks(&self) -> i32 {
        match self {
            ImpostorRange::Off => 0,
            ImpostorRange::Plus16 => 16,
            ImpostorRange::Plus24 => 24,
            ImpostorRange::Plus32 => 32,
            ImpostorRange::Plus40 => 40,
        }
    }

    /// Size of impostor quads in pixels (scales with distance)
    pub fn pixel_size(&self) -> f32 {
        match self {
            ImpostorRange::Off => 0.0,
            ImpostorRange::Plus16 => 2.0,
            ImpostorRange::Plus24 => 2.0,
            ImpostorRange::Plus32 => 3.0,
            ImpostorRange::Plus40 => 3.0,
        }
    }

    /// Whether extended view is enabled
    pub fn enabled(&self) -> bool {
        !matches!(self, ImpostorRange::Off)
    }

    /// Convert from string (for settings loading)
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "+16" | "PLUS16" => ImpostorRange::Plus16,
            "+24" | "PLUS24" => ImpostorRange::Plus24,
            "+32" | "PLUS32" => ImpostorRange::Plus32,
            "+40" | "PLUS40" => ImpostorRange::Plus40,
            _ => ImpostorRange::Off,
        }
    }
}

/// Instance data for a single impostor (GPU format)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImpostorInstance {
    /// World position of chunk center
    pub position: [f32; 3],
    /// Padding for alignment
    pub _pad0: f32,
    /// RGBA color [0-1 range]
    pub color: [f32; 4],
}

impl ImpostorInstance {
    pub fn new(chunk_pos: ChunkPos, color: [u8; 4]) -> Self {
        // Calculate chunk center in world space
        let center = chunk_pos.center();

        Self {
            position: [center.x, center.y, center.z],
            _pad0: 0.0,
            color: [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
                color[3] as f32 / 255.0,
            ],
        }
    }
}

/// Uniforms for impostor rendering
/// Note: WGSL vec3 is aligned to 16 bytes, so we use [f32; 4] for camera_pos
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImpostorUniforms {
    /// View-projection matrix (64 bytes)
    pub view_proj: [[f32; 4]; 4],
    /// Camera position + padding (16 bytes, vec3 aligned to 16 in WGSL)
    pub camera_pos: [f32; 4],
    /// Pixel size, screen dimensions, fog (16 bytes)
    pub pixel_size: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub fog_density: f32,
    /// Sky brightness, chunk size, padding (16 bytes)
    pub sky_brightness: f32,
    pub chunk_size: f32,
    pub _pad: [f32; 2],
}

impl Default for ImpostorUniforms {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 4],
            pixel_size: 2.0,
            screen_width: 1920.0,
            screen_height: 1080.0,
            fog_density: 0.0,
            sky_brightness: 1.0,
            chunk_size: 16.0,
            _pad: [0.0; 2],
        }
    }
}

/// Impostor renderer for extended view distance
pub struct ImpostorRenderer {
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout (kept for potential future extension)
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Instance buffer for impostor data
    instance_buffer: wgpu::Buffer,
    /// Current instance count
    instance_count: u32,
    /// Extended view range setting
    pub range: ImpostorRange,
    /// Statistics: impostors rendered last frame
    pub impostors_rendered: u32,
}

impl ImpostorRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impostor Shader"),
            source: wgpu::ShaderSource::Wgsl(IMPOSTOR_SHADER.into()),
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Impostor Uniforms"),
            contents: bytemuck::cast_slice(&[ImpostorUniforms::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Impostor Instance Buffer"),
            size: (MAX_IMPOSTORS as usize * std::mem::size_of::<ImpostorInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Impostor Bind Group Layout"),
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
            ],
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Impostor Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Impostor Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout for instances
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ImpostorInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // _pad0 (skip)
                // color
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
            ],
        };

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Impostor Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[instance_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Draw both sides of quads
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
            instance_buffer,
            instance_count: 0,
            range: ImpostorRange::default(),
            impostors_rendered: 0,
        }
    }

    /// Update uniforms for this frame
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        camera_pos: Vec3,
        screen_width: f32,
        screen_height: f32,
        sky_brightness: f32,
    ) {
        let uniforms = ImpostorUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.extend(0.0).to_array(),
            pixel_size: self.range.pixel_size(),
            screen_width,
            screen_height,
            fog_density: 0.001, // Light fog for distant chunks
            sky_brightness,
            chunk_size: 16.0,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Upload impostor instances for this frame
    pub fn upload_instances(&mut self, queue: &wgpu::Queue, instances: &[ImpostorInstance]) {
        let count = instances.len().min(MAX_IMPOSTORS as usize);
        if count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances[..count]),
            );
        }
        self.instance_count = count as u32;
    }

    /// Render impostors
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.instance_count == 0 || !self.range.enabled() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));

        // Draw 6 vertices (2 triangles = quad) per instance
        render_pass.draw(0..6, 0..self.instance_count);
    }

    /// Get the number of impostors that will be rendered
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
}

/// Impostor shader (WGSL)
const IMPOSTOR_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,  // xyz = position, w = unused
    pixel_size: f32,
    screen_width: f32,
    screen_height: f32,
    fog_density: f32,
    sky_brightness: f32,
    chunk_size: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
};

// Get quad vertex offset based on vertex index (2 triangles = 6 vertices)
fn get_quad_offset(vid: u32) -> vec2<f32> {
    switch vid {
        case 0u: { return vec2<f32>(-0.5, -0.5); }
        case 1u: { return vec2<f32>(0.5, -0.5); }
        case 2u: { return vec2<f32>(0.5, 0.5); }
        case 3u: { return vec2<f32>(-0.5, -0.5); }
        case 4u: { return vec2<f32>(0.5, 0.5); }
        case 5u: { return vec2<f32>(-0.5, 0.5); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) inst_pos: vec3<f32>,
    @location(1) inst_color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    // Project chunk center to clip space
    let clip = uniforms.view_proj * vec4<f32>(inst_pos, 1.0);

    // Calculate screen-space quad size
    // Use pixel_size as base, but scale with distance for consistent appearance
    let size_px = max(uniforms.pixel_size, 1.0);
    let offset_px = get_quad_offset(vid) * size_px;

    // Convert pixel offset to NDC
    let screen_w = max(uniforms.screen_width, 1.0);
    let screen_h = max(uniforms.screen_height, 1.0);
    let ndc_offset = vec2<f32>(
        offset_px.x * 2.0 / screen_w,
        offset_px.y * 2.0 / screen_h
    );

    // Apply offset in clip space (multiply by w to get proper screen alignment)
    out.position = clip + vec4<f32>(ndc_offset * clip.w, 0.0, 0.0);
    out.color = inst_color;
    out.world_pos = inst_pos;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Skip fully transparent impostors
    if input.color.a < 0.01 {
        discard;
    }

    // Apply distance fog
    let dist = length(uniforms.camera_pos.xyz - input.world_pos);
    let fog_factor = exp(-uniforms.fog_density * dist);

    // Fog color based on sky brightness (matches main shader)
    let fog_base = mix(
        vec3<f32>(0.02, 0.02, 0.03),  // Dark fog at night
        vec3<f32>(0.7, 0.8, 0.9),      // Light fog during day
        uniforms.sky_brightness
    );

    // Apply basic ambient lighting (impostors are simplified)
    let lit_color = input.color.rgb * (0.5 + 0.5 * uniforms.sky_brightness);

    // Blend with fog
    let final_color = mix(lit_color, fog_base, 1.0 - fog_factor);

    return vec4<f32>(final_color, input.color.a);
}
"#;
