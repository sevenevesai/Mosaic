//! Render pipeline creation

use crate::world::ChunkVertex;

/// Simple UI vertex for crosshair
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
}

impl UiVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Line vertex for block outline
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
}

impl LineVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// UI shader for crosshair (screen-space, no depth)
const UI_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Position is already in clip space (-1 to 1)
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 0.9); // White with slight transparency
}
"#;

/// Line shader for block outline
const LINE_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.8); // Black with slight transparency
}
"#;

/// Sky shader with gradient and sun
const SKY_SHADER: &str = r#"
struct SkyUniform {
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    screen_size: vec4<f32>,  // xy = screen size, zw = unused
};

@group(0) @binding(0)
var<uniform> sky: SkyUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// Fullscreen triangle (no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle covering entire screen
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 0.9999, 1.0);  // Far depth
    out.ndc = vec2<f32>(x, y);  // Pass NDC directly
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Unproject from NDC to world space to get view direction
    // Use two points along the ray: near and far plane
    let near_point = sky.inv_view_proj * vec4<f32>(in.ndc, -1.0, 1.0);
    let far_point = sky.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);

    let near_world = near_point.xyz / near_point.w;
    let far_world = far_point.xyz / far_point.w;

    let view_dir = normalize(far_world - near_world);

    // Sky gradient (blue at top, lighter at horizon)
    let sky_top = vec3<f32>(0.4, 0.65, 0.95);      // Deep blue
    let sky_horizon = vec3<f32>(0.7, 0.82, 0.95);  // Light blue
    let horizon_blend = pow(1.0 - max(view_dir.y, 0.0), 2.5);
    var sky_color = mix(sky_top, sky_horizon, horizon_blend);

    // Sun direction (world space)
    let sun_dir = normalize(sky.sun_direction.xyz);
    let sun_dot = dot(view_dir, sun_dir);

    // Sun disc
    let sun_size = 0.9975;  // Tight sun disc
    if (sun_dot > sun_size) {
        let sun_intensity = smoothstep(sun_size, 0.999, sun_dot);
        let sun_color = vec3<f32>(1.0, 0.98, 0.9);  // Bright warm white
        sky_color = mix(sky_color, sun_color, sun_intensity);
    }

    // Sun glow (larger, softer)
    let glow_size = 0.95;
    if (sun_dot > glow_size) {
        let glow_intensity = smoothstep(glow_size, 0.999, sun_dot) * 0.5;
        let glow_color = vec3<f32>(1.0, 0.92, 0.75);
        sky_color = mix(sky_color, glow_color, glow_intensity);
    }

    // Outer glow
    let outer_glow_size = 0.85;
    if (sun_dot > outer_glow_size) {
        let outer_intensity = smoothstep(outer_glow_size, 0.98, sun_dot) * 0.2;
        let outer_color = vec3<f32>(1.0, 0.88, 0.7);
        sky_color = mix(sky_color, outer_color, outer_intensity);
    }

    // Subtle horizon glow near sun
    if (view_dir.y < 0.3 && view_dir.y > -0.1) {
        let horizon_glow = max(sun_dot, 0.0) * (1.0 - abs(view_dir.y) * 3.0) * 0.12;
        sky_color += vec3<f32>(1.0, 0.85, 0.6) * horizon_glow;
    }

    return vec4<f32>(sky_color, 1.0);
}
"#;

/// Shadow pass shader (depth-only, renders from light's perspective)
const SHADOW_SHADER: &str = r#"
struct LightUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> light: LightUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_packed: vec4<i32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao_tile_packed: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return light.view_proj * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main() {
    // Depth-only pass, no color output needed
}
"#;

/// Chunk shader source (WGSL)
const CHUNK_SHADER: &str = r#"
// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
};

// Light/shadow uniform
struct LightUniform {
    view_proj: mat4x4<f32>,
    direction: vec4<f32>,  // xyz = direction, w = unused
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@group(2) @binding(0)
var<uniform> light: LightUniform;
@group(2) @binding(1)
var t_shadow: texture_depth_2d;
@group(2) @binding(2)
var s_shadow: sampler_comparison;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_packed: vec4<i32>,  // Sint8x4 unpacked to i32
    @location(2) uv: vec2<f32>,             // Unorm16x2 auto-normalized
    @location(3) ao_tile_packed: vec4<f32>, // Unorm8x4: [ao, tile_u/255, tile_v/255, 0]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao: f32,
    @location(4) tile_count: vec2<f32>,     // Tile repeat count for greedy meshing
    @location(5) shadow_pos: vec4<f32>,     // Position in light space for shadow lookup
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    // Unpack and normalize normal (Sint8 range is -128 to 127, divide by 127)
    out.normal = vec3<f32>(
        f32(in.normal_packed.x) / 127.0,
        f32(in.normal_packed.y) / 127.0,
        f32(in.normal_packed.z) / 127.0
    );
    out.uv = in.uv;
    out.ao = in.ao_tile_packed.x;
    // Tile counts are stored as bytes 1,2 normalized to 0-1, scale back to 1-255 range
    out.tile_count = vec2<f32>(
        max(in.ao_tile_packed.y * 255.0, 1.0),
        max(in.ao_tile_packed.z * 255.0, 1.0)
    );
    // Transform position to light space for shadow lookup
    out.shadow_pos = light.view_proj * vec4<f32>(in.position, 1.0);
    return out;
}

// PCF shadow sampling for soft shadows
fn calculate_shadow(shadow_pos: vec4<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Perspective divide
    let proj_coords = shadow_pos.xyz / shadow_pos.w;

    // Transform from [-1,1] to [0,1] UV space
    let shadow_uv = vec2<f32>(
        proj_coords.x * 0.5 + 0.5,
        -proj_coords.y * 0.5 + 0.5  // Flip Y
    );

    // If outside shadow map, no shadow
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }

    // Current depth from light's perspective
    let current_depth = proj_coords.z;

    // If behind light frustum, no shadow
    if (current_depth > 1.0 || current_depth < 0.0) {
        return 1.0;
    }

    // Slope-based bias to prevent shadow acne
    let bias = max(0.003 * (1.0 - dot(normal, light_dir)), 0.001);
    let biased_depth = current_depth - bias;

    // PCF 3x3 sampling for soft shadows
    let texel_size = 1.0 / 2048.0;  // Shadow map size
    var shadow = 0.0;

    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + offset, biased_depth);
        }
    }
    shadow /= 9.0;

    return shadow;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate tiled UV for greedy meshing
    let tiles_per_row = 4.0;  // 4x4 grid of 256x256 tiles = 1024x1024 atlas
    let tile_size = 1.0 / tiles_per_row;

    // Get the base tile corner (floor to tile boundary)
    let tile_base = floor(in.uv / tile_size) * tile_size;

    // Get local position within quad (0-1 range scaled by tile count)
    let local_uv = (in.uv - tile_base) / tile_size;
    let tiled_local = fract(local_uv * in.tile_count);

    // Final UV = base tile + tiled local position
    let tiled_uv = tile_base + tiled_local * tile_size;

    // Sample texture
    let tex_color = textureSample(t_diffuse, s_diffuse, tiled_uv);

    // === LIGHTING SETUP ===
    // Sun direction from uniform (normalized)
    let sun_dir = normalize(light.direction.xyz);
    let sun_color = vec3<f32>(1.0, 0.93, 0.8);  // Slightly warmer

    // Sky light (soft blue from above)
    let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
    let sky_color = vec3<f32>(0.65, 0.78, 0.95);  // Soft blue

    // Ground bounce (subtle warm from below)
    let ground_color = vec3<f32>(0.35, 0.28, 0.2);  // Warm brown

    // === SHADOW CALCULATION ===
    let shadow = calculate_shadow(in.shadow_pos, in.normal, sun_dir);

    // === FACE-BASED BRIGHTNESS ===
    // Minecraft-style: top=1.0, north/south=0.85, east/west=0.7, bottom=0.5
    var face_brightness = 0.8;
    if (in.normal.y > 0.5) {
        face_brightness = 1.0;      // Top - full brightness
    } else if (in.normal.y < -0.5) {
        face_brightness = 0.5;      // Bottom - dark
    } else if (abs(in.normal.x) > 0.5) {
        face_brightness = 0.7;      // East/West - darker sides
    } else {
        face_brightness = 0.85;     // North/South - medium sides
    }

    // === DIRECT LIGHTING ===
    // Sun (main directional light) - fully affected by shadow
    let sun_ndotl = max(dot(in.normal, sun_dir), 0.0);
    let sun_contribution = sun_color * sun_ndotl * 0.7 * shadow;

    // Sky light (hemisphere from above) - partially affected by shadow
    let sky_ndotl = max(dot(in.normal, sky_dir), 0.0) * 0.5 + 0.5;
    let sky_shadow = mix(0.6, 1.0, shadow);  // Sky light reduced in shadow
    let sky_contribution = sky_color * sky_ndotl * 0.2 * sky_shadow;

    // Ground bounce (subtle upward fill)
    let ground_ndotl = max(dot(in.normal, -sky_dir), 0.0);
    let ground_contribution = ground_color * ground_ndotl * 0.08;

    // Ambient base - reduced in shadow for more pronounced effect
    let ambient_shadow = mix(0.7, 1.0, shadow);  // Darken ambient in shadow
    let ambient = vec3<f32>(0.28, 0.28, 0.32) * ambient_shadow;

    // === AMBIENT OCCLUSION ===
    // Stronger AO contrast for better depth
    let ao_factor = mix(0.25, 1.0, in.ao);

    // === COMBINE LIGHTING ===
    let total_light = (ambient + sun_contribution + sky_contribution + ground_contribution)
                      * ao_factor * face_brightness;
    var final_color = tex_color.rgb * total_light;

    // === ATMOSPHERIC DEPTH ===
    let dist = length(camera.view_pos.xyz - in.world_position);

    // Exponential fog - very minimal, only at extreme distance
    let fog_density = 0.0003;  // Extremely low density
    let fog_factor = 1.0 - exp(-dist * fog_density);
    let fog_factor_clamped = clamp(fog_factor, 0.0, 0.15);  // Max 15% fog

    // Horizon color (warm sky)
    let horizon_color = vec3<f32>(0.75, 0.82, 0.92);

    // Atmospheric scattering - minimal blue shift only at extreme distance
    let scatter_factor = clamp((dist - 300.0) / 800.0, 0.0, 0.08);  // Starts at 300 blocks
    let scatter_color = vec3<f32>(0.7, 0.76, 0.88);

    // Minimal desaturation at extreme distance
    let luminance = dot(final_color, vec3<f32>(0.299, 0.587, 0.114));
    let desat_factor = clamp((dist - 400.0) / 1000.0, 0.0, 0.08);  // Starts at 400 blocks
    final_color = mix(final_color, vec3<f32>(luminance), desat_factor);

    // Apply atmospheric effects
    final_color = mix(final_color, scatter_color, scatter_factor);
    final_color = mix(final_color, horizon_color, fog_factor_clamped);

    return vec4<f32>(final_color, tex_color.a);
}
"#;

pub fn create_chunk_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    shadow_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Chunk Shader"),
        source: wgpu::ShaderSource::Wgsl(CHUNK_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Chunk Pipeline Layout"),
        bind_group_layouts: &[camera_bind_group_layout, texture_bind_group_layout, shadow_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Chunk Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[ChunkVertex::desc()],
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
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            // Small depth bias to prevent 1px gaps at block edges
            // Pushes geometry slightly toward camera, preventing seam artifacts
            // at grazing angles where adjacent faces fight for pixels
            bias: wgpu::DepthBiasState {
                constant: -1,        // Slight bias toward camera
                slope_scale: -1.0,   // Scale with slope for angled surfaces
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

/// Shadow map size (2048x2048 provides good quality)
pub const SHADOW_MAP_SIZE: u32 = 2048;

/// Create shadow pass pipeline (depth-only, renders from light perspective)
pub fn create_shadow_pipeline(
    device: &wgpu::Device,
    light_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shadow Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Shadow Pipeline Layout"),
        bind_group_layouts: &[light_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shadow Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[ChunkVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[],  // No color output for shadow pass
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,       // Slight bias to prevent shadow acne
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Create sky pipeline for rendering sky background with sun
pub fn create_sky_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sky_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sky Shader"),
        source: wgpu::ShaderSource::Wgsl(SKY_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Sky Pipeline Layout"),
        bind_group_layouts: &[sky_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Sky Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],  // No vertex buffer - fullscreen triangle from vertex_index
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,  // Render at far depth
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Create UI pipeline for crosshair rendering
pub fn create_ui_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("UI Shader"),
        source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("UI Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("UI Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[UiVertex::desc()],
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
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always, // Always pass, ignore depth
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Create line pipeline for block outline rendering
pub fn create_line_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Line Shader"),
        source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Line Pipeline Layout"),
        bind_group_layouts: &[camera_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Line Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[LineVertex::desc()],
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
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false, // Don't write depth, just test
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -2, // Slight bias to render on top of block faces
                slope_scale: 0.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}
