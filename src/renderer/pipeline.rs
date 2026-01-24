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

/// Sky shader with gradient, sun, and moon (day/night cycle)
const SKY_SHADER: &str = r#"
struct SkyUniform {
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,     // xyz = direction, w = intensity
    moon_direction: vec4<f32>,    // xyz = direction, w = intensity
    sky_top_color: vec4<f32>,     // rgb = color, a = time_of_day
    sky_horizon_color: vec4<f32>, // rgb = color, a = unused
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
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 0.9999, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Unproject from NDC to world space to get view direction
    let near_point = sky.inv_view_proj * vec4<f32>(in.ndc, -1.0, 1.0);
    let far_point = sky.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);

    let near_world = near_point.xyz / near_point.w;
    let far_world = far_point.xyz / far_point.w;

    let view_dir = normalize(far_world - near_world);

    // Time-based sky gradient from uniform
    let sky_top = sky.sky_top_color.rgb;
    let sky_horizon = sky.sky_horizon_color.rgb;
    let horizon_blend = pow(1.0 - max(view_dir.y, 0.0), 2.5);
    var sky_color = mix(sky_top, sky_horizon, horizon_blend);

    // Sun properties
    let sun_dir = normalize(sky.sun_direction.xyz);
    let sun_intensity = sky.sun_direction.w;
    let sun_dot = dot(view_dir, sun_dir);

    // Moon properties
    let moon_dir = normalize(sky.moon_direction.xyz);
    let moon_intensity = sky.moon_direction.w;
    let moon_dot = dot(view_dir, moon_dir);

    // Sun rendering (only when above horizon)
    if (sun_intensity > 0.05) {
        // Sun disc - warm color that shifts with time
        let sun_size = 0.9975;
        if (sun_dot > sun_size) {
            let disc_intensity = smoothstep(sun_size, 0.999, sun_dot) * sun_intensity;
            let sun_color = vec3<f32>(1.0, 0.95, 0.85);
            sky_color = mix(sky_color, sun_color, disc_intensity);
        }

        // Sun glow
        let glow_size = 0.95;
        if (sun_dot > glow_size) {
            let glow_strength = smoothstep(glow_size, 0.999, sun_dot) * 0.5 * sun_intensity;
            let glow_color = vec3<f32>(1.0, 0.90, 0.70);
            sky_color = mix(sky_color, glow_color, glow_strength);
        }

        // Outer glow
        let outer_glow_size = 0.85;
        if (sun_dot > outer_glow_size) {
            let outer_strength = smoothstep(outer_glow_size, 0.98, sun_dot) * 0.2 * sun_intensity;
            let outer_color = vec3<f32>(1.0, 0.85, 0.65);
            sky_color = mix(sky_color, outer_color, outer_strength);
        }

        // Horizon glow near sun
        if (view_dir.y < 0.3 && view_dir.y > -0.1) {
            let horizon_glow = max(sun_dot, 0.0) * (1.0 - abs(view_dir.y) * 3.0) * 0.15 * sun_intensity;
            sky_color += vec3<f32>(1.0, 0.80, 0.55) * horizon_glow;
        }
    }

    // Moon rendering (only when above horizon) - sharp crisp moon, not sun-like
    if (moon_intensity > 0.05) {
        // Moon disc - crisp edge, warm pale color
        let moon_size = 0.9985;
        if (moon_dot > moon_size) {
            // Sharp edge transition
            let disc_intensity = smoothstep(moon_size, 0.9992, moon_dot);
            let moon_color = vec3<f32>(0.95, 0.90, 0.80);  // Pale warm moon
            sky_color = mix(sky_color, moon_color, disc_intensity);
        }

        // Thin tight glow right around moon edge
        let moon_glow_size = 0.995;
        if (moon_dot > moon_glow_size && moon_dot <= moon_size) {
            let glow_strength = smoothstep(moon_glow_size, moon_size, moon_dot) * 0.15;
            let glow_color = vec3<f32>(0.85, 0.82, 0.75);
            sky_color = mix(sky_color, glow_color, glow_strength);
        }
    }

    // Add subtle stars at night (when sun is below horizon)
    if (sun_intensity < 0.2 && view_dir.y > 0.0) {
        let star_noise = fract(sin(dot(view_dir.xz * 500.0, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let star_threshold = 0.997;
        if (star_noise > star_threshold) {
            let star_brightness = (star_noise - star_threshold) / (1.0 - star_threshold);
            let night_factor = 1.0 - sun_intensity * 5.0;
            sky_color += vec3<f32>(0.9, 0.85, 0.75) * star_brightness * 0.5 * night_factor * view_dir.y;
        }
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
    // Animation data: x = current time, y = spawn time, z = wave speed, w = drop height
    anim_data: vec4<f32>,
};

// Light/shadow uniform (day/night cycle)
struct LightUniform {
    view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,    // xyz = direction, w = intensity
    sun_color: vec4<f32>,        // rgb = color
    moon_direction: vec4<f32>,   // xyz = direction, w = intensity
    moon_color: vec4<f32>,       // rgb = warm amber
    ambient_color: vec4<f32>,    // rgb = ambient, a = time_of_day
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
    @location(3) ao_tile_packed: vec4<f32>, // Unorm8x4: [ao, tile_u/255, tile_v/255, light/255]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao: f32,
    @location(4) tile_count: vec2<f32>,     // Tile repeat count for greedy meshing
    @location(5) shadow_pos: vec4<f32>,     // Position in light space for shadow lookup
    @location(6) block_light: f32,          // Block light level (0.0-1.0, from torches etc)
    @location(7) fade_alpha: f32,           // Fade-in alpha for chunk loading
};

// Simple hash function for randomization
fn hash_vec2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Unpack normal first (we need it for top-face detection)
    let normal = vec3<f32>(
        f32(in.normal_packed.x) / 127.0,
        f32(in.normal_packed.y) / 127.0,
        f32(in.normal_packed.z) / 127.0
    );

    // Animation parameters from camera uniform
    let current_time = camera.anim_data.x;
    let spawn_time = camera.anim_data.y;
    let wave_speed = camera.anim_data.z;  // Blocks per second for wave expansion

    // Position unchanged - no geometry animation
    var pos = in.position;

    // Use CHUNK position for consistent fade across entire chunk
    let chunk_pos = floor(pos / 16.0) * 16.0;

    // Distance from camera position (horizontal only for wave effect)
    let cam_pos = camera.view_pos.xyz;
    let dist = length(chunk_pos.xz - cam_pos.xz);

    // Calculate when this chunk "spawns" based on wave expansion
    let chunk_birth = spawn_time + dist / wave_speed;

    // Age of this chunk (time since it should have appeared)
    let age = current_time - chunk_birth;

    // Simple alpha fade-in: 0 -> 1 over 0.5 seconds
    let fade_duration = 0.5;
    var alpha = 1.0;
    if (age >= 0.0 && age < fade_duration) {
        alpha = age / fade_duration;
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_position = in.position;
    out.normal = normal;
    out.uv = in.uv;
    out.ao = in.ao_tile_packed.x;
    // Tile counts are stored as bytes 1,2 normalized to 0-1, scale back to 1-255 range
    out.tile_count = vec2<f32>(
        max(in.ao_tile_packed.y * 255.0, 1.0),
        max(in.ao_tile_packed.z * 255.0, 1.0)
    );
    // Block light level (byte 3, 0-255 scaled to 0-1)
    out.block_light = in.ao_tile_packed.w;
    // Transform position to light space for shadow lookup
    out.shadow_pos = light.view_proj * vec4<f32>(pos, 1.0);
    // Fade-in alpha
    out.fade_alpha = alpha;
    return out;
}

// PCF shadow sampling with rotated kernel for smooth diagonal shadows
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

    // Minimal slope-based bias - small values for sharp shadow contact at object bases
    let ndotl = dot(normal, light_dir);
    let bias = max(0.0008 * (1.0 - ndotl), 0.0002);
    let biased_depth = current_depth - bias;

    // Texel size for 4K shadow map
    let texel_size = 1.0 / 4096.0;

    // Rotated PCF kernel - samples at 45 degree angles for smooth diagonal shadows
    // This avoids the axis-aligned stair-stepping of regular grid sampling
    var shadow = 0.0;

    // Center sample
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv, biased_depth);

    // Inner ring (rotated 45 degrees, radius ~1 texel)
    let r1 = texel_size * 0.707;  // sqrt(2)/2 for 45 degree rotation
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(r1, r1), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(-r1, r1), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(r1, -r1), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(-r1, -r1), biased_depth);

    // Outer ring (axis-aligned, radius ~1.5 texels)
    let r2 = texel_size * 1.5;
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(r2, 0.0), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(-r2, 0.0), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(0.0, r2), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(0.0, -r2), biased_depth);

    // Third ring (rotated 45 degrees, radius ~2 texels)
    let r3 = texel_size * 1.414;  // sqrt(2) for larger diagonal
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(r3, r3), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(-r3, r3), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(r3, -r3), biased_depth);
    shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + vec2<f32>(-r3, -r3), biased_depth);

    shadow /= 13.0;

    return shadow;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate tiled UV for greedy meshing using WORLD POSITION
    // This ensures textures align to world grid regardless of how quads are merged
    let tiles_per_row = 4.0;  // 4x4 grid of 256x256 tiles = 1024x1024 atlas
    let tile_size = 1.0 / tiles_per_row;

    // Get the base tile corner from UV (tells us WHICH texture to use)
    let tile_base = floor(in.uv / tile_size) * tile_size;

    // Use world position for tiling (tells us WHERE in the texture)
    // fract gives us 0-1 position within each block, aligned to world grid
    let world_frac = fract(in.world_position);

    // Select the right 2D coordinates based on face normal (triplanar-lite)
    var tiled_local: vec2<f32>;
    if (abs(in.normal.y) > 0.5) {
        // Top/bottom face: use X and Z
        tiled_local = vec2<f32>(world_frac.x, world_frac.z);
    } else if (abs(in.normal.x) > 0.5) {
        // Left/right face: use Z and Y
        tiled_local = vec2<f32>(world_frac.z, world_frac.y);
    } else {
        // Front/back face: use X and Y
        tiled_local = vec2<f32>(world_frac.x, world_frac.y);
    }

    // Final UV = base tile + world-aligned local position
    let tiled_uv = tile_base + tiled_local * tile_size;

    // Sample texture
    let tex_color = textureSample(t_diffuse, s_diffuse, tiled_uv);

    // Alpha test: discard transparent pixels (cutout transparency like Minecraft leaves)
    if (tex_color.a < 0.5) {
        discard;
    }

    // === DAY/NIGHT LIGHTING SETUP ===
    // Sun properties from uniform
    let sun_dir = normalize(light.sun_direction.xyz);
    let sun_intensity = light.sun_direction.w;
    let sun_color = light.sun_color.rgb;

    // Moon properties from uniform
    let moon_dir = normalize(light.moon_direction.xyz);
    let moon_intensity = light.moon_direction.w;
    let moon_color = light.moon_color.rgb;

    // Time-based ambient from uniform
    let ambient_base = light.ambient_color.rgb;

    // Sky light direction (always from above)
    let sky_dir = vec3<f32>(0.0, 1.0, 0.0);

    // Ground bounce color (warm brown, subtle)
    let ground_color = vec3<f32>(0.35, 0.28, 0.2);

    // === SHADOW CALCULATION ===
    // Shadows are from sun only - fade out as sun sets, no moon shadows
    let raw_shadow = calculate_shadow(in.shadow_pos, in.normal, sun_dir);
    // Fade shadow influence with sun intensity - no harsh transition
    // When sun_intensity < 0.3, shadows start fading; at 0, no shadows
    let shadow_fade = smoothstep(0.0, 0.3, sun_intensity);
    let shadow = mix(1.0, raw_shadow, shadow_fade);

    // === FACE-BASED BRIGHTNESS ===
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
    // Sun contribution (fades with sun_intensity, affected by shadows)
    let sun_ndotl = max(dot(in.normal, sun_dir), 0.0);
    let sun_contribution = sun_color * sun_ndotl * 0.7 * sun_intensity * shadow;

    // Moon contribution - NO shadows, just soft ambient light
    let moon_ndotl = max(dot(in.normal, moon_dir), 0.0);
    let moon_contribution = moon_color * moon_ndotl * 0.25 * moon_intensity;

    // Sky light (hemisphere from above) - reduced at night
    let sky_ndotl = max(dot(in.normal, sky_dir), 0.0) * 0.5 + 0.5;
    let sky_brightness = max(sun_intensity * 0.8, moon_intensity * 0.1);
    // Sky shadows also fade with sun
    let sky_shadow = mix(1.0, mix(0.6, 1.0, shadow), shadow_fade);
    let sky_contribution = ambient_base * sky_ndotl * 0.2 * sky_shadow * (0.2 + sky_brightness * 0.8);

    // Ground bounce (subtle upward fill) - fades at night
    let ground_ndotl = max(dot(in.normal, -sky_dir), 0.0);
    let ground_contribution = ground_color * ground_ndotl * 0.05 * sun_intensity;

    // Ambient base - shadow influence also fades at night
    let ambient_shadow = mix(1.0, mix(0.7, 1.0, shadow), shadow_fade);
    let ambient = ambient_base * ambient_shadow;

    // === BLOCK LIGHT (Torches, etc.) ===
    let block_light_level = in.block_light;
    let block_light_curve = pow(block_light_level, 1.4);
    let torch_color = vec3<f32>(1.0, 0.85, 0.6);
    // Block light is MORE visible at night (relative to ambient)
    let night_boost = 1.0 + (1.0 - sun_intensity) * 0.5;
    let block_light_contribution = torch_color * block_light_curve * 0.9 * night_boost;

    // === AMBIENT OCCLUSION ===
    let ao_factor = mix(0.25, 1.0, in.ao);

    // === COMBINE LIGHTING ===
    let natural_light = ambient + sun_contribution + moon_contribution + sky_contribution + ground_contribution;
    let natural_lit = natural_light * face_brightness;

    let block_lit = block_light_contribution;

    let total_light = (natural_lit + block_lit) * ao_factor;
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

    // Apply fade-in alpha
    return vec4<f32>(final_color, tex_color.a * in.fade_alpha);
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
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

/// Shadow map size (4096x4096 for sharp diagonal shadows)
pub const SHADOW_MAP_SIZE: u32 = 4096;

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
                constant: 1,       // Minimal bias for sharp shadow contact
                slope_scale: 1.0,  // Reduced for tighter shadows at object bases
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
