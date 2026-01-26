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

// ═══════════════════════════════════════════════════════════════════════════
// PROFESSIONAL PCF SHADOW SAMPLING
// Uses adaptive bias, Vogel disk sampling, and smooth edge transitions
// ═══════════════════════════════════════════════════════════════════════════

// Vogel disk - provides well-distributed samples for soft PCF shadows
fn vogel_disk_sample(index: i32, samples: i32, rotation: f32) -> vec2<f32> {
    let golden_angle = 2.4; // Approximation of golden angle in radians
    let r = sqrt((f32(index) + 0.5) / f32(samples));
    let theta = f32(index) * golden_angle + rotation;
    return vec2<f32>(cos(theta), sin(theta)) * r;
}

fn calculate_shadow(shadow_pos: vec4<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Perspective divide
    let proj_coords = shadow_pos.xyz / shadow_pos.w;

    // Transform from [-1,1] to [0,1] UV space
    let shadow_uv = vec2<f32>(
        proj_coords.x * 0.5 + 0.5,
        -proj_coords.y * 0.5 + 0.5  // Flip Y
    );

    // If outside shadow map, return fully lit (no shadow)
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }

    // Current depth from light's perspective
    let current_depth = proj_coords.z;

    // If behind or in front of light frustum, no shadow
    if (current_depth > 1.0 || current_depth < 0.0) {
        return 1.0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADAPTIVE BIAS SYSTEM
    // Prevents shadow acne while maintaining sharp contact shadows
    // ═══════════════════════════════════════════════════════════════════════
    let ndotl = max(dot(normal, light_dir), 0.0);

    // Receiver plane bias: prevents self-shadowing on steep surfaces
    // Uses both constant and slope-scaled components
    let constant_bias = 0.0003;
    let slope_bias = 0.0015 * (1.0 - ndotl * ndotl); // Quadratic falloff for grazing angles
    let normal_offset_bias = 0.0005 * sqrt(1.0 - ndotl * ndotl); // Normal-based offset

    let total_bias = constant_bias + slope_bias + normal_offset_bias;
    let biased_depth = current_depth - total_bias;

    // ═══════════════════════════════════════════════════════════════════════
    // SOFT EDGE HANDLING
    // Prevents harsh shadow map boundary artifacts
    // With larger shadow coverage (280 units), we can use tighter edge fading
    // ═══════════════════════════════════════════════════════════════════════
    let edge_fade_start = 0.005;  // Start fading at 0.5% from edge (was 3%)
    let edge_fade_end = 0.025;    // Fully faded at 2.5% from edge (was 12%)
    let edge_x = min(shadow_uv.x, 1.0 - shadow_uv.x);
    let edge_y = min(shadow_uv.y, 1.0 - shadow_uv.y);
    let edge_dist = min(edge_x, edge_y);
    let shadow_strength = smoothstep(edge_fade_start, edge_fade_end, edge_dist);

    // ═══════════════════════════════════════════════════════════════════════
    // VOGEL DISK PCF SAMPLING
    // 16 samples in a well-distributed disk pattern for smooth soft shadows
    // ═══════════════════════════════════════════════════════════════════════
    let texel_size = 1.0 / 4096.0;
    let filter_radius = texel_size * 1.8; // Slightly larger filter for softer edges

    // Rotation based on position to break up banding patterns
    let rotation = fract(sin(dot(shadow_uv, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.28318;

    var shadow = 0.0;
    let num_samples = 16;

    // Vogel disk sampling - provides excellent coverage without regular patterns
    for (var i = 0; i < num_samples; i++) {
        let offset = vogel_disk_sample(i, num_samples, rotation) * filter_radius;
        shadow += textureSampleCompare(t_shadow, s_shadow, shadow_uv + offset, biased_depth);
    }
    shadow /= f32(num_samples);

    // ═══════════════════════════════════════════════════════════════════════
    // CONTACT HARDENING (PCSS-lite)
    // Shadows are sharper near occluder, softer further away
    // ═══════════════════════════════════════════════════════════════════════
    // Approximate penumbra - sharpen shadows that are mostly lit or mostly shadowed
    let penumbra_sharpness = 1.0 - abs(shadow - 0.5) * 2.0; // 0 at extremes, 1 in middle
    let sharpened_shadow = mix(
        shadow,
        smoothstep(0.25, 0.75, shadow), // Sharpen the transition
        0.3 * (1.0 - penumbra_sharpness)
    );

    // Blend to fully lit at shadow map edges
    return mix(1.0, sharpened_shadow, shadow_strength);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ═══════════════════════════════════════════════════════════════════════
    // TEXTURE ATLAS SAMPLING WITH MIP-AWARE ANTI-BLEEDING
    // Calculates proper margins based on which mip level is being sampled
    // ═══════════════════════════════════════════════════════════════════════

    let tiles_per_row = 4.0;  // 4x4 grid of 256x256 tiles = 1024x1024 atlas
    let tile_size = 1.0 / tiles_per_row;  // 0.25 in UV space
    let tile_pixels = 256.0;  // Pixels per tile in base mip

    // Get the base tile corner from UV (tells us WHICH texture to use)
    let tile_base = floor(in.uv / tile_size) * tile_size;

    // Select the right 2D world coordinates based on face normal (triplanar-lite)
    // Calculate derivatives BEFORE fract() to avoid discontinuity artifacts at block edges
    var world_2d: vec2<f32>;
    if (abs(in.normal.y) > 0.5) {
        // Top/bottom face: use X and Z
        world_2d = vec2<f32>(in.world_position.x, in.world_position.z);
    } else if (abs(in.normal.x) > 0.5) {
        // Left/right face: use Z and Y
        world_2d = vec2<f32>(in.world_position.z, in.world_position.y);
    } else {
        // Front/back face: use X and Y
        world_2d = vec2<f32>(in.world_position.x, in.world_position.y);
    }

    // Get derivatives from continuous world coords (no fract discontinuity)
    let world_dx = dpdx(world_2d) * tile_size;
    let world_dy = dpdy(world_2d) * tile_size;

    // Now apply fract for the actual UV coordinate
    let tiled_local = fract(world_2d);

    // Calculate UV within tile (0 to tile_size range)
    let uv_in_tile = tiled_local * tile_size;

    // ═══════════════════════════════════════════════════════════════════════
    // MIP-AWARE MARGIN CALCULATION
    // Use screen-space derivatives to estimate mip level and scale margin
    // ═══════════════════════════════════════════════════════════════════════

    // Use pre-fract derivatives for correct mip calculation
    let dx = world_dx;
    let dy = world_dy;

    // Maximum rate of change determines the mip level being used
    let max_deriv = max(
        max(abs(dx.x), abs(dx.y)),
        max(abs(dy.x), abs(dy.y))
    );

    // Convert derivative to approximate mip level
    // At mip 0, 1 pixel = 1/1024 UV. At mip N, 1 pixel = 2^N / 1024 UV
    // So mip level ≈ log2(deriv * atlas_size)
    let approx_mip = max(0.0, log2(max_deriv * 1024.0 + 1.0));

    // Scale margin based on mip level - need larger margin at higher mips
    // Base margin is 1 texel, scaled up by 2^mip
    let mip_scale = pow(2.0, min(approx_mip, 6.0));  // Cap at mip 6
    let margin = (mip_scale * 1.5) / (tile_pixels * tiles_per_row);  // 1.5 texels at current mip

    // Clamp UV to stay safely inside tile at current mip level
    let clamped_uv = clamp(
        uv_in_tile,
        vec2<f32>(margin),
        vec2<f32>(tile_size - margin)
    );

    // Final UV = base tile + clamped position within tile
    let tiled_uv = tile_base + clamped_uv;

    // Sample texture with clamped derivatives to prevent bleeding during filtering
    // Use textureSampleGrad with clamped gradients for maximum control
    let clamped_dx = clamp(dx, vec2<f32>(-tile_size * 0.5), vec2<f32>(tile_size * 0.5));
    let clamped_dy = clamp(dy, vec2<f32>(-tile_size * 0.5), vec2<f32>(tile_size * 0.5));
    let tex_color = textureSampleGrad(t_diffuse, s_diffuse, tiled_uv, clamped_dx, clamped_dy);

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
    // Professional-quality AO with smooth cubic interpolation
    // Uses hermite smoothstep for natural-looking soft shadows in corners
    // Floor at 65% to prevent overly dark corners while maintaining definition
    let ao_raw = in.ao;
    // Apply smoothstep curve for more natural falloff (prevents harsh banding)
    let ao_smooth = smoothstep(0.0, 1.0, ao_raw);
    // Second smoothstep pass for even softer transitions (hermite^2)
    let ao_curved = smoothstep(0.0, 1.0, ao_smooth);
    // Map to final range: 65% minimum brightness to prevent over-darkening
    let ao_factor = mix(0.65, 1.0, ao_curved);

    // === COMBINE LIGHTING ===
    let natural_light = ambient + sun_contribution + moon_contribution + sky_contribution + ground_contribution;
    let natural_lit = natural_light * face_brightness;

    let block_lit = block_light_contribution;

    let total_light = (natural_lit + block_lit) * ao_factor;
    var final_color = tex_color.rgb * total_light;

    // === ATMOSPHERIC DEPTH & HEIGHT-BASED FOG ===
    let dist = length(camera.view_pos.xyz - in.world_position);
    let is_underwater = camera.anim_data.w > 0.5;

    // Different fog for underwater vs normal
    var fog_factor: f32;
    var fog_color: vec3<f32>;

    if (is_underwater) {
        // Underwater: heavy blue-green fog, very limited visibility
        let underwater_density = 0.08;
        fog_factor = 1.0 - exp(-dist * underwater_density);
        fog_factor = clamp(fog_factor, 0.0, 0.95);
        fog_color = vec3<f32>(0.1, 0.3, 0.4);
    } else {
        // ═══════════════════════════════════════════════════════════════════════
        // REALISTIC VALLEY FOG (Radiation Fog)
        // Based on real meteorology: cold air sinks into valleys, moisture from
        // nearby water bodies condenses. Fog pools near water level (Y=32).
        // Hills stay clear as warm air rises and moisture evaporates.
        // ═══════════════════════════════════════════════════════════════════════

        // Water/sea level where fog is most likely to form
        let water_level = 32.0;
        let fog_ceiling = 55.0;  // Fog dissipates above this

        let frag_height = in.world_position.y;

        // Height-based fog: strongest near water, fades as elevation increases
        let height_factor = 1.0 - smoothstep(water_level, fog_ceiling, frag_height);
        // Smooth cubic falloff for natural look
        let height_influence = height_factor * height_factor * (3.0 - 2.0 * height_factor);

        // Light distant haze (atmospheric perspective)
        let base_fog_density = 0.00005;

        // Valley/water fog - denser near sea level
        let valley_fog_density = 0.0003;

        // Blend based on height
        let effective_density = mix(base_fog_density, valley_fog_density, height_influence);

        // Push fog further from player - minimal fog nearby, builds up at distance
        // At night, fog creeps closer for atmospheric effect
        let time_of_day = light.ambient_color.a;
        let is_night = smoothstep(0.25, 0.0, time_of_day) + smoothstep(0.75, 1.0, time_of_day);
        let fog_start_dist = mix(80.0, 35.0, is_night);  // 80 blocks day, 35 blocks night
        let dist_factor = smoothstep(fog_start_dist, fog_start_dist + 150.0, dist);

        // Calculate fog with distance, scaled by distance factor
        fog_factor = (1.0 - exp(-dist * effective_density)) * dist_factor;

        // Fog caps: clear on hills (3%), misty in valleys (15%)
        let max_fog = mix(0.03, 0.15, height_influence);
        fog_factor = clamp(fog_factor, 0.0, max_fog);

        // Fog color - cool morning mist near water, warmer haze at distance
        let mist_color = vec3<f32>(0.78, 0.83, 0.90);  // Cool valley mist
        let haze_color = vec3<f32>(0.82, 0.85, 0.90);  // Warm distant haze
        fog_color = mix(haze_color, mist_color, height_influence);
    }

    // Atmospheric scattering (Rayleigh) - subtle blue shift at distance
    let scatter_factor = clamp((dist - 400.0) / 1200.0, 0.0, 0.05);
    let scatter_color = vec3<f32>(0.75, 0.82, 0.90);

    // Subtle desaturation at extreme distance (aerial perspective)
    let luminance = dot(final_color, vec3<f32>(0.299, 0.587, 0.114));
    let desat_factor = clamp((dist - 500.0) / 1400.0, 0.0, 0.04);
    final_color = mix(final_color, vec3<f32>(luminance), desat_factor);

    // Apply atmospheric effects
    if (!is_underwater) {
        final_color = mix(final_color, scatter_color, scatter_factor);
    }
    final_color = mix(final_color, fog_color, fog_factor);

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
            // Professional shadow bias settings:
            // - Small constant bias prevents surface acne
            // - Moderate slope scale handles grazing angles
            // - Clamp prevents over-biasing at steep angles
            bias: wgpu::DepthBiasState {
                constant: 2,       // Small constant offset (in depth buffer units)
                slope_scale: 1.5,  // Scale bias by surface slope for grazing angles
                clamp: 0.01,       // Maximum bias to prevent peter-panning
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

// ═══════════════════════════════════════════════════════════════════════════
// WATER SHADER
// ═══════════════════════════════════════════════════════════════════════════

/// Water shader with Gerstner waves, fresnel reflection, and foam
const WATER_SHADER: &str = r#"
// Camera uniform (same as chunk shader)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,  // x = current time
};

// Light/shadow uniform (same as chunk shader)
struct LightUniform {
    view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    moon_direction: vec4<f32>,
    moon_color: vec4<f32>,
    ambient_color: vec4<f32>,
};

// Water-specific uniform
struct WaterUniform {
    // Wave parameters
    time: f32,
    wave_speed: f32,
    wave_scale: f32,
    wave_strength: f32,
    // Colors
    shallow_color: vec4<f32>,
    deep_color: vec4<f32>,
    // Visual parameters
    fresnel_power: f32,
    foam_threshold: f32,
    refraction_strength: f32,
    transparency: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> light: LightUniform;

@group(2) @binding(0)
var<uniform> water: WaterUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_packed: u32,
    @location(2) uv_data: u32,       // [foam, depth, flow_lo, flow_hi]
    @location(3) light_data: u32,    // [block_light, sky_light, 0, 0]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) foam_factor: f32,
    @location(3) water_depth: f32,
    @location(4) block_light: f32,
    @location(5) sky_light: f32,
    @location(6) view_dir: vec3<f32>,
};

// Gerstner wave function for realistic water motion
fn gerstner_wave(pos: vec2<f32>, time: f32, dir: vec2<f32>, wavelength: f32, steepness: f32) -> vec3<f32> {
    let k = 2.0 * 3.14159 / wavelength;
    let c = sqrt(9.8 / k);
    let d = normalize(dir);
    let f = k * (dot(d, pos) - c * time);
    let a = steepness / k;

    return vec3<f32>(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

// Unified wave calculation - all waves flow in similar direction with variation
// This prevents the grid/X-pattern from opposing wave directions
fn calculate_waves(pos: vec2<f32>, time: f32) -> vec3<f32> {
    // Primary wave direction (roughly toward +X with slight angle)
    let primary_dir = vec2<f32>(0.9, 0.4);

    // All waves flow in similar directions - subtle angular variation only
    var total = vec3<f32>(0.0);

    // Primary wave - longest, visible but controlled height
    total += gerstner_wave(pos, time, primary_dir, 14.0, 0.22);

    // Secondary wave - adds natural variation
    let secondary_dir = vec2<f32>(0.85, 0.55);  // ~15 degrees off primary
    total += gerstner_wave(pos, time * 0.9, secondary_dir, 9.0, 0.15);

    // Tertiary wave - smaller ripples for detail
    let tertiary_dir = vec2<f32>(0.95, 0.25);   // ~10 degrees other way
    total += gerstner_wave(pos, time * 1.15, tertiary_dir, 6.0, 0.10);

    return total;
}

// Calculate combined wave height at a position (for foam calculation)
fn gerstner_wave_height(pos: vec2<f32>, time: f32) -> f32 {
    return calculate_waves(pos, time).y;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos = in.position;
    let world_xz = pos.xz;
    let t = camera.anim_data.x * water.wave_speed;

    // Apply unified wave system - all waves flow together
    let wave_offset = calculate_waves(world_xz, t);

    // Scale wave displacement
    pos.y += wave_offset.y * water.wave_strength;
    pos.x += wave_offset.x * water.wave_strength * 0.15;
    pos.z += wave_offset.z * water.wave_strength * 0.15;

    // Calculate smooth normal from wave derivatives
    let epsilon = 0.5;  // Larger epsilon for smoother normals
    let wave_at_x_plus = calculate_waves(world_xz + vec2(epsilon, 0.0), t);
    let wave_at_x_minus = calculate_waves(world_xz - vec2(epsilon, 0.0), t);
    let wave_at_z_plus = calculate_waves(world_xz + vec2(0.0, epsilon), t);
    let wave_at_z_minus = calculate_waves(world_xz - vec2(0.0, epsilon), t);

    // Central difference for smoother derivatives
    let dx = (wave_at_x_plus.y - wave_at_x_minus.y) / (2.0 * epsilon);
    let dz = (wave_at_z_plus.y - wave_at_z_minus.y) / (2.0 * epsilon);

    var wave_normal = normalize(vec3<f32>(-dx * water.wave_strength, 1.0, -dz * water.wave_strength));

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_position = pos;
    out.normal = wave_normal;

    // Unpack foam factor and depth from uv_data
    let uv_bytes = unpack4x8unorm(in.uv_data);
    out.foam_factor = uv_bytes.x;
    out.water_depth = uv_bytes.y * 255.0;  // Scale back to 0-255 range

    // Unpack light data
    let light_bytes = unpack4x8unorm(in.light_data);
    out.block_light = light_bytes.x;
    out.sky_light = light_bytes.y;

    // View direction
    out.view_dir = normalize(camera.view_pos.xyz - pos);

    return out;
}

// Schlick's fresnel approximation - physically accurate
fn schlick_fresnel(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// Smooth caustic-like light pattern
fn caustic_noise(p: vec2<f32>, time: f32) -> f32 {
    // Multiple overlapping smooth sine waves for organic look
    let p1 = p * 0.25 + vec2<f32>(time * 0.015, time * 0.012);
    let p2 = p * 0.35 + vec2<f32>(-time * 0.013, time * 0.016);
    let p3 = p * 0.18 + vec2<f32>(time * 0.01, -time * 0.008);

    let n1 = sin(p1.x * 2.5 + p1.y * 2.0) * cos(p1.y * 2.2 - p1.x * 1.3);
    let n2 = sin(p2.x * 1.8 - p2.y * 2.4) * cos(p2.y * 1.6 + p2.x * 2.1);
    let n3 = sin(p3.x * 1.4 + p3.y * 1.7) * cos(p3.y * 1.2 - p3.x * 0.9);

    // Smooth blend with softened edges
    let combined = (n1 + n2 * 0.7 + n3 * 0.5) / 2.2;
    return smoothstep(-0.3, 0.3, combined) * 0.6 + 0.2;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let view_dir = normalize(in.view_dir);
    let time = camera.anim_data.x;

    // === VIEW-ANGLE DEPTH ===
    // Looking at an angle means light travels through more water
    let view_dot = max(dot(view_dir, vec3<f32>(0.0, 1.0, 0.0)), 0.1);
    let optical_depth = in.water_depth / view_dot;  // Effective depth based on view angle

    // === FRESNEL (Schlick approximation) ===
    // Water has F0 of about 0.02
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = schlick_fresnel(n_dot_v, 0.02);

    // === SKY/ENVIRONMENT REFLECTION ===
    let reflect_dir = reflect(-view_dir, normal);

    // Dynamic sky color based on sun position
    let sky_brightness = max(light.sun_direction.w, 0.15);
    let sky_up = mix(
        vec3<f32>(0.15, 0.25, 0.45),   // Night sky
        vec3<f32>(0.40, 0.65, 0.95),   // Day sky zenith
        sky_brightness
    );
    let sky_horizon = mix(
        vec3<f32>(0.10, 0.15, 0.25),   // Night horizon
        vec3<f32>(0.55, 0.70, 0.85),   // Day horizon
        sky_brightness
    );
    let sky_blend = pow(max(reflect_dir.y, 0.0), 0.8);
    var sky_reflection = mix(sky_horizon, sky_up, sky_blend);

    // Sun specular - smooth blended highlights
    let sun_dir = normalize(light.sun_direction.xyz);
    let sun_intensity = light.sun_direction.w;
    let sun_reflect = max(dot(reflect_dir, sun_dir), 0.0);

    // Use smoothstep for softer falloff instead of harsh pow
    let spec_tight = smoothstep(0.995, 1.0, sun_reflect) * 1.5;      // Core highlight
    let spec_medium = smoothstep(0.95, 0.995, sun_reflect) * 0.4;    // Inner glow
    let spec_wide = smoothstep(0.85, 0.96, sun_reflect) * 0.15;      // Outer soft glow
    let sun_specular = (spec_tight + spec_medium + spec_wide) * sun_intensity;
    sky_reflection += vec3<f32>(1.0, 0.97, 0.90) * sun_specular;

    // === WATER COLOR WITH ABSORPTION ===
    // Real water absorbs red first, then green - deeper = more blue
    let absorption_depth = clamp(optical_depth / 12.0, 0.0, 1.0);

    // Natural water colors - shallow is nearly invisible, color builds with depth
    // Real water: shallow = clear, deep = blue from light absorption
    let surface_color = vec3<f32>(0.08, 0.12, 0.14);   // Very dark, nearly invisible
    let mid_color = vec3<f32>(0.12, 0.20, 0.28);       // Subtle blue tint emerging
    let deep_color = vec3<f32>(0.06, 0.12, 0.22);      // Deep water blue-gray

    // Smooth depth gradient using smoothstep for soft transitions
    let shallow_blend = smoothstep(0.0, 0.5, absorption_depth);
    let deep_blend = smoothstep(0.3, 1.0, absorption_depth);
    var water_color = mix(surface_color, mid_color, shallow_blend);
    water_color = mix(water_color, deep_color, deep_blend);

    // === SUBSURFACE SCATTERING ===
    // Light penetrating through water creates subtle glow
    let sun_through = max(dot(-view_dir, sun_dir), 0.0);
    let scatter_amount = pow(sun_through, 4.0) * (1.0 - absorption_depth) * 0.25;
    let scatter_color = vec3<f32>(0.25, 0.35, 0.32);  // Subtle warm-gray with hint of green
    water_color += scatter_color * scatter_amount * sun_intensity;

    // === CAUSTICS SHIMMER ===
    // Subtle animated light patterns on the water surface
    let caustic = caustic_noise(in.world_position.xz, time);
    let caustic_strength = (1.0 - absorption_depth) * sun_intensity * 0.08;
    water_color += vec3<f32>(0.20, 0.22, 0.20) * caustic * caustic_strength;

    // === COMBINE REFLECTION AND WATER COLOR ===
    // Fresnel controls the blend - water is highly reflective at grazing angles
    // Shallow water reflects less (you see through it), deep water reflects more
    let depth_reflection_boost = smoothstep(0.0, 0.4, absorption_depth) * 0.25;
    let reflection_strength = smoothstep(0.0, 1.0, fresnel) * (0.55 + depth_reflection_boost) + 0.05;
    var final_color = mix(water_color, sky_reflection, reflection_strength);

    // === FOAM ===
    let base_foam = in.foam_factor;
    let wave_foam_sample = gerstner_wave_height(in.world_position.xz, time * water.wave_speed);
    let wave_peak_foam = smoothstep(0.15, 0.25, wave_foam_sample) * 0.08;
    let total_foam = clamp(base_foam + wave_peak_foam, 0.0, 1.0);

    if (total_foam > water.foam_threshold) {
        let foam_intensity = (total_foam - water.foam_threshold) / (1.0 - water.foam_threshold);
        let noise_pos = in.world_position.xz * 0.3 + vec2<f32>(time * 0.05, time * 0.04);
        let foam_noise = fract(sin(dot(noise_pos, vec2<f32>(12.9898, 78.233))) * 43758.5453);

        // Foam - mostly white with subtle warm tint
        let foam_color = mix(vec3<f32>(0.92, 0.94, 0.92), water_color * 1.2, 0.10);
        let foam_blend = foam_intensity * foam_intensity * (0.35 + foam_noise * 0.2);
        final_color = mix(final_color, foam_color, foam_blend * 0.55);
    }

    // === LIGHTING ===
    let ambient = light.ambient_color.rgb * 0.5;
    let sun_diffuse = max(dot(normal, sun_dir), 0.0) * 0.3;
    let sun_contrib = light.sun_color.rgb * sun_diffuse * sun_intensity;

    // Block light (torches) - warm glow in water
    let torch_color = vec3<f32>(1.0, 0.80, 0.55);
    let block_contrib = torch_color * in.block_light * 0.7;

    // Sky light contribution
    let sky_contrib = vec3<f32>(0.5, 0.6, 0.8) * in.sky_light * 0.25;

    let total_light = ambient + sun_contrib + block_contrib + sky_contrib;
    final_color *= total_light;

    // === ATMOSPHERIC FOG ===
    let dist = length(camera.view_pos.xyz - in.world_position);
    let fog_factor = 1.0 - exp(-dist * 0.00012);
    // Fog color matches sky for consistency
    let fog_color = mix(sky_horizon, sky_up, 0.3);
    final_color = mix(final_color, fog_color, clamp(fog_factor, 0.0, 0.2));

    // === TRANSPARENCY ===
    // Shallow water is very transparent, opacity builds with depth
    let depth_opacity = 1.0 - exp(-optical_depth * 0.05);  // Gradual opacity buildup
    let base_alpha = mix(0.15, 0.65, depth_opacity);       // Very clear shallow, solid deep
    let alpha = mix(base_alpha, 0.80, fresnel * 0.5);      // Fresnel adds some opacity at edges

    return vec4<f32>(final_color, alpha);
}
"#;

/// Water uniform buffer structure (must match shader)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniform {
    pub time: f32,
    pub wave_speed: f32,
    pub wave_scale: f32,
    pub wave_strength: f32,
    pub shallow_color: [f32; 4],
    pub deep_color: [f32; 4],
    pub fresnel_power: f32,
    pub foam_threshold: f32,
    pub refraction_strength: f32,
    pub transparency: f32,
}

impl Default for WaterUniform {
    fn default() -> Self {
        Self {
            time: 0.0,
            wave_speed: 0.45,           // Slightly faster for more life
            wave_scale: 8.0,
            wave_strength: 0.09,        // Balanced - visible but won't clip terrain
            // Natural water palette - desaturated blue-gray
            shallow_color: [0.38, 0.42, 0.44, 0.30],  // Near-neutral gray
            deep_color: [0.08, 0.14, 0.22, 0.70],     // Deep muted blue-gray
            fresnel_power: 4.0,  // Balanced reflection curve
            foam_threshold: 0.55, // Slightly lower - foam a bit more visible
            refraction_strength: 0.04,
            transparency: 0.80,
        }
    }
}

/// Create water bind group layout for water-specific uniforms
pub fn create_water_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Bind Group Layout"),
        entries: &[
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
    })
}

/// Create water render pipeline
pub fn create_water_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    water_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    use crate::world::WaterVertex;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Shader"),
        source: wgpu::ShaderSource::Wgsl(WATER_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Pipeline Layout"),
        bind_group_layouts: &[camera_bind_group_layout, light_bind_group_layout, water_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[WaterVertex::desc()],
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
            cull_mode: None,  // No culling for water (need to see from both sides)
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,  // Don't write depth (transparent)
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
