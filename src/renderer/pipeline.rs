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

    // Sun rendering - warmer at horizon, larger near sunrise/sunset
    if (sun_intensity > 0.05) {
        // Calculate how close sun is to horizon (0 = horizon, 1 = high in sky)
        let sun_height = sun_dir.y;
        let horizon_factor = 1.0 - smoothstep(0.0, 0.35, sun_height);

        // Sun appears larger near horizon (atmospheric magnification)
        let base_sun_size = 0.9985;
        let sun_size = base_sun_size - horizon_factor * 0.002;  // Larger disc near horizon

        // Color shifts warm/orange near horizon, whiter when high
        let sun_color_high = vec3<f32>(1.0, 0.98, 0.92);      // Bright warm white
        let sun_color_horizon = vec3<f32>(1.0, 0.65, 0.25);   // Deep orange/amber
        let sun_tint = mix(sun_color_high, sun_color_horizon, horizon_factor);

        // Outer atmospheric bloom - wider and warmer near horizon
        let bloom_outer = pow(max(sun_dot, 0.0), 6.0) * (0.12 + horizon_factor * 0.15) * sun_intensity;
        sky_color += sun_tint * bloom_outer;

        // Middle bloom layer
        let bloom_mid = pow(max(sun_dot, 0.0), 24.0) * (0.20 + horizon_factor * 0.1) * sun_intensity;
        sky_color += mix(sun_tint, vec3<f32>(1.0, 0.95, 0.85), 0.5) * bloom_mid;

        // Inner bloom - tighter, brighter
        let bloom_inner = pow(max(sun_dot, 0.0), 96.0) * 0.35 * sun_intensity;
        sky_color += vec3<f32>(1.0, 0.97, 0.88) * bloom_inner;

        // Sun disc core
        if (sun_dot > sun_size) {
            let disc_intensity = smoothstep(sun_size, sun_size + 0.001, sun_dot) * sun_intensity;
            let core_color = mix(vec3<f32>(1.0, 1.0, 0.95), sun_tint, horizon_factor * 0.4);
            sky_color = mix(sky_color, core_color, disc_intensity);
            // Overbright center
            sky_color += vec3<f32>(0.25, 0.20, 0.10) * disc_intensity * (1.0 - horizon_factor * 0.3);
        }

        // Rich horizon glow during sunrise/sunset
        if (sun_height < 0.2 && sun_height > -0.1) {
            let glow_strength = 1.0 - smoothstep(-0.1, 0.2, sun_height);
            let angular_glow = pow(max(sun_dot, 0.0), 3.0);
            let horizon_glow = angular_glow * glow_strength * 0.3 * sun_intensity;
            // Gradient from orange to pink/magenta further from sun
            let glow_color = mix(vec3<f32>(1.0, 0.55, 0.20), vec3<f32>(1.0, 0.70, 0.50), 1.0 - angular_glow);
            sky_color += glow_color * horizon_glow;
        }
    }

    // Moon rendering - solid and luminous
    if (moon_intensity > 0.05) {
        // Subtle outer halo - very soft atmospheric glow
        let moon_halo = pow(max(moon_dot, 0.0), 12.0) * 0.06;
        sky_color += vec3<f32>(0.75, 0.78, 0.85) * moon_halo;

        // Moon disc - solid, bright, well-defined
        let moon_size = 0.9990;
        let moon_edge = 0.9995;

        if (moon_dot > moon_size) {
            // Sharp edge with slight softness
            let edge_blend = smoothstep(moon_size, moon_edge, moon_dot);

            // Solid moon surface - pale cream/silver
            let moon_surface = vec3<f32>(0.92, 0.90, 0.85);

            // Slight limb darkening for 3D appearance
            let limb_factor = smoothstep(moon_size, moon_edge, moon_dot);
            let shaded_surface = mix(moon_surface * 0.85, moon_surface, limb_factor);

            // Apply moon with full opacity for solid look
            sky_color = mix(sky_color, shaded_surface, edge_blend * moon_intensity);

            // Subtle bright edge highlight
            if (moon_dot > moon_edge - 0.0003 && moon_dot < moon_edge) {
                sky_color += vec3<f32>(0.1, 0.1, 0.08) * moon_intensity;
            }
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

// ═══════════════════════════════════════════════════════════════════════════
// CHUNK SHADER
// Uses DDA ray marching for shadows + Voxel Cone Tracing for GI
// ═══════════════════════════════════════════════════════════════════════════
const CHUNK_SHADER: &str = r#"
// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
};

// Light uniform
struct LightUniform {
    view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    moon_direction: vec4<f32>,
    moon_color: vec4<f32>,
    ambient_color: vec4<f32>,
};

// Voxel volume uniform (144 bytes for proper WGSL alignment)
struct VoxelVolumeUniform {
    volume_center: vec4<f32>,
    volume_size: vec4<f32>,
    inv_volume_size: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    time_of_day: vec4<f32>,  // x = time, y = shadow_mode (0=hard, 1=soft), z = shadow_debug, w = padding
    player_light: vec4<f32>, // xyz = position, w = intensity (0-1, where 1 = 15 emission)
    volume_origin: vec4<i32>, // xyz = world position of voxel [0,0,0] for toroidal wrapping
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@group(2) @binding(0)
var<uniform> light: LightUniform;

// Voxel GI group
@group(3) @binding(0)
var<uniform> voxel_volume: VoxelVolumeUniform;
@group(3) @binding(1)
var t_voxel: texture_3d<f32>;
@group(3) @binding(2)
var t_radiance: texture_3d<f32>;
@group(3) @binding(3)
var s_voxel: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_packed: vec4<i32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao_tile_packed: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao: f32,
    @location(4) tile_count: vec2<f32>,
    @location(5) block_light: f32,
    @location(6) fade_alpha: f32,
    @location(7) is_emissive: f32,  // 1.0 for emissive blocks (torches, etc.)
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let normal = vec3<f32>(
        f32(in.normal_packed.x) / 127.0,
        f32(in.normal_packed.y) / 127.0,
        f32(in.normal_packed.z) / 127.0
    );

    // Flags in 4th byte of normal_packed: bit 0 = is_foliage, bit 1 = is_emissive
    let is_foliage = (in.normal_packed.w & 1) != 0;
    let is_emissive = (in.normal_packed.w & 2) != 0;

    let current_time = camera.anim_data.x;
    let spawn_time = camera.anim_data.y;
    let wave_speed = camera.anim_data.z;

    var pos = in.position;

    // Gentle breeze animation for foliage blocks (leaves)
    if (is_foliage) {
        // Use world position for consistent wind across vertices
        let wind_time = current_time * 0.7;  // Slower, more peaceful
        let world_pos = pos;

        // Gentle overlapping sine waves for subtle swaying
        let wind_x = sin(wind_time + world_pos.x * 0.3 + world_pos.z * 0.2) * 0.025
                   + sin(wind_time * 0.6 + world_pos.z * 0.4) * 0.012;
        let wind_z = sin(wind_time * 0.5 + world_pos.z * 0.3 + world_pos.x * 0.15) * 0.02
                   + cos(wind_time * 0.8 + world_pos.x * 0.35) * 0.01;

        // Very subtle vertical movement
        let wind_y = sin(wind_time * 0.9 + world_pos.x * 0.25 + world_pos.z * 0.25) * 0.006;

        pos.x += wind_x;
        pos.y += wind_y;
        pos.z += wind_z;
    }

    let chunk_pos = floor(pos / 16.0) * 16.0;
    let cam_pos = camera.view_pos.xyz;
    let dist = length(chunk_pos.xz - cam_pos.xz);
    let chunk_birth = spawn_time + dist / wave_speed;
    let age = current_time - chunk_birth;
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
    out.tile_count = vec2<f32>(
        max(in.ao_tile_packed.y * 255.0, 1.0),
        max(in.ao_tile_packed.z * 255.0, 1.0)
    );
    out.block_light = in.ao_tile_packed.w;
    out.fade_alpha = alpha;
    out.is_emissive = select(0.0, 1.0, is_emissive);
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// DDA RAY MARCHING FOR SHADOWS
// ═══════════════════════════════════════════════════════════════════════════

// Convert world position to voxel UV using toroidal wrapping
// The volume wraps around, so coordinates outside [0,256) map back into the volume
fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    // Compute local position relative to origin (floating point for smooth sampling)
    let origin = vec3<f32>(voxel_volume.volume_origin.xyz);
    let local = world_pos - origin;

    // Toroidal wrap using floating-point modulo
    // ((x % 256) + 256) % 256 handles negative values correctly
    let wrapped = ((local % 256.0) + 256.0) % 256.0;

    // Convert to UV (0-1 range)
    return wrapped / 256.0;
}

// With toroidal wrapping, UV is always in valid range
fn is_in_volume(uv: vec3<f32>) -> bool {
    return all(uv >= vec3<f32>(0.0)) && all(uv < vec3<f32>(1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// SHADOW MODE: HARD (new style - solid fill, sharp edges)
// ═══════════════════════════════════════════════════════════════════════════
fn ray_march_shadow_hard(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let max_steps = 112;  // Slightly reduced from 128, but not too aggressive

    var accumulated_opacity = 0.0;
    var solid_hits = 0;

    var pos = world_pos;
    var total_distance = 0.0;

    for (var i = 0; i < max_steps; i++) {
        // Adaptive step size: small near (accuracy), larger far (range)
        // Optimized step progression while maintaining accuracy
        var current_step: f32;
        if (i < 6) {
            current_step = 0.25;  // Fine detail near surface
        } else if (i < 24) {
            current_step = 0.5;   // Medium detail
        } else if (i < 56) {
            current_step = 1.0;   // Coarse
        } else {
            current_step = 2.0;   // Very coarse for long shadows
        }

        pos += light_dir * current_step;
        total_distance += current_step;

        // With toroidal wrapping, check WORLD distance from volume center (not UV bounds)
        // Valid data is within ~120 blocks of center (half of 256, with margin)
        let center_dist = length(pos - voxel_volume.volume_center.xyz);
        if (center_dist > 120.0) { break; }

        // Fade near edge of valid region (100-120 blocks from center)
        let edge_fade = 1.0 - smoothstep(100.0, 120.0, center_dist);

        let uv = world_to_voxel_uv(pos);
        let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
        let opacity = voxel.r * edge_fade;

        if (opacity > 0.5) {
            solid_hits += 1;
            accumulated_opacity += 0.4;
            // EARLY EXIT: Fully occluded after 3 solid hits (was continuing indefinitely)
            if (solid_hits >= 3) { break; }
        } else if (opacity > 0.1) {
            accumulated_opacity += opacity * 0.2;
        }

        // Early exit when clearly in shadow
        if (accumulated_opacity >= 1.1) { break; }
    }

    var shadow: f32;
    if (solid_hits > 0) {
        let shadow_strength = smoothstep(0.3, 0.7, clamp(accumulated_opacity, 0.0, 1.0));
        shadow = mix(1.0, 0.25, shadow_strength);  // Darker shadows
    } else if (accumulated_opacity > 0.0) {
        shadow = 1.0 - clamp(accumulated_opacity * 0.5, 0.0, 0.5);
    } else {
        shadow = 1.0;
    }

    if (accumulated_opacity > 1.0) {
        let depth_darkness = clamp((accumulated_opacity - 1.0) * 0.08, 0.0, 0.15);
        shadow *= (1.0 - depth_darkness);
    }

    return shadow;
}

// ═══════════════════════════════════════════════════════════════════════════
// SHADOW MODE: SOFT (old style - gradient, painterly)
// ═══════════════════════════════════════════════════════════════════════════
fn ray_march_shadow_soft_core(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let max_steps = 64;  // Reduced from 96 - early termination handles dense scenes

    var accumulated_opacity = 0.0;
    var min_distance_to_occluder = 1000.0;
    var solid_hits = 0;

    var pos = world_pos + light_dir * 0.5;
    var total_distance = 0.5;

    for (var i = 0; i < max_steps; i++) {
        // Optimized adaptive step size - faster progression
        var step_size: f32;
        if (i < 8) {
            step_size = 0.5;
        } else if (i < 32) {
            step_size = 1.5;
        } else {
            step_size = 3.0;
        }

        pos += light_dir * step_size;
        total_distance += step_size;

        // With toroidal wrapping, check WORLD distance from volume center
        let center_dist = length(pos - voxel_volume.volume_center.xyz);
        if (center_dist > 120.0) { break; }

        let edge_fade = 1.0 - smoothstep(100.0, 120.0, center_dist);

        let uv = world_to_voxel_uv(pos);
        let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
        let opacity = voxel.r * edge_fade;

        if (opacity > 0.3) {
            if (min_distance_to_occluder > 999.0) {
                min_distance_to_occluder = total_distance;
            }
            solid_hits += 1;
            // EARLY EXIT: Fully occluded after hitting 3 solid voxels
            if (solid_hits >= 3) {
                accumulated_opacity = 1.0;
                break;
            }
        }

        accumulated_opacity += opacity * 0.2;  // Faster accumulation

        // AGGRESSIVE EARLY EXIT
        if (accumulated_opacity >= 0.85) { break; }
    }

    var shadow = 1.0 - clamp(accumulated_opacity, 0.0, 1.0);

    if (min_distance_to_occluder < 999.0) {
        let penumbra_softness = smoothstep(1.0, 12.0, min_distance_to_occluder);
        shadow = mix(shadow * 0.3, shadow, penumbra_softness * 0.5 + 0.5);
    }

    return shadow;
}

// Multi-tap soft shadow - optimized 2-tap version (was 4-tap)
// 2 taps provides good softness while halving the shadow ray cost
fn ray_march_shadow_soft_multitap(world_pos: vec3<f32>, light_dir: vec3<f32>, softness: f32) -> f32 {
    var tangent: vec3<f32>;
    if (abs(light_dir.y) < 0.9) {
        tangent = normalize(cross(light_dir, vec3<f32>(0.0, 1.0, 0.0)));
    } else {
        tangent = normalize(cross(light_dir, vec3<f32>(1.0, 0.0, 0.0)));
    }
    let bitangent = cross(light_dir, tangent);

    // Optimized 2-tap: diagonal offsets for best coverage
    let jitter_scale = softness * 1.2;  // Slightly reduced for 2-tap
    let o0 = vec2<f32>(-0.6, -0.4) * jitter_scale;
    let o1 = vec2<f32>( 0.6,  0.4) * jitter_scale;

    let d0 = normalize(light_dir + tangent * o0.x + bitangent * o0.y);
    let d1 = normalize(light_dir + tangent * o1.x + bitangent * o1.y);

    // 2 taps instead of 4 - still smooth, half the cost
    let shadow_sum = ray_march_shadow_soft_core(world_pos, d0)
                   + ray_march_shadow_soft_core(world_pos, d1);

    return shadow_sum * 0.5;
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED SHADOW FUNCTION - picks mode based on uniform + distance LOD
// ═══════════════════════════════════════════════════════════════════════════
fn ray_march_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>, shadow_mode: f32) -> f32 {
    if (shadow_mode > 0.5) {
        // Soft mode (old style)
        return ray_march_shadow_soft_multitap(world_pos, light_dir, 0.3);
    } else {
        // Hard mode (new style)
        return ray_march_shadow_hard(world_pos, light_dir);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VOXEL CONE TRACING FOR GI
// ═══════════════════════════════════════════════════════════════════════════

fn cone_trace(origin: vec3<f32>, direction: vec3<f32>, cone_ratio: f32, max_dist: f32) -> vec4<f32> {
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    let start_dist = 1.5;
    var dist = start_dist;

    while (dist < max_dist && accumulated_alpha < 0.95) {
        let diameter = max(1.0, 2.0 * cone_ratio * dist);
        let sample_pos = origin + direction * dist;
        let uv = world_to_voxel_uv(sample_pos);

        if (!is_in_volume(uv)) {
            break;
        }

        let mip = log2(diameter);
        let sample = textureSampleLevel(t_radiance, s_voxel, uv, mip);

        let alpha = 1.0 - accumulated_alpha;
        accumulated_color += sample.rgb * sample.a * alpha;
        accumulated_alpha += sample.a * alpha;

        dist += diameter * 0.5;
    }

    return vec4<f32>(accumulated_color, accumulated_alpha);
}

fn trace_diffuse_gi(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cone_angle = 0.577;
    let max_distance = 48.0;

    var tangent: vec3<f32>;
    if (abs(normal.y) > 0.9) {
        tangent = normalize(cross(normal, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        tangent = normalize(cross(normal, vec3<f32>(0.0, 1.0, 0.0)));
    }
    let bitangent = cross(normal, tangent);

    var gi = vec3<f32>(0.0);

    let center = cone_trace(world_pos, normal, cone_angle, max_distance);
    gi += center.rgb * 0.4;

    let side_weight = 0.15;
    let tilt = 0.707;

    let dir1 = normalize(normal * tilt + tangent * tilt);
    let dir2 = normalize(normal * tilt - tangent * tilt);
    let dir3 = normalize(normal * tilt + bitangent * tilt);
    let dir4 = normalize(normal * tilt - bitangent * tilt);

    gi += cone_trace(world_pos, dir1, cone_angle, max_distance).rgb * side_weight;
    gi += cone_trace(world_pos, dir2, cone_angle, max_distance).rgb * side_weight;
    gi += cone_trace(world_pos, dir3, cone_angle, max_distance).rgb * side_weight;
    gi += cone_trace(world_pos, dir4, cone_angle, max_distance).rgb * side_weight;

    return gi;
}

// ═══════════════════════════════════════════════════════════════════════════
// VOXEL AMBIENT OCCLUSION
// ═══════════════════════════════════════════════════════════════════════════

fn trace_voxel_ao(world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let max_distance = 6.0;
    let step_size = 1.0;
    var occlusion = 0.0;

    var tangent: vec3<f32>;
    if (abs(normal.y) > 0.9) {
        tangent = normalize(cross(normal, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        tangent = normalize(cross(normal, vec3<f32>(0.0, 1.0, 0.0)));
    }
    let bitangent = cross(normal, tangent);

    // Sample in a small cone around the normal
    for (var i = 1; i <= 4; i++) {
        let dist = f32(i) * step_size;
        let falloff = 1.0 - dist / max_distance;

        let sample_pos = world_pos + normal * dist;
        let uv = world_to_voxel_uv(sample_pos);

        if (is_in_volume(uv)) {
            let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
            occlusion += voxel.r * falloff * 0.25;
        }
    }

    return 1.0 - clamp(occlusion, 0.0, 0.5);
}

// ═══════════════════════════════════════════════════════════════════════════
// SKY OCCLUSION - Check if position has sky access (for cave darkness)
// ═══════════════════════════════════════════════════════════════════════════

fn trace_sky_occlusion(world_pos: vec3<f32>) -> f32 {
    // Trace upward to check for sky access
    // Returns 0.0 = fully enclosed (cave), 1.0 = full sky access
    let up_dir = vec3<f32>(0.0, 1.0, 0.0);
    let max_steps = 32;
    let step_size = 2.0;  // Larger steps for efficiency

    var accumulated_opacity = 0.0;
    var pos = world_pos + up_dir * 1.0;  // Start slightly above surface

    for (var i = 0; i < max_steps; i++) {
        pos += up_dir * step_size;
        let uv = world_to_voxel_uv(pos);

        // If we exit the volume upward without hitting anything, we have sky access
        if (uv.y > 1.0) {
            break;
        }

        // If outside volume horizontally, assume some sky access
        if (!is_in_volume(uv)) {
            accumulated_opacity += 0.2;
            break;
        }

        let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
        // Gradual occlusion - 0.08 per solid voxel means ~12 blocks for full darkness
        // This allows light to penetrate further into cave openings
        accumulated_opacity += voxel.r * 0.08;

        if (accumulated_opacity >= 1.0) {
            break;
        }
    }

    return 1.0 - clamp(accumulated_opacity, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// VOXEL-BASED TORCH/EMISSIVE LIGHTING
// ═══════════════════════════════════════════════════════════════════════════

// Trace shadow ray from surface toward held light
fn trace_held_light_shadow(world_pos: vec3<f32>, light_pos: vec3<f32>) -> f32 {
    let to_light = light_pos - world_pos;
    let dist_to_light = length(to_light);
    let light_dir = to_light / dist_to_light;

    // Start slightly off surface to avoid self-shadowing
    var pos = world_pos + light_dir * 0.3;
    let max_steps = 32;
    let step_size = max(dist_to_light / f32(max_steps), 0.5);

    var accumulated_opacity = 0.0;

    for (var i = 0; i < max_steps; i++) {
        pos += light_dir * step_size;

        // Check if we've reached or passed the light
        if (distance(world_pos, pos) >= dist_to_light - 0.5) {
            break;
        }

        let uv = world_to_voxel_uv(pos);
        if (!is_in_volume(uv)) {
            continue;
        }

        let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
        let opacity = voxel.r;

        // Solid block blocks light
        if (opacity > 0.5) {
            accumulated_opacity += 0.5;
        } else if (opacity > 0.1) {
            accumulated_opacity += opacity * 0.3;
        }

        if (accumulated_opacity >= 1.0) {
            return 0.0;  // Fully blocked
        }
    }

    return 1.0 - clamp(accumulated_opacity, 0.0, 1.0);
}

// Calculate light from player's held torch (with shadow casting)
fn calculate_held_light(world_pos: vec3<f32>) -> vec3<f32> {
    let torch_color = vec3<f32>(1.0, 0.85, 0.6);
    let player_pos = voxel_volume.player_light.xyz;
    let intensity = voxel_volume.player_light.w;

    if (intensity <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Distance-based falloff
    let dist = distance(world_pos, player_pos);
    let max_radius = intensity * 24.0;  // ~24 blocks for full torch

    if (dist >= max_radius) {
        return vec3<f32>(0.0);
    }

    // Trace shadow from surface toward held light
    let shadow = trace_held_light_shadow(world_pos, player_pos);

    // Quartic falloff: smaller bright core, dims quickly, long subtle tail to edge
    let falloff = 1.0 - (dist / max_radius);
    let light_strength = pow(falloff, 4.0) * intensity;

    // Apply shadow - light blocked by geometry between surface and player
    return torch_color * light_strength * shadow * 2.5;
}

fn sample_torch_light(world_pos: vec3<f32>) -> vec3<f32> {
    // Sample pre-propagated light from the voxel volume
    // Light is propagated on CPU and stored in the A channel
    let torch_color = vec3<f32>(1.0, 0.85, 0.6);

    let uv = world_to_voxel_uv(world_pos);
    if (!is_in_volume(uv)) {
        return vec3<f32>(0.0);
    }

    // Sample the voxel - A channel contains propagated light intensity
    // Linear filtering provides smooth interpolation between voxels
    let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
    let light_intensity = voxel.a;  // A channel = propagated light (0-1)

    // Apply softer gamma curve for longer falloff trail
    let curved_light = pow(light_intensity, 1.2);

    // Much brighter placed torches
    return torch_color * curved_light * 3.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Texture sampling (same as original)
    let tiles_per_row = 8.0;
    let tile_size = 1.0 / tiles_per_row;
    let tile_pixels = 256.0;
    let tile_base = floor(in.uv / tile_size) * tile_size;

    var world_2d: vec2<f32>;
    if (abs(in.normal.y) > 0.5) {
        world_2d = vec2<f32>(in.world_position.x, in.world_position.z);
    } else if (abs(in.normal.x) > 0.5) {
        world_2d = vec2<f32>(in.world_position.z, in.world_position.y);
    } else {
        world_2d = vec2<f32>(in.world_position.x, in.world_position.y);
    }

    let world_dx = dpdx(world_2d) * tile_size;
    let world_dy = dpdy(world_2d) * tile_size;
    let tiled_local = fract(world_2d);
    let uv_in_tile = tiled_local * tile_size;

    let dx = world_dx;
    let dy = world_dy;
    let max_deriv = max(max(abs(dx.x), abs(dx.y)), max(abs(dy.x), abs(dy.y)));
    let approx_mip = max(0.0, log2(max_deriv * 1024.0 + 1.0));
    let mip_scale = pow(2.0, min(approx_mip, 6.0));
    let margin = (mip_scale * 1.5) / (tile_pixels * tiles_per_row);

    let clamped_uv = clamp(uv_in_tile, vec2<f32>(margin), vec2<f32>(tile_size - margin));
    let tiled_uv = tile_base + clamped_uv;

    let clamped_dx = clamp(dx, vec2<f32>(-tile_size * 0.5), vec2<f32>(tile_size * 0.5));
    let clamped_dy = clamp(dy, vec2<f32>(-tile_size * 0.5), vec2<f32>(tile_size * 0.5));
    let tex_color = textureSampleGrad(t_diffuse, s_diffuse, tiled_uv, clamped_dx, clamped_dy);

    if (tex_color.a < 0.5) {
        discard;
    }

    // === EMISSIVE BLOCKS (torches, uranium, etc.) ===
    // Render at full brightness regardless of lighting - visible at any distance
    if (in.is_emissive > 0.5) {
        // Warm emissive color that will be picked up by bloom
        let emissive_tint = vec3<f32>(1.2, 1.0, 0.8);  // Slightly warm
        let emissive_boost = 1.5;  // Boost brightness for bloom
        var emissive_color = tex_color.rgb * emissive_tint * emissive_boost;

        // Subtle face-based shading so it's not completely flat
        var face_mult = 0.9;
        if (in.normal.y > 0.5) { face_mult = 1.0; }
        else if (in.normal.y < -0.5) { face_mult = 0.8; }
        emissive_color *= face_mult;

        // Apply distance fog but reduced (emissive cuts through fog)
        let dist = length(camera.view_pos.xyz - in.world_position);
        let fog_factor = 1.0 - exp(-dist * 0.00003);  // Very light fog
        let fog_color = vec3<f32>(0.8, 0.82, 0.85);
        emissive_color = mix(emissive_color, fog_color, fog_factor * 0.3);

        return vec4<f32>(emissive_color, tex_color.a * in.fade_alpha);
    }

    // === VOXEL-BASED LIGHTING ===
    let sun_dir = normalize(voxel_volume.sun_direction.xyz);
    let sun_intensity = voxel_volume.sun_direction.w;
    let sun_color = voxel_volume.sun_color.rgb;
    let ambient_base = voxel_volume.ambient_color.rgb;

    let moon_dir = normalize(light.moon_direction.xyz);
    let moon_intensity = light.moon_direction.w;
    let moon_color = light.moon_color.rgb;

    // DEBUG: Check if we're in the voxel volume
    let debug_uv = world_to_voxel_uv(in.world_position);
    let in_volume = is_in_volume(debug_uv);

    // DDA Ray March Shadow (replaces shadow map)
    let ndotl = max(dot(in.normal, sun_dir), 0.0);
    let shadow_mode = voxel_volume.time_of_day.y;   // 0 = hard, 1 = soft
    let shadow_debug = voxel_volume.time_of_day.z;  // 0 = off, 1 = on

    var shadow = 1.0;  // Default: fully lit
    var was_traced = 0.0;  // Track if we traced or not

    // Always trace shadow for all faces - shadow determines if the AREA is lit,
    // not whether the face receives direct sun (that's what ndotl is for).
    // Previously, side faces weren't traced in hard mode, causing them to appear
    // lighter than shadowed top faces because they got full ambient.
    {
        was_traced = 1.0;
        // Offset along normal to prevent self-shadowing
        // Use a larger offset for grazing angles to avoid acne
        let grazing_factor = 1.0 - max(ndotl, 0.0);
        let normal_offset = mix(0.15, 0.5, grazing_factor);
        let shadow_origin = in.world_position + in.normal * normal_offset;
        shadow = ray_march_shadow(shadow_origin, sun_dir, shadow_mode);
    }

    // === SHADOW DEBUG MODE ===
    // Press U to toggle
    // RED channel   = ndotl (how much face sees sun: 0=away, 1=facing sun)
    // GREEN channel = shadow value (0=full shadow, 1=fully lit)
    // BLUE channel  = was traced (0=skipped, 1=traced)
    //
    // What to look for:
    // - Bright RED tops = faces that see sun (should have shadows cast ON them)
    // - GREEN shows shadow: dark green = shadowed, bright green = lit
    // - BLUE shows tracing: blue = traced, no blue = skipped
    // - A face with RED but no GREEN shadow = shadow not being cast properly
    if (shadow_debug > 0.5) {
        return vec4<f32>(ndotl, shadow, was_traced, 1.0);
    }

    // Shadow fade with sun intensity - stronger shadows during day
    let shadow_fade = smoothstep(0.0, 0.3, sun_intensity);
    // Smooth shadow curve (not too harsh, not too soft)
    let shadow_smooth = smoothstep(0.0, 1.0, shadow);
    let final_shadow = mix(1.0, shadow_smooth, shadow_fade);

    // Face-based brightness
    var face_brightness = 0.8;
    if (in.normal.y > 0.5) {
        face_brightness = 1.0;
    } else if (in.normal.y < -0.5) {
        face_brightness = 0.5;
    } else if (abs(in.normal.x) > 0.5) {
        face_brightness = 0.7;
    } else {
        face_brightness = 0.85;
    }

    // Direct lighting
    let sun_ndotl = max(dot(in.normal, sun_dir), 0.0);
    let sun_contribution = sun_color * sun_ndotl * 0.7 * sun_intensity * final_shadow;

    let moon_ndotl = max(dot(in.normal, moon_dir), 0.0);
    let moon_contribution = moon_color * moon_ndotl * 0.25 * moon_intensity;

    // Voxel Cone Traced GI - disabled for now, too expensive
    // TODO: Re-enable once shadows are stable
    let gi = vec3<f32>(0.0);

    // Voxel AO - short range occlusion from nearby geometry
    let voxel_ao_raw = trace_voxel_ao(in.world_position, in.normal);
    let voxel_ao = mix(0.85, 1.0, voxel_ao_raw);  // Subtle effect (0.85-1.0 range)

    // Sky occlusion - check if we're in an enclosed space (cave)
    let sky_access = trace_sky_occlusion(in.world_position);

    // Per-vertex AO
    let ao_raw = in.ao;
    let ao_smooth = smoothstep(0.0, 1.0, ao_raw);
    let ao_curved = smoothstep(0.0, 1.0, ao_smooth);
    let vertex_ao = mix(0.65, 1.0, ao_curved);

    // Combined AO
    let combined_ao = vertex_ao * voxel_ao;

    // Block light (torches) - now using voxel-based sampling
    // Sample emissive light from nearby voxels in the volume
    let voxel_torch_light = sample_torch_light(in.world_position);

    // Add player's held torch light (dynamic, follows player)
    let held_torch_light = calculate_held_light(in.world_position);

    // Night boost: torch light is more prominent when sun is down
    let night_boost = 1.0 + (1.0 - sun_intensity) * 0.5;
    let block_light_contribution = (voxel_torch_light + held_torch_light) * night_boost;

    // Combine all lighting
    // Sky access affects ambient - caves get much less ambient light
    let cave_ambient_factor = mix(0.10, 1.0, sky_access);  // 10% ambient in full cave, 100% with sky
    let sky_contrib = ambient_base * 0.2 * (0.2 + sun_intensity * 0.8) * cave_ambient_factor;
    let ambient_with_sky = ambient_base * cave_ambient_factor;
    let natural_light_raw = ambient_with_sky + sun_contribution + moon_contribution + sky_contrib + gi * 0.3;

    // Shadow affects the whole natural light result
    // In caves (no sky access), shadow multiplier is lower
    let min_shadow_light = mix(0.07, 0.3, sky_access);  // 7% minimum in caves, 30% with sky

    // Shadow applies to ALL faces - if a block is in shadow, all its faces are darkened
    var shadow_multiplier: f32;
    if (shadow_mode > 0.5) {
        // Soft mode: shadow applies everywhere (painterly look)
        shadow_multiplier = mix(min_shadow_light, 1.0, final_shadow);
    } else {
        // Hard mode: shadow applies to ALL faces equally
        // This prevents side faces from being lighter than shadowed top faces
        shadow_multiplier = mix(min_shadow_light, 1.0, final_shadow);
    }
    let natural_light = natural_light_raw * shadow_multiplier;
    let natural_lit = natural_light * face_brightness;

    let total_light = (natural_lit + block_light_contribution) * combined_ao;
    var final_color = tex_color.rgb * total_light;

    // === FOG AND DISTANCE FADE ===
    let dist = length(camera.view_pos.xyz - in.world_position);
    let is_underwater = camera.anim_data.w > 0.5;

    var fog_factor: f32;
    var fog_color: vec3<f32>;
    var alpha_fade: f32 = 1.0;

    if (is_underwater) {
        let underwater_density = 0.08;
        fog_factor = 1.0 - exp(-dist * underwater_density);
        fog_factor = clamp(fog_factor, 0.0, 0.95);
        fog_color = vec3<f32>(0.1, 0.3, 0.4);
    } else {
        let water_level = 32.0;
        let fog_ceiling = 55.0;
        let frag_height = in.world_position.y;

        let height_factor = 1.0 - smoothstep(water_level, fog_ceiling, frag_height);
        let height_influence = height_factor * height_factor * (3.0 - 2.0 * height_factor);

        let base_fog_density = 0.00005;
        let valley_fog_density = 0.0003;
        let effective_density = mix(base_fog_density, valley_fog_density, height_influence);

        let time_of_day = voxel_volume.time_of_day.x;
        let is_night = smoothstep(0.25, 0.0, time_of_day) + smoothstep(0.75, 1.0, time_of_day);
        let fog_start_dist = mix(80.0, 35.0, is_night);
        let dist_factor = smoothstep(fog_start_dist, fog_start_dist + 150.0, dist);

        fog_factor = (1.0 - exp(-dist * effective_density)) * dist_factor;
        let max_fog = mix(0.03, 0.15, height_influence);
        fog_factor = clamp(fog_factor, 0.0, max_fog);

        // === RENDER DISTANCE FADE ===
        // Fade chunks out at render distance - use alpha fade, not bright fog
        // Render distance: 24 chunks = 384 blocks
        let render_dist = 384.0;
        let fade_start = render_dist - 64.0;  // Start fading 4 chunks before edge
        let edge_fade = smoothstep(fade_start, render_dist, dist);

        // Calculate fog color based on actual scene lighting (time-aware)
        // Use ambient color as base - it's already dark at night
        let ambient = voxel_volume.ambient_color.rgb;
        let sun_contribution = voxel_volume.sun_color.rgb * max(0.0, voxel_volume.sun_direction.w);

        // Fog color is a desaturated, slightly brighter version of ambient
        // At night this will be very dark, during day it will be light
        let scene_brightness = (ambient.r + ambient.g + ambient.b) / 3.0 +
                               (sun_contribution.r + sun_contribution.g + sun_contribution.b) * 0.1;

        // Base fog is tinted based on time of day
        let day_fog = vec3<f32>(0.75, 0.82, 0.92);   // Light blue-gray for day
        let night_fog = vec3<f32>(0.08, 0.10, 0.15); // Very dark blue for night
        let base_fog = mix(day_fog, night_fog, is_night);

        // Blend fog color with ambient to match scene
        fog_color = mix(base_fog, ambient * 1.5, 0.3);
        fog_color = fog_color * (0.3 + scene_brightness * 0.7);

        // For distant chunks, use alpha fadeout instead of fog color blend
        // This makes chunks disappear rather than turn bright
        alpha_fade = 1.0 - edge_fade * edge_fade; // Quadratic for smoother fade

        // Only apply minimal atmospheric fog at distance (not the edge fade)
        // The alpha fade handles the render distance transition
        fog_factor = fog_factor * (1.0 - edge_fade * 0.8);
    }

    // Apply atmospheric fog (subtle, distance-based)
    final_color = mix(final_color, fog_color, fog_factor);

    return vec4<f32>(final_color, tex_color.a * in.fade_alpha * alpha_fade);
}
"#;

/// Create voxel-lit chunk pipeline (uses DDA ray marching + VCT instead of shadow maps)
pub fn create_voxel_lit_chunk_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    voxel_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Voxel-Lit Chunk Shader"),
        source: wgpu::ShaderSource::Wgsl(CHUNK_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Voxel-Lit Chunk Pipeline Layout"),
        bind_group_layouts: &[
            camera_bind_group_layout,
            texture_bind_group_layout,
            light_bind_group_layout,  // Reuse light uniform for moon/etc
            voxel_bind_group_layout,  // Voxel GI system
        ],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Voxel-Lit Chunk Pipeline"),
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
    @location(7) flow_angle: f32,  // Flow direction in radians (0 = +X)
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

// Rotate a 2D vector by an angle (in radians)
fn rotate_2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(
        v.x * c - v.y * s,
        v.x * s + v.y * c
    );
}

// Unified wave calculation - all waves flow in direction based on flow_angle
// flow_angle: direction of water flow in radians (0 = +X, PI/2 = +Z)
fn calculate_waves(pos: vec2<f32>, time: f32, flow_angle: f32) -> vec3<f32> {
    // Base direction from flow simulation
    let flow_dir = vec2<f32>(cos(flow_angle), sin(flow_angle));

    // Primary wave follows flow direction
    let primary_dir = flow_dir;

    // All waves flow in similar directions - subtle angular variation only
    var total = vec3<f32>(0.0);

    // Primary wave - longest, visible but controlled height
    total += gerstner_wave(pos, time, primary_dir, 14.0, 0.22);

    // Secondary wave - adds natural variation (~30 degrees offset)
    let secondary_dir = rotate_2d(flow_dir, 0.52);  // ~30 degrees
    total += gerstner_wave(pos, time * 0.9, secondary_dir, 9.0, 0.15);

    // Tertiary wave - smaller ripples for detail (~-15 degrees offset)
    let tertiary_dir = rotate_2d(flow_dir, -0.26);   // ~-15 degrees
    total += gerstner_wave(pos, time * 1.15, tertiary_dir, 6.0, 0.10);

    return total;
}

// Legacy version for code that doesn't have flow_angle yet
fn calculate_waves_default(pos: vec2<f32>, time: f32) -> vec3<f32> {
    return calculate_waves(pos, time, 0.0);  // Default to +X direction
}

// Calculate combined wave height at a position (for foam calculation)
fn gerstner_wave_height(pos: vec2<f32>, time: f32, flow_angle: f32) -> f32 {
    return calculate_waves(pos, time, flow_angle).y;
}

// Smooth noise for subtle water surface variation
fn water_noise(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Multi-octave smooth ripple noise for realistic water surface
fn ripple_height(pos: vec2<f32>, time: f32) -> f32 {
    var height = 0.0;

    // Large gentle swells
    let p1 = pos * 0.04 + vec2<f32>(time * 0.02, time * 0.015);
    height += sin(p1.x * 2.0 + p1.y * 1.5) * 0.4;
    height += sin(p1.x * 1.2 - p1.y * 2.1) * 0.3;

    // Medium ripples
    let p2 = pos * 0.12 + vec2<f32>(-time * 0.03, time * 0.025);
    height += sin(p2.x * 3.0 + p2.y * 2.5) * 0.2;

    // Small detail ripples
    let p3 = pos * 0.25 + vec2<f32>(time * 0.04, -time * 0.03);
    height += sin(p3.x * 4.0 + p3.y * 3.5) * 0.1;

    return height;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos = in.position;
    let world_xz = pos.xz;
    let t = camera.anim_data.x * water.wave_speed;

    // Unpack foam factor, depth, and flow_angle from uv_data
    let uv_bytes = unpack4x8unorm(in.uv_data);
    out.foam_factor = uv_bytes.x;
    out.water_depth = uv_bytes.y * 255.0;

    // Unpack flow angle (kept for compatibility but using subtle ripples now)
    let flow_lo = u32(uv_bytes.z * 255.0);
    let flow_hi = u32(uv_bytes.w * 255.0);
    let flow_u16 = flow_lo | (flow_hi << 8u);
    let flow_angle = f32(flow_u16) / 65535.0 * 6.28318;
    out.flow_angle = flow_angle;

    // Subtle height displacement - very gentle surface movement
    let ripple = ripple_height(world_xz, t);
    pos.y += ripple * water.wave_strength * 0.3;

    // Calculate normal from ripple derivatives for realistic reflections
    let epsilon = 0.5;
    let h_px = ripple_height(world_xz + vec2<f32>(epsilon, 0.0), t);
    let h_nx = ripple_height(world_xz - vec2<f32>(epsilon, 0.0), t);
    let h_pz = ripple_height(world_xz + vec2<f32>(0.0, epsilon), t);
    let h_nz = ripple_height(world_xz - vec2<f32>(0.0, epsilon), t);

    let dx = (h_px - h_nx) / (2.0 * epsilon);
    let dz = (h_pz - h_nz) / (2.0 * epsilon);

    // Subtle normal perturbation for gentle rippling reflections
    let normal_strength = water.wave_strength * 0.5;
    var ripple_normal = normalize(vec3<f32>(-dx * normal_strength, 1.0, -dz * normal_strength));

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_position = pos;
    out.normal = ripple_normal;

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
    let view_dot = max(dot(view_dir, vec3<f32>(0.0, 1.0, 0.0)), 0.1);
    let optical_depth = in.water_depth / view_dot;

    // === FRESNEL (Schlick approximation) ===
    // Water F0 = 0.02, but boost for more visible reflections
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = schlick_fresnel(n_dot_v, 0.02);

    // === SKY/ENVIRONMENT REFLECTION ===
    let reflect_dir = reflect(-view_dir, normal);

    // Rich sky gradient for reflections
    let sky_brightness = max(light.sun_direction.w, 0.15);
    let sky_up = mix(
        vec3<f32>(0.12, 0.18, 0.35),   // Night sky - deeper blue
        vec3<f32>(0.45, 0.68, 0.98),   // Day sky zenith - vibrant blue
        sky_brightness
    );
    let sky_horizon = mix(
        vec3<f32>(0.08, 0.12, 0.22),   // Night horizon
        vec3<f32>(0.65, 0.78, 0.92),   // Day horizon - bright warm
        sky_brightness
    );

    // Smooth gradient based on reflection direction
    let sky_blend = pow(max(reflect_dir.y, 0.0), 0.6);
    var sky_reflection = mix(sky_horizon, sky_up, sky_blend);

    // Sun specular highlight - bright and sharp
    let sun_dir = normalize(light.sun_direction.xyz);
    let sun_intensity = light.sun_direction.w;
    let sun_reflect = max(dot(reflect_dir, sun_dir), 0.0);

    // Layered specular for realistic sun reflection
    let spec_core = smoothstep(0.997, 1.0, sun_reflect) * 2.5;     // Bright core
    let spec_inner = smoothstep(0.98, 0.997, sun_reflect) * 0.8;   // Inner glow
    let spec_outer = smoothstep(0.92, 0.98, sun_reflect) * 0.25;   // Soft outer halo
    let sun_specular = (spec_core + spec_inner + spec_outer) * sun_intensity;
    sky_reflection += vec3<f32>(1.0, 0.98, 0.92) * sun_specular;

    // === WATER COLOR WITH DEPTH ===
    let absorption_depth = clamp(optical_depth / 10.0, 0.0, 1.0);

    // Clean water colors - emphasize clarity
    let surface_color = vec3<f32>(0.06, 0.10, 0.12);   // Nearly clear
    let mid_color = vec3<f32>(0.10, 0.18, 0.26);       // Subtle teal
    let deep_color = vec3<f32>(0.05, 0.10, 0.18);      // Deep blue-gray

    let shallow_blend = smoothstep(0.0, 0.4, absorption_depth);
    let deep_blend = smoothstep(0.25, 1.0, absorption_depth);
    var water_color = mix(surface_color, mid_color, shallow_blend);
    water_color = mix(water_color, deep_color, deep_blend);

    // === SUBSURFACE SCATTERING ===
    let sun_through = max(dot(-view_dir, sun_dir), 0.0);
    let scatter_amount = pow(sun_through, 3.0) * (1.0 - absorption_depth) * 0.2;
    let scatter_color = vec3<f32>(0.20, 0.30, 0.28);
    water_color += scatter_color * scatter_amount * sun_intensity;

    // === COMBINE REFLECTION AND WATER COLOR ===
    // Strong Fresnel-based reflections - realistic water behavior
    // At grazing angles, water is almost mirror-like
    let fresnel_boost = pow(fresnel, 0.7);  // Boost Fresnel effect
    let depth_boost = smoothstep(0.0, 0.5, absorption_depth) * 0.15;
    let reflection_strength = fresnel_boost * 0.85 + depth_boost + 0.08;
    var final_color = mix(water_color, sky_reflection, clamp(reflection_strength, 0.0, 0.95));

    // === SUBTLE FOAM (shorelines only) ===
    let base_foam = in.foam_factor;
    if (base_foam > water.foam_threshold) {
        let foam_intensity = (base_foam - water.foam_threshold) / (1.0 - water.foam_threshold);
        let noise_pos = in.world_position.xz * 0.4 + vec2<f32>(time * 0.03, time * 0.025);
        let foam_noise = fract(sin(dot(noise_pos, vec2<f32>(12.9898, 78.233))) * 43758.5453);

        let foam_color = vec3<f32>(0.95, 0.96, 0.94);
        let foam_blend = foam_intensity * foam_intensity * (0.4 + foam_noise * 0.15);
        final_color = mix(final_color, foam_color, foam_blend * 0.5);
    }

    // === LIGHTING ===
    let ambient = light.ambient_color.rgb * 0.6;
    let sun_diffuse = max(dot(normal, sun_dir), 0.0) * 0.25;
    let sun_contrib = light.sun_color.rgb * sun_diffuse * sun_intensity;

    let torch_color = vec3<f32>(1.0, 0.80, 0.55);
    let block_contrib = torch_color * in.block_light * 0.6;
    let sky_contrib = vec3<f32>(0.5, 0.6, 0.8) * in.sky_light * 0.2;

    let total_light = ambient + sun_contrib + block_contrib + sky_contrib;
    final_color *= total_light;

    // === ATMOSPHERIC FOG ===
    let dist = length(camera.view_pos.xyz - in.world_position);
    let fog_factor = 1.0 - exp(-dist * 0.00010);
    let fog_color = mix(sky_horizon, sky_up, 0.35);
    final_color = mix(final_color, fog_color, clamp(fog_factor, 0.0, 0.15));

    // === TRANSPARENCY ===
    // Clearer shallow water, more opaque with depth
    let depth_opacity = 1.0 - exp(-optical_depth * 0.06);
    let base_alpha = mix(0.20, 0.70, depth_opacity);
    let alpha = mix(base_alpha, 0.85, fresnel * 0.4);

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
            wave_speed: 0.45,           // Gentle wave speed
            wave_scale: 8.0,
            wave_strength: 0.09,        // Subtle - visible but won't clip terrain
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

// ═══════════════════════════════════════════════════════════════════════════
// DIMMER SHADER - Fullscreen overlay for settings menu
// ═══════════════════════════════════════════════════════════════════════════

/// Dimmer shader - renders a semi-transparent dark overlay
const DIMMER_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle from vertex index
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.6); // 60% black overlay
}
"#;

/// Create dimmer pipeline for settings menu overlay
pub fn create_dimmer_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dimmer Shader"),
        source: wgpu::ShaderSource::Wgsl(DIMMER_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Dimmer Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Dimmer Pipeline"),
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
