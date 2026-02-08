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

@group(0) @binding(1)
var t_moon: texture_2d<f32>;

@group(0) @binding(2)
var s_moon: sampler;

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

// ============================================================================
// CUBE-MAP BASED SKY UTILITIES (for stable stars and moon)
// ============================================================================

struct CubeFace {
    face_id: u32,
    uv: vec2<f32>,
}

fn get_cube_face(dir: vec3<f32>) -> CubeFace {
    let abs_dir = abs(dir);
    var result: CubeFace;

    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        if (dir.x > 0.0) {
            result.face_id = 0u;
            result.uv = vec2<f32>(-dir.z / dir.x, dir.y / dir.x);
        } else {
            result.face_id = 1u;
            result.uv = vec2<f32>(dir.z / dir.x, dir.y / dir.x);
        }
    } else if (abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z) {
        if (dir.y > 0.0) {
            result.face_id = 2u;
            result.uv = vec2<f32>(dir.x / dir.y, -dir.z / dir.y);
        } else {
            result.face_id = 3u;
            result.uv = vec2<f32>(dir.x / dir.y, dir.z / dir.y);
        }
    } else {
        if (dir.z > 0.0) {
            result.face_id = 4u;
            result.uv = vec2<f32>(dir.x / dir.z, dir.y / dir.z);
        } else {
            result.face_id = 5u;
            result.uv = vec2<f32>(-dir.x / dir.z, dir.y / dir.z);
        }
    }

    result.uv = result.uv * 0.5 + 0.5;
    return result;
}

fn cube_face_to_dir(face_id: u32, uv: vec2<f32>) -> vec3<f32> {
    let mapped_uv = uv * 2.0 - 1.0;

    switch (face_id) {
        case 0u: { return normalize(vec3<f32>( 1.0, mapped_uv.y, -mapped_uv.x)); }
        case 1u: { return normalize(vec3<f32>(-1.0, mapped_uv.y,  mapped_uv.x)); }
        case 2u: { return normalize(vec3<f32>(mapped_uv.x,  1.0, -mapped_uv.y)); }
        case 3u: { return normalize(vec3<f32>(mapped_uv.x, -1.0,  mapped_uv.y)); }
        case 4u: { return normalize(vec3<f32>(mapped_uv.x, mapped_uv.y,  1.0)); }
        default: { return normalize(vec3<f32>(-mapped_uv.x, mapped_uv.y, -1.0)); }
    }
}

fn hash_star(seed: f32) -> f32 {
    return fract(sin(seed * 127.1 + 311.7) * 43758.5453);
}

fn hash_star_2d(seed: f32) -> vec2<f32> {
    return vec2<f32>(
        fract(sin(seed * 127.1 + 311.7) * 43758.5453),
        fract(sin(seed * 269.5 + 183.3) * 43758.5453)
    );
}

// Cube-map based star layer calculation
fn calculate_star_layer(
    view_dir: vec3<f32>,
    face: CubeFace,
    grid_scale: f32,
    threshold: f32,
    star_size: f32,
    intensity: f32,
    time: f32,
    seed_offset: f32
) -> vec3<f32> {
    var result_color = vec3<f32>(0.0);

    let cell_uv = face.uv * grid_scale;
    let cell_x = floor(cell_uv.x);
    let cell_y = floor(cell_uv.y);

    // Check 3x3 neighborhood to prevent boundary popping
    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let check_x = cell_x + f32(dx);
            let check_y = cell_y + f32(dy);

            // Skip out-of-bounds cells
            if (check_x < 0.0 || check_x >= grid_scale ||
                check_y < 0.0 || check_y >= grid_scale) {
                continue;
            }

            let cell_id = f32(face.face_id) * 1000000.0 + check_x * 1000.0 + check_y + seed_offset;
            let star_hash = hash_star(cell_id);

            if (star_hash > threshold) {
                let star_offset = hash_star_2d(cell_id + 0.5);
                let star_uv = (vec2<f32>(check_x, check_y) + star_offset) / grid_scale;
                let star_dir = cube_face_to_dir(face.face_id, star_uv);

                let star_dot = dot(view_dir, star_dir);
                let angle_to_star = acos(clamp(star_dot, -1.0, 1.0));

                if (angle_to_star < star_size) {
                    let star_brightness = 1.0 - angle_to_star / star_size;
                    let brightness_factor = (star_hash - threshold) / (1.0 - threshold);
                    let twinkle = 0.85 + 0.15 * sin(star_hash * 100.0 + time * 2.0);

                    var this_star_color = vec3<f32>(1.0, 1.0, 1.0);
                    let color_hash = hash_star(cell_id + 100.0);
                    if (color_hash < 0.3) {
                        this_star_color = vec3<f32>(1.0, 0.95, 0.85);  // Warm
                    } else if (color_hash < 0.4) {
                        this_star_color = vec3<f32>(0.9, 0.95, 1.0);   // Blue
                    }

                    let height_factor = max(view_dir.y + 0.1, 0.0);
                    result_color += this_star_color * star_brightness * brightness_factor *
                                  intensity * twinkle * height_factor;
                }
            }
        }
    }

    return result_color;
}

// Main cube-map star function
fn calculate_stars_stable(view_dir: vec3<f32>, sun_intensity: f32, time: f32) -> vec3<f32> {
    if (sun_intensity >= 0.3 || view_dir.y <= -0.1) {
        return vec3<f32>(0.0);
    }

    let night_factor = 1.0 - sun_intensity * 3.5;
    var star_color = vec3<f32>(0.0);

    let face = get_cube_face(view_dir);

    // Primary stars - 20x20 grid per face
    star_color += calculate_star_layer(view_dir, face, 20.0, 0.85, 0.003, 0.9 * night_factor, time, 0.0);

    // Secondary dimmer stars - 30x30 grid per face
    star_color += calculate_star_layer(view_dir, face, 30.0, 0.90, 0.002, 0.4 * night_factor, time, 1000000.0);

    return star_color;
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

    // Moon phase determines how much of the moon is illuminated
    // Used to scale halo/glow and affect night sky brightness
    let moon_phase = sky.sky_horizon_color.a;
    let phase_angle_global = moon_phase * 6.28318;
    // lit_fraction: 0 at new moon, 1 at full moon, 0.5 at quarters
    let moon_lit_fraction = (1.0 - cos(phase_angle_global)) * 0.5;

    // Darken night sky when moon provides less light
    // Only applies at night (when sun is down and moon is up)
    let night_factor = (1.0 - sun_intensity) * moon_intensity;
    let moon_sky_dim = mix(0.4, 1.0, moon_lit_fraction); // 40% brightness at new moon
    sky_color *= mix(1.0, moon_sky_dim, night_factor);

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

    // Moon rendering - textured with warm glow (stable UV calculation)
    // Only render when moon is up AND has visible illumination
    if (moon_intensity > 0.05 && moon_lit_fraction > 0.02) {
        // Subtle outer halo - warm atmospheric glow, scaled by how lit the moon is
        let moon_halo = pow(max(moon_dot, 0.0), 8.0) * 0.08 * moon_lit_fraction;
        sky_color += vec3<f32>(0.95, 0.80, 0.55) * moon_halo * moon_intensity;

        // Moon angular radius (in radians, ~0.08 = larger moon)
        let moon_angular_radius = 0.08;

        // Calculate angular distance from view direction to moon center
        let angle_to_moon = acos(clamp(moon_dot, -1.0, 1.0));

        if (angle_to_moon < moon_angular_radius) {
            // Project view_dir onto plane perpendicular to moon_dir
            let offset_vec = view_dir - moon_dir * moon_dot;
            let offset_len = length(offset_vec);

            // === WORLD-SPACE TANGENT FRAME ===
            // Use a fixed world-space reference to avoid any flipping at zenith.
            // "Right" on the moon = East direction (1,0,0) projected onto moon's tangent plane
            // This gives consistent orientation regardless of where moon is in sky.
            let world_east = vec3<f32>(1.0, 0.0, 0.0);

            // Project east onto plane perpendicular to moon_dir
            var moon_right = world_east - moon_dir * dot(world_east, moon_dir);
            let right_len = length(moon_right);

            if (right_len > 0.001) {
                moon_right = moon_right / right_len;
            } else {
                // Moon is directly east or west - use north as fallback
                let world_north = vec3<f32>(0.0, 0.0, -1.0);
                moon_right = world_north - moon_dir * dot(world_north, moon_dir);
                moon_right = normalize(moon_right);
            }

            // moon_up completes the orthonormal frame (perpendicular to both moon_dir and moon_right)
            let moon_up = cross(moon_dir, moon_right);

            var uv: vec2<f32>;
            var disk_x: f32 = 0.0;
            var disk_y: f32 = 0.0;
            if (offset_len < 0.0001) {
                uv = vec2<f32>(0.5, 0.5);
            } else {
                let offset_norm = offset_vec / offset_len;
                let local_x = dot(offset_norm, moon_right);
                let local_y = dot(offset_norm, moon_up);
                // Position on unit disk (-1 to 1)
                disk_x = local_x * offset_len / moon_angular_radius;
                disk_y = local_y * offset_len / moon_angular_radius;
                uv = vec2<f32>(
                    disk_x * 0.5 + 0.5,
                    -disk_y * 0.5 + 0.5
                );
            }

            let moon_tex = textureSample(t_moon, s_moon, uv);

            let edge_softness = 0.01;
            let edge_blend = 1.0 - smoothstep(moon_angular_radius - edge_softness, moon_angular_radius, angle_to_moon);

            // === PHASE ILLUMINATION (in horizon-aligned local coords) ===
            let moon_phase = sky.sky_horizon_color.a;

            // Calculate 3D position on unit sphere for this point on the visible disk
            let r_sq = disk_x * disk_x + disk_y * disk_y;
            let sphere_z = sqrt(max(1.0 - r_sq, 0.0));

            // Phase determines light direction in the horizon-aligned local frame
            // This is now stable because moon_right is always horizontal
            let phase_angle = moon_phase * 6.28318;
            let light_x = sin(phase_angle);   // ±1 at quarters, 0 at new/full
            let light_z = -cos(phase_angle);  // +1 at full (front), -1 at new (behind)

            // Illumination on spherical surface
            // disk_x = horizontal position, sphere_z = depth toward viewer
            let illumination = disk_x * light_x + sphere_z * light_z;

            // Soft terminator for realistic falloff
            let lit_amount = smoothstep(-0.05, 0.15, illumination);

            // Very subtle earthshine for shadow side - mostly invisible
            let earthshine = 0.015;

            if (moon_tex.a > 0.1) {
                let lit_color = moon_tex.rgb * 1.3;
                let shadow_color = moon_tex.rgb * earthshine * vec3<f32>(0.7, 0.8, 1.0);
                let moon_surface = mix(shadow_color, lit_color, lit_amount);

                // Visibility depends on how much of this point is lit
                // Unlit portions fade toward invisible (blend with sky)
                let point_visibility = lit_amount + earthshine;
                let alpha = edge_blend * moon_intensity * moon_tex.a * point_visibility;
                sky_color = mix(sky_color, moon_surface, alpha);
            }
        }
    }

    // Stars at night - using cube-map based grid for stability
    sky_color += calculate_stars_stable(view_dir, sun_intensity, sky.sky_top_color.a);

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

// Voxel volume uniform
struct VoxelVolumeUniform {
    volume_center: vec4<f32>,
    volume_size: vec4<f32>,
    inv_volume_size: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    time_of_day: vec4<f32>,  // x = time, y = shadow_mode (0=hard, 1=soft), z = shadow_debug, w = shadow_blend (0=no shadow, 1=full)
    player_light: vec4<f32>, // xyz = position, w = intensity (0-1, where 1 = 15 emission)
    volume_origin: vec4<i32>, // xyz = world position of voxel [0,0,0] for toroidal wrapping
    player_shadow_box: vec4<f32>, // xyz = feet position, w = half_width (0.3)
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

    // Foliage animation disabled for now
    // if (is_foliage) { ... }

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

// Convert world position to voxel UV (no toroidal wrapping)
// Positions outside the 256-block volume will have UV outside [0,1]
fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    // Compute local position relative to volume center
    let center = voxel_volume.volume_center.xyz;
    let half_size = 128.0;
    let local = world_pos - center + vec3<f32>(half_size);

    // Convert to UV (0-1 range) - NO wrapping
    // Positions outside volume will have UV outside [0,1] and be rejected
    return local / 256.0;
}

// Check if position is within the voxel volume bounds
fn is_in_volume(uv: vec3<f32>) -> bool {
    return all(uv >= vec3<f32>(0.0)) && all(uv <= vec3<f32>(1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Convert world position to voxel integer coordinates
// ═══════════════════════════════════════════════════════════════════════════
fn world_to_voxel_coords(world_pos: vec3<f32>) -> vec3<i32> {
    let center = voxel_volume.volume_center.xyz;
    let local = world_pos - center + vec3<f32>(128.0);
    return vec3<i32>(floor(local));
}

fn voxel_in_bounds(v: vec3<i32>) -> bool {
    return all(v >= vec3<i32>(0)) && all(v < vec3<i32>(256));
}

// ═══════════════════════════════════════════════════════════════════════════
// ANALYTICAL PLAYER SHADOW: Ray-box intersection for smooth player shadow
// No grid snapping — uses exact floating-point player position.
// ═══════════════════════════════════════════════════════════════════════════

fn ray_intersects_player_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> bool {
    // Build player AABB from feet position
    let feet = voxel_volume.player_shadow_box.xyz;
    let half_w = voxel_volume.player_shadow_box.w; // 0.3
    let half_h = 0.9; // half of 1.8 block height

    // Box center is at feet + half_height
    let box_min = vec3<f32>(feet.x - half_w, feet.y, feet.z - half_w);
    let box_max = vec3<f32>(feet.x + half_w, feet.y + half_h * 2.0, feet.z + half_w);

    // Slab method ray-AABB intersection
    let inv_dir = vec3<f32>(1.0) / max(abs(ray_dir), vec3<f32>(1e-8)) * sign(ray_dir);
    let t1 = (box_min - ray_origin) * inv_dir;
    let t2 = (box_max - ray_origin) * inv_dir;

    let t_near = min(t1, t2);
    let t_far = max(t1, t2);

    let t_enter = max(t_near.x, max(t_near.y, t_near.z));
    let t_exit = min(t_far.x, min(t_far.y, t_far.z));

    // Hit if: ray enters before it exits, and exit is in front of ray origin
    // t_enter can be negative (ray starts inside box) — that's fine, still a hit
    return t_enter < t_exit && t_exit > 0.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// CORE DDA: Amanatides & Woo voxel traversal with integer textureLoad
// Steps exactly one voxel boundary per iteration - never skips a voxel.
// Uses textureLoad with integer coordinates for exact per-voxel sampling.
// No interpolation = no banding from partial opacity at voxel boundaries.
// ═══════════════════════════════════════════════════════════════════════════

fn dda_trace_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>, max_steps: i32) -> vec2<f32> {
    // Returns: x = number of solid voxels hit, y = distance to first occluder

    let center = voxel_volume.volume_center.xyz;

    // Start slightly offset along light direction to escape source voxel
    // Keep tight: with textureLoad (integer sampling), we only need to clear
    // the surface boundary, not avoid interpolation bleed
    let start = world_pos + light_dir * 0.3;

    // Local position in volume space
    let local_start = start - center + vec3<f32>(128.0);

    // Current integer voxel position
    var voxel = vec3<i32>(floor(local_start));

    // Step direction: +1 or -1 per axis
    let step_sign = vec3<i32>(sign(light_dir));

    // tDelta: parametric distance to cross one voxel per axis
    let abs_dir = max(abs(light_dir), vec3<f32>(1e-8));
    let t_delta = vec3<f32>(1.0) / abs_dir;

    // tMax: parametric distance to first boundary crossing per axis
    // Branchless: select positive/negative direction per component
    let frac = local_start - vec3<f32>(floor(local_start));
    let dir_pos = vec3<bool>(light_dir.x >= 0.0, light_dir.y >= 0.0, light_dir.z >= 0.0);
    var t_max = select(frac * t_delta, (vec3<f32>(1.0) - frac) * t_delta, dir_pos);

    var solid_hits = 0.0;
    var min_dist_to_occluder = 1000.0;

    for (var i = 0; i < max_steps; i++) {
        // Bounds check
        if (!voxel_in_bounds(voxel)) { break; }

        // Distance from voxel to volume center (cheap: max-component, no sqrt)
        let vd = abs(vec3<f32>(voxel) - vec3<f32>(127.5));
        let center_dist = max(vd.x, max(vd.y, vd.z));
        if (center_dist > 120.0) { break; }

        // Integer textureLoad: exact voxel value, no interpolation
        let voxel_data = textureLoad(t_voxel, voxel, 0);
        let opacity = voxel_data.r;

        // Binary test: solid voxels cast shadow, everything else doesn't
        // opacity is 0.0 (air) or 1.0 (solid) or specific values (leaves=0.75, water=0.25)
        if (opacity > 0.5) {
            solid_hits += 1.0;
            if (min_dist_to_occluder > 999.0) {
                min_dist_to_occluder = min(t_max.x, min(t_max.y, t_max.z));
            }
        }

        // Early exit: one solid voxel is enough for full shadow
        if (solid_hits >= 1.0) { break; }

        // Branchless DDA step: advance along the axis with smallest t_max
        // Uses comparison masks instead of nested if/else to reduce GPU warp divergence.
        // All threads execute the same instructions regardless of which axis steps.
        let x_le_y = t_max.x <= t_max.y;
        let x_le_z = t_max.x <= t_max.z;
        let y_le_z = t_max.y <= t_max.z;

        // Exactly one mask is true: step_x if x is smallest, step_y if y is, step_z otherwise
        let step_x = x_le_y && x_le_z;
        let step_y = !x_le_y && y_le_z;
        let step_z = !step_x && !step_y;

        // Apply step using select: branchless per-component conditional add
        let mask = vec3<bool>(step_x, step_y, step_z);
        voxel += select(vec3<i32>(0), step_sign, mask);
        t_max += select(vec3<f32>(0.0), t_delta, mask);
    }

    return vec2<f32>(solid_hits, min_dist_to_occluder);
}

// ═══════════════════════════════════════════════════════════════════════════
// QUALITY PRESETS: control step count
//   LOW    = 48 steps   (fast, shorter range)
//   MEDIUM = 96 steps   (balanced)
//   HIGH   = 128 steps  (full range)
//   ULTRA  = 128 steps + 2-tap PCF (smoothest)
// ═══════════════════════════════════════════════════════════════════════════
fn get_shadow_max_steps() -> i32 {
    let packed_flags = voxel_volume.volume_origin.w;
    let quality = (packed_flags >> 1) & 7;
    if (quality <= 1) { return 48; }
    if (quality == 2) { return 96; }
    return 128;  // HIGH and ULTRA
}

// ═══════════════════════════════════════════════════════════════════════════
// SHADOW MODE: HARD (DDA with integer sampling - crisp, exact shadows)
// ═══════════════════════════════════════════════════════════════════════════
fn ray_march_shadow_hard(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let max_steps = get_shadow_max_steps();
    let result = dda_trace_shadow(world_pos, light_dir, max_steps);
    let hit_count = result.x;

    // Binary: hit a solid voxel = in shadow, otherwise fully lit
    if (hit_count >= 1.0) {
        return 0.2;  // Shadow darkness
    }
    return 1.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SHADOW MODE: SOFT (DDA core with penumbra from occluder distance)
// ═══════════════════════════════════════════════════════════════════════════
fn ray_march_shadow_soft_core(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let max_steps = get_shadow_max_steps();
    let result = dda_trace_shadow(world_pos, light_dir, max_steps);
    let hit_count = result.x;
    let min_distance_to_occluder = result.y;

    if (hit_count < 1.0) {
        return 1.0;  // No occlusion
    }

    // In shadow - darken based on occluder distance (closer = darker)
    var shadow = 0.2;
    if (min_distance_to_occluder < 999.0) {
        let penumbra_softness = smoothstep(1.0, 15.0, min_distance_to_occluder);
        shadow = mix(0.15, 0.3, penumbra_softness);
    }

    return shadow;
}

// PCF-style multi-tap: offsets the shadow ORIGIN perpendicular to sun direction
// This straddles voxel boundaries across taps, creating anti-aliased shadow edges
fn ray_march_shadow_pcf(world_pos: vec3<f32>, light_dir: vec3<f32>, spread: f32) -> f32 {
    var tangent: vec3<f32>;
    if (abs(light_dir.y) < 0.9) {
        tangent = normalize(cross(light_dir, vec3<f32>(0.0, 1.0, 0.0)));
    } else {
        tangent = normalize(cross(light_dir, vec3<f32>(1.0, 0.0, 0.0)));
    }
    let bitangent = cross(light_dir, tangent);

    // 2-tap PCF: offset origins perpendicular to sun (straddle voxel boundaries)
    let o0 = tangent * (-0.35 * spread) + bitangent * (-0.2 * spread);
    let o1 = tangent * ( 0.35 * spread) + bitangent * ( 0.2 * spread);

    let max_steps = get_shadow_max_steps();
    let s0 = dda_trace_shadow(world_pos + o0, light_dir, max_steps);
    let s1 = dda_trace_shadow(world_pos + o1, light_dir, max_steps);

    // Average the binary results: 0, 0.5, or 1.0 hits
    let total_hits = s0.x + s1.x;

    if (total_hits >= 2.0) {
        return 0.2;   // Both taps hit = full shadow
    } else if (total_hits >= 1.0) {
        return 0.6;   // One tap hit = partial shadow (edge)
    }
    return 1.0;       // No hits = fully lit
}

// Soft multi-tap: offsets origins for penumbra effect
fn ray_march_shadow_soft_multitap(world_pos: vec3<f32>, light_dir: vec3<f32>, spread: f32) -> f32 {
    var tangent: vec3<f32>;
    if (abs(light_dir.y) < 0.9) {
        tangent = normalize(cross(light_dir, vec3<f32>(0.0, 1.0, 0.0)));
    } else {
        tangent = normalize(cross(light_dir, vec3<f32>(1.0, 0.0, 0.0)));
    }
    let bitangent = cross(light_dir, tangent);

    let o0 = tangent * (-0.4 * spread) + bitangent * (-0.25 * spread);
    let o1 = tangent * ( 0.4 * spread) + bitangent * ( 0.25 * spread);

    let s0 = ray_march_shadow_soft_core(world_pos + o0, light_dir);
    let s1 = ray_march_shadow_soft_core(world_pos + o1, light_dir);

    return (s0 + s1) * 0.5;
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED SHADOW FUNCTION - picks mode based on quality + shadow mode
// ═══════════════════════════════════════════════════════════════════════════
fn ray_march_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>, shadow_mode: f32) -> f32 {
    // Check analytical player shadow first (smooth, no grid snapping)
    if (voxel_volume.player_shadow_box.w > 0.0) {
        if (ray_intersects_player_box(world_pos, light_dir)) {
            return 0.2; // Player shadow
        }
    }

    let packed_flags = voxel_volume.volume_origin.w;
    let quality = (packed_flags >> 1) & 7;

    if (shadow_mode > 0.5) {
        // Soft mode
        if (quality >= 4) {
            return ray_march_shadow_soft_multitap(world_pos, light_dir, 1.0);
        }
        return ray_march_shadow_soft_core(world_pos, light_dir);
    }

    // Hard mode
    if (quality >= 4) {
        // ULTRA: 2-tap PCF for anti-aliased hard edges
        return ray_march_shadow_pcf(world_pos, light_dir, 0.5);
    }

    return ray_march_shadow_hard(world_pos, light_dir);
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
// VOXEL AMBIENT OCCLUSION — DDA Hemisphere Tracing
// Uses Amanatides & Woo DDA through the 256^3 voxel volume with textureLoad
// (integer sampling, no interpolation artifacts). Fixed cosine-weighted
// hemisphere directions ensure deterministic, camera-independent results.
// Quality presets: OFF(0), LOW(1)=3 dirs, MEDIUM(2)=5, HIGH(3)=7, ULTRA(4)=9
// ═══════════════════════════════════════════════════════════════════════════

fn get_ao_quality() -> i32 {
    let packed_flags = voxel_volume.volume_origin.w;
    return (packed_flags >> 4) & 7;
}

// DDA trace a single AO ray: returns occlusion amount (0=clear, 1=fully occluded)
fn dda_trace_ao(start_pos: vec3<f32>, direction: vec3<f32>, max_steps: i32) -> f32 {
    let center = voxel_volume.volume_center.xyz;

    let local_start = start_pos - center + vec3<f32>(128.0);
    var voxel = vec3<i32>(floor(local_start));

    if (!voxel_in_bounds(voxel)) { return 0.0; }

    let step_sign = vec3<i32>(sign(direction));
    let abs_dir = max(abs(direction), vec3<f32>(1e-8));
    let t_delta = vec3<f32>(1.0) / abs_dir;

    let frac = local_start - vec3<f32>(floor(local_start));
    let dir_pos = vec3<bool>(direction.x >= 0.0, direction.y >= 0.0, direction.z >= 0.0);
    var t_max = select(frac * t_delta, (vec3<f32>(1.0) - frac) * t_delta, dir_pos);

    // Falloff range: occlusion drops to zero at this distance (in blocks).
    // Decoupled from step count — steps control search range, falloff controls shadow width.
    // AO should darken within 1-3 blocks of geometry, not 5-10.
    let falloff_range = 4.0;
    var occlusion = 0.0;

    for (var i = 0; i < max_steps; i++) {
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

        if (!voxel_in_bounds(voxel)) { break; }

        let voxel_data = textureLoad(t_voxel, voxel, 0);

        if (voxel_data.r > 0.5) {
            let dist = min(t_max.x, min(t_max.y, t_max.z));
            // Linear falloff: smooth gradient, squared for natural decay
            let falloff = max(0.0, 1.0 - dist / falloff_range);
            // DDA traverses in distance order — first hit is closest, dominates max().
            // No later hit can exceed this value, so break immediately.
            return falloff * falloff;
        }
    }

    return 0.0;
}

// Stable world-space hash for AO spatial jitter
fn ao_hash31(p: vec3<f32>) -> f32 {
    let q = fract(p * 127.1);
    return fract(dot(q, q.yzx + 31.17));
}
fn ao_hash32(p: vec3<f32>) -> vec2<f32> {
    let q = fract(p * vec3<f32>(127.1, 311.7, 74.7));
    let d = dot(q, q.yzx + 33.33);
    return fract(vec2<f32>(d * 1.37, d * 2.79));
}

fn trace_voxel_ao(world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let quality = get_ao_quality();
    if (quality <= 0) { return 1.0; }

    // Build tangent frame
    var tangent: vec3<f32>;
    if (abs(normal.y) > 0.9) {
        tangent = normalize(cross(normal, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        tangent = normalize(cross(normal, vec3<f32>(0.0, 1.0, 0.0)));
    }
    let bitangent = cross(normal, tangent);

    // Spatial jitter: shift sample position in the surface plane.
    // World-space hash = stable (no flicker), breaks up voxel-aligned shadow edges.
    let noise = ao_hash32(world_pos);
    let jitter = tangent * (noise.x - 0.5) * 0.25 + bitangent * (noise.y - 0.5) * 0.25;

    // Trace start: barely above surface (0.1 blocks along normal).
    // Block surfaces sit on integer coordinates, so 0.1 safely clears the surface
    // while staying close enough that angled rays hit adjacent walls correctly.
    let trace_origin = world_pos + jitter + normal * 0.1;

    // Step count per quality — with first-hit early break, most rays terminate
    // in 1-3 steps near geometry. Steps only matter in open areas (miss case).
    // Falloff at step N: (1-N/max)^2. At step 8 of 10: 0.04 — negligible.
    var max_steps: i32;
    if (quality <= 1) { max_steps = 6; }
    else if (quality == 2) { max_steps = 8; }
    else if (quality == 3) { max_steps = 8; }
    else { max_steps = 10; }

    var total_occlusion = 0.0;
    var total_weight = 0.0;

    // === Direction 1: Normal (straight out from surface) ===
    // Detects ceilings, overhangs directly above
    let w_center = 0.20;
    total_occlusion += dda_trace_ao(trace_origin, normal, max_steps) * w_center;
    total_weight += w_center;

    // === Directions 2-3: Tangent pair at 40° tilt from normal ===
    // LOW+ — mid-hemisphere, catches nearby walls and overhangs
    let tilt40_n = 0.766;  // cos(40°)
    let tilt40_s = 0.643;  // sin(40°)
    let w_mid = 0.18;

    let dir_t_pos = normalize(normal * tilt40_n + tangent * tilt40_s);
    let dir_t_neg = normalize(normal * tilt40_n - tangent * tilt40_s);
    total_occlusion += dda_trace_ao(trace_origin, dir_t_pos, max_steps) * w_mid;
    total_occlusion += dda_trace_ao(trace_origin, dir_t_neg, max_steps) * w_mid;
    total_weight += w_mid * 2.0;

    // === Directions 4-5: Bitangent pair at 40° tilt ===
    // MEDIUM+ — same tier, orthogonal axis
    if (quality >= 2) {
        let dir_b_pos = normalize(normal * tilt40_n + bitangent * tilt40_s);
        let dir_b_neg = normalize(normal * tilt40_n - bitangent * tilt40_s);
        total_occlusion += dda_trace_ao(trace_origin, dir_b_pos, max_steps) * w_mid;
        total_occlusion += dda_trace_ao(trace_origin, dir_b_neg, max_steps) * w_mid;
        total_weight += w_mid * 2.0;
    }

    // === Directions 6-7: Grazing tangent pair at 72° tilt ===
    // HIGH+ — near-surface rays that reliably detect adjacent walls at distance.
    // These stay close to the surface plane and catch geometry the steeper rays miss.
    if (quality >= 3) {
        let tilt72_n = 0.309;  // cos(72°) — small upward component
        let tilt72_s = 0.951;  // sin(72°) — mostly sideways
        let w_graze = 0.16;

        let dir_g1 = normalize(normal * tilt72_n + tangent * tilt72_s);
        let dir_g2 = normalize(normal * tilt72_n - tangent * tilt72_s);
        total_occlusion += dda_trace_ao(trace_origin, dir_g1, max_steps) * w_graze;
        total_occlusion += dda_trace_ao(trace_origin, dir_g2, max_steps) * w_graze;
        total_weight += w_graze * 2.0;
    }

    // === Directions 8-9: Grazing bitangent pair at 72° tilt ===
    // ULTRA — complete near-surface coverage
    if (quality >= 4) {
        let tilt72_n = 0.309;
        let tilt72_s = 0.951;
        let w_graze = 0.16;

        let dir_g3 = normalize(normal * tilt72_n + bitangent * tilt72_s);
        let dir_g4 = normalize(normal * tilt72_n - bitangent * tilt72_s);
        total_occlusion += dda_trace_ao(trace_origin, dir_g3, max_steps) * w_graze;
        total_occlusion += dda_trace_ao(trace_origin, dir_g4, max_steps) * w_graze;
        total_weight += w_graze * 2.0;
    }

    // Normalize — each ray returns 0-1 so weighted average is 0-1
    let ao = total_occlusion / total_weight;

    // Power curve: < 1.0 strengthens near-crease darkening, preserves smooth gradient
    let ao_shaped = pow(saturate(ao), 0.65);

    return 1.0 - ao_shaped;
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

// === TORCH FLICKER UTILITIES ===
// Hash function for per-position variation (returns 0-1)
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Multi-octave smooth noise for natural torch flicker
// torch_pos should be the actual torch block position (or estimate)
// Returns a multiplier around 1.0 (range: ~0.88 to ~1.12)
fn torch_flicker_at_source(torch_pos: vec3<f32>, time: f32) -> f32 {
    // Per-torch phase offset based on torch position (so each torch flickers differently)
    let phase_offset = hash31(torch_pos) * 6.28318;

    // Multiple frequency components for natural look:
    // - Slow breathing (like fire expanding/contracting)
    // - Medium flicker (like flame movement)
    // - Fast shimmer (like heat distortion)
    let slow_wave = sin(time * 2.5 + phase_offset) * 0.06;
    let med_wave = sin(time * 7.3 + phase_offset * 1.3) * 0.04;
    let fast_wave = sin(time * 15.7 + phase_offset * 2.1) * 0.02;

    // Combine: base 1.0 + variations (range ~0.88 to ~1.12)
    return 1.0 + slow_wave + med_wave + fast_wave;
}

// Simple time-based flicker for when we don't know torch position
// Provides subtle ambient variation without position-based artifacts
fn torch_flicker_simple(time: f32) -> f32 {
    let slow_wave = sin(time * 2.5) * 0.04;
    let med_wave = sin(time * 7.3) * 0.03;
    return 1.0 + slow_wave + med_wave;
}

// Get torch color (warm orange, constant - flicker applied to intensity only)
fn get_torch_color() -> vec3<f32> {
    return vec3<f32>(1.0, 0.85, 0.6);  // Warm orange
}

// Calculate light from player's held torch (with shadow casting and flicker)
fn calculate_held_light(world_pos: vec3<f32>) -> vec3<f32> {
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

    // Get animation time and calculate flicker (based on player position)
    let time = camera.anim_data.x;
    let flicker = torch_flicker_at_source(floor(player_pos), time);
    let torch_color = get_torch_color();

    // Quartic falloff: smaller bright core, dims quickly, long subtle tail to edge
    let falloff = 1.0 - (dist / max_radius);
    let light_strength = pow(falloff, 4.0) * intensity * flicker;

    // Apply shadow - light blocked by geometry between surface and player
    return torch_color * light_strength * shadow * 2.5;
}

// Trace shadow toward estimated torch light source
// Uses light gradient to find direction toward the brightest source
fn trace_torch_shadow(world_pos: vec3<f32>, max_dist: f32) -> f32 {
    // Sample light gradient to find direction toward light source
    let step = 2.0;
    let uv = world_to_voxel_uv(world_pos);
    let inv_size = 1.0 / 256.0;

    // Sample neighbors to compute gradient
    let sample_x_pos = textureSampleLevel(t_voxel, s_voxel, uv + vec3<f32>(step * inv_size, 0.0, 0.0), 0.0).a;
    let sample_x_neg = textureSampleLevel(t_voxel, s_voxel, uv - vec3<f32>(step * inv_size, 0.0, 0.0), 0.0).a;
    let sample_y_pos = textureSampleLevel(t_voxel, s_voxel, uv + vec3<f32>(0.0, step * inv_size, 0.0), 0.0).a;
    let sample_y_neg = textureSampleLevel(t_voxel, s_voxel, uv - vec3<f32>(0.0, step * inv_size, 0.0), 0.0).a;
    let sample_z_pos = textureSampleLevel(t_voxel, s_voxel, uv + vec3<f32>(0.0, 0.0, step * inv_size), 0.0).a;
    let sample_z_neg = textureSampleLevel(t_voxel, s_voxel, uv - vec3<f32>(0.0, 0.0, step * inv_size), 0.0).a;

    // Gradient points toward higher light (toward source)
    let gradient = vec3<f32>(
        sample_x_pos - sample_x_neg,
        sample_y_pos - sample_y_neg,
        sample_z_pos - sample_z_neg
    );

    let grad_len = length(gradient);
    if (grad_len < 0.01) {
        return 1.0;  // No clear gradient, assume no shadow
    }

    let light_dir = normalize(gradient);

    // Short ray march toward light source
    var accumulated_opacity = 0.0;
    var pos = world_pos + light_dir * 0.5;  // Start slightly offset
    let step_size = 1.0;
    let max_steps = i32(min(max_dist / step_size, 12.0));  // Limit for performance

    for (var i = 0; i < max_steps; i++) {
        let sample_uv = world_to_voxel_uv(pos);
        if (!is_in_volume(sample_uv)) {
            break;
        }

        let sample_voxel = textureSampleLevel(t_voxel, s_voxel, sample_uv, 0.0);

        // If we hit an emissive voxel, we reached the source
        if (sample_voxel.g > 0.5) {
            break;
        }

        // Accumulate opacity from solid blocks
        if (sample_voxel.r > 0.5) {
            accumulated_opacity += 0.25;
        } else if (sample_voxel.r > 0.2) {
            accumulated_opacity += sample_voxel.r * 0.15;
        }

        if (accumulated_opacity >= 1.0) {
            return 0.0;  // Fully blocked
        }

        pos += light_dir * step_size;
    }

    return 1.0 - clamp(accumulated_opacity, 0.0, 1.0);
}

fn sample_torch_light(world_pos: vec3<f32>) -> vec3<f32> {
    // Sample pre-propagated light from the voxel volume
    // Light is propagated via GPU compute shader and stored in the A channel

    let uv = world_to_voxel_uv(world_pos);
    if (!is_in_volume(uv)) {
        return vec3<f32>(0.0);
    }

    // Sample the voxel - A channel contains propagated light intensity
    // Linear filtering provides smooth interpolation between voxels
    let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
    let light_intensity = voxel.a;  // A channel = propagated light (0-1)

    if (light_intensity < 0.01) {
        return vec3<f32>(0.0);
    }

    // Estimate torch position by following light gradient
    // This ensures all surfaces lit by the same torch flicker together
    let inv_size = 1.0 / 256.0;
    let step = 2.0;
    let sample_x_pos = textureSampleLevel(t_voxel, s_voxel, uv + vec3<f32>(step * inv_size, 0.0, 0.0), 0.0).a;
    let sample_x_neg = textureSampleLevel(t_voxel, s_voxel, uv - vec3<f32>(step * inv_size, 0.0, 0.0), 0.0).a;
    let sample_y_pos = textureSampleLevel(t_voxel, s_voxel, uv + vec3<f32>(0.0, step * inv_size, 0.0), 0.0).a;
    let sample_y_neg = textureSampleLevel(t_voxel, s_voxel, uv - vec3<f32>(0.0, step * inv_size, 0.0), 0.0).a;
    let sample_z_pos = textureSampleLevel(t_voxel, s_voxel, uv + vec3<f32>(0.0, 0.0, step * inv_size), 0.0).a;
    let sample_z_neg = textureSampleLevel(t_voxel, s_voxel, uv - vec3<f32>(0.0, 0.0, step * inv_size), 0.0).a;

    let gradient = vec3<f32>(
        sample_x_pos - sample_x_neg,
        sample_y_pos - sample_y_neg,
        sample_z_pos - sample_z_neg
    );

    // Estimate torch position: current position + normalized gradient * estimated distance
    // Distance estimate: light_intensity maps to ~14 blocks radius, so inverse
    let estimated_dist = (1.0 - light_intensity) * 14.0;
    var estimated_torch_pos = world_pos;
    let grad_len = length(gradient);
    if (grad_len > 0.01) {
        estimated_torch_pos = world_pos + normalize(gradient) * estimated_dist;
    }

    // Check if torch shadows are enabled (stored in volume_origin.w)
    let torch_shadows_enabled = (voxel_volume.volume_origin.w & 1) > 0;
    var shadow_factor = 1.0;
    if (torch_shadows_enabled && grad_len > 0.01) {
        // Trace shadow toward light source
        shadow_factor = trace_torch_shadow(world_pos, estimated_dist);
    }

    // Get animation time and calculate flicker based on estimated torch position
    // This ensures all surfaces lit by the same torch flicker together
    let time = camera.anim_data.x;
    let flicker = torch_flicker_at_source(floor(estimated_torch_pos), time);
    let torch_color = get_torch_color();

    // Apply softer gamma curve for longer falloff trail, modulated by flicker and shadow
    let curved_light = pow(light_intensity, 1.2) * flicker * shadow_factor;

    // Brighter placed torches
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

    // Unpack shadow quality from volume_origin.w: bit 0 = torch_shadows, bits 1-3 = quality
    let packed_flags = voxel_volume.volume_origin.w;
    let shadow_quality = (packed_flags >> 1) & 7;  // 0=OFF, 1=LOW, 2=MEDIUM, 3=HIGH, 4=ULTRA

    var shadow = 1.0;  // Default: fully lit
    var was_traced = 0.0;  // Track if we traced or not

    // Skip shadow tracing if shadows are disabled (quality = OFF)
    // Also skip when sun is too low for shadows to matter:
    // shadow_fade = smoothstep(0.0, 0.3, sun_intensity) drives final_shadow → 1.0
    // when sun_intensity < ~0.02, so DDA result is completely unused.
    if (shadow_quality > 0 && sun_intensity > 0.02) {
        was_traced = 1.0;

        // Shadow origin offset strategy:
        // Sun-facing faces need offset along normal to prevent self-shadow acne.
        // Non-sun-facing faces need MINIMAL offset — large offsets push the ray
        // past adjacent occluders, creating false "lit" strips near edges.
        //
        // Note: dda_trace_shadow() adds its own light_dir * 0.3 escape offset.
        // For sun-facing faces: total offset ~0.4 along sun_dir (prevents acne).
        // For non-sun-facing faces: we SUBTRACT to nearly cancel the DDA's 0.3,
        // keeping total ~0.01 so the ray starts flush with the surface.
        let normal_offset = mix(0.05, 0.3, max(ndotl, 0.0));
        let sun_offset = mix(-0.29, 0.1, max(ndotl, 0.0));
        let shadow_origin = in.world_position + in.normal * normal_offset + sun_dir * sun_offset;
        shadow = ray_march_shadow(shadow_origin, sun_dir, shadow_mode);
    }

    // Edge-distance based shadow fade
    // Soften shadows near volume boundaries (where new data appears)
    // This creates a smooth transition at the edges without affecting the center
    let volume_center = voxel_volume.volume_center.xyz;
    let half_size = 128.0;  // Volume is 256, half is 128
    let edge_fade_distance = 2.0;  // Fade within 2 blocks of edge

    // Distance from position to volume center on each axis
    let delta = abs(in.world_position - volume_center);
    // Distance to nearest edge (minimum across all axes)
    let dist_to_edge = half_size - max(delta.x, max(delta.y, delta.z));
    // Fade factor: 0 at edge, 1 at edge_fade_distance or more inside
    let edge_blend = saturate(dist_to_edge / edge_fade_distance);
    // Apply edge fade - shadows soften near edges
    shadow = mix(1.0, shadow, edge_blend);

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

    // Sky occlusion - check if we're in an enclosed space (cave)
    // Must be calculated early as it affects moon, ambient, and torch lighting
    let sky_access = trace_sky_occlusion(in.world_position);

    // Direct lighting
    let sun_ndotl = max(dot(in.normal, sun_dir), 0.0);
    let sun_contribution = sun_color * sun_ndotl * 0.7 * sun_intensity * final_shadow;

    // Enhanced moon lighting - soft ambient fill + directional
    let moon_ndotl = max(dot(in.normal, moon_dir), 0.0);
    // Moon provides both directional light and soft ambient fill
    let moon_direct = moon_ndotl * 0.2;
    let moon_ambient = 0.15;  // Soft fill from all directions (scattered moonlight)
    // Moon only illuminates areas with sky access
    let moon_contribution = moon_color * (moon_direct + moon_ambient) * moon_intensity * sky_access;

    // Voxel Cone Traced GI - disabled for now, too expensive
    // TODO: Re-enable once shadows are stable
    let gi = vec3<f32>(0.0);

    // Per-vertex AO — direct contact shadows at block corners.
    // AO-aware mesh splitting prevents gradient smearing across large quads,
    // so we can use the baked values directly for strong crease darkening.
    // Table: [1.0, 0.85, 0.65, 0.40] — 3 occluders = 60% dark.
    let vertex_ao = in.ao;

    // Voxel AO - smooth world-space occlusion from nearby geometry.
    // Skip the expensive trace if vertex AO says no blocks are nearby (ao=1.0).
    // This saves 100% of trace cost on open flat terrain (majority of fragments).
    var voxel_ao = 1.0;
    if (vertex_ao < 0.99) {
        let voxel_ao_raw = trace_voxel_ao(in.world_position, in.normal);
        // Mapping: 0.30 in deep corners/caves, 1.0 in open areas
        voxel_ao = mix(0.30, 1.0, voxel_ao_raw);
    }

    // Combined AO — both layers multiply for corner darkness.
    // Reduce AO strength in shadowed areas to prevent double-darkening.
    // In full shadow (final_shadow≈0), AO effect is halved.
    // In full light (final_shadow=1), AO applies at full strength.
    let raw_ao = vertex_ao * voxel_ao;
    let ao_strength = mix(0.5, 1.0, final_shadow);
    let combined_ao = mix(1.0, raw_ao, ao_strength);

    // Block light (torches) - now using voxel-based sampling
    // Sample emissive light from nearby voxels in the volume
    let voxel_torch_light = sample_torch_light(in.world_position);

    // Add player's held torch light (dynamic, follows player)
    let held_torch_light = calculate_held_light(in.world_position);

    // Context-aware torch lighting:
    // - In caves (low sky_access): torches are PRIMARY light source, boost significantly
    // - On surface at night: torches complement moonlight, moderate boost
    // - During day: torches barely visible (sun overpowers)
    let night_factor = 1.0 - sun_intensity;
    let cave_boost = mix(1.5, 3.0, 1.0 - sky_access);  // 1.5x surface, 3x in caves
    let time_boost = 1.0 + night_factor * 0.8;          // Up to 1.8x at night
    let torch_boost = cave_boost * time_boost;
    let block_light_contribution = (voxel_torch_light + held_torch_light) * torch_boost;

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
        let render_dist = voxel_volume.volume_center.w;
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

/// Crack overlay vertex (position + UV for textured crack faces)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrackVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl CrackVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CrackVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// WGSL shader for crack overlay rendering
const CRACK_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var crack_texture: texture_2d<f32>;

@group(1) @binding(1)
var crack_sampler: sampler;

struct CrackUniform {
    stage: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(2)
var<uniform> crack_data: CrackUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample from the correct stage column in the 4x1 atlas (256x64 = 4 stages x 64x64)
    let stage_offset = f32(crack_data.stage) * 0.25;
    let atlas_uv = vec2<f32>(in.uv.x * 0.25 + stage_offset, in.uv.y);
    let crack_color = textureSample(crack_texture, crack_sampler, atlas_uv);
    // Dark cracks with the texture's alpha
    return vec4<f32>(0.0, 0.0, 0.0, crack_color.a);
}
"#;

/// Create pipeline for crack overlay rendering
pub fn create_crack_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    crack_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Crack Shader"),
        source: wgpu::ShaderSource::Wgsl(CRACK_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Crack Pipeline Layout"),
        bind_group_layouts: &[camera_bind_group_layout, crack_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Crack Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[CrackVertex::desc()],
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
            cull_mode: None, // Render both sides for crack overlay
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -3, // Stronger bias than outline to render on top of block faces
                slope_scale: 0.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

// Note: Water shader moved to src/renderer/water.rs (ray-marched water system)

// ═══════════════════════════════════════════════════════════════════════════
// LEGACY WATER CODE REMOVED
// The old mesh-based water shader and pipeline have been replaced by:
// - src/renderer/water.rs: Ray-marched infinite water plane
// - All water rendering now uses fullscreen ray-plane intersection
// This eliminates chunk boundary seams and camera jitter
// ═══════════════════════════════════════════════════════════════════════════

// Note: Old water mesh code was here - replaced by ray-marched water in water.rs

// Old water code removed from here - see water.rs for the new ray-marched implementation

// The following structs/functions were in the old water system and are now removed:
// - WATER_SHADER (old mesh-based water shader)
// - WaterUniform (uniform buffer for old water system)
// - create_water_bind_group_layout
// - create_water_pipeline

// ═══════════════════════════════════════════════════════════════════════════
// LEGACY WATER SHADER CONTENT REMOVED
// All water rendering now uses src/renderer/water.rs
// ═══════════════════════════════════════════════════════════════════════════

// The content that was here (old WATER_SHADER and related code) has been
// replaced by the ray-marched water system in water.rs

// ═══════════════════════════════════════════════════════════════════════════
// DIMMER SHADER - Fullscreen overlay for settings menu
// ═══════════════════════════════════════════════════════════════════════════

// [LEGACY WATER CODE REMOVED]
// The mesh-based water shader, WaterUniform struct, create_water_bind_group_layout,
// and create_water_pipeline functions have been replaced by the ray-marched water
// system in src/renderer/water.rs

// ═══════════════════════════════════════════════════════════════════════════
// DIMMER SHADER - Fullscreen overlay for settings menu
// ═══════════════════════════════════════════════════════════════════════════

/// Menu blur shader - applies subtle Gaussian blur and darkening for menu overlay
const MENU_BLUR_SHADER: &str = r#"
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle from vertex index
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Convert from NDC to UV (flip Y)
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_tex));
    let texel = 1.0 / tex_size;

    // Subtle 9-tap Gaussian blur (sigma ~1.5)
    // Weights: center=0.25, adjacent=0.125, diagonal=0.0625
    var color = vec3<f32>(0.0);

    // Center
    color += textureSample(scene_tex, scene_sampler, in.uv).rgb * 0.25;

    // Adjacent (up, down, left, right)
    let blur_radius = 2.0; // Subtle blur
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(0.0, -texel.y * blur_radius)).rgb * 0.125;
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(0.0, texel.y * blur_radius)).rgb * 0.125;
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(-texel.x * blur_radius, 0.0)).rgb * 0.125;
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(texel.x * blur_radius, 0.0)).rgb * 0.125;

    // Diagonals
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(-texel.x * blur_radius, -texel.y * blur_radius)).rgb * 0.0625;
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(texel.x * blur_radius, -texel.y * blur_radius)).rgb * 0.0625;
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(-texel.x * blur_radius, texel.y * blur_radius)).rgb * 0.0625;
    color += textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(texel.x * blur_radius, texel.y * blur_radius)).rgb * 0.0625;

    // Darken the blurred result
    let darken = 0.5; // 50% darker
    color = color * darken;

    return vec4<f32>(color, 1.0);
}
"#;

/// Create bind group layout for menu blur (scene texture + sampler)
pub fn create_menu_blur_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Menu Blur Bind Group Layout"),
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
    })
}

/// Create bind group for menu blur
pub fn create_menu_blur_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Menu Blur Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Create a pipeline for menu blur overlay
pub fn create_menu_blur_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Menu Blur Shader"),
        source: wgpu::ShaderSource::Wgsl(MENU_BLUR_SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Menu Blur Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Menu Blur Pipeline"),
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
                format,
                blend: None, // No blending - fully replace
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None, // No depth test for post-process
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

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
    // Darker overlay - 60% opacity black
    return vec4<f32>(0.0, 0.0, 0.0, 0.6);
}
"#;

/// Create a pipeline for dimmer overlay (settings menu background) - fallback
pub fn create_dimmer_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dimmer Shader"),
        source: wgpu::ShaderSource::Wgsl(DIMMER_SHADER.into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Dimmer Pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
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
        primitive: wgpu::PrimitiveState::default(),
        // Must match the render pass depth format (UI pass has depth attachment)
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

// End of pipeline.rs - Legacy water code has been removed and replaced by water.rs
