// Procedural Grass Rendering Shader
// Tile-based placement with vertex-pulling blade generation
// Includes voxel-based shadow sampling for terrain occlusion

// ============================================================================
// STRUCTURES
// ============================================================================

struct GrassUniforms {
    view_proj: mat4x4<f32>,
    wind: vec4<f32>,           // xy = direction, z = strength, w = time
    camera_pos: vec4<f32>,     // xyz = position
    lod_distances: vec4<f32>,  // x = full_detail, y = low_lod, z = cull, w = chunk_size
    quality: vec4<u32>,        // x = blades_per_tile, y = blade_segments, z = max_tiles, w = min_blade_threshold
    sun_direction: vec4<f32>,  // xyz = direction, w = intensity
    sun_color: vec4<f32>,      // rgb = color, a = unused
    ambient_color: vec4<f32>,  // rgb = ambient, a = shadow_mode (0=off, 1=simple)
}

struct GrassTile {
    position: vec2<f32>,       // World XZ position
    density_seed: u32,         // blade_count | height_var | seed
    biome_height: u32,         // biome_id | surface_y (fixed point)
}

struct BiomeColor {
    base: vec4<f32>,
    tip: vec4<f32>,
}

// Voxel GI uniform (matches VoxelVolumeUniform in voxel_gi.rs)
struct VoxelVolumeUniform {
    volume_center: vec4<f32>,
    volume_size: vec4<f32>,
    inv_volume_size: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    time_of_day: vec4<f32>,    // x = time, y = shadow_mode, z = shadow_debug, w = shadow_blend
    player_light: vec4<f32>,   // xyz = position, w = intensity
    volume_origin: vec4<i32>,  // xyz = world position of voxel [0,0,0]
}

// ============================================================================
// BINDINGS
// ============================================================================

// Grass-specific bindings
@group(0) @binding(0) var<uniform> uniforms: GrassUniforms;
@group(0) @binding(1) var<storage, read> tiles: array<GrassTile>;
@group(0) @binding(2) var<uniform> biome_colors: array<BiomeColor, 8>;

// Voxel GI bindings (shared with terrain for shadow sampling)
@group(1) @binding(0) var<uniform> voxel_volume: VoxelVolumeUniform;
@group(1) @binding(1) var t_voxel: texture_3d<f32>;
@group(1) @binding(2) var t_radiance: texture_3d<f32>;
@group(1) @binding(3) var s_voxel: sampler;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Unpack blade count from density_seed
fn unpack_blade_count(density_seed: u32) -> u32 {
    return density_seed & 0xFFu;
}

// Unpack height variation from density_seed (0-63 mapped to 0-1)
fn unpack_height_var(density_seed: u32) -> f32 {
    return f32((density_seed >> 8u) & 0x3Fu) / 63.0;
}

// Unpack seed from density_seed
fn unpack_seed(density_seed: u32) -> u32 {
    return density_seed >> 16u;
}

// Unpack biome ID from biome_height
fn unpack_biome(biome_height: u32) -> u32 {
    return biome_height & 0xFFu;
}

// Unpack surface Y from biome_height (8.16 fixed point)
fn unpack_surface_y(biome_height: u32) -> f32 {
    return f32((biome_height >> 8u) & 0xFFFFFFu) / 256.0;
}

// Hash function for deterministic randomness
fn hash(n: u32) -> u32 {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

// Hash to float [0, 1]
fn hash_float(seed: u32) -> f32 {
    return f32(hash(seed) & 0xFFFFu) / 65535.0;
}

// Hash to float [-1, 1]
fn hash_float_signed(seed: u32) -> f32 {
    return hash_float(seed) * 2.0 - 1.0;
}

// Quadratic Bezier curve evaluation
fn bezier(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, t: f32) -> vec3<f32> {
    let one_minus_t = 1.0 - t;
    return one_minus_t * one_minus_t * p0 + 2.0 * one_minus_t * t * p1 + t * t * p2;
}

// Rotate point around Y axis
fn rotate_y(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(
        p.x * c - p.z * s,
        p.y,
        p.x * s + p.z * c
    );
}

// ============================================================================
// VOXEL SHADOW SAMPLING
// ============================================================================

// Convert world position to voxel UV (no toroidal wrapping)
fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    let center = voxel_volume.volume_center.xyz;
    let half_size = 128.0;
    let local = world_pos - center + vec3<f32>(half_size);

    // Convert to UV (0-1 range) - no wrapping
    return local / 256.0;
}

fn voxel_in_bounds(v: vec3<i32>) -> bool {
    return all(v >= vec3<i32>(0)) && all(v < vec3<i32>(256));
}

// Sample shadow using DDA with bilinear filtering for smooth edges
// Returns 0.0 = fully in shadow, 1.0 = fully lit
fn sample_grass_shadow(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let max_steps = 48;
    let center = voxel_volume.volume_center.xyz;

    let start = world_pos + sun_dir * 0.55;
    let center_dist = length(start - center);
    if (center_dist > 120.0) {
        return 1.0;
    }

    let local_start = start - center + vec3<f32>(128.0);
    var voxel = vec3<i32>(floor(local_start));
    let step_sign = vec3<i32>(sign(sun_dir));
    let abs_dir = max(abs(sun_dir), vec3<f32>(1e-8));
    let t_delta = vec3<f32>(1.0) / abs_dir;
    let frac = local_start - vec3<f32>(floor(local_start));
    var t_max: vec3<f32>;
    if (sun_dir.x >= 0.0) { t_max.x = (1.0 - frac.x) * t_delta.x; } else { t_max.x = frac.x * t_delta.x; }
    if (sun_dir.y >= 0.0) { t_max.y = (1.0 - frac.y) * t_delta.y; } else { t_max.y = frac.y * t_delta.y; }
    if (sun_dir.z >= 0.0) { t_max.z = (1.0 - frac.z) * t_delta.z; } else { t_max.z = frac.z * t_delta.z; }

    var accumulated_opacity = 0.0;
    var t_entry = 0.0;

    for (var i = 0; i < max_steps; i++) {
        if (!voxel_in_bounds(voxel)) { break; }

        let voxel_world = vec3<f32>(voxel) - vec3<f32>(128.0) + vec3<f32>(0.5) + center;
        let dist = length(voxel_world - center);
        if (dist > 120.0) { break; }

        // Bilinear sampling at ray midpoint for smooth edges
        let t_exit = min(t_max.x, min(t_max.y, t_max.z));
        let t_mid = (t_entry + t_exit) * 0.5;
        let ray_pos = local_start + sun_dir * t_mid;
        let ray_uv = ray_pos / 256.0;
        let opacity = textureSampleLevel(t_voxel, s_voxel, ray_uv, 0.0).r;

        accumulated_opacity += opacity * 0.8;

        if (accumulated_opacity > 0.9) {
            return 0.1;
        }

        t_entry = t_exit;

        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                voxel.x += step_sign.x;
                t_max.x += t_delta.x;
            } else {
                voxel.z += step_sign.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                voxel.y += step_sign.y;
                t_max.y += t_delta.y;
            } else {
                voxel.z += step_sign.z;
                t_max.z += t_delta.z;
            }
        }
    }

    return 1.0 - saturate(accumulated_opacity);
}

// ============================================================================
// VERTEX SHADER - GRASS BLADE GENERATION
// ============================================================================

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) height_factor: f32,
}

@vertex
fn vs_grass(
    @builtin(instance_index) instance_id: u32,
    @builtin(vertex_index) vertex_id: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Decode tile and blade indices
    let blades_per_tile = uniforms.quality.x;
    let segments = uniforms.quality.y;

    let tile_idx = instance_id / blades_per_tile;
    let blade_idx = instance_id % blades_per_tile;

    // Bounds check
    if (tile_idx >= uniforms.quality.z) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        out.height_factor = 0.0;
        return out;
    }

    // Get tile data
    let tile = tiles[tile_idx];

    // Unpack tile data
    let actual_blade_count = unpack_blade_count(tile.density_seed);
    let surface_y = unpack_surface_y(tile.biome_height);
    let biome_id = unpack_biome(tile.biome_height);
    let tile_seed = unpack_seed(tile.density_seed);
    let height_var = unpack_height_var(tile.density_seed);

    // Quality-based density threshold from uniforms (quality.w = min_blade_threshold)
    // Lower quality = higher threshold = only show denser patches
    let min_density_blades = uniforms.quality.w;

    // Skip sparse tiles at lower quality
    if (actual_blade_count < min_density_blades) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        out.height_factor = 0.0;
        return out;
    }

    // Skip blades beyond actual count for this tile
    if (blade_idx >= actual_blade_count) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        out.height_factor = 0.0;
        return out;
    }

    // Per-blade deterministic values using hash
    let blade_seed = hash(tile_seed * 7919u + blade_idx * 12345u);

    // Per-block placement: spread blades within the single block (±0.4 from center)
    let offset_x = hash_float_signed(blade_seed) * 0.4;
    let offset_z = hash_float_signed(blade_seed + 1u) * 0.4;
    let rotation = hash_float(blade_seed + 2u) * 6.28318;         // 0 to 2PI
    let height_mult = 0.85 + hash_float(blade_seed + 3u) * 0.3;   // 0.85 to 1.15
    let curve_dir = hash_float_signed(blade_seed + 4u);           // Curve direction

    // Blade base position (block center + offset)
    let base_pos = vec3<f32>(
        tile.position.x + offset_x,
        surface_y,
        tile.position.y + offset_z
    );

    // Distance-based culling
    let dist_to_camera = distance(base_pos.xz, uniforms.camera_pos.xz);
    if (dist_to_camera > uniforms.lod_distances.z) {
        // Beyond cull distance - degenerate triangle
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        out.height_factor = 0.0;
        return out;
    }

    // Distance-based LOD: increase width at distance for anti-aliasing
    let lod_factor = saturate(dist_to_camera / uniforms.lod_distances.z);
    let width_scale = mix(1.0, 1.5, lod_factor);    // Wider at distance

    // Blade parameters: SHORT and THICK grass, minimal curve (not noodly)
    let base_height = 0.15 + height_var * 0.15;   // 0.15 to 0.30 blocks (much shorter)
    let blade_height = base_height * height_mult;
    let blade_width = 0.06 * width_scale;          // Thicker blades
    let curve_amount = 0.02 + hash_float(blade_seed + 5u) * 0.04; // Very slight curve (0.02-0.06)

    // Vertex position along blade
    let segment = vertex_id / 2u;
    let side = f32(vertex_id % 2u) - 0.5; // -0.5 or 0.5
    let t = f32(segment) / f32(segments - 1u);

    // Bezier control points for blade curve
    let p0 = vec3<f32>(0.0, 0.0, 0.0);
    let p1 = vec3<f32>(curve_amount * curve_dir, blade_height * 0.5, 0.0);
    let p2 = vec3<f32>(curve_amount * curve_dir * 2.0, blade_height, 0.0);

    // Wind effect - multi-octave with per-blade variation for natural movement
    let wind_time = uniforms.wind.w;
    let wind_strength = uniforms.wind.z;

    // Per-blade phase offset for desynchronization (0 to 2π)
    let blade_phase = hash_float(blade_seed + 7u) * 6.28318;

    // Spatial frequency - lower = larger waves traveling across landscape
    let spatial_freq = 0.08;
    let wind_phase = (base_pos.x * uniforms.wind.x + base_pos.z * uniforms.wind.y) * spatial_freq;

    // Gust variation - slow-moving areas of stronger/weaker wind
    let gust_phase = base_pos.x * 0.03 + base_pos.z * 0.04;
    let gust = sin(gust_phase + wind_time * 0.4) * 0.3 + 0.7;  // 0.4 to 1.0

    // Multi-octave wind waves at different speeds and scales
    // Primary: slow, large-scale swaying
    let wind1 = sin(wind_phase + wind_time * 1.2 + blade_phase) * 0.6;
    // Secondary: medium turbulence
    let wind2 = sin(wind_phase * 2.3 + wind_time * 2.0 + blade_phase * 1.3) * 0.25;
    // Tertiary: fast flutter (small amplitude)
    let wind3 = sin(wind_phase * 4.1 + wind_time * 3.5 + blade_phase * 0.7) * 0.15;

    // Combine with gust modulation - creates traveling waves effect
    let wind_wave = (wind1 + wind2 + wind3) * gust;

    // Apply wind: stronger at blade tip (t*t curve), scaled by strength
    let wind_offset = wind_wave * t * t * wind_strength * 0.20;

    // Evaluate curve
    var curve_pos = bezier(p0, p1, p2, t);

    // Apply wind
    curve_pos.x += wind_offset * uniforms.wind.x;
    curve_pos.z += wind_offset * uniforms.wind.y;

    // Width tapers toward tip
    let width = blade_width * (1.0 - t * 0.85);

    // Apply width offset perpendicular to blade
    curve_pos.x += side * width;

    // Rotate blade around Y axis and position in world
    let rotated = rotate_y(curve_pos, rotation);
    let world_pos = base_pos + rotated;

    // Calculate normal (simplified - mostly up with slight outward tilt)
    let normal = normalize(vec3<f32>(side * 0.3, 1.0, 0.0));
    let rotated_normal = rotate_y(normal, rotation);

    // Biome color interpolation
    let biome_color = biome_colors[biome_id];
    let color = mix(biome_color.base.rgb, biome_color.tip.rgb, t);

    // Add slight color variation per blade for natural look
    let color_var = (hash_float(blade_seed + 6u) - 0.5) * 0.08;
    let final_color = color + vec3<f32>(color_var, color_var * 0.5, -color_var * 0.3);

    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = final_color;
    out.world_pos = world_pos;
    out.normal = rotated_normal;
    out.height_factor = t;

    return out;
}

// ============================================================================
// FRAGMENT SHADER
// ============================================================================

@fragment
fn fs_grass(in: VertexOutput) -> @location(0) vec4<f32> {
    // Day/night aware lighting from scene uniforms
    let sun_dir = normalize(uniforms.sun_direction.xyz);
    let sun_intensity = uniforms.sun_direction.w;
    let sun_color = uniforms.sun_color.rgb;
    let ambient_base = uniforms.ambient_color.rgb;
    let shadow_mode = uniforms.ambient_color.a;  // 0=off, 1=simple

    let normal = normalize(in.normal);

    // === SHADOW SAMPLING ===
    // When shadow_mode > 0, sample voxel volume for occlusion
    var shadow = 1.0;
    if (shadow_mode > 0.5 && sun_intensity > 0.1) {
        shadow = sample_grass_shadow(in.world_pos, sun_dir);
        // Blend shadow based on sun intensity (less shadow at dawn/dusk)
        shadow = mix(1.0, shadow, sun_intensity);
    }

    // Directional sun lighting (only when sun is up)
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let sun_contrib = sun_color * ndotl * sun_intensity * 0.4 * shadow;

    // Face-based shading for grass blades
    // Upward-facing parts brighter, sides darker
    let face_brightness = mix(0.6, 0.9, max(normal.y, 0.0));

    // Ambient light (provides base illumination, especially at night)
    // Shadow affects ambient more strongly to darken shadowed areas
    let ambient_shadow = mix(0.5, 1.0, shadow);  // Stronger shadow on ambient
    let ambient = ambient_base * 0.45 * ambient_shadow;

    // Combine lighting
    let lighting = (ambient + sun_contrib) * face_brightness;

    // Subtle subsurface scattering for grass (reduced at night and in shadow)
    let view_dir = normalize(uniforms.camera_pos.xyz - in.world_pos);
    let back_light = max(dot(-view_dir, sun_dir), 0.0);
    let sss = back_light * sun_intensity * shadow * 0.04 * in.height_factor;

    // Apply lighting to grass color
    var final_color = in.color * (lighting + sss);

    // Ensure grass doesn't go completely black at night (minimum visibility)
    let min_brightness = 0.03;
    final_color = max(final_color, vec3<f32>(min_brightness));

    // Distance fade
    let dist = distance(in.world_pos.xz, uniforms.camera_pos.xz);
    let fade_start = uniforms.lod_distances.z * 0.5;
    let fade = 1.0 - smoothstep(fade_start, uniforms.lod_distances.z, dist);

    return vec4<f32>(final_color, fade);
}
