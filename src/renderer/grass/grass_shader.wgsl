// Procedural Grass Rendering Shader
// Tile-based placement with vertex-pulling blade generation

// ============================================================================
// STRUCTURES
// ============================================================================

struct GrassUniforms {
    view_proj: mat4x4<f32>,
    wind: vec4<f32>,           // xy = direction, z = strength, w = time
    camera_pos: vec4<f32>,     // xyz = position
    lod_distances: vec4<f32>,  // x = full_detail, y = low_lod, z = cull, w = chunk_size
    quality: vec4<u32>,        // x = blades_per_tile, y = blade_segments, z = max_tiles
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

// ============================================================================
// BINDINGS
// ============================================================================

@group(0) @binding(0) var<uniform> uniforms: GrassUniforms;
@group(0) @binding(1) var<storage, read> tiles: array<GrassTile>;
@group(0) @binding(2) var<uniform> biome_colors: array<BiomeColor, 8>;

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

    // Skip blades beyond actual count for this tile (noise-driven density)
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

    // Wind effect - world position based for coherent waves
    let wind_freq = 0.15;
    let wind_phase = (base_pos.x * uniforms.wind.x + base_pos.z * uniforms.wind.y) * wind_freq;
    let wind_time = uniforms.wind.w * 2.5;
    let wind_strength = uniforms.wind.z;

    // Primary wind wave + secondary faster wave for variation
    let wind_wave = sin(wind_phase + wind_time) * 0.7 + sin(wind_phase * 1.7 + wind_time * 1.3) * 0.3;
    let wind_offset = wind_wave * t * t * wind_strength * 0.25;

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
    // Simple directional lighting matching main scene
    let sun_dir = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let normal = normalize(in.normal);
    let ndotl = max(dot(normal, sun_dir), 0.0);

    // Ambient + diffuse lighting (darker than before)
    let ambient = 0.35;
    let diffuse = 0.45;
    let lighting = ambient + diffuse * ndotl;

    // Reduced subsurface scattering (less glow)
    let view_dir = normalize(uniforms.camera_pos.xyz - in.world_pos);
    let back_light = max(dot(-view_dir, sun_dir), 0.0);
    let sss = back_light * 0.06 * in.height_factor;

    let final_color = in.color * (lighting + sss);

    // Stronger distance fade starting earlier
    let dist = distance(in.world_pos.xz, uniforms.camera_pos.xz);
    let fade_start = uniforms.lod_distances.z * 0.5;  // Start fade at 50%
    let fade = 1.0 - smoothstep(fade_start, uniforms.lod_distances.z, dist);

    return vec4<f32>(final_color, fade);
}
