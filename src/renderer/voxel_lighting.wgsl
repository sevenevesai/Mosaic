// ═══════════════════════════════════════════════════════════════════════════
// VOXEL-BASED LIGHTING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════
//
// Modern voxel lighting using:
// - DDA ray marching for shadows (replaces shadow maps)
// - Voxel Cone Tracing for global illumination
// - Beer-Lambert for translucency
//
// Based on: "Voxel Global Illumination" research and lightingideas.txt

// ═══════════════════════════════════════════════════════════════════════════
// VOXEL VOLUME UNIFORM
// ═══════════════════════════════════════════════════════════════════════════

struct VoxelVolumeUniform {
    volume_center: vec4<f32>,      // xyz = world center of volume
    volume_size: vec4<f32>,        // xyz = size in voxels (256)
    inv_volume_size: vec4<f32>,    // xyz = 1/size
    sun_direction: vec4<f32>,      // xyz = direction, w = intensity
    sun_color: vec4<f32>,          // rgb = color
    ambient_color: vec4<f32>,      // rgb = ambient
    time_of_day: f32,
    _padding: vec3<f32>,
};

// Voxel volume texture (256^3 RGBA8)
// R = opacity (0=air, 255=solid)
// G = emissive
// B = material type
// A = flags

// ═══════════════════════════════════════════════════════════════════════════
// DDA RAY MARCHING FOR SHADOWS
// ═══════════════════════════════════════════════════════════════════════════
// Fast integer-based traversal through voxel grid
// Reference: "A Fast Voxel Traversal Algorithm" - Amanatides & Woo

fn world_to_voxel_uv(world_pos: vec3<f32>, volume_center: vec3<f32>, volume_size: f32) -> vec3<f32> {
    // Convert world position to voxel UV coordinates [0, 1]
    let half_size = volume_size * 0.5;
    let local_pos = world_pos - volume_center + vec3<f32>(half_size);
    return local_pos / volume_size;
}

fn is_in_volume(uv: vec3<f32>) -> bool {
    return all(uv >= vec3<f32>(0.0)) && all(uv <= vec3<f32>(1.0));
}

// DDA ray march through voxel volume
// Returns: shadow factor (0.0 = full shadow, 1.0 = fully lit)
fn ray_march_shadow(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    world_pos: vec3<f32>,
    light_dir: vec3<f32>,
    volume_center: vec3<f32>,
    volume_size: f32
) -> f32 {
    let max_steps = 128;
    let step_size = 1.0;  // One voxel per step

    var accumulated_opacity = 0.0;
    var pos = world_pos + light_dir * 0.5;  // Start slightly above surface

    // Convert to voxel space for bounds checking
    let voxel_start = world_to_voxel_uv(pos, volume_center, volume_size);
    if (!is_in_volume(voxel_start)) {
        return 1.0;  // Outside volume = no shadow
    }

    // DDA-style stepping through voxels
    let inv_dir = 1.0 / max(abs(light_dir), vec3<f32>(0.0001));
    let step_dir = sign(light_dir);

    for (var i = 0; i < max_steps; i++) {
        pos += light_dir * step_size;

        let uv = world_to_voxel_uv(pos, volume_center, volume_size);

        // Exit if outside volume
        if (!is_in_volume(uv)) {
            break;
        }

        // Sample voxel opacity
        let voxel = textureSampleLevel(voxel_tex, voxel_sampler, uv, 0.0);
        let opacity = voxel.r;

        // Accumulate opacity (Beer-Lambert style)
        accumulated_opacity += opacity * 0.8;

        // Early termination if fully shadowed
        if (accumulated_opacity >= 1.0) {
            return 0.0;
        }
    }

    // Convert accumulated opacity to shadow factor
    return 1.0 - clamp(accumulated_opacity, 0.0, 1.0);
}

// Soft shadow version with multiple samples
fn ray_march_shadow_soft(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    world_pos: vec3<f32>,
    light_dir: vec3<f32>,
    volume_center: vec3<f32>,
    volume_size: f32,
    softness: f32
) -> f32 {
    // Main ray
    var shadow = ray_march_shadow(voxel_tex, voxel_sampler, world_pos, light_dir, volume_center, volume_size);

    // Add soft penumbra with offset rays (simple 4-tap)
    if (softness > 0.0) {
        let tangent = normalize(cross(light_dir, vec3<f32>(0.0, 1.0, 0.0)));
        let bitangent = cross(light_dir, tangent);

        let offset = softness;
        shadow += ray_march_shadow(voxel_tex, voxel_sampler, world_pos + tangent * offset, light_dir, volume_center, volume_size);
        shadow += ray_march_shadow(voxel_tex, voxel_sampler, world_pos - tangent * offset, light_dir, volume_center, volume_size);
        shadow += ray_march_shadow(voxel_tex, voxel_sampler, world_pos + bitangent * offset, light_dir, volume_center, volume_size);
        shadow += ray_march_shadow(voxel_tex, voxel_sampler, world_pos - bitangent * offset, light_dir, volume_center, volume_size);

        shadow /= 5.0;  // Average all samples
    }

    return shadow;
}

// ═══════════════════════════════════════════════════════════════════════════
// VOXEL CONE TRACING FOR GLOBAL ILLUMINATION
// ═══════════════════════════════════════════════════════════════════════════
// Traces cones through voxel volume to gather indirect lighting
// Reference: "Interactive Indirect Illumination Using Voxel Cone Tracing" - Crassin et al.

fn cone_trace(
    radiance_tex: texture_3d<f32>,
    radiance_sampler: sampler,
    origin: vec3<f32>,
    direction: vec3<f32>,
    cone_ratio: f32,  // tan(half_angle)
    max_dist: f32,
    volume_center: vec3<f32>,
    volume_size: f32
) -> vec4<f32> {
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    let start_dist = 1.0;  // Start 1 voxel away to avoid self-lighting
    var dist = start_dist;

    while (dist < max_dist && accumulated_alpha < 0.95) {
        // Cone diameter at current distance
        let diameter = max(1.0, 2.0 * cone_ratio * dist);

        // Sample position
        let sample_pos = origin + direction * dist;
        let uv = world_to_voxel_uv(sample_pos, volume_center, volume_size);

        if (!is_in_volume(uv)) {
            break;
        }

        // Calculate mip level based on cone diameter
        // mip = log2(diameter) where diameter is in voxels
        let mip = log2(diameter);

        // Sample radiance at appropriate mip level
        let sample = textureSampleLevel(radiance_tex, radiance_sampler, uv, mip);

        // Front-to-back alpha compositing
        let alpha = 1.0 - accumulated_alpha;
        accumulated_color += sample.rgb * sample.a * alpha;
        accumulated_alpha += sample.a * alpha;

        // Step size increases with cone diameter for efficiency
        dist += diameter * 0.5;
    }

    return vec4<f32>(accumulated_color, accumulated_alpha);
}

// Trace diffuse cones in hemisphere around normal
fn trace_diffuse_gi(
    radiance_tex: texture_3d<f32>,
    radiance_sampler: sampler,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    volume_center: vec3<f32>,
    volume_size: f32
) -> vec3<f32> {
    let cone_angle = 0.577;  // tan(30 degrees) - wide cones for diffuse
    let max_distance = 64.0;  // Trace up to 64 voxels

    // Build tangent frame
    let tangent = normalize(cross(normal, vec3<f32>(0.0, 1.0, 0.0)));
    let bitangent = cross(normal, tangent);

    // 5 cones: 1 up, 4 around at 45 degrees
    var gi = vec3<f32>(0.0);

    // Center cone (strongest contribution)
    let center = cone_trace(radiance_tex, radiance_sampler, world_pos, normal, cone_angle, max_distance, volume_center, volume_size);
    gi += center.rgb * 0.4;

    // Side cones (tilted 45 degrees from normal)
    let side_weight = 0.15;
    let tilt = 0.707;  // cos(45)
    let side_dirs = array<vec3<f32>, 4>(
        normalize(normal * tilt + tangent * tilt),
        normalize(normal * tilt - tangent * tilt),
        normalize(normal * tilt + bitangent * tilt),
        normalize(normal * tilt - bitangent * tilt)
    );

    for (var i = 0; i < 4; i++) {
        let side = cone_trace(radiance_tex, radiance_sampler, world_pos, side_dirs[i], cone_angle, max_distance, volume_center, volume_size);
        gi += side.rgb * side_weight;
    }

    return gi;
}

// Trace specular cone for reflections
fn trace_specular(
    radiance_tex: texture_3d<f32>,
    radiance_sampler: sampler,
    world_pos: vec3<f32>,
    reflect_dir: vec3<f32>,
    roughness: f32,
    volume_center: vec3<f32>,
    volume_size: f32
) -> vec3<f32> {
    // Cone angle based on roughness (sharper for smooth surfaces)
    let cone_angle = mix(0.1, 0.6, roughness);
    let max_distance = 128.0;  // Longer traces for reflections

    let reflection = cone_trace(radiance_tex, radiance_sampler, world_pos, reflect_dir, cone_angle, max_distance, volume_center, volume_size);
    return reflection.rgb;
}

// ═══════════════════════════════════════════════════════════════════════════
// AMBIENT OCCLUSION FROM VOXELS
// ═══════════════════════════════════════════════════════════════════════════
// Short-range cone traces for ambient occlusion

fn trace_voxel_ao(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    volume_center: vec3<f32>,
    volume_size: f32
) -> f32 {
    let cone_angle = 0.5;  // Medium-wide cone
    let max_distance = 8.0;  // Short range for AO

    // Build tangent frame
    var tangent: vec3<f32>;
    if (abs(normal.y) > 0.9) {
        tangent = normalize(cross(normal, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        tangent = normalize(cross(normal, vec3<f32>(0.0, 1.0, 0.0)));
    }
    let bitangent = cross(normal, tangent);

    // Trace AO in multiple directions
    var ao = 0.0;
    let weight_sum = 3.0;

    // Center (up normal)
    ao += trace_ao_cone(voxel_tex, voxel_sampler, world_pos, normal, cone_angle, max_distance, volume_center, volume_size) * 1.0;

    // Sides (tilted)
    let tilt = 0.8;
    ao += trace_ao_cone(voxel_tex, voxel_sampler, world_pos, normalize(normal * tilt + tangent * 0.6), cone_angle, max_distance, volume_center, volume_size) * 0.5;
    ao += trace_ao_cone(voxel_tex, voxel_sampler, world_pos, normalize(normal * tilt - tangent * 0.6), cone_angle, max_distance, volume_center, volume_size) * 0.5;
    ao += trace_ao_cone(voxel_tex, voxel_sampler, world_pos, normalize(normal * tilt + bitangent * 0.6), cone_angle, max_distance, volume_center, volume_size) * 0.5;
    ao += trace_ao_cone(voxel_tex, voxel_sampler, world_pos, normalize(normal * tilt - bitangent * 0.6), cone_angle, max_distance, volume_center, volume_size) * 0.5;

    ao /= weight_sum;

    // Invert: 1.0 = no occlusion, 0.0 = full occlusion
    return 1.0 - clamp(ao, 0.0, 1.0);
}

fn trace_ao_cone(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    origin: vec3<f32>,
    direction: vec3<f32>,
    cone_ratio: f32,
    max_dist: f32,
    volume_center: vec3<f32>,
    volume_size: f32
) -> f32 {
    var occlusion = 0.0;
    var dist = 0.5;  // Start close

    while (dist < max_dist) {
        let diameter = max(0.5, 2.0 * cone_ratio * dist);
        let sample_pos = origin + direction * dist;
        let uv = world_to_voxel_uv(sample_pos, volume_center, volume_size);

        if (!is_in_volume(uv)) {
            break;
        }

        // Sample voxel at appropriate mip level
        let mip = log2(max(1.0, diameter));
        let voxel = textureSampleLevel(voxel_tex, voxel_sampler, uv, mip);

        // Accumulate occlusion with distance falloff
        let falloff = 1.0 - dist / max_dist;
        occlusion += voxel.r * falloff * 0.3;

        if (occlusion >= 1.0) {
            break;
        }

        dist += diameter * 0.5;
    }

    return clamp(occlusion, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// BEER-LAMBERT TRANSLUCENCY
// ═══════════════════════════════════════════════════════════════════════════
// For light passing through translucent materials (water, leaves, etc.)

fn beer_lambert_transmittance(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    start_pos: vec3<f32>,
    direction: vec3<f32>,
    distance: f32,
    volume_center: vec3<f32>,
    volume_size: f32,
    absorption_coeff: vec3<f32>  // RGB absorption coefficients
) -> vec3<f32> {
    let step_size = 0.5;
    let steps = i32(distance / step_size);

    var total_density = vec3<f32>(0.0);

    for (var i = 0; i < min(steps, 64); i++) {
        let pos = start_pos + direction * (f32(i) * step_size);
        let uv = world_to_voxel_uv(pos, volume_center, volume_size);

        if (!is_in_volume(uv)) {
            break;
        }

        let voxel = textureSampleLevel(voxel_tex, voxel_sampler, uv, 0.0);

        // Material type determines absorption
        // G channel = emissive, B channel = material type
        let is_translucent = voxel.r > 0.0 && voxel.r < 0.9;  // Semi-transparent

        if (is_translucent) {
            total_density += absorption_coeff * voxel.r * step_size;
        } else if (voxel.r > 0.9) {
            // Fully opaque - block all light
            return vec3<f32>(0.0);
        }
    }

    // Beer-Lambert: transmittance = e^(-density)
    return exp(-total_density);
}

// ═══════════════════════════════════════════════════════════════════════════
// COMBINED LIGHTING FUNCTION
// ═══════════════════════════════════════════════════════════════════════════
// Main entry point for voxel-based lighting

fn calculate_voxel_lighting(
    voxel_tex: texture_3d<f32>,
    radiance_tex: texture_3d<f32>,
    tex_sampler: sampler,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    volume_uniform: VoxelVolumeUniform,
    use_gi: bool,
    use_ao: bool
) -> vec3<f32> {
    let volume_center = volume_uniform.volume_center.xyz;
    let volume_size = volume_uniform.volume_size.x;
    let sun_dir = normalize(volume_uniform.sun_direction.xyz);
    let sun_intensity = volume_uniform.sun_direction.w;
    let sun_color = volume_uniform.sun_color.rgb;
    let ambient = volume_uniform.ambient_color.rgb;

    // === DDA RAY MARCH SHADOW ===
    let shadow = ray_march_shadow_soft(
        voxel_tex, tex_sampler,
        world_pos, sun_dir,
        volume_center, volume_size,
        0.3  // Soft shadow radius
    );

    // === DIRECT LIGHTING ===
    let ndotl = max(dot(normal, sun_dir), 0.0);
    var direct = sun_color * ndotl * sun_intensity * shadow;

    // === GLOBAL ILLUMINATION (VCT) ===
    var indirect = vec3<f32>(0.0);
    if (use_gi) {
        indirect = trace_diffuse_gi(
            radiance_tex, tex_sampler,
            world_pos, normal,
            volume_center, volume_size
        );
    }

    // === VOXEL AMBIENT OCCLUSION ===
    var ao = 1.0;
    if (use_ao) {
        ao = trace_voxel_ao(
            voxel_tex, tex_sampler,
            world_pos, normal,
            volume_center, volume_size
        );
    }

    // === SPECULAR REFLECTION (for glossy surfaces) ===
    var specular = vec3<f32>(0.0);
    if (roughness < 0.8 && use_gi) {
        let reflect_dir = reflect(-view_dir, normal);
        specular = trace_specular(
            radiance_tex, tex_sampler,
            world_pos, reflect_dir, roughness,
            volume_center, volume_size
        );

        // Fresnel for reflection intensity
        let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 5.0);
        specular *= fresnel * (1.0 - roughness);
    }

    // === COMBINE ===
    let total_light = direct + indirect + ambient * ao;
    var final_color = base_color * total_light + specular;

    return final_color;
}
