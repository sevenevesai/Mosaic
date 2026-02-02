# Water Rendering System

Ray-marched infinite water plane with SSR reflections and Fresnel effects.

## Architecture Overview

Water renders as a **fullscreen post-process pass** using ray-plane intersection:
- No per-chunk mesh generation (eliminates seams, jitter, mesh overhead)
- Single water level at y=32 (supports up to 16 levels for future use)
- Depth buffer sampling for occlusion testing
- Scene texture sampling for refraction

## Key Files

| File | Purpose |
|------|---------|
| `renderer/water.rs` | Ray-marched water system (~900 LOC) |
| `renderer/mod.rs` | Water pass integration, texture copy for refraction |
| `world/water_flow.rs` | Pipe-model flow simulation (parked, not used visually) |

## Data Structures

```rust
// Water quality preset
pub enum WaterQuality { Off, Low, Medium, High, Ultra }

// Uniform buffer for water rendering
pub struct WaterUniforms {
    view_proj, inv_view_proj, inv_view, inv_proj: mat4x4,
    camera_pos: vec4,           // xyz = position, w = time
    sun_direction: vec4,        // xyz = direction, w = intensity
    shallow_color, deep_color: vec4,
    wave_speed, wave_strength, fresnel_power: f32,
    ssr_steps, wave_octaves: u32,
    water_levels: [WaterLevel; 16],  // Multiple water heights
}

// Per-level water data
pub struct WaterLevel {
    height: f32,      // Y coordinate of water surface
    water_type: f32,  // 0=water, 1=lava (future)
}
```

## Rendering Pipeline

1. Opaque chunks render → depth buffer + scene_texture
2. **Copy scene_texture** → refraction_source_texture (if water enabled)
3. **Water pass** (fullscreen triangle):
   - Ray-plane intersection for each pixel
   - Distance-based depth test against scene geometry
   - Terrain height check (skip if terrain above water level)
   - Apply waves, Fresnel, SSR, refraction, foam
4. Alpha blend onto scene

## Shader Algorithm

```wgsl
// For each pixel:
1. Reconstruct view ray from UV + inverse matrices
2. Sample depth buffer → reconstruct scene world position
3. Ray-plane intersection: t = (water_height - cam.y) / ray.y
4. Depth test: skip if terrain above water OR water behind terrain
5. Calculate wave normal (multi-octave sine)
6. Fresnel: blend reflection vs refraction based on view angle
7. SSR: ray march in screen space for nearby reflections
8. Fallback: procedural sky color for distant reflections
9. Refraction: sample scene with wave-based UV distortion
10. Output: blended color with depth-based alpha
```

## Quality Presets

| Preset | SSR Steps | Wave Octaves | Refraction | Chromatic |
|--------|-----------|--------------|------------|-----------|
| Off | 0 | - | 0% | None |
| Low | 8 | 2 | 2% | None |
| Medium | 16 | 3 | 4% | None |
| High | 24 | 3 | 6% | 0.3% |
| Ultra | 32 | 4 | 8% | 0.6% |

Access: ESC menu → WATER REFRACTION

## Key Parameters (in shader)

```wgsl
wave_speed: 0.4        // Animation speed
wave_strength: 0.08    // Normal perturbation intensity
fresnel_power: 5.0     // Fresnel falloff exponent
F0: 0.04               // Water reflectivity at normal incidence
ssr_max_distance: 50.0 // Max SSR ray travel (blocks)
ssr_thickness: 0.5     // Depth threshold for SSR hit
```

## Depth Testing

Two-stage depth test for robustness:

1. **Terrain height check**: `if scene_world_pos.y > water_height + 0.5` → skip
   - Prevents water overlay on blocks above water level

2. **Distance comparison**: `if water_dist < scene_dist + bias` → render water
   - Uses world-space distance (more robust than NDC depth at distance)
   - Proportional bias (0.5% of max distance) prevents z-fighting

## Shore Handling

- **Vertical faces**: Skip water on steep viewing angles at water level
  ```wgsl
  if (scene_at_water_level && looking_horizontal) { discard; }
  ```
- **Shore fade**: Alpha smoothstep based on water depth (0-2 blocks)
- **Minimum alpha**: 0.7 ensures water always visible

## Fresnel & Reflections

- Fresnel uses mostly-flat normal (70% flat / 30% wave) for stability
- n_dot_v clamped to 0.85 max to prevent extreme transparency looking down
- Reflection strength: 35% minimum, 90% maximum
- Wave normals affect reflection direction but not intensity

## SSR (Screen-Space Reflections)

- Ray march from water surface along reflection direction
- Binary search refinement (4 iterations) when hit detected
- Edge fade: smooth falloff at screen boundaries
- Distance fade: SSR contribution reduces with ray travel distance
- Fallback: procedural sky gradient when SSR misses

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Water on terrain above sea level | Depth test failing | Terrain height check added |
| Circular transparent area around player | Fresnel too strong looking down | Clamped n_dot_v, higher min alpha |
| Bouncy reflections on camera move | Wave normal affecting Fresnel | Separate normals for Fresnel vs reflection |
| Water level changes with camera Y | NDC depth precision issues | Switched to world-space distance test |
| Water on vertical shore faces | Ray hits water plane at shore edge | Skip when looking_horizontal at water level |

## Performance Notes

- Single fullscreen pass (3 vertices, no mesh upload)
- SSR is main cost (ray marching per pixel)
- OFF preset skips entire water pass (zero GPU cost)
- Scene texture copy only when water enabled

## Future: Multiple Water Levels

System supports 16 water levels for:
- Underground lakes at different depths
- Waterfalls (multiple connected levels)
- Lava pools (water_type = 1.0)

Currently only sea level (y=32) is used. To add levels:
```rust
water_system.set_water_levels(&[
    (32.0, 0.0),   // Sea level, water
    (20.0, 0.0),   // Underground lake
    (10.0, 1.0),   // Lava pool
]);
```
