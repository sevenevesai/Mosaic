# Water Rendering System

Ray-marched infinite water plane with SSR reflections and Fresnel effects (`renderer/water.rs`, ~900 LOC).

## Architecture

Fullscreen post-process pass using ray-plane intersection per pixel. No mesh generation (eliminates seams/jitter). Single level at y=32, supports up to 16 levels for future use.

## Pipeline Integration

1. Opaque chunks render -> depth buffer + scene_texture
2. Copy scene_texture -> refraction_source_texture (if water enabled)
3. Water pass: ray-plane intersection, depth test, waves, Fresnel, SSR, refraction
4. Alpha blend onto scene

## Quality Presets (ESC menu -> WATER REFRACTION)

| Preset | SSR Steps | Waves | Refraction | Chromatic |
|--------|-----------|-------|------------|-----------|
| Off | 0 | - | 0% | None |
| Low | 8 | 2 oct | 2% | None |
| Medium | 16 | 3 oct | 4% | None |
| High | 24 | 3 oct | 6% | 0.3% |
| Ultra | 32 | 4 oct | 8% | 0.6% |

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Water on terrain above sea level | Depth test failing | Terrain height check: `scene_world_pos.y > water_height + 0.5` |
| Transparent circle around player | Fresnel too strong looking down | Clamped n_dot_v to 0.85, min alpha 0.7 |
| Bouncy reflections on camera move | Wave normal affecting Fresnel | Separate normals: 70% flat / 30% wave for Fresnel |
| Water level changes with camera Y | NDC depth precision | Switched to world-space distance test |
| Water on vertical shore faces | Ray hits water plane at shore | Skip when looking_horizontal at water level |

## Gotchas

- **OFF preset skips entire pass** - zero GPU cost
- **SSR is main performance cost** - ray marching per pixel
- **Scene texture copy required** - can't read/write same texture for refraction
- **Future levels**: `water_system.set_water_levels(&[(32.0, 0.0), (20.0, 0.0)])` for multiple heights
