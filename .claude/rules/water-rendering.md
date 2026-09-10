---
paths:
  - "src/renderer/water.rs"
  - "src/renderer/refraction.rs"
  - "src/world/water_meshing.rs"
  - "src/world/water_flow.rs"
---

# Water rendering

Fullscreen post-process in `renderer/water.rs`: a ray-plane intersection per pixel against an
infinite water plane, so there is no water mesh and therefore no seams or vertex jitter. One
level at y = 32; `MAX_WATER_LEVELS` is 16 and `set_water_levels(&[(32.0, 0.0), ...])` feeds
them, unused so far.

Pass order: opaque chunks fill depth and `scene_texture`, that texture is copied into
`refraction_source_texture`, the water pass runs (depth test, waves, Fresnel, SSR,
refraction), and the result alpha-blends back.

| Preset | SSR steps | Wave octaves | Refraction | Chromatic |
|---|---|---|---|---|
| Off | 0 | - | 0% | none |
| Low | 8 | 2 | 2% | none |
| Medium | 16 | 3 | 4% | none |
| High | 24 | 3 | 6% | 0.3% |
| Ultra | 32 | 4 | 8% | 0.6% |

## Gotchas

Each row is a bug that recurred until the stated fix; the fix is easy to undo by accident.

| Symptom | Cause | Fix in place |
|---|---|---|
| Water drawn over terrain above sea level | depth test alone is not enough | terrain height check `scene_world_pos.y > water_height + 0.5` |
| Transparent disc around the player | Fresnel goes to zero looking straight down | clamp `n_dot_v` to 0.85, floor alpha at 0.7 |
| Reflections bounce as the camera moves | wave normal fed into Fresnel | split normals, Fresnel uses 70% flat / 30% wave |
| Water level shifts with camera Y | NDC depth precision | compare world-space distance instead |
| Water on vertical shore faces | ray still hits the plane at the shore | skip when looking horizontally at water level |

- The Off preset skips the whole pass, so it costs nothing; SSR is the dominant cost of the
  others.
- The scene texture must be copied before the pass — a shader cannot sample the target it
  writes.
- `world/water_flow.rs` holds a pipe-model flow simulation that is parked and not wired to
  rendering. Do not treat it as live behaviour.
