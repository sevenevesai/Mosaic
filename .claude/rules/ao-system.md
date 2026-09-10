---
paths:
  - "src/renderer/pipeline.rs"
  - "src/renderer/ssao.rs"
  - "src/renderer/bloom.rs"
  - "src/renderer/voxel_gi.rs"
  - "src/renderer/voxel_lighting.wgsl"
  - "src/world/meshing.rs"
---

# Ambient occlusion

Two layers multiply in the chunk fragment shader: per-vertex AO baked during meshing (1-2
block corner darkening, free, camera-independent) and voxel AO traced in the shader
(`trace_voxel_ao()` in `pipeline.rs`, 5-8 blocks). Mechanism and the hemisphere direction
layout: `docs/ambient-occlusion-design.md`.

| Preset | Directions | Steps per direction | Falloff |
|---|---|---|---|
| OFF | 0 | 0 | returns 1.0, zero cost |
| LOW | 3 | 6 | 4 blocks |
| MEDIUM (default) | 5 | 8 | 5 blocks |
| HIGH | 7 | 12 | 7 blocks |
| ULTRA | 9 | 16 | 9 blocks |

## SSAO is disabled, and why

Screen-space AO drew grid lines at every block boundary that popped in and out as the camera
turned. The cause is structural: half-resolution rendering plus a linear upscale across depth
discontinuities, and block edges put a discontinuity everywhere. Five sessions of SSAO and
GTAO variants did not fix it, so voxel AO replaced it. `ssao.rs` stays for reference with
quality defaulting to Off and its uniform update and render calls guarded by `is_enabled()`.
Turning it back on programmatically brings the grid lines back, and the ESC menu's
AMBIENT OCCLUSION item now sets voxel AO quality rather than SSAO.

## Gotchas

- The bloom composite shader samples the scene with `textureLoad`, not `textureSample`. It
  was part of the same fix; reverting it reintroduces edge smearing.
- AO quality occupies bits 4-6 of `volume_origin.w`, alongside shadow quality in 1-3 and torch
  shadows in bit 0.
- The AO ray starts 0.1 blocks along the normal. Block surfaces sit on integer coordinates,
  exactly where voxel boundaries are, so a ray starting on the surface occludes itself.
- Voxel AO is traced only where baked `vertex_ao < 0.99`, so a flat open face costs nothing.
- AO strength is halved in shadow (`mix(0.5, 1.0, final_shadow)`) so shadowed corners are not
  darkened twice.
- Tune intensity through the `mix(0.40, 1.0, ao_raw)` mapping in `pipeline.rs`: 0.40 is deep
  corners and caves, 1.0 is open ground.
- Per-vertex AO is the stable base layer. Keep it even when changing the traced layer.
