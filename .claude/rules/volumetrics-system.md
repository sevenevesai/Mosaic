---
paths:
  - "src/renderer/volumetrics.rs"
  - "src/renderer/mod.rs"
---

# Volumetric light shafts

Three half-resolution passes in `renderer/volumetrics.rs`: a DDA raymarch through the 256^3
voxel volume into `volumetric_texture`, a bilateral blur into `blur_texture`, then a
full-resolution additive composite onto `scene_texture`. Runs after SSAO and before DoF,
bloom and AA. Mechanism: `docs/volumetrics-design.md`.

| Preset | March steps | Shadow DDA steps | Blur | Max distance |
|---|---|---|---|---|
| OFF | 0 | 0 | none | zero cost |
| LOW | 24 | 16 | H only | 48 |
| MEDIUM (default) | 40 | 32 | H+V | 80 |
| HIGH | 56 | 48 | 2x H+V | 112 |
| ULTRA | 72 | 64 | 2x H+V | 128 |

Quality lives in `VolumetricUniform.screen_params.z`, deliberately separate from the shadow
and AO bits in `volume_origin.w`. The V key and the Graphics tab both cycle it, and
`settings.rs` stores `volumetrics_quality` as a string; the older boolean
`volumetrics_enabled` still parses, true as MEDIUM and false as OFF.

## Gotchas

- Crossing the LOW/MEDIUM boundary changes which texture the composite reads (LOW ends on
  `blur_texture`, MEDIUM and above end back on `volumetric_texture`), so
  `rebuild_composite_bind_group()` must run on that transition or the screen keeps the old
  source.
- Blur taps are unrolled because WGSL cannot index an `array<>` with a runtime value. Do not
  refactor them into a loop.
- `update_blur_uniforms()` has to run on resize; stale texel sizes blur by the wrong radius.
- Density is `sun_color.a`, currently 0.006, and is very sensitive — small changes swing the
  image hard.
- Sky pixels are marched rather than skipped, so shafts stay visible against the sky, capped
  at `max_distance`.
- Open-air wash is a known unsolved issue: the forward-scatter peak (about 2.4 at g = 0.76)
  washes the screen out when staring at the sun in the open. Capping the phase near 0.35 helps
  but dims the shafts; neither has been settled on.
