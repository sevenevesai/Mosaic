# Ambient occlusion design

## Mechanism (from .claude/rules/ao-system.md)

`trace_voxel_ao()` in `src/renderer/pipeline.rs` traces the 256^3 voxel volume with
Amanatides & Woo DDA — the same integer-exact traversal the shadow system uses. It is
world-space and camera-independent, which is what removes the artifacts that defeated the
screen-space attempts.

Per fragment:

1. Build a tangent frame from the surface normal.
2. Fire 3-9 DDA rays in a cosine-weighted hemisphere around the normal, the count set by the
   quality preset.
3. Each ray samples with `textureLoad`, so hits are exact voxels with no interpolation.
4. The first solid voxel a ray hits contributes occlusion, weighted by distance with a
   quadratic falloff.
5. Normalize by total weight and apply a power curve.
6. Map through `mix(0.40, 1.0, ao_raw)`.

### Direction layout

| Ring | Added at | Directions | Tilt | Weight each |
|---|---|---|---|---|
| Centre | always | 1 (the normal) | 0 | 0.20 |
| Ring 1 | LOW | 2 (±tangent) | 40 deg | 0.18 |
| Ring 1 | MEDIUM | 2 (±bitangent) | 40 deg | 0.18 |
| Ring 2 | HIGH | 2 (±tangent) | 55 deg | 0.16 |
| Ring 2 | ULTRA | 2 (±bitangent) | 70 deg | 0.16 |

Falloff range grows with quality (4/5/7/9 blocks) and is passed into `dda_trace_ao()`, so a
higher preset darkens over a longer range rather than only sampling more finely.

### Why DDA rather than `textureSampleLevel`

`textureLoad` reads one voxel exactly. `textureSampleLevel` filters linearly across voxel
boundaries, and since block surfaces lie on those boundaries the blend produces visible lines
along every block edge. The shadow system reached the same conclusion first; AO follows it.

### Where the pieces live

| File | Role |
|---|---|
| `src/renderer/pipeline.rs` | `dda_trace_ao()`, `trace_voxel_ao()`, and where AO is applied |
| `src/renderer/voxel_gi.rs` | packs AO quality into `volume_origin.w` bits 4-6 |
| `src/renderer/mod.rs` | `ao_quality` field and the cycling methods |
| `src/settings.rs`, `src/settings_menu.rs` | `ao_quality` setting, AMBIENT OCCLUSION item |
| `src/renderer/ssao.rs` | disabled screen-space implementation, kept for reference |
| `src/world/meshing.rs` | baked per-vertex AO |
