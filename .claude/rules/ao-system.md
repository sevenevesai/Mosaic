# Ambient Occlusion System

## Current State (as of Feb 2026)

Two AO layers exist, multiplied together in the chunk fragment shader (`pipeline.rs:~1600`):

| Layer | Source | Scale | Status |
|-------|--------|-------|--------|
| Per-vertex AO | Baked during meshing (`meshing.rs`) | 1-2 blocks (corners) | Working, stable, free |
| Voxel AO (DDA hemisphere) | `trace_voxel_ao()` in fragment shader | 5-8 blocks | **Working — DDA hemisphere tracing** |
| SSAO (post-process) | `ssao.rs` hemisphere sampling | N/A | **Broken — disabled by default, legacy** |

## Voxel AO: DDA Hemisphere Tracing (Current Implementation)

`trace_voxel_ao()` in `pipeline.rs:~965` traces the 256^3 voxel volume using Amanatides & Woo DDA — the same integer-exact traversal algorithm used for shadow tracing. This is world-space, camera-independent, and artifact-free.

### How It Works

1. Build tangent frame from surface normal
2. Fire 3-9 DDA rays (quality-dependent) in a cosine-weighted hemisphere around the normal
3. Each ray uses `textureLoad` (integer sampling, no interpolation) for exact voxel hits
4. First solid voxel hit contributes occlusion weighted by distance (quadratic falloff)
5. Results are normalized by total weight and power-curved for natural feel
6. Applied as `mix(0.4, 1.0, ao_raw)` — strong darkening in corners, clear in open areas

### Direction Layout (Cosine-Weighted Hemisphere)

| Quality | Directions | Description |
|---------|-----------|-------------|
| Center (always) | 1 | Normal direction, weight 0.20 |
| Ring 1 (LOW+) | 2 | ±tangent at 40° tilt, weight 0.18 each |
| Ring 1 (MED+) | 2 | ±bitangent at 40° tilt, weight 0.18 each |
| Ring 2 (HIGH+) | 2 | ±tangent at 55° tilt, weight 0.16 each |
| Ring 2 (ULTRA) | 2 | ±bitangent at 70° tilt, weight 0.16 each |

### Quality Presets (AMBIENT OCCLUSION in ESC menu)

| Preset | Directions | Steps/Dir | Falloff | Notes |
|--------|-----------|-----------|---------|-------|
| OFF | 0 | 0 | — | Zero cost, returns 1.0 |
| LOW | 3 | 6 | 4 blocks | Center + tangent pair |
| MEDIUM (default) | 5 | 8 | 5 blocks | + bitangent pair |
| HIGH | 7 | 12 | 7 blocks | + 55° tangent pair (longer range, more contrast) |
| ULTRA | 9 | 16 | 9 blocks | + 70° bitangent pair (maximum range/coverage) |

Quality is packed into `volume_origin.w` bits 4-6 (alongside shadow_quality in bits 1-3 and torch_shadows in bit 0).

### Why DDA, Not `textureSampleLevel`

- `textureLoad`: Integer sampling, exact voxel value, no interpolation artifacts
- `textureSampleLevel`: Linear filtering blends across voxel boundaries, creates visible lined shadows
- DDA matches the shadow system pattern — proven, battle-tested, no artifacts

## SSAO: Root Cause of Grid Lines (CONFIRMED — LEGACY)

**Problem**: Visible grid lines at block boundaries that pop in/out when rotating camera. Persisted across 5+ sessions attempting SSAO, GTAO, and other screen-space AO approaches.

**Root cause**: SSAO half-res rendering + linear upscale + depth discontinuities at every block edge = smeared grid lines. Camera rotation changes depth sampling, causing per-frame popping. Fundamental to screen-space approach, not fixable.

**Status**: SSAO defaults to OFF and is kept for legacy reference only. The menu item was repurposed to control voxel AO quality ("AMBIENT OCCLUSION" setting).

**Fixes applied**:
- `ssao.rs`: Default quality changed from Medium to **Off**
- `mod.rs`: SSAO uniform update and render calls guarded by `is_enabled()` (zero GPU cost when off)
- `bloom.rs`: Composite shader changed from `textureSample` to `textureLoad` for scene texture

## Files

| File | Role |
|------|------|
| `src/renderer/pipeline.rs:~965` | `trace_voxel_ao()` — DDA hemisphere AO (main implementation) |
| `src/renderer/pipeline.rs:~1590` | Where voxel AO is applied in fragment shader |
| `src/renderer/voxel_gi.rs:~1118` | Packs ao_quality into `volume_origin.w` bits 4-6 |
| `src/renderer/mod.rs:~209` | `ao_quality` field + quality methods |
| `src/renderer/mod.rs:~2445` | `apply_settings()` — parses ao_quality from settings |
| `src/settings.rs:~33` | `ao_quality` field in Settings struct |
| `src/settings_menu.rs:~339` | "AMBIENT OCCLUSION" menu item |
| `src/renderer/ssao.rs` | SSAO system (disabled legacy, kept for reference) |
| `src/renderer/bloom.rs:~775` | Bloom composite shader (textureLoad fix applied) |
| `src/world/meshing.rs:~1816` | Per-vertex AO calculation (baked, working) |

## Gotchas

- **SSAO is disabled by default** and no longer has a menu item. Enabling it programmatically WILL show grid lines.
- **Bloom composite uses textureLoad** for scene texture. Do not revert to textureSample.
- **Block surfaces sit on integer coordinates**, same as voxel volume boundaries. The AO DDA start offset (0.1 blocks along normal) prevents self-occlusion.
- **Per-vertex AO is stable** — baked during meshing, no camera-dependent artifacts. Keep it as the base AO layer.
- **Falloff range scales with quality** (4/5/7/9 blocks for LOW/MED/HIGH/ULTRA). Passed as parameter to `dda_trace_ao()`. Higher quality = longer-range darkening.
- **AO quality packed into bits 4-6** of `volume_origin.w`. Shadow quality uses bits 1-3, torch_shadows uses bit 0.
- **The 0.40-1.0 mapping range** gives strong visual darkening in corners/caves. Adjust `mix(0.40, 1.0, ...)` in pipeline.rs:~1625 to tune intensity.
