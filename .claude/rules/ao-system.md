# Ambient Occlusion System

## Current State (as of Feb 2026)

Three AO layers exist, multiplied together in the chunk fragment shader (`pipeline.rs:~1480`):

| Layer | Source | Scale | Status |
|-------|--------|-------|--------|
| Per-vertex AO | Baked during meshing (`meshing.rs`) | 1-2 blocks (corners) | Working, stable, free |
| Voxel AO | `trace_voxel_ao()` in fragment shader | 1-6 blocks | Working but too simplistic (see below) |
| SSAO (post-process) | `ssao.rs` hemisphere sampling | 1.5-3.0 blocks | **Broken - disabled by default** |

## SSAO: Root Cause of Grid Lines (CONFIRMED)

**Problem**: Visible grid lines at block boundaries that pop in/out when rotating camera. Persisted across 5+ sessions attempting SSAO, GTAO, and other screen-space AO approaches.

**Root cause**: The SSAO post-process pipeline fundamentally cannot work cleanly in this voxel engine:

1. SSAO renders at half resolution to an AO texture (R16Float)
2. The depth buffer has hard 90-degree discontinuities at EVERY block edge
3. SSAO sampling creates sharp AO transitions at these depth edges
4. The **apply pass** upscales from half-res using `textureSample` with **linear filtering**
5. Linear filtering smears the sharp AO transitions into visible lines at block boundaries
6. Camera rotation changes depth buffer sampling, making lines shift/pop per-frame
7. The bloom composite (also linear filtering) further amplified these artifacts

**Confirmation method**: Toggling SSAO OFF in settings eliminates ALL grid lines. Toggling SSAO to LOW or above immediately reintroduces them. No other combination of settings (AA, DoF, bloom, volumetrics, shadows) produces the lines when SSAO is off.

**Fixes applied**:
- `ssao.rs`: Default quality changed from Medium to **Off**
- `mod.rs`: SSAO uniform update and render calls guarded by `is_enabled()` (zero GPU cost when off)
- `bloom.rs`: Composite shader changed from `textureSample` to `textureLoad` for scene texture (prevents bilinear bleeding at quad boundaries for any future effects)

**Why screen-space AO fails for voxel engines**: The depth buffer is composed almost entirely of hard right-angle edges. Screen-space techniques reconstruct normals from depth derivatives (incorrect at discontinuities), use noise textures tied to screen coordinates (causes rotation-dependent jitter), and render at reduced resolution (linear upscale smears edge artifacts). These are fundamental to the screen-space approach, not fixable bugs.

## Voxel AO: Current Limitations

`trace_voxel_ao()` in `pipeline.rs:~965` traces the 256^3 voxel volume from the fragment shader. It works in world space (no screen-space artifacts, no grid lines from this system), but has issues:

**Single-direction tracing**: Only samples straight along the surface normal (4 steps at 1-block intervals). This creates sharp rectangular shadow projections of overhead blocks onto surfaces below, rather than smooth natural-looking occlusion gradients.

**Linear sampler on voxel volume**: Uses `textureSampleLevel` which blends across voxel boundaries. Combined with single-direction tracing, this creates visible lined shadows at block layout transitions.

**Narrow effect range**: Currently set to 0.85-1.0 (barely visible). When amplified to 0.4-1.0 for testing, the single-direction artifacts become clearly visible.

## Plan: Replace SSAO with Enhanced Voxel AO

The voxel AO approach is correct (world-space, stable, no grid lines from this system), but needs better sampling:

### Multi-Direction Hemisphere Sampling
Instead of 1 direction (straight up), trace 5-6 directions spread in a cone around the normal. Each direction contributes a partial occlusion vote. This:
- Smooths out sharp block-projection shadows into natural gradients
- Averages voxel boundary transitions (no lines)
- Darkens near walls/overhangs/ceilings realistically

### Quality Presets
Control direction count and step count per direction:
- Off: skip entirely
- Low: 3 directions, 3 steps
- Medium: 5 directions, 4 steps
- High: 6 directions, 5 steps

### Key Design Decisions Still Needed
- `textureLoad` (integer, exact) vs `textureSampleLevel` (linear, smooth) for voxel sampling
- How many directions provide smooth enough results without performance regression
- Intensity tuning to complement per-vertex AO without double-darkening
- Whether to integrate into existing settings menu or replace the SSAO setting

## Files

| File | Role |
|------|------|
| `src/renderer/ssao.rs` | SSAO system (disabled, kept for reference) |
| `src/renderer/pipeline.rs:~965` | `trace_voxel_ao()` - current voxel AO |
| `src/renderer/pipeline.rs:~1475` | Where voxel AO is applied in fragment shader |
| `src/renderer/bloom.rs:~775` | Bloom composite shader (textureLoad fix applied) |
| `src/world/meshing.rs:~1816` | Per-vertex AO calculation (baked, working) |

## Gotchas

- **SSAO is disabled by default** but still toggleable in settings. Enabling it WILL show grid lines.
- **Bloom composite uses textureLoad** for scene texture to prevent bilinear bleeding at quad boundaries. Do not revert to textureSample.
- **Block surfaces sit on integer coordinates**, same as voxel volume boundaries. Any per-pixel effect sampling the volume near surfaces must account for boundary-crossing artifacts.
- **Per-vertex AO is stable** - baked during meshing, no camera-dependent artifacts. Keep it as the base AO layer.
- **The voxel AO does NOT cause grid lines** - confirmed by testing at amplified range (0.4-1.0) with SSAO off. The visual issues it has (block-shaped shadow projections) are a quality problem, not an artifact problem.
