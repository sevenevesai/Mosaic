# Volumetric Light Shafts (God Rays)

DDA-based world-space raymarching through the 256^3 voxel volume. 3-pass pipeline at half resolution.

## Architecture

```
Pass 1: DDA Raymarch (half-res) -> volumetric_texture [Rgba16Float]
Pass 2: Bilateral Blur (half-res) -> blur_texture [Rgba16Float]
Pass 3: Composite (full-res, additive blend) -> scene_texture
```

Runs after SSAO, before DoF/Bloom/AA in the render pipeline.

## Files

| File | Role |
|------|------|
| `src/renderer/volumetrics.rs` | VolumetricLighting struct, 3 pipelines, 3 WGSL shaders (~1160 LOC) |
| `src/renderer/mod.rs:~790` | Constructor, uniform writes (`screen_params.z=quality, w=frame`) |
| `src/renderer/mod.rs:~2970` | VolumetricUniform update each frame |
| `src/renderer/mod.rs:~3772` | Render call in pass sequence |
| `src/renderer/mod.rs:~1324` | Quality cycling methods |
| `src/settings.rs:~19` | `volumetrics_quality: String` (OFF/LOW/MEDIUM/HIGH/ULTRA) |
| `src/settings_menu.rs:~316` | VOLUMETRICS EnumCycle menu item |
| `src/app.rs:~423` | V key cycles quality presets |

## DDA Shadow Test (in raymarch shader)

Adapted from `pipeline.rs:684-769` — same Amanatides & Woo pattern:
- `textureLoad` with integer coords (no interpolation, no artifacts)
- Branchless DDA stepping with `select()`
- Skip starting voxel to prevent self-shadowing
- Binary result: 0.0 (shadow) or 1.0 (lit), early exit on first solid hit
- Volume edge termination at 120 blocks from center

## Phase Function

Dual-lobe Henyey-Greenstein:
- 85% forward scattering (g=0.76) — bright shafts when looking toward sun
- 15% backward fill (g=-0.25) — subtle fill when looking away
- Combined via `mix(forward, back, 0.15)`

## Temporal Dithering

Interleaved Gradient Noise (Jorge Jimenez, SIGGRAPH 2014):
- Per-pixel jitter rotated each frame by golden angle (~5.588 rad)
- `screen_params.w` = frame_counter (wrapping u32)
- Breaks banding from discrete march steps

## Bilateral Blur

Depth-aware 9-tap Gaussian (unrolled — WGSL requires constant array indexing):
- Weights: 0.227, 0.195, 0.122, 0.054, 0.016
- Bilateral factor: `step(abs(sample_depth - center_depth), threshold)` rejects samples across depth edges
- Depth threshold: 2.0 blocks (prevents bleeding at block boundaries)
- Separable H+V passes; ping-pong between volumetric_texture and blur_texture

## Quality Presets

| Preset | March Steps | Shadow DDA Steps | Blur Passes | Max Distance | Control |
|--------|------------|-----------------|-------------|-------------|---------|
| OFF | 0 | 0 | 0 | 0 | Zero cost |
| LOW | 24 | 16 | H only | 48 | `quality=1` |
| MEDIUM | 40 | 32 | H+V | 80 | `quality=2` (default) |
| HIGH | 56 | 48 | 2x H+V | 112 | `quality=3` |
| ULTRA | 72 | 64 | 2x H+V | 128 | `quality=4` |

Quality packed into `VolumetricUniform.screen_params.z` (NOT in `volume_origin.w` — separate from shadow/AO bits).

## Composite Texture Routing

- **LOW** (H blur only): composite reads `blur_texture`
- **MEDIUM+** (H+V): composite reads `volumetric_texture` (V pass writes back to it)
- `rebuild_composite_bind_group()` swaps the source when crossing the LOW/MEDIUM boundary

## Settings Integration

- `settings.rs`: `volumetrics_quality: String` (default "MEDIUM")
- Backward compat: old `volumetrics_enabled=true` parsed as "MEDIUM", `false` as "OFF"
- V key: cycles OFF -> LOW -> MEDIUM -> HIGH -> ULTRA -> OFF
- ESC menu: Graphics tab, EnumCycle control

## Gotchas

- **WGSL array indexing** — Cannot use runtime variable to index `array<>`. Blur taps are unrolled, not looped.
- **Composite bind group** must be rebuilt when quality crosses LOW/MEDIUM boundary (texture source changes)
- **Blur uniforms** need updating on resize (`update_blur_uniforms()` writes new texel sizes)
- **Frame counter** uses `wrapping_add(1)` — IGN uses `fract()` so overflow is harmless
- **Sky pixels** are marched (not skipped) — god rays visible against sky background, capped at max_distance
- **Sun fade** — `smoothstep(0.05, 0.3, sun_intensity)` prevents abrupt pop at sunset/sunrise
- **Density** is `sun_color.a` (currently 0.006) — very sensitive, small changes have large visual impact
- **Open-air wash** — Forward scatter peak (~2.4 at g=0.76) causes uniform bright wash when staring at sun in open areas. Known issue. Phase capping at ~0.35 helps but reduces shaft brightness. Needs further tuning.
