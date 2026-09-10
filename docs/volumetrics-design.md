# Volumetric light shafts design

## Mechanism (from .claude/rules/volumetrics-system.md)

```
Pass 1  DDA raymarch      half-res -> volumetric_texture [Rgba16Float]
Pass 2  bilateral blur    half-res -> blur_texture       [Rgba16Float]
Pass 3  composite         full-res, additive blend -> scene_texture
```

### DDA shadow test

Adapted from the chunk shader's sun shadow trace and identical in pattern: `textureLoad` with
integer coordinates so nothing is interpolated, branchless stepping with `select()`, the
starting voxel skipped to avoid self-shadowing, a binary 0.0/1.0 result with an early exit on
the first solid hit, and termination at 120 blocks from the volume centre.

### Phase function

Dual-lobe Henyey-Greenstein: `mix(henyey_greenstein(cos_theta, 0.76),
henyey_greenstein(cos_theta, -0.25), 0.15)` — 85% strong forward scattering, which is what
makes shafts bright when looking toward the sun, plus 15% backward fill so the effect does not
disappear looking away.

`smoothstep(0.05, 0.3, sun_intensity)` fades the whole effect out, so it does not pop at
sunrise and sunset.

### Temporal dithering

Interleaved Gradient Noise (Jimenez, SIGGRAPH 2014), jittered per pixel and rotated each frame
by the golden angle 5.588238 radians, breaks up the banding that discrete march steps produce.
The frame counter in `screen_params.w` uses `wrapping_add(1)`; IGN takes `fract()` of it, so
overflow is harmless.

### Bilateral blur

Depth-aware 9-tap Gaussian, weights 0.227027, 0.195, 0.122, 0.054, 0.016, separable into
horizontal and vertical passes that ping-pong between `volumetric_texture` and `blur_texture`.
The bilateral factor `step(abs(sample_depth - center_depth), depth_threshold)` with a
threshold of 2.0 blocks rejects samples across a depth edge, which stops light bleeding across
block boundaries.

### Where the pieces live

| File | Role |
|---|---|
| `src/renderer/volumetrics.rs` | `VolumetricLighting`, three pipelines and three WGSL shaders |
| `src/renderer/mod.rs` | constructor, per-frame `VolumetricUniform` update, quality cycling, the render call in the pass sequence |
| `src/settings.rs`, `src/settings_menu.rs` | `volumetrics_quality`, the VOLUMETRICS EnumCycle item |
| `src/app.rs` | V key cycles the presets |
