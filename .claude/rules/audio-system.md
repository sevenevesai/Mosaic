---
paths:
  - "src/audio/**"
  - "src/bin/generate_sounds.rs"
---

# Audio

`AudioManager` in `src/audio/mod.rs`: rodio `SpatialSink` per source, ten `SoundMaterial`
variants (Grass, Stone, Sand, Wood, Snow, Metal, Gravel, Foliage, Crystal, Generic) mapped by
`SoundMaterial::from_block()`, assets under `assets/sfx/{footstep,break,place,ambient}/`.

Volumes multiply in three tiers: Master (default 0.8) times SFX (1.0) or Ambient (0.6), all
in the ESC menu and persisted to `settings.cfg`.

## Occlusion

`calculate_occlusion()` DDA-marches from listener to source and accumulates 0.4 per solid
block and 0.1 per foliage or water block, clamped to 1.0. Playback volume is
`1.0 - occlusion * 0.7`, so a fully walled-off source is attenuated by at most 70% and never
goes silent — an inaudible occluded sound means that factor was changed, not that the raycast
failed.

## Ambient

Crossfades over `AMBIENT_CROSSFADE_SECS` 2.5 between environments. Cave is `sky_light < 4`
(assembled in `app.rs`), underwater is player Y below sea level and inside water, otherwise a
biome surface bed.

## Adding sounds

1. Map the block in `SoundMaterial::from_block()`.
2. Add `assets/sfx/{footstep,break,place}/material-N.wav`.
3. `cargo run --bin generate_sounds` writes placeholders if real assets are not ready.

Missing files are handled without a crash, so a silent material means a missing or misnamed
file rather than an error.

## Unfinished

- Reverb is generated but not applied. The cave impulse response exists and `reverb_mix`
  interpolates toward 0.4 in caves, but nothing reads it into the signal path.
- `biome_id` is hardcoded to 0 in `app.rs`, so every surface ambient is the grasslands bed
  regardless of biome. Wiring it up needs the biome from world generation.
- Legacy `footstep-grass-*.wav` files are still loaded as a fallback.
