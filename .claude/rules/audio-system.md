# Audio System

3D spatial audio with material-based sounds, ambient soundscapes, and DDA-based occlusion.

## Files

| File | Purpose |
|------|---------|
| `src/audio/mod.rs` | AudioManager, SoundMaterial, ambient system (~900 LOC) |
| `src/bin/generate_sounds.rs` | Procedural WAV generator utility |

## Sound Materials

10 materials (Grass, Stone, Sand, Wood, Snow, Metal, Gravel, Foliage, Crystal, Generic). Mapped via `SoundMaterial::from_block()`. Assets in `assets/sfx/{footstep,break,place,ambient}/`.

## Volume System

Three-tier: Master (80%) x SFX (100%) or Ambient (60%). All in ESC menu, persisted to settings.cfg.

## Spatial Audio & Occlusion

Uses rodio `SpatialSink`. DDA raycast accumulates occlusion: +0.4 per solid block, +0.1 per foliage/water, max 70% reduction.

## Ambient System

Environment-aware crossfading (2.5s transitions):
- Cave detection: `sky_light < 4`
- Underwater: player Y below sea level AND inside water
- Biome-specific surface ambients (birds, wind, insects, etc.)

## Adding New Sounds

1. Add mapping in `SoundMaterial::from_block()`
2. Add WAV files: `assets/sfx/{footstep,break,place}/material-N.wav`
3. Optional: `cargo run --bin generate_sounds` for placeholders

## Gotchas

- **Reverb not connected**: Cave impulse response generated but not applied to audio sources yet
- **Biome ID hardcoded**: Currently always 0 (grasslands) - needs world gen integration
- **Legacy fallback**: Old `footstep-grass-*.wav` files still loaded as fallback
- **Missing files**: System handles gracefully, no crash
