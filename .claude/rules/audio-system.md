# Audio System

Comprehensive 3D spatial audio with material-based sounds, ambient soundscapes, and environment effects.

## Architecture

| File | LOC | Purpose |
|------|-----|---------|
| `src/audio/mod.rs` | ~900 | AudioManager, SoundMaterial, ambient system |
| `src/bin/generate_sounds.rs` | ~300 | Procedural WAV generator utility |

## Sound Materials

Blocks are mapped to sound materials for footsteps, breaking, and placing:

| Material | Blocks | Sound Character |
|----------|--------|-----------------|
| Grass | Grass, Dirt, JungleGrass, Mud, Peat | Soft, organic |
| Stone | Stone, Brick, Sandstone, Ores (non-metal) | Hard, clicking |
| Sand | Sand | Granular, shuffling |
| Wood | Wood, Logs, Chest | Hollow, resonant |
| Snow | Snow, Ice, SnowGrass | Crunchy, compressed |
| Metal | Iron, Gold, Copper, Silver ores | Metallic clink |
| Gravel | Gravel | Loose, rocky |
| Foliage | Leaves, Vines, DeadBush | Rustling |
| Crystal | Diamond, Emerald, Amethyst ores | Crystalline |
| Generic | Fallback | Default sounds |

## Asset Structure

```
assets/sfx/
├── footstep/
│   ├── grass-1.wav, grass-2.wav
│   ├── stone-1.wav, stone-2.wav
│   ├── sand-1.wav, sand-2.wav
│   ├── wood-1.wav, wood-2.wav
│   ├── snow-1.wav, snow-2.wav
│   ├── metal-1.wav, metal-2.wav
│   ├── gravel-1.wav, gravel-2.wav
│   └── generic-1.wav, generic-2.wav
├── break/
│   ├── stone-1.wav, stone-2.wav
│   ├── wood-1.wav, wood-2.wav
│   ├── dirt-1.wav, dirt-2.wav
│   └── generic-1.wav, generic-2.wav
├── place/
│   └── (same structure as break)
└── ambient/
    ├── cave-drips.wav
    ├── surface-birds.wav
    ├── desert-wind.wav
    ├── jungle-insects.wav
    ├── tundra-wind.wav
    ├── swamp-frogs.wav
    ├── ocean-waves.wav
    └── underwater.wav
```

## Volume Settings

Three-tier volume system accessible from ESC menu:

| Setting | Default | Range | Persistence |
|---------|---------|-------|-------------|
| Master Volume | 80% | 0-100% | settings.cfg |
| SFX Volume | 100% | 0-100% | settings.cfg |
| Ambient Volume | 60% | 0-100% | settings.cfg |

Effective volume = master × category volume × base volume

## 3D Spatial Audio

Uses rodio's `SpatialSink` with DDA-based occlusion:

1. **Ear positions**: Calculated perpendicular to look direction (±0.1 blocks)
2. **Occlusion**: DDA raycast accumulates solid blocks (0.4 per block, max 70% reduction)
3. **Foliage/Water**: Light muffling (0.1 per block)

```rust
// Occlusion calculation in calculate_occlusion()
if block.is_solid() { occlusion += 0.4; }
else if block.is_foliage() || block == Water { occlusion += 0.1; }
```

## Ambient System

Environment-aware ambient soundscapes with smooth crossfading:

**Environment Detection:**
- `in_cave`: sky_light < 4
- `underwater`: Player Y below sea level AND inside water block
- `biome_id`: Determines surface ambient (0-7)

**Crossfade:** 2.5 second transitions between ambient tracks

**Biome Ambients:**
| Biome ID | Name | Ambient |
|----------|------|---------|
| 0 | Grasslands | Birds, wind |
| 2 | Desert | Wind, distant sounds |
| 3 | Tundra | Cold wind |
| 4 | Jungle | Insects, birds |
| 5 | Swamp | Frogs |
| 7 | Ocean | Waves |

## Reverb System (Infrastructure)

Cave impulse response generated at startup:
- 1.5 second reverb tail
- 8 early reflections (15-200ms)
- Diffuse late reverb with exponential decay
- Mix controlled by `reverb_mix` (0.0-0.4)

**Note:** Full convolution reverb infrastructure exists but audio routing not yet connected.

## Adding New Sounds

1. **Add material mapping** in `SoundMaterial::from_block()`:
   ```rust
   Block::NewBlock => SoundMaterial::Stone,
   ```

2. **Add sound files** following naming convention:
   ```
   assets/sfx/footstep/material-N.wav  (N = 1, 2, ...)
   assets/sfx/break/material-N.wav
   assets/sfx/place/material-N.wav
   ```

3. **Run generator for placeholders** (optional):
   ```bash
   cargo run --bin generate_sounds
   ```

## Generate Placeholder Sounds

```bash
cargo run --bin generate_sounds
```

Generates 32 procedural WAV files for all materials. Replace with production sounds.

## Key Functions

| Function | Purpose |
|----------|---------|
| `AudioManager::new()` | Init, load sounds, generate impulse |
| `update_footsteps()` | Material-aware footsteps |
| `play_block_break()` | 3D spatial break sound |
| `play_block_place()` | 3D spatial place sound |
| `update_ambient()` | Environment transitions |
| `update_reverb()` | Smooth cave reverb mix |

## Gotchas

- **Legacy sounds**: Old `footstep-grass-*.wav` files still loaded as fallback
- **Missing ambient**: No crash - system handles missing files gracefully
- **Reverb not connected**: Cave impulse generated but not applied to sources yet
- **Biome ID**: Currently hardcoded to 0 (grasslands) - needs world gen integration
