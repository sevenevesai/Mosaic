# Procedural Grass Rendering System

GPU-driven grass using "grass bomb" density, per-block placement, and vertex-pulling blade generation.

## Architecture

| File | LOC | Purpose |
|------|-----|---------|
| `src/renderer/grass/mod.rs` | ~1050 | GrassSystem, tile buffer, quality/distance/shadow enums |
| `src/renderer/grass/grass_shader.wgsl` | ~320 | Vertex-pulling blade generation, wind, LOD, day/night lighting |
| `src/world/generation/vegetation.rs` | +150 | Grass bomb generation in `generate_grass_tiles()` |
| `src/world/chunk.rs` | +30 | ChunkGrassData storage |

## Grass Bomb System

Grass density uses a "bomb" pattern for natural clustering:

1. **Bomb Grid**: 6-block spacing, 70% probability per cell
2. **Jittered Position**: Random offset within cell for organic placement
3. **Radial Falloff**: Smoothstep from center (5 block radius)
4. **Overlapping Bombs**: Densities add for extra-dense intersections
5. **Fine Noise**: Subtle detail variation layered on top

```
Density = sum(smoothstep_falloff(dist_to_bomb) * bomb_strength) + detail_noise
```

## Data Structures

**GrassTile** (16 bytes - GPU upload):
```rust
position: [f32; 2],    // World XZ of block center
density_seed: u32,     // blade_count(8) | height_var(8) | seed(16)
biome_height: u32,     // biome_id(8) | surface_y fixed-point(24)
```

**Per-block placement**: Each grass-capable block gets a tile with exact Y height.

## Settings

Three independent settings in ESC menu:

| Setting | Controls | Values |
|---------|----------|--------|
| **GRASS** | Density (min blade threshold) | OFF, LOW (5+), MEDIUM (3+), HIGH (2+), ULTRA (1+) |
| **GRASS DISTANCE** | Render range | NEAR (6), MEDIUM (12), FAR (18), ULTRA (26 chunks) |
| **GRASS SHADOW** | Shadow mode | OFF, SIMPLE |

Quality affects which tiles are shown via shader threshold. Distance affects cull range.

## Wind Animation

Multi-octave wind system for natural movement:

1. **Per-blade phase offset**: Each blade has unique timing via hash(seed)
2. **Gust variation**: Slow-moving regions of stronger/weaker wind
3. **Multi-octave waves**:
   - Primary: slow, large-scale swaying
   - Secondary: medium turbulence
   - Tertiary: fast flutter (small amplitude)
4. **Traveling waves**: Wind waves move across landscape, not synchronized

```wgsl
wind1 = sin(phase + time * 1.2 + blade_offset) * 0.6    // Primary
wind2 = sin(phase * 2.3 + time * 2.0 + offset) * 0.25  // Secondary
wind3 = sin(phase * 4.1 + time * 3.5 + offset) * 0.15  // Flutter
wind_wave = (wind1 + wind2 + wind3) * gust_factor
```

## Day/Night Lighting

Grass responds to scene lighting via uniforms:

- **Sun direction + intensity**: Directional lighting
- **Sun color**: Warm at dawn/dusk, white at noon
- **Ambient color**: Base illumination, varies with time
- **Face brightness**: Upward-facing parts brighter than sides

Night: grass darkens with environment, no "glow" effect.

## Biome Colors

8-color palette matching terrain grass surface (#2F441A):

| Biome | Base Color | Tip Color |
|-------|------------|-----------|
| Grasslands | (0.16, 0.23, 0.08) | (0.184, 0.267, 0.102) |
| Highlands | Similar | Slight variation |
| Jungle | Darker | Deep green |
| Tundra | Tan/brown | Dead grass |
| Swamp | Murky | Dark olive |
| Desert | Sandy | Sparse |
| Mountains | Gray-green | Alpine |

## Shader Pipeline

1. **Vertex Pulling**: No vertex buffer - blade shape from `instance_id` + `vertex_id`
2. **Density Threshold**: Skip tiles where `blade_count < min_threshold` (quality-based)
3. **Blade Count Check**: Skip blades beyond tile's actual count
4. **Distance Culling**: Degenerate triangles beyond cull distance
5. **Bezier Curves**: Blade shape via quadratic Bezier
6. **Wind Animation**: Multi-octave with per-blade variation
7. **Day/Night Lighting**: Scene-aware ambient + directional
8. **Distance LOD**: Wider blades at distance for anti-aliasing

## Render Integration

**Per-frame in `render()`**:
1. Collect visible grass chunks within cull distance
2. Upload new chunk tiles to GPU buffer
3. Remove tiles from chunks outside cull distance (`retain_chunks`)
4. Update uniforms (view_proj, camera_pos, wind_time, LOD distances, **sun/ambient**)
5. Single draw call: `draw(0..verts_per_blade, 0..total_blades)`

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `generate_grass_tiles()` | vegetation.rs | Grass bomb generation per chunk |
| `GrassSystem::update()` | grass/mod.rs | Per-frame uniform updates (incl. lighting) |
| `GrassSystem::render()` | grass/mod.rs | Draw call |
| `tile_buffer.upload_chunk()` | grass/mod.rs | GPU buffer management |
| `tile_buffer.retain_chunks()` | grass/mod.rs | Remove distant chunks |

## Gotchas

- **Quality toggle**: Clears tile buffer to force re-upload with new threshold
- **Distance toggle**: Takes effect immediately (shader reads from uniforms)
- **Per-block tiles**: More tiles than old 4x4 system, but noise skips ~30% of blocks
- **Blade count**: Stored in tile, capped by quality's `blades_per_tile`
- **Terrain sync**: Grass tiles generated during chunk generation, stored in chunk
- **Lighting**: Grass uses same DayNightCycle as terrain for consistency

## Performance

- ~70% of grass blocks have tiles (noise threshold)
- 1-8 blades per block based on bomb density
- Single draw call for all grass (vertex pulling)
- Distance culling in shader (degenerate triangles)
- Max tiles: 64K-512K based on quality setting
