# Procedural Grass System

GPU-driven grass using "grass bomb" density, per-block placement, and vertex-pulling blade generation.

## Files

| File | Purpose |
|------|---------|
| `renderer/grass/mod.rs` | GrassSystem, tile buffer, settings (~1050 LOC) |
| `renderer/grass/grass_shader.wgsl` | Vertex-pulling, wind, LOD, lighting (~320 LOC) |
| `world/generation/vegetation.rs` | Grass bomb generation per chunk |

## Grass Bomb Concept

Natural clustering via overlapping density "bombs": 6-block grid spacing, 70% probability, 5-block radius with smoothstep falloff. Overlapping bombs = extra-dense patches.

## Settings (3 independent ESC menu controls)

| Setting | Controls | Values |
|---------|----------|--------|
| GRASS | Density threshold | OFF, LOW (5+), MEDIUM (3+), HIGH (2+), ULTRA (1+) |
| GRASS DISTANCE | Render range | NEAR (6), MEDIUM (12), FAR (18), ULTRA (26 chunks) |
| GRASS SHADOW | Shadow mode | OFF, SIMPLE |

## Gotchas

- **Quality toggle clears tile buffer** - forces full re-upload with new threshold
- **Distance toggle is instant** - shader reads from uniforms, no re-upload
- **Terrain sync** - Grass tiles generated during chunk generation, stored in `ChunkGrassData`
- **Lighting** - Uses same `DayNightCycle` uniforms as terrain for consistency
- **Single draw call** - All grass rendered via vertex pulling, no vertex buffer
