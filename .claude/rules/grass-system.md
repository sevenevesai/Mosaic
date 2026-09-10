---
paths:
  - "src/renderer/grass/**"
  - "src/world/generation/vegetation.rs"
---

# Procedural grass

GPU-driven: one draw call, blades built by vertex pulling in `grass_shader.wgsl` from a tile
buffer; no vertex buffer exists. Placement comes from chunk generation in `vegetation.rs` and
is stored in `ChunkGrassData`.

## Grass bombs

Density is the sum of overlapping "bombs" on a grid, which is what makes clumps look natural
rather than uniform: `bomb_spacing` 6 blocks, `bomb_probability` 0.7 per cell,
`bomb_radius` 5.0 blocks with a smoothstep falloff, and the centre jittered up to 0.8 of a
cell. Two overlapping bombs give an extra-dense patch. All of it is hashed from position, so
regeneration is deterministic.

## Settings

Three independent controls in the ESC menu.

| Setting | Controls | Values |
|---|---|---|
| GRASS | blade-count threshold | OFF, LOW, MEDIUM, HIGH, ULTRA |
| GRASS DISTANCE | render range | NEAR, MEDIUM, FAR, ULTRA |
| GRASS SHADOW | shadow mode | OFF, SIMPLE |

`GrassDistance` has three tiers per preset — full detail / low LOD / cull, in chunks:
NEAR 3/5/6, MEDIUM 6/10/12, FAR 10/16/18, ULTRA 14/22/26.

## Gotchas

- Changing quality clears the tile buffer and forces a full re-upload, because the threshold
  is baked in at upload time. Changing distance is instant: the shader reads it from uniforms.
- Grass tiles are produced during chunk generation. A change to placement rules only shows on
  chunks generated afterwards; delete the save to see it everywhere.
- Grass reads the same `DayNightCycle` uniforms as terrain. Lighting that diverges between
  the two means one of them was updated alone.
