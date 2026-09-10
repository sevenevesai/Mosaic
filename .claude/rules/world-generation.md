---
paths:
  - "src/world/generation/**"
---

# World generation

`TerrainGenerator` in `world/generation/` produces height, caves, ores, biomes, trees, ruins
and formations. Generation runs on the chunk worker pool and must stay deterministic from
seed and position: it is re-run to rebuild chunks that were never saved, so a nondeterministic
generator makes terrain change under the player.

## Biomes

`BiomeId` has eight variants and all eight are registered in `BiomeId` order in `biome.rs`.
`select_biome()` is Whittaker-style on temperature and humidity, with the shares its own
comments record: Tundra about 15% below -0.3 temperature, Jungle about 20% above 0.3
temperature with humidity above 0.2, Cavernous Highlands about 15% above 0.4 temperature,
Grasslands about 50% otherwise.

Desert, Swamp, Mountains and Ocean are implemented and registered, and `formations.rs`
branches on them, but `select_biome()` never returns them — they are unreachable through
normal generation today. Treat them as unfinished, not as dead code to delete.

Grasslands keeps caves underground; Cavernous Highlands opens caverns at the surface.

## Gotchas

- Tree placement has to go through `is_ground_stable_for_tree()`. Skipping it produces trees
  floating over caves and overhangs.
- Ores come from `get_ore_or_stone_fast()`, noise clustering plus a hash slot system. It is on
  the hot path for every stone block, so changes there move generation time directly.
- Mountains are decided twice and the two do not agree. `is_mountain_region()`
  (`mountain_noise` at frequency 0.001, threshold 0.4) is computed and handed to
  `select_biome()`, which ignores it as `_is_mountain`; relief comes from
  `get_mountain_mask()` (`height_noise` at 0.001, ramping above 0.25) in `mod.rs`. Mountain
  terrain therefore appears inside other biomes.
- Grass placement lives in `vegetation.rs` and is documented in `grass-system.md`.
