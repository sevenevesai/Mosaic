---
paths:
  - "src/world/block_registry.rs"
  - "src/renderer/texture.rs"
  - "src/world/chunk.rs"
---

# Blocks and the texture registry

`Block` in `world/chunk.rs` is the enum of block types and owns their IDs. `From<u8>` must
round-trip every variant: saved chunks store the raw discriminant, so a reused or renumbered
ID silently rewrites existing worlds rather than failing to load.

`BlockRegistry` (`world/block_registry.rs`) is the single source of truth for textures. It
scans `assets/blocks/*.png` at startup, sorts the names, and assigns atlas slots in that
order, then appends the procedural slots `_torch` and `_water` (a leading underscore marks
procedural). `build_block_textures()` maps each block ID to six texture names, so the mapping
survives files being added or renamed; `Block::texture_indices()` just forwards to it.
`init_registry()` must run early in `App::new()`, before any rendering or world generation
reads it.

The atlas is `TILES_PER_ROW` 16 (256 slots), declared in both `src/lib.rs` and
`block_registry.rs`, and hardcoded as 16 in the chunk, item and inventory shaders. All of
them describe one grid and change together.

## Adding a block type

1. Add the variant with its ID to the `Block` enum in `world/chunk.rs`.
2. Add it to the `From<u8>` impl.
3. Drop the texture PNG into `assets/blocks/` and name it in `build_block_textures()`. There
   is no hardcoded file list to edit.
4. Foliage goes in `is_foliage()`, light emitters in `light_emission()`, and anything light
   should pass through in `is_light_transparent()`.

## Gotchas

- An unknown texture name resolves to index 0 with a warning rather than an error, so a typo
  shows up as the wrong texture, not a failure.
- Adding PNGs to `assets/blocks/` shifts every later slot, since slots are assigned by sorted
  filename. That is safe only because lookups go through names; index-based assumptions
  written anywhere else will break.
- `find_texture_dir()` tries `assets/blocks` under the working directory, then beside the
  executable, then three levels above it, so the game finds textures whether it is run from
  the repo root or from `target/release`.
