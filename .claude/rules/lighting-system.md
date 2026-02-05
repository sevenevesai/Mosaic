# Lighting System

BFS flood fill light propagation. 12 tests including cross-chunk propagation.

## Data Storage

Each chunk: `light: Box<[u8; 4096]>` - upper 4 bits = sky light (0-15), lower 4 bits = block light (0-15). Packed into vertex `ao_tile_packed.w` as 0-255.

## Files

- `world/lighting.rs` - `LightEngine` with BFS propagation/removal
- `world/chunk.rs` - `Block::light_emission()`, `Block::is_light_transparent()`
- `world/meshing.rs` - `get_face_light()` samples light for vertices
- `renderer/pipeline.rs` - Warm torch color with gamma: `pow(block_light, 2.2) * vec3(1.0, 0.7, 0.4)`

## Adding Light Sources

1. Add block variant in `Block` enum
2. Set emission in `Block::light_emission()` (torch=14, max=15)
3. Add to `Block::is_light_transparent()` so light passes through it

## Gotchas

- **Cross-chunk**: Propagation checks neighbor chunks at boundaries, triggers remesh
- **Removal**: BFS removal collects affected positions, then re-propagates from all nearby sources
- **Performance**: Torch radius 14 = ~11K blocks worst case. Batch multiple changes.
