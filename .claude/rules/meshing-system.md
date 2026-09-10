---
paths:
  - "src/world/meshing.rs"
  - "src/world/water_meshing.rs"
  - "src/renderer/chunk_pool.rs"
---

# Binary greedy meshing

Bitwise face culling and quad merging in `world/meshing.rs`, the most intricate module in the
tree (19 tests). Entry points: `build_chunk_mesh()`, `build_chunk_mesh_lod()`, and the private
`build_chunk_mesh_simple()` fallback.

Per axis: `visible = current & !neighbor` culls faces, `trailing_zeros()` finds horizontal
runs, one AND per row extends them vertically, then merged quads are emitted with UVs and AO.
Greedy meshing yields roughly 800-1500 quads per chunk against about 6000 for the simple path
— more CPU per chunk, far fewer draw primitives.

## Winding

Front faces are CCW. Quad corners are laid out `3--2 / 0--1` and emit triangles 0-2-1 and
0-3-2. Left and right faces are the exception: their UVs must be reordered to
`[uv0, uv3, uv2, uv1]`. To check a change, `(v1-v0) x (v2-v0)` has to point along the face
normal; if it does not, the face is culled away and reads as a hole in the terrain.

## LOD

`Renderer::lod_thresholds()` derives the pair each frame from the ESC menu LOD DISTANCE
setting: full detail from 0 to that distance, half detail out to
`lod + (render_distance - lod) / 2`, quarter beyond. `LOD_FULL_DETAIL` 12 and
`LOD_HALF_DETAIL` 22 in `src/lib.rs` are only the defaults; OFF passes `i32::MAX` so
everything meshes at full detail. Sampling is every block, every second block, every fourth.

## Gotchas

- `ChunkVertex` is 24 bytes: position 12, packed normal 4, UV 4, `ao_tile_packed` 4. That last
  word is `[AO, tile_u, tile_v, light_level]` — the struct comment calling the fourth byte
  unused is wrong; the shader reads it as `block_light`.
- Merged quads tile with `fract(uv * tile_size)` and inset by half a texel, which is what
  keeps neighbouring atlas tiles from bleeding into a merged face.
- Foliage, cross billboards and props are excluded from the greedy pass and emitted
  separately by `add_foliage_blocks()`, which draws all six faces of each block. The
  neighbour tests exclude the same three, so a stone face touching leaves survives culling;
  drop that check and terrain behind foliage turns transparent.
- Mesh statistics go through the atomic `MESH_STATS` static, since meshing runs on the worker
  pool.
- To see backfaces while debugging, set `cull_mode: None` in `pipeline.rs`.
