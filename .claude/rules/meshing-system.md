# Binary Greedy Meshing System

Pure bitwise face culling and quad merging. Most complex module (~2,619 LOC, 22 tests).

## Algorithm

1. Face culling: `visible = current & !neighbor` per axis
2. Run detection: `trailing_zeros()` for horizontal runs
3. Vertical extension: single AND per row
4. Quad emission: merged quads with proper UVs and AO

Entry: `build_chunk_mesh()` / `build_chunk_mesh_lod()` / `build_chunk_mesh_simple()` in `world/meshing.rs`.

## Winding Order (Critical)

CCW front faces. Quad layout: `3--2 / | | / 0--1`. Triangles: 0-2-1 and 0-3-2.

**Left/Right Face Exception**: UVs must be reordered: `[uv0, uv3, uv2, uv1]` instead of `[uv0, uv1, uv2, uv3]`. Verify: `(v1-v0) x (v2-v0)` should match face normal.

## LOD System

Configurable via ESC menu -> LOD DISTANCE (OFF, 6-32):

| LOD | Block Sampling |
|-----|----------------|
| Full | Every block (0 to LOD setting) |
| Half | Every 2nd block (LOD to midpoint) |
| Quarter | Every 4th block (beyond midpoint) |

Dynamic: `Half = LOD + (render_distance - LOD) / 2`

## Vertex Format

`ChunkVertex`: 24 bytes. `ao_tile_packed` byte layout: [AO, tile_u, tile_v, light_level].

## Gotchas

- **Texture tiling**: Merged quads use `fract(uv * tile_size)`. Atlas anti-bleed via half-texel inset.
- **Thread safety**: Mesh stats use atomics (`AtomicMeshStats`)
- **Debug backface**: Set `cull_mode: None` in pipeline.rs
- **Performance**: Greedy = ~800-1500 quads/chunk vs ~6000 simple. Trades CPU for GPU efficiency.
