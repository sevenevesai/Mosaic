# Binary Greedy Meshing System

Pure bitwise operations for face culling and quad merging. Most complex module (~2,619 LOC, 22 tests).

## Algorithm Overview

1. **Face Culling**: For each axis, determine visible faces via `visible = current & !neighbor`
2. **Run Detection**: Find horizontal runs using `trailing_zeros()`
3. **Vertical Extension**: Extend runs vertically via single AND per row
4. **Quad Emission**: Output merged quads with proper UVs and AO

## Key Files

| File | Function | Purpose |
|------|----------|---------|
| `world/meshing.rs` | `build_chunk_mesh()` | Entry point, calls greedy |
| `world/meshing.rs` | `build_chunk_mesh_lod()` | LOD-aware greedy meshing |
| `world/meshing.rs` | `build_chunk_mesh_simple()` | Non-greedy fallback |
| `world/meshing.rs` | `add_quad()` | Emits vertices for merged quad |

## Vertex Format

```rust
#[repr(C)]
pub struct ChunkVertex {
    position: [f32; 3],      // 12 bytes
    normal_packed: u32,      // 4 bytes - face direction encoded
    uv: [f32; 2],            // 8 bytes - REMOVED, now in ao_tile_packed
    ao_tile_packed: u32,     // 4 bytes - [ao, tile_u, tile_v, light]
}
// Total: 24 bytes (was 36 with separate UVs)
```

**Packing scheme for ao_tile_packed:**
- Byte 0: AO value (0-255)
- Byte 1: Tile U coordinate
- Byte 2: Tile V coordinate
- Byte 3: Light level (0-255, scaled from 0-15)

## Winding Order

**Critical for backface culling with CCW front faces.**

Quad vertex layout:
```
   3----2      Triangles: 0-2-1 and 0-3-2
   |    |      This produces CCW winding when viewed from outside
   0----1
```

**Left/Right Face Exception:**
These faces have vertices in different order. UVs must be reordered to match:
```rust
// Normal faces: [uv0, uv1, uv2, uv3]
// Left/Right:   [uv0, uv3, uv2, uv1]
```

To verify winding: compute `(v1-v0) × (v2-v0)` - result should point same direction as face normal.

## LOD System

Three detail levels based on distance from player:

| LOD | Distance | Block Sampling |
|-----|----------|----------------|
| Full | < 8 chunks | Every block |
| Half | 8-16 chunks | Every 2nd block |
| Quarter | 16-24 chunks | Every 4th block |

LOD affects greedy mesh generation - coarser sampling = larger quads = fewer vertices.

## AO Calculation

Per-vertex ambient occlusion based on neighboring blocks:

```rust
fn calculate_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 { 0 }      // Fully occluded
    else { 3 - (side1 as u8 + side2 as u8 + corner as u8) }
}
```

Values: 0 (darkest) to 3 (brightest), scaled to 0-255 for vertex.

## Texture Tiling

For merged quads larger than 1x1, texture must tile:
- `tile_u` and `tile_v` in `ao_tile_packed` store quad dimensions
- Shader uses `fract(uv * tile_size)` for tiling
- Atlas anti-bleeding: half-texel inset at edges

## Thread Safety

```rust
static MESH_STATS: AtomicMeshStats = ...;
```

Mesh generation happens on worker threads. Statistics use atomics for thread-safe accumulation.

## Performance Characteristics

| Metric | Simple Meshing | Greedy Meshing |
|--------|----------------|----------------|
| Quads per chunk | ~6000 | ~800-1500 |
| Vertices | ~24000 | ~3200-6000 |
| Generation time | ~1ms | ~2-3ms |
| GPU draw time | Higher | Lower |

Greedy meshing trades CPU time for GPU efficiency. Net win for rendering.

## Debugging

1. **Missing faces**: Check winding order, verify `visible = current & !neighbor`
2. **Z-fighting**: Check vertex positions, ensure no duplicate quads
3. **Wrong textures**: Check `Block::texture_indices()` mapping
4. **AO artifacts**: Check neighbor sampling, ensure chunk boundaries handled

To disable backface culling for debugging:
```rust
// In pipeline.rs
cull_mode: None,  // Was: Some(wgpu::Face::Back)
```
