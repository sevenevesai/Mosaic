# Virtual Shadow Maps (VSM) - Status and Issues

## Current Status: DISABLED
VSM is currently disabled in `renderer/mod.rs:use_vsm = false` due to performance issues (no per-tile culling) and incomplete debugging.

## What's Implemented

### Infrastructure (Working)
- `VsmRenderer` - Main VSM interface
- `TileCache` - LRU cache for physical tiles
- `PageTable` - Virtual-to-physical page mapping
- `Clipmap` - Multi-resolution shadow coverage (8 levels)
- Shader code for page table lookup and shadow sampling

### Tile Rendering Loop (Partially Working)
- Per-tile uniform buffers and bind groups
- Viewport/scissor per tile
- Render pass to tile atlas

## Session Findings (2026-01-26)

### Issues Fixed
1. **Level selection mismatch** - Shader and Rust now use same threshold formula (0.9 * half_size)
2. **Page gathering** - Now requests ALL pages in range, not just corners
3. **Multi-level requests** - Now requests tiles for ALL levels that could be sampled (min to max distance)
4. **Texel snapping** - Fixed `units_per_texel` calculation (was dividing by pages_per_dim twice)
5. **Tile budget** - Increased `max_tiles_per_frame` from 128 to 512

### Issues Remaining

#### 1. Performance (Critical)
**Problem**: Tile rendering loops over ALL visible chunks for EVERY tile, causing O(tiles × chunks) draw calls.

**Current code** (renderer/mod.rs ~line 1100):
```rust
for (chunk_mesh, _chunk_min, _chunk_max) in &all_shadow_chunks {
    // Renders ALL chunks to EVERY tile - no culling!
}
```

**Fix needed**: Re-enable per-tile culling using `VsmRenderer::chunk_intersects_tile()`:
```rust
if !VsmRenderer::chunk_intersects_tile(chunk_min, chunk_max, job) {
    continue;  // Skip chunks outside tile's shadow volume
}
```

#### 2. Shadow Sampling (Unknown)
The depth comparison may still have issues. With per-tile culling fixed, need to verify:
- Depth values are being written correctly
- Depth comparison in shader produces correct results
- Page table lookups return valid entries

### Debug Observations
- VSM shader IS being executed (FPS drops significantly when enabled)
- Tile cache shows 100% hit rate after warmup
- 6000+ tiles requested but only 512 processed per frame
- With shadow=0.7 for invalid pages, no visible darkening observed

## Architecture Overview

```
World Space → Level View-Proj → NDC [-1,1] → Page Table Lookup → Physical Tile UV → Sample

Tile Rendering:
  For each tile in render_queue:
    1. Set viewport to tile's atlas position
    2. Set per-tile view-proj (scales level's VP to tile region)
    3. Render geometry (NEEDS CULLING!)
    4. Update page table entry

Shader Sampling:
  1. Transform world pos using level's view-proj → virtual UV
  2. Calculate page coordinate from virtual UV
  3. Look up page table → physical tile coordinates
  4. Compute in-tile UV, sample from atlas
  5. Compare depth with comparison sampler
```

## Recommended Next Steps

1. **Fix per-tile culling** - Re-enable `chunk_intersects_tile()` check in tile render loop
2. **Test with culling** - Should have acceptable performance with proper culling
3. **Debug depth values** - Add RenderDoc capture to verify tile depth content
4. **Verify page table** - Add debug output showing page table entries being set/read

## Files Modified This Session

| File | Changes |
|------|---------|
| `vsm/clipmap.rs` | Fixed `gather_needed_pages` to request multi-level, all pages in range; fixed `units_per_texel` |
| `vsm/shaders.rs` | Fixed level selection threshold; added debug returns (now partially reverted) |
| `vsm/mod.rs` | Increased `max_tiles_per_frame` to 512 |
| `renderer/mod.rs` | Added debug logging for VSM stats |

## Testing Checklist

- [ ] Per-tile culling re-enabled and working
- [ ] Performance acceptable (< 5ms additional) with culling
- [ ] Tiles render correctly (verify via RenderDoc)
- [ ] Page tables populated for all visible geometry
- [ ] Shadows appear during day
- [ ] No crashes during night transition
