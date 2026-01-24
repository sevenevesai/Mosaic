# Blocks Engine - Claude Session Guide

A modern voxel engine built in Rust + wgpu, designed for scalability testing and learning from Minecraft's design.

## Quick Start

```bash
cargo build --release
./target/release/blocks.exe
```

**Controls:** Click to capture mouse, WASD move, Space jump/fly-up, Shift fly-down, Ctrl sprint, F toggle fly mode, F3 debug overlay, 1-7 select block, Left-click break, Right-click place, P log performance, Escape release mouse.

---

## Architecture Overview

```
src/
├── main.rs              # Entry point, logging setup
├── lib.rs               # Module exports, constants (RENDER_DISTANCE, LOD, physics)
├── app.rs               # Main loop, window events, input handling
├── renderer/
│   ├── mod.rs           # GPU state, render pass, frustum culling
│   ├── camera.rs        # FPS camera, Frustum for culling
│   ├── pipeline.rs      # WGSL shaders (packed vertex, lighting, fog, texture tiling)
│   ├── texture.rs       # Procedural texture atlas (16x16 tiles)
│   ├── debug_overlay.rs # F3 overlay with bitmap font rendering
│   └── chunk_pool.rs    # MDI mega-buffers (disabled, needs work)
├── world/
│   ├── mod.rs           # WorldManager, chunk workers, LOD, raycast, set_block
│   ├── chunk.rs         # Chunk data (16³), Block enum (15 types)
│   ├── generation.rs    # Noise terrain, caves, ores, biomes
│   └── meshing.rs       # Greedy meshing, packed ChunkVertex (24 bytes)
├── player/
│   └── mod.rs           # Player state, AABB collision, block interaction
└── utils/
    ├── profiler.rs      # CPU frame timing, section profiling
    └── gpu_profiler.rs  # GPU timestamp queries (shows 0.00ms)
```

---

## Current State

### Working
- Procedural terrain with caves, ores, mountains
- Chunk streaming with 8 worker threads
- **Greedy meshing** with proper texture tiling (2.5x triangle reduction)
- **Packed vertex format** (24 bytes vs 36 = 33% GPU memory reduction)
- **LOD system** - Distance-based mesh detail (Full/Half/Quarter)
- Frustum culling (skips off-screen chunks)
- Low-latency rendering (Mailbox present mode)
- FPS camera with mouse look, fly mode (F key)
- Block placement/breaking with raycast
- AABB collision detection
- F3 debug overlay (on-screen stats)
- Performance logging to `performance_log.txt`

### Known Issues
- **GPU profiler readback** - Shows 0.00ms due to wgpu 0.19 async buffer mapping complexity
- **MDI disabled** - Multi-draw indirect code exists in `renderer/chunk_pool.rs` but causes visual bugs (missing chunks). Needs debugging before enabling.

### Performance (RTX 2080 Ti, render distance 24)
- ~550-600 FPS with greedy meshing + LOD
- ~570K triangles, ~2.3K chunks rendered
- ~3.7M vertices in GPU memory

---

## Key Constants (src/lib.rs)

```rust
CHUNK_SIZE: 16           // Blocks per chunk dimension
RENDER_DISTANCE: 24      // Chunks to load around player
LOD_FULL_DETAIL: 8       // Full mesh within 8 chunks
LOD_HALF_DETAIL: 16      // Half detail 8-16 chunks
LOD_QUARTER_DETAIL: 24   // Quarter detail 16-24 chunks
```

---

## Development Workflow

### Build & Run
```bash
cargo build --release
./target/release/blocks.exe
```

### Quick Iteration
```bash
taskkill //F //IM blocks.exe 2>&1; cargo build --release && ./target/release/blocks.exe
```

### Check for Errors
```bash
cargo check 2>&1 | head -40
```

---

## Common Tasks

### Adding a New Block Type
1. Add variant to `Block` enum in `src/world/chunk.rs`
2. Add texture indices in `Block::texture_indices()`
3. Add generation rules in `src/world/generation.rs`
4. Add color in `TextureAtlas::generate_tile()` in `src/renderer/texture.rs`

### Modifying Terrain Generation
Edit `src/world/generation.rs` - `TerrainGenerator` struct handles height, caves, ores.

### Changing Render Distance
Edit `RENDER_DISTANCE` in `src/lib.rs`. Higher = more chunks = more GPU/CPU load.

### Debugging Mesh Issues
- Check `src/world/meshing.rs` - greedy meshing in `build_chunk_mesh_lod()`, simple in `build_chunk_mesh_simple()`
- Current: `build_chunk_mesh()` calls greedy meshing with Full LOD
- Packed vertex: position (12B) + normal_packed (4B) + uv (4B) + ao_tile_packed (4B) = 24 bytes
- Texture tiling: `ao_tile_packed` contains [ao, tile_u, tile_v, 0] for proper tiling on merged quads
- To test winding: set `cull_mode: None` in `pipeline.rs`

---

## Winding Order Reference

For backface culling with CCW front faces (`wgpu::FrontFace::Ccw`), the triangle's computed normal must point toward the camera.

**Current implementation** in `add_quad()`:
```
   3----2      Triangles: 0-2-1 and 0-3-2 (reversed from naive order)
   |    |      This produces CCW winding when viewed from outside
   0----1
```

**Important:** Left and right faces (`add_face_left`, `add_face_right`) have vertices in a different order than other faces, with UVs reordered to match: `[uv0, uv3, uv2, uv1]`. This was necessary to achieve consistent CCW winding across all 6 faces.

To verify winding: compute `(v1-v0) × (v2-v0)` - result should point in same direction as face normal.

---

## Dependencies

Key crates:
- `wgpu 0.19` - Graphics (Vulkan/DX12/Metal backend)
- `winit 0.29` - Windowing
- `noise` - Terrain generation
- `rayon` - Parallel iteration
- `crossbeam-channel` - Thread communication
- `parking_lot` - Fast mutexes
- `glam` - Math (vectors, matrices)

---

## Session Handoff Checklist

When starting a new session:
1. Read this file first
2. Check `ROADMAP.md` for priorities
3. Run `cargo check` to verify build state
4. Test current build before making changes
5. Focus on one task at a time
6. Update this doc if architecture changes significantly
