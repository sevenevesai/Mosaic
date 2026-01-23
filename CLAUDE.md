# Blocks Engine - Claude Session Guide

A modern voxel engine built in Rust + wgpu, designed for scalability testing and learning from Minecraft's design.

## Quick Start

```bash
cargo build --release
./target/release/blocks.exe
```

**Controls:** Click to capture mouse, WASD move, Space/Shift up/down, Ctrl sprint, F3 stats, Escape release mouse.

---

## Architecture Overview

```
src/
├── main.rs              # Entry point, logging setup
├── lib.rs               # Module exports, constants (RENDER_DISTANCE, LOD levels, CHUNK_SIZE)
├── app.rs               # Main loop, window events, input handling
├── renderer/
│   ├── mod.rs           # GPU state, render pass, frustum culling
│   ├── camera.rs        # FPS camera, Frustum for culling
│   ├── pipeline.rs      # WGSL shaders (lighting, fog, AO)
│   └── texture.rs       # Procedural texture atlas (16x16 tiles)
├── world/
│   ├── mod.rs           # WorldManager, chunk worker threads, LOD
│   ├── chunk.rs         # Chunk data (16³), Block enum (15 types)
│   ├── generation.rs    # Noise terrain, caves, ores, biomes
│   └── meshing.rs       # Face meshing, greedy meshing (WIP)
├── player/
│   └── mod.rs           # Player state, movement, input
└── utils/
    ├── profiler.rs      # CPU frame timing, section profiling
    └── gpu_profiler.rs  # GPU timestamp queries (partially working)
```

---

## Current State

### Working
- Procedural terrain with caves, ores, mountains
- Chunk streaming with 8 worker threads
- Simple face-culled meshing with correct winding order (all 6 faces render properly)
- Frustum culling (skips off-screen chunks)
- Low-latency rendering (Mailbox present mode)
- FPS camera with mouse look
- Back-face culling enabled (CCW front faces)

### Known Issues
- **Greedy meshing disabled** - `build_chunk_mesh_lod()` exists but has winding bugs; using `build_chunk_mesh()` instead
- **GPU profiler readback** - Disabled due to wgpu 0.19 async buffer mapping complexity
- **LOD system** - Infrastructure exists but currently always returns Full LOD
- **F3 overlay** - Logs to console only, no on-screen display

### Performance Baseline
- Render distance: 24 chunks
- ~8 worker threads for generation/meshing
- Frustum culling active
- Simple meshing (more triangles than greedy)

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
- Check `src/world/meshing.rs` - face functions `add_face_*`
- To test if faces are being generated: temporarily set `cull_mode: None` in `pipeline.rs:112`
- If faces appear with culling off but not on: winding order is wrong
- If faces are missing even with culling off: visibility check (`is_face_visible`) is wrong
- World manager calls `build_chunk_mesh()` (simple) not `build_chunk_mesh_lod()` (greedy)

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
