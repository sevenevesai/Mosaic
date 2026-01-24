# Neolith Engine - Claude Session Guide

A high-performance voxel engine built in Rust + wgpu, designed for scalability from the metal.

## Quick Start

```bash
cargo build --release
./target/release/neolith.exe
```

**Controls:** Click to capture mouse, WASD move, Space jump/fly-up, Shift fly-down, Ctrl sprint, F toggle fly mode, F3 debug overlay, T cycle day/night speed, 1-7 select block, 8 torch, Left-click break, Right-click place, P log performance, Escape release mouse.

---

## Architecture Overview

```
src/
├── main.rs              # Entry point, logging setup
├── lib.rs               # Module exports, constants (RENDER_DISTANCE, LOD, physics)
├── app.rs               # Main loop, window events, input handling
├── audio/
│   └── mod.rs           # AudioManager, footstep sounds (rodio)
├── renderer/
│   ├── mod.rs           # GPU state, render pass, frustum culling, shadow pass, day/night cycle
│   ├── camera.rs        # FPS camera, Frustum for culling
│   ├── pipeline.rs      # WGSL shaders (packed vertex, lighting, fog, shadows, sky)
│   ├── texture.rs       # Texture atlas from PNG files (16 tiles)
│   ├── debug_overlay.rs # F3 overlay with bitmap font rendering
│   └── chunk_pool.rs    # MDI mega-buffers for multi-draw indirect rendering
├── world/
│   ├── mod.rs           # WorldManager, chunk workers, LOD, raycast, set_block
│   ├── chunk.rs         # Chunk data (16³), Block enum (16 types), light storage
│   ├── generation.rs    # Noise terrain, caves, ores, biomes, trees
│   ├── lighting.rs      # BFS flood fill light propagation system
│   ├── meshing.rs       # Binary greedy meshing, packed ChunkVertex (24 bytes)
│   └── storage/         # World persistence system
│       ├── mod.rs       # WorldStorage API, LRU region cache
│       ├── region.rs    # Region files (32³ chunks, 4KB sectors)
│       ├── chunk_io.rs  # Chunk serialization + zstd compression
│       ├── world_meta.rs# World metadata (seed, spawn, timestamps)
│       └── dirty_tracker.rs # Tracks modified chunks for saving
├── player/
│   └── mod.rs           # Player state, AABB collision, block interaction
└── utils/
    ├── profiler.rs      # CPU frame timing, section profiling
    └── gpu_profiler.rs  # GPU timestamp queries (shows 0.00ms)
```

---

## Current State

### Working
- Procedural terrain with caves, ores, mountains, trees
- Chunk streaming with 8 worker threads
- **Binary greedy meshing** - Pure bitwise operations for face culling and quad merging
- **Packed vertex format** (24 bytes vs 36 = 33% GPU memory reduction)
- **LOD system** - Distance-based mesh detail (Full/Half/Quarter)
- **BFS flood fill lighting** - Minecraft-style block light propagation
- **Shadow mapping** with frustum-culled shadow pass
- **Multi-draw indirect (MDI)** - Single GPU draw call for all pooled chunks
- **Day/night cycle** - Warm color palette, smooth sun/moon transitions (T key to adjust speed)
- **Chunk fade-in animation** - Smooth alpha fade when chunks load
- Frustum culling (skips off-screen chunks)
- Low-latency rendering (Mailbox present mode)
- FPS camera with mouse look, fly mode (F key)
- Block placement/breaking with raycast
- **Torch block** (key 8) - Light source with 14-block radius warm glow
- AABB collision detection
- F3 debug overlay (on-screen stats + light levels + shadow stats)
- Performance logging to `performance_log.txt`
- **World persistence** - Region-based save/load with zstd compression
- **Player position saving** - Spawn at last position on reload
- **Audio system** - Footstep sounds (rodio)

### Known Issues
- **GPU profiler readback** - Shows 0.00ms due to wgpu 0.19 async buffer mapping complexity

### Performance (RTX 2080 Ti, render distance 24)
- ~300-450 FPS with shadows + greedy meshing + LOD
- ~265K triangles, ~2.4K chunks rendered
- F3 overlay shows shadow pass timing separately

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
./target/release/neolith.exe
```

### Quick Iteration
```bash
taskkill //F //IM neolith.exe 2>&1; cargo build --release && ./target/release/neolith.exe
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
- Texture tiling: `ao_tile_packed` contains [ao, tile_u, tile_v, light_level] for proper tiling on merged quads
- To test winding: set `cull_mode: None` in `pipeline.rs`

### Lighting System
The lighting uses Minecraft-style BFS flood fill propagation:

**Data Storage:**
- Each chunk has `light: Box<[u8; 4096]>` - 1 byte per block
- Upper 4 bits = sky light (0-15), lower 4 bits = block light (0-15)
- Light packed into vertex `ao_tile_packed.w` byte (0-255 scaled from 0-15)

**Key Components:**
- `src/world/lighting.rs` - `LightEngine` with BFS propagation/removal algorithms
- `src/world/chunk.rs` - `Block::light_emission()` returns emission level (Torch = 14)
- `src/world/meshing.rs` - `get_face_light()` samples light for mesh vertices
- `src/renderer/pipeline.rs` - Shader applies warm torch color with gamma curve

**Light Propagation:**
1. When torch placed: BFS from source, decrementing level by 1 per block
2. When torch removed: BFS removal, then re-propagate from nearby sources
3. Cross-chunk propagation handled via neighbor chunk access

**Adding New Light Sources:**
1. Add variant to `Block` enum in `chunk.rs`
2. Add emission value in `Block::light_emission()` (0-15)
3. Ensure `Block::is_light_transparent()` returns appropriate value

### World Persistence System
Region-based chunk storage with zstd compression:

**Directory Structure:**
```
saves/world1/
  world.dat              # Metadata: seed, spawn position, timestamps (224 bytes)
  region/
    r.{x}.{y}.{z}.region # 32³ chunks per region file
```

**Region File Format:**
- Header: 8 bytes (magic "BLKR" + version)
- Chunk table: 256KB (32768 entries × 8 bytes)
- Chunk data: 4KB sectors with zstd-compressed payloads

**Key Components:**
- `WorldStorage` - Main API with LRU region cache (max 16 open files)
- `DirtyChunkTracker` - Tracks modified chunks for efficient saving
- Sector reuse optimization - Reuses allocation when chunk fits in same space

**Integration:**
- `WorldManager::new_with_save(path, seed)` - Creates world with persistence
- `set_block()` automatically marks chunks dirty
- `save_all()` called on shutdown (in Drop impl)
- Workers try disk load before generation

### Day/Night Cycle
Full day/night system with warm color palette:

**Time System:**
- `time_of_day: f32` (0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk)
- `cycle_speed: f32` multiplier (T key cycles: 1x → 5x → 20x → 50x → 100x → paused)
- Base cycle: 600 seconds (10 minutes) for full day

**Key Components:**
- `DayNightCycle` struct in `renderer/mod.rs` - calculates sun/moon positions and colors
- `SkyUniform` - passes sky gradient colors, sun/moon directions to sky shader
- `LightUniform` - passes sun/moon colors, ambient to chunk shader

**Color Transitions:**
- Warm palette throughout (coral dawn, golden noon, amber dusk, warm indigo night)
- Night is darker with subtle warm moon (no harsh shadows)
- Fast transitions at dawn/dusk (~5% of cycle), longer day/night periods

**Shadows:**
- Sun shadows only (no moon shadows)
- Shadows fade smoothly with `smoothstep(0.0, 0.3, sun_intensity)` at sunset
- No jarring direction flips during transitions

**Moon:**
- Crisp sharp edge (not sun-like glow)
- Warm amber color, very dim (max intensity 0.25)
- Provides soft ambient light, no directional shadows

### Audio System
Simple footstep audio using rodio:

**Files:**
- `assets/sfx/footstep-grass-1.wav`
- `assets/sfx/footstep-grass-2.wav`

**Behavior:**
- Plays alternating footstep sounds when walking on ground
- 400ms interval (280ms when sprinting)
- 15% volume (subtle)
- Disabled in fly mode

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
- `zstd 0.13` - Chunk compression for persistence
- `lru 0.12` - Region file caching
- `rodio 0.19` - Audio playback

---

## Session Handoff Checklist

When starting a new session:
1. Read this file first
2. Check `ROADMAP.md` for priorities
3. Run `cargo check` to verify build state
4. Test current build before making changes
5. Focus on one task at a time
6. Update this doc if architecture changes significantly
