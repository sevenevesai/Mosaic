# Neolith Engine

High-performance voxel engine in Rust + wgpu. ~12K LOC, 8 worker threads, 5 render passes.

## Quick Start

```bash
cargo build --release
./target/release/neolith.exe
```

**Controls:** WASD move, Space jump/fly-up, Shift fly-down, Ctrl sprint, F fly mode, F3 debug, T day/night speed, 1-8 blocks (1=Stone, 2=Dirt, 3=Grass, 4=StoneDirt, 5=Brick, 6=Wood, 7=Leaves, 8=Torch), LMB break, RMB place, P perf log, Esc release mouse.

## Commands

| Action | Command |
|--------|---------|
| Build | `cargo build --release` |
| Check | `cargo check 2>&1 \| head -40` |
| Run | `./target/release/neolith.exe` |
| Quick iteration | `taskkill //F //IM neolith.exe 2>&1; cargo build --release && ./target/release/neolith.exe` |

## Architecture

```
src/
├── app.rs           # Main loop, input handling
├── lib.rs           # Constants: RENDER_DISTANCE, LOD thresholds, physics
├── renderer/
│   ├── mod.rs       # GPU state, render passes, day/night cycle
│   ├── pipeline.rs  # WGSL shaders (chunk, water, sky, UI)
│   ├── voxel_gi.rs  # Voxel GI: 256³ volume, DDA shadows, VCT
│   └── chunk_pool.rs # MDI mega-buffers
├── world/
│   ├── mod.rs       # WorldManager, 8 worker threads, chunk lifecycle
│   ├── meshing.rs   # Binary greedy meshing (1,793 LOC) - most complex
│   ├── lighting.rs  # BFS flood fill propagation
│   └── storage/     # Region files, zstd compression
└── player/mod.rs    # AABB collision, raycast, block interaction
```

## Key Constants (src/lib.rs)

```rust
CHUNK_SIZE: 16           // Blocks per chunk dimension
RENDER_DISTANCE: 24      // Chunks around player
LOD_FULL_DETAIL: 8       // Full mesh within 8 chunks
WATER_SEA_LEVEL: 32      // Y level for water generation
```

## Code Patterns

**Packed Vertex Format** (24 bytes):
- position [f32; 3] + normal_packed u32 + uv u32 + ao_tile_light u32
- Light stored as: upper 4 bits = sky, lower 4 bits = block

**Thread Communication**:
- `crossbeam_channel` for chunk tasks/results
- Generation counter discards stale mesh results
- `Arc<RwLock<AHashMap>>` for thread-safe chunk storage

**Winding Order**: CCW front faces. Triangles: 0-2-1 and 0-3-2. Left/right faces have reordered UVs.

## Gotchas & Warnings

- **GPU profiler shows 0.00ms** - wgpu 0.19 async buffer mapping issue, not a bug in your code
- **Greedy meshing winding** - Left/right faces need UV reorder `[uv0, uv3, uv2, uv1]` for correct CCW
- **Cross-chunk lighting** - Light propagation requires neighbor chunk access; ensure neighbors loaded
- **Region file corruption** - Always use zstd compression; raw writes can corrupt sector alignment
- **Chunk fade-in** - Alpha-based; ensure blend state enabled in pipeline

## Lighting System

Voxel-based global illumination using `renderer/voxel_gi.rs`:
- **DDA Ray Marching Shadows**: Traces rays through 256³ voxel volume
- **Voxel Cone Tracing (VCT)**: 5-cone diffuse GI from radiance volume
- **Voxel AO**: Short-range cone traces for ambient occlusion
- No shadow map artifacts, infinite shadow range within volume

**Key files:**
- `renderer/voxel_gi.rs` - VoxelGI struct, 3D texture management
- `renderer/voxel_lighting.wgsl` - DDA + VCT shader functions
- `renderer/pipeline.rs:CHUNK_SHADER` - Chunk shader with voxel lighting

## Common Tasks

**Add Block Type:**
1. Add variant to `Block` enum in `world/chunk.rs`
2. Add texture indices in `Block::texture_indices()`
3. Add generation rules in `world/generation.rs`
4. Add color in `TextureAtlas::generate_tile()` in `renderer/texture.rs`

**Add Light Source:**
1. Add block variant in `world/chunk.rs`
2. Set emission in `Block::light_emission()` (0-15, torch=14)
3. Set `Block::is_light_transparent()` appropriately

**Modify Terrain:**
Edit `world/generation.rs` - `TerrainGenerator` handles height, caves, ores, biomes.

## Verification

Before committing changes:
```bash
cargo check                    # Compilation errors
cargo build --release          # Full build
./target/release/neolith.exe   # Manual smoke test
```

Check F3 overlay for: FPS, chunk count, triangle count, light levels.

## Dependencies

Core: wgpu 0.19, winit 0.29, glam 0.25, noise 0.9, rayon 1.8, crossbeam-channel 0.5, zstd 0.13, rodio 0.19

## Session Checklist

1. Read this file and `ROADMAP.md`
2. Run `cargo check` to verify build state
3. Test current build before changes
4. Check `.claude/rules/` for detailed system docs if needed
