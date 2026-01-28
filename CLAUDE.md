# Neolith Engine

High-performance voxel engine in Rust + wgpu. ~13K LOC, 8 worker threads, 5 render passes.

## Quick Start

```bash
cargo build --release
./target/release/neolith.exe
```

**Controls:** WASD move, Space jump/fly-up, Shift fly-down, Ctrl sprint, F fly mode, F3 debug, T day/night speed, Y shadow mode, U shadow debug, 1-8 blocks (1=Stone, 2=Dirt, 3=Grass, 4=StoneDirt, 5=Brick, 6=Wood, 7=Leaves, 8=Torch), LMB break, RMB place, P perf log, Esc release mouse.

## Commands

| Action | Command |
|--------|---------|
| Build | `cargo build --release` |
| Check | `cargo check 2>&1 \| head -40` |
| Run | `./target/release/neolith.exe` |
| Quick iteration | `taskkill //F //IM neolith.exe 2>&1; cargo build --release && ./target/release/neolith.exe` |
| New world | `rm -rf saves/world1` before running |

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
│   ├── meshing.rs   # Binary greedy meshing (~1,800 LOC)
│   ├── generation.rs # Biomes, caves, terrain, trees
│   ├── lighting.rs  # BFS flood fill propagation
│   └── storage/     # Region files, zstd compression
└── player/mod.rs    # AABB collision, raycast, block interaction
```

## Biome System

Two biomes with distinct cave behavior:
- **Grasslands** (~75%): Rolling hills, caves stay underground (5+ blocks below surface), no surface openings
- **Cavernous Highlands** (~25%): Green hilly terrain with surface caverns, large open spaces

## Block Types

| ID | Block | Texture | Notes |
|----|-------|---------|-------|
| 0 | Air | - | |
| 1 | Dirt | 256_Dirt Pebbles 01 | |
| 2 | Grass | Top/side variants | |
| 3 | Stone | 256_Stone 2 | Main cave/underground |
| 4 | StoneDirt | 256_Stone Dirt 01 | Dirty stone variant |
| 5 | Brick | 256_Brick 06 | Placeable building block |
| 6 | Water | - | Heightfield rendering |
| 7 | Wood | Planks textures | |
| 8 | Leaves | - | Foliage with wind sway |
| 9 | StoneRubble2 | 256_Stone Rubble 02 | Stone variation |
| 10 | IronOre | 64_Ore_Iron | Y:10-64, common |
| 11 | GoldOre | 64_Ore_Gold | Y:5-32, uncommon |
| 12 | DiamondOre | 64_Ore_Diamond | Y:0-16, rare |
| 13 | RedstoneOre | 64_Ore_Redstone | Y:0-24, uncommon |
| 14 | Bedrock | - | |
| 15 | Torch | Procedural | Light emission 14 |
| 16 | Sand | 256_Sand 01 | Beach generation |
| 17 | CoalOre | 64_Ore_Coal | Y:20-90, common, chunky veins |
| 18 | CopperOre | 64_Ore_Copper | Y:20-70, common |
| 19 | EmeraldOre | 64_Ore_Emerald | Y:0-32, rare (1%) |
| 20 | SapphireOre | 64_Ore_Sapphire | Y:0-20, rare |
| 21 | QuartzOre | 64_Ore_Quartz | Y:10-50, uncommon |
| 22 | AmethystOre | 64_Ore_Amethyst | Y:0-28, uncommon |
| 23 | SilverOre | 64_Ore_Silver | Y:10-50, uncommon |
| 24 | TinOre | 64_Ore_Tin | Y:25-70, common |
| 25 | NickelOre | 64_Ore_Nickel | Y:8-50, uncommon |
| 26 | LeadOre | 64_Ore_Lead | Y:5-32, uncommon |
| 27 | TitaniumOre | 64_Ore_Titanium | Y:0-14, very rare |
| 28 | UraniumOre | 64_Ore_Uranium | Y:0-10, very rare, **glows!** |

## Key Constants (src/lib.rs)

```rust
CHUNK_SIZE: 16           // Blocks per chunk dimension
RENDER_DISTANCE: 24      // Chunks around player
LOD_FULL_DETAIL: 8       // Full mesh within 8 chunks
WATER_SEA_LEVEL: 32      // Y level for water generation
```

## Code Patterns

**Block Placement Priority System**:
- Player block changes use priority channel (skip generation queue)
- Modified chunk meshes synchronously for instant feedback
- Neighbor chunks queued with urgent upload flag
- `urgent_mesh_uploads` bypasses 16/frame throttle

**Thread Communication**:
- `crossbeam_channel` for chunk tasks/results
- `priority_task_sender` for player actions (checked first by workers)
- Generation counter discards stale mesh results

**Winding Order**: CCW front faces. Triangles: 0-2-1 and 0-3-2. Left/right faces have reordered UVs.

## Gotchas & Warnings

- **Texture atlas** is 6x6 grid (36 tiles), `tiles_per_row` must match in texture.rs, pipeline.rs, meshing.rs
- **Foliage blocks** (leaves) render all 6 faces via `add_foliage_blocks()`, excluded from greedy meshing
- **Foliage neighbors** - Solid blocks render faces toward foliage (check `!nb.is_foliage()` in neighbor occupancy)
- **Greedy meshing winding** - Left/right faces need UV reorder `[uv0, uv3, uv2, uv1]`
- **Cross-chunk lighting** - Light propagation requires neighbor chunk access
- **Tree placement** - Must check `is_ground_stable_for_tree()` to prevent floating trees
- **Ore generation** - Uses noise clustering + hash slot system in `get_ore_or_stone_fast()`

## Lighting System

Voxel-based global illumination using `renderer/voxel_gi.rs`:
- **DDA Ray Marching Shadows**: Traces rays through 256³ voxel volume (max ~170 blocks)
- **Sky Occlusion**: Gradual light falloff into caves (0.08 opacity per voxel)
- **Cave Ambient**: 10% minimum ambient, 7% minimum shadow light
- **Voxel Cone Tracing (VCT)**: 5-cone diffuse GI from radiance volume

**Shadow Modes** (Y key to toggle):
| Mode | Style | Best For |
|------|-------|----------|
| **Hard** | Solid fill, sharp edges, lighter caves | Realism, accuracy |
| **Soft** | Gradient, painterly, darker caves | Atmosphere, artistic |

**Shadow Debug** (U key): Visualizes R=sun-facing, G=shadow value, B=was-traced

## Foliage System

Leaves render as blocks (not X-billboards) with gentle wind sway animation:
- **Rendering**: All 6 faces rendered via `add_foliage_blocks()` - no neighbor culling
- **Wind animation**: Shader-based sway in `pipeline.rs` vertex shader (foliage flag in normal_packed byte 3)
- **Neighbor handling**: Solid blocks always render faces toward foliage to prevent gaps during sway

## Ore Generation

Depth-based distribution in `get_ore_or_stone_fast()`:
- **Noise clustering**: Ores spawn in pockets where `ore_noise > 0.55`
- **Hash slots**: Each position gets ONE ore type based on `hash % 1000` slot ranges
- **Coal special**: Uses lower frequency noise (0.06) for chunkier 4-8 block veins
- **Depth tiers**: Very rare (Y<16), rare (Y<32), uncommon (Y<50), common (Y<70)

## Common Tasks

**Add Block Type:**
1. Add variant to `Block` enum in `world/chunk.rs` (with ID)
2. Add texture indices in `Block::texture_indices()`
3. Add to `From<u8>` impl in `world/chunk.rs`
4. Add texture file to `TEXTURE_FILES` in `renderer/texture.rs`
5. Update `tiles_per_row` if atlas needs expansion (texture.rs, pipeline.rs, meshing.rs)
6. For foliage: add to `is_foliage()`, for light emitters: add to `light_emission()`

**Add Biome:**
1. Add variant to `Biome` enum in `world/generation.rs`
2. Update `get_biome()` with noise-based selection
3. Add biome-specific cave logic in `is_cave_biome_aware()`

**Add Ore Type:**
1. Add block variant and texture (see "Add Block Type" above)
2. Add depth/slot logic in `get_ore_or_stone_fast()` in `world/generation.rs`
3. Smaller slot range = rarer ore, check depth with `y < N` conditions

**Modify Terrain:**
Edit `world/generation.rs` - `TerrainGenerator` handles height, caves, ores, biomes.

## Verification

Before committing changes:
```bash
cargo check                    # Compilation errors
cargo test                     # Run 50 unit tests
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
