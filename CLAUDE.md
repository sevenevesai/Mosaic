# Mosaic Engine

High-performance voxel engine in Rust + wgpu. ~13K LOC, 8 worker threads, 5 render passes.

## Quick Start

```bash
cargo build --release
./target/release/mosaic.exe
```

**Controls:** WASD move, Space jump/fly-up, Shift fly-down, Ctrl sprint, F fly mode, E inventory, 1-9 hotbar slots, Scroll wheel cycle slots, LMB break, RMB place, F3 debug, F4 body part editor, T day/night speed, Y shadow mode, U shadow debug, V volumetrics quality cycle, P perf log, Esc menu.

## Commands

| Action | Command |
|--------|---------|
| Build | `cargo build --release` |
| Check | `cargo check 2>&1 \| head -40` |
| Run | `./target/release/mosaic.exe` |
| Quick iteration | `taskkill //F //IM mosaic.exe 2>&1; cargo build --release && ./target/release/mosaic.exe` |
| New world | `rm -rf saves/world1` before running |

## Architecture

```
src/
├── app.rs           # Main loop, input handling
├── lib.rs           # Constants: RENDER_DISTANCE, LOD thresholds, physics
├── settings_menu.rs # Main menu + tabbed settings UI (~950 LOC)
├── audio/
│   └── mod.rs       # 3D spatial audio, material sounds, ambient system
├── inventory/
│   ├── item.rs      # Item types (Block + Tool), tags, rarity, stacking, durability
│   └── inventory.rs # Slots, hotbar, sorting, Quick Stack
├── renderer/
│   ├── mod.rs       # GPU state, render passes, day/night cycle
│   ├── pipeline.rs  # WGSL shaders (chunk, sky, UI, crack overlay)
│   ├── water.rs     # Ray-marched water (fullscreen post-process)
│   ├── clouds.rs    # Procedural cloud system
│   ├── grass/       # Procedural grass rendering (vertex pulling)
│   ├── character_renderer.rs # Skinned character rendering, bone matrices, voxel GI shadows + torch lighting
│   ├── inventory_ui.rs # Hotbar/inventory UI rendering (tools: colored rect + letter + durability bar)
│   ├── gpu_culling.rs # GPU frustum culling + visibility cache
│   ├── voxel_gi.rs  # Voxel GI: 256^3 volume, DDA shadows, torch buffer (256 lights), VCT, player shadow
│   ├── ssao.rs      # Screen-space ambient occlusion
│   ├── aa.rs        # Hybrid Morphological-Temporal AA
│   ├── volumetrics.rs # Volumetric light shafts: DDA raymarch + bilateral blur + composite
│   └── chunk_pool.rs # MDI mega-buffers
├── entity/
│   ├── mod.rs       # EntityManager, Skeleton, AnimationComponent (looping/clamp), player_entity tracking
│   └── loader.rs    # FBX loading, body part merging, skeleton binding
├── world/
│   ├── mod.rs       # WorldManager, 8 worker threads, chunk lifecycle
│   ├── meshing.rs   # Binary greedy meshing (~1,800 LOC)
│   ├── water_flow.rs # Pipe-model water flow simulation (parked)
│   ├── generation/  # Biomes, caves, terrain, trees, ores
│   ├── lighting.rs  # BFS flood fill propagation
│   └── storage/     # Region files, zstd compression
├── character_styler.rs # Character appearance: outfit, hair, head selection (default: outfit 010)
└── player/mod.rs    # AABB collision, raycast, 3rd person camera, mining (tools+hardness), inventory
```

## Key Constants (src/lib.rs)

```rust
CHUNK_SIZE: 16           // Blocks per chunk dimension
RENDER_DISTANCE: 32      // Chunks around player
LOD_FULL_DETAIL: 12      // Default full mesh distance (configurable via settings)
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

**Render Pipeline Order**: Sky pass -> Opaque geometry (MDI) -> Character entities (w/ voxel GI shadows) -> Block outline -> Crack overlay -> Water pass -> SSAO -> Volumetrics (raymarch + blur + composite) -> Post-process (DoF, AA, Bloom) -> UI

**3rd Person Camera**:
- Camera orbits behind player: `position = player_center - forward * 10.0`
- Camera-terrain collision: 10-step ray sampling, min distance 1.0 block
- `facing_yaw` computed from WASD input relative to camera yaw
- Animation selected by `MovementState`: Idle(0), Walk(1), Jumping(3), Falling(3 slow+clamp)
- Jumping/Falling split: Jumping = rising (velocity.y > 0.5), Falling = descending/apex
- Jump animation clamps at last frame (no loop), falling slows to 0.3x and holds pose
- Binary search ground contact on landing for pixel-accurate placement
- Root motion disabled for player (physics drives position)

## Gotchas & Warnings

- **Texture atlas** is 6x6 grid (36 tiles), `tiles_per_row` must match in texture.rs, pipeline.rs, meshing.rs
- **Foliage blocks** (leaves) render all 6 faces via `add_foliage_blocks()`, excluded from greedy meshing
- **Foliage neighbors** - Solid blocks render faces toward foliage (check `!nb.is_foliage()`)
- **Greedy meshing winding** - Left/right faces need UV reorder `[uv0, uv3, uv2, uv1]`
- **Cross-chunk lighting** - Light propagation requires neighbor chunk access, triggers remesh
- **Tree placement** - Must check `is_ground_stable_for_tree()` to prevent floating trees
- **Ore generation** - Noise clustering + hash slot system in `get_ore_or_stone_fast()`
- **water_flow.rs** exists but is parked - visual flow not currently used
- **Biomes**: Grasslands (~75%) caves underground only; Cavernous Highlands (~25%) have surface caverns
- **Player shadow voxels** - Injected AFTER queue.submit() to avoid GPU fill overwrite. 1-frame latency, block-snapping (known limitation)
- **Crack overlay** - 4-stage procedural texture (256x64), rendered after block outline pass with depth bias -3
- **Tool items** - Stack size 1, rendered as colored rect + letter (P/A/S/W) + durability bar in hotbar
- **Character lighting** - Receives sun/moon shadows via voxel GI DDA + torch point-lights from torch buffer (group 3 binding 4); casts via inject_player_shadow (2 voxels)
- **Shadow quality** - 5 presets (OFF/LOW/MEDIUM/HIGH/ULTRA) control DDA max_steps (0/48/96/128/128+2tap). Packed into `volume_origin.w` bits 1-3. ULTRA forces 2-tap multi-sampling regardless of shadow mode
- **Shadow DDA** - Uses Amanatides & Woo true voxel traversal with `textureLoad` (integer sampling). Skips starting voxel (one DDA step before loop) to prevent self-shadowing without positional offset. Shadow origin uses minimal `normal * 0.02` offset for float precision only. Only traces for sun-facing faces (`ndotl > 0`)
- **Torch lighting** - GPU storage buffer of up to 256 light positions (group 3 binding 4). Fragment shader loops over buffer, computes point-light falloff + DDA shadow per torch. Mount-block skip: solid blocks within 1 voxel of light source are ignored (prevents torch's own support block from casting shadow). `torch_count` packed into `sun_color.w` of VoxelVolumeUniform. Both chunk and character shaders use the same system

## Common Tasks

**Add Block Type:**
1. Add variant to `Block` enum in `world/chunk.rs` (with ID)
2. Add texture indices in `Block::texture_indices()`
3. Add to `From<u8>` impl in `world/chunk.rs`
4. Add texture file to `TEXTURE_FILES` in `renderer/texture.rs`
5. Update `tiles_per_row` if atlas needs expansion (texture.rs, pipeline.rs, meshing.rs)
6. For foliage: add to `is_foliage()`, for light emitters: add to `light_emission()`

**Add Setting:** See `.claude/rules/settings-menu.md` for the pattern.

**Modify Terrain:** Edit `world/generation/` - `TerrainGenerator` handles height, caves, ores, biomes.

## Verification

```bash
cargo check                    # Compilation errors
cargo test                     # Run 192 unit tests
cargo build --release          # Full build
./target/release/mosaic.exe    # Manual smoke test
```

Check F3 overlay for: FPS, chunk count, triangle count, light levels.

## Dependencies

Core: wgpu 0.19, winit 0.29, glam 0.25, noise 0.9, rayon 1.8, crossbeam-channel 0.5, zstd 0.13, rodio 0.19

## Session Checklist

1. Read this file
2. Run `cargo check` to verify build state
3. Test current build before changes
4. Check `.claude/rules/` for detailed system docs if needed
