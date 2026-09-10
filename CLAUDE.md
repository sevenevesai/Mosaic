# Mosaic — voxel engine (Rust + wgpu)

Real-time voxel world: threaded chunk generation, greedy meshing, a 256^3 GI volume,
multi-pass rendering. Key bindings: README.md.

## Commands

- `cargo check` — clean compile; the gate for any edit.
- `cargo test` — 209 tests; a green run says nothing about shaders or the renderer.
- `cargo build --release && ./target/release/mosaic.exe` — smoke test; F3 shows FPS, chunk,
  triangle and light counts. `rm -rf saves/world1` first when generation or saves changed.
- Background runs: redirect stderr (`2>session_stderr.log`) and read `performance_log.txt`;
  raw tracing is thousands of lines.

## Invariants

- Every trace through the voxel volume — sun and torch shadows, AO, volumetrics — uses
  Amanatides & Woo DDA with `textureLoad` and skips the starting voxel.
  `textureSampleLevel` blends across voxel boundaries and draws lines at every block
  edge; not skipping the first voxel self-shadows the surface.
- Quality flags share `volume_origin.w`: bit 0 torch shadows, bits 1-3 shadow, bits 4-6 AO;
  volumetrics keeps its own in `screen_params.z`. A new field must claim unused bits or it
  corrupts its neighbours.
- Front faces are CCW; quads emit triangles 0-2-1 and 0-3-2, and left/right faces reorder
  UVs to `[uv0, uv3, uv2, uv1]`. Wrong winding makes the face vanish under culling.
- `Block` in `src/world/chunk.rs` owns block IDs and `From<u8>` must round-trip them: saves
  store the raw discriminant, so reusing an ID rewrites existing worlds.
- The atlas is `TILES_PER_ROW` 16, declared in `src/lib.rs` and `src/world/block_registry.rs`
  and hardcoded as 16 in the chunk, item and inventory shaders; all change together.
- Player block edits go through `priority_task_sender`, which workers check ahead of the
  generation queue, and remesh synchronously; a generation counter discards mesh results
  arriving after the chunk changed.

## Conventions

- Textures resolve by name: a new block texture is a PNG in `assets/blocks/` named in
  `build_block_textures()`, not an edit to a file list.
- World tuning constants (`CHUNK_SIZE` 16, `RENDER_DISTANCE` 32, `LOD_FULL_DETAIL` 12,
  `WATER_SEA_LEVEL` 32) live in `src/lib.rs`.

## Subsystem docs, path-scoped to their source

| Area | Rule (`.claude/rules/`) | Mechanism (docs/) |
|---|---|---|
| Blocks, atlas | block-registry.md | |
| World generation | world-generation.md | |
| Meshing, LOD | meshing-system.md | |
| Frame, uploads | render-passes.md | |
| Light, torches | lighting-system.md | lighting-design.md |
| Ambient occlusion | ao-system.md | ambient-occlusion-design.md |
| Volumetrics | volumetrics-system.md | volumetrics-design.md |
| Day/night | day-night-cycle.md | |
| Water | water-rendering.md | |
| Grass | grass-system.md | |
| Player, camera | player-camera.md | |
| Audio | audio-system.md | |
| Settings menu | settings-menu.md | |
| Save format | world-persistence.md | |
| Tests, benches | testing.md | |
| Perf logging | performance-logging.md | |
