# Mosaic — voxel engine (Rust + wgpu)

Real-time voxel world: threaded chunk generation and greedy meshing, a 256^3 voxel GI volume,
a multi-pass renderer. Key bindings: README.md.

## Commands

- `cargo check` — compiles clean; the gate for any edit.
- `cargo test` — 209 `#[test]` functions over world, inventory, player, meshing, lighting.
- `cargo build --release && ./target/release/mosaic.exe` — smoke test; F3 shows FPS, chunk,
  triangle and light counts. `rm -rf saves/world1` first when generation or saves changed.
- Background runs: redirect stderr (`2>session_stderr.log`), read `performance_log.txt`; raw tracing
  output is thousands of lines.

## Invariants

- Every trace through the voxel volume — sun and torch shadows, AO, volumetrics — uses
  Amanatides & Woo DDA with `textureLoad` and skips the starting voxel.
  `textureSampleLevel` interpolates across voxel boundaries and draws lines at every block
  edge; not skipping the first voxel self-shadows the surface.
- Quality flags share `volume_origin.w`: bit 0 torch shadows, bits 1-3 shadow, bits 4-6 AO;
  volumetrics keeps its own in `screen_params.z`. A new field must claim unused bits or it
  corrupts its neighbours.
- Front faces are CCW; quads emit triangles 0-2-1 and 0-3-2, and left/right faces reorder
  UVs to `[uv0, uv3, uv2, uv1]`. Wrong winding makes the face vanish under culling.
- The atlas is 16 tiles per row: `TILES_PER_ROW` in `src/lib.rs` and
  `src/world/block_registry.rs` and the literal `16` in the chunk, item and inventory
  shaders describe one grid; change together.
- Player block edits go through `priority_task_sender`, which workers check ahead of the
  generation queue, and remesh synchronously; a generation counter discards mesh
  results arriving after the chunk changed.

## Conventions

- `Block` in `src/world/chunk.rs` owns block IDs; `From<u8>` must round-trip them, since
  saves store the raw discriminant and reusing an ID rewrites existing worlds.
- Textures resolve by name: `BlockRegistry` scans `assets/blocks/*.png` sorted into atlas
  slots (`_torch`, `_water` procedural) and `build_block_textures` maps a block ID to six
  names. Adding a texture means adding the PNG, not editing a list.
- World tuning constants (`CHUNK_SIZE` 16, `RENDER_DISTANCE` 32, `LOD_FULL_DETAIL` 12,
  `WATER_SEA_LEVEL` 32) live in `src/lib.rs`.

## Subsystem docs — path-scoped, load with their source

| Area | Rule (`.claude/rules/`) | Mechanism (docs/) |
|---|---|---|
| Meshing, LOD | meshing-system.md | |
| Light, torches | lighting-system.md | lighting-design.md |
| Ambient occlusion | ao-system.md | ambient-occlusion-design.md |
| Volumetrics | volumetrics-system.md | volumetrics-design.md |
| Day/night | day-night-cycle.md | |
| Water | water-rendering.md | |
| Grass | grass-system.md | |
| Audio | audio-system.md | |
| Settings menu | settings-menu.md | |
| Save format | world-persistence.md | |
| Tests, benches | testing.md | |
| Perf logging | performance-logging.md | |
