---
paths:
  - "src/world/storage/**"
  - "src/world/item_persistence.rs"
  - "src/player/persistence.rs"
---

# World persistence

Region-based chunk storage with zstd compression. 8 tests under `src/world/storage`.

```
saves/world1/
  world.dat                  # magic "WDAT", seed, spawn, timestamps
  region/r.{x}.{y}.{z}.region  # 32^3 chunks per file, 4 KB sectors, zstd
```

| File | Role |
|---|---|
| `storage/mod.rs` | `WorldStorage`: the API, plus an LRU of open region files |
| `storage/region.rs` | `RegionFile`: sector allocation, read and write |
| `storage/chunk_io.rs` | serialization: 4096 blocks + 4096 light + optional water |
| `storage/dirty_tracker.rs` | `DirtyChunkTracker` |

`set_block()` marks the chunk dirty; workers check disk before generating and only dirty
chunks are written, from `WorldManager::drop()` via `save_all()`. A chunk that still fits its
existing sectors reuses them rather than reallocating.

## Gotchas

- The region-file LRU is capped at 16 entries. Raising it risks exhausting file handles; the
  cap is why long sessions across a wide world stay stable.
- `ZSTD_LEVEL` is 3 (`chunk_io.rs`): about 3:1 and under a millisecond per chunk. Higher
  levels cost frame time on the save path for little size gain.
- A corrupt region is logged and the chunk is regenerated from the seed, so corruption shows
  up as terrain silently reverting, not as a crash.
- On a full disk the error propagates and unsaved chunks stay in memory — they are lost on
  exit, so treat a write failure as data loss, not a warning.
- Chunk payload layout is positional. Adding a field means bumping the header `version` in
  `chunk_io.rs`; without it, old saves deserialize into garbage blocks.
