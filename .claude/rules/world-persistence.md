# World Persistence System

Region-based chunk storage with zstd compression. 8 tests.

## Structure

```
saves/world1/
├── world.dat              # Metadata: magic "WDAT", seed, spawn, timestamps
└── region/
    └── r.{x}.{y}.{z}.region  # 32^3 chunks per file, 4KB sectors, zstd compressed
```

## Key Files

- `storage/mod.rs` - `WorldStorage`: main API, LRU cache (16 regions max)
- `storage/region.rs` - `RegionFile`: read/write region files
- `storage/chunk_io.rs` - Serialization: [4096 blocks + 4096 light + optional water], zstd level 3
- `storage/dirty_tracker.rs` - `DirtyChunkTracker`: modified chunk tracking

## How It Works

- `set_block()` automatically marks chunk dirty
- Workers check disk before generating (load if exists, generate if not)
- `save_all()` called in `WorldManager::drop()` - only dirty chunks written
- Sector reuse: if chunk fits in same space, reuses allocation

## Gotchas

- **Region corruption**: Logged, chunk regenerated from seed
- **Disk full**: Error propagated, unsaved chunks remain in memory
- **LRU cache**: 16 regions max prevents file handle exhaustion
- **zstd level 3**: ~3:1 ratio, <1ms per chunk
