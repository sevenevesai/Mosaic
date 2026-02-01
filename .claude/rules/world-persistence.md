# World Persistence System

Region-based chunk storage with zstd compression. 8 tests for serialization and region files.

## Directory Structure

```
saves/world1/
├── world.dat              # Metadata (224 bytes)
└── region/
    └── r.{x}.{y}.{z}.region  # 32³ chunks per file
```

## File Formats

### world.dat (224 bytes)
- Magic: "WDAT" (4 bytes)
- Version: u32
- Seed: i64
- Spawn position: [f64; 3]
- Last played: timestamp
- Play time: duration

### Region Files
- Header: 8 bytes (magic "BLKR" + version)
- Chunk table: 256KB (32768 entries × 8 bytes)
  - Each entry: sector_offset (u32) + sector_count (u32)
- Chunk data: 4KB sectors
  - Payload: zstd-compressed block data

## Key Components

| File | Struct | Purpose |
|------|--------|---------|
| `storage/mod.rs` | `WorldStorage` | Main API, LRU cache (16 regions max) |
| `storage/region.rs` | `RegionFile` | Read/write region files |
| `storage/chunk_io.rs` | - | Serialization + zstd compression |
| `storage/world_meta.rs` | `WorldMeta` | Metadata handling |
| `storage/dirty_tracker.rs` | `DirtyChunkTracker` | Modified chunk tracking |

## Integration Points

**Creating World with Persistence:**
```rust
let world = WorldManager::new_with_save("saves/world1", seed);
```

**Block Changes:**
- `set_block()` automatically marks chunk dirty
- No manual tracking needed

**Saving:**
- `save_all()` called in `WorldManager::drop()`
- Only dirty chunks written to disk
- Sector reuse: if chunk fits in same space, reuses allocation

**Loading:**
- Workers check disk before generating
- If chunk exists on disk, load + decompress
- If not, generate from seed

## Chunk Serialization Format

```
[block_data: 4096 bytes]     // 16³ blocks, 1 byte each
[light_data: 4096 bytes]     // 16³ light values
[water_data: optional]       // If chunk has water
```

Compressed with zstd level 3 (fast compression, good ratio).

## Common Operations

**Force Save All Chunks:**
```rust
world_manager.save_all();
```

**Check if Chunk Modified:**
```rust
dirty_tracker.is_dirty(chunk_pos);
```

**Clear Dirty Flag:**
```rust
dirty_tracker.mark_clean(chunk_pos);
```

## Error Handling

- Region file corruption: logged, chunk regenerated
- Disk full: error propagated, unsaved chunks remain in memory
- Invalid magic: file ignored, new region created

## Performance Notes

- LRU cache prevents file handle exhaustion
- zstd level 3: ~3:1 compression ratio, <1ms per chunk
- Sector reuse avoids file fragmentation
- Dirty tracking prevents unnecessary writes
