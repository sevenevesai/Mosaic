# Testing & Benchmarking

175 unit/integration tests across core systems.

## Running Tests

```bash
cargo test                     # All 175 tests
cargo test world::             # World module (108 tests)
cargo test inventory::         # Inventory module (32 tests)
cargo test player::            # Player module (18 tests)
cargo test meshing             # Meshing tests (22 tests)
cargo test lighting            # Lighting tests (12 tests)
```

## Test Coverage

| Module | Tests | Key Coverage |
|--------|-------|--------------|
| `world/meshing.rs` | 22 | Greedy merge, LOD, winding order, vertices |
| `world/lighting.rs` | 12 | BFS propagation, cross-chunk, removal |
| `world/chunk.rs` | 12 | Block properties, coordinates, fixtures |
| `world/generation/` | 24 | Caves, caverns, biomes, trees, ruins |
| `world/storage/` | 8 | Serialization, region files, dirty tracking |
| `world/item_entity.rs` | 6 | Spawn, merge, despawn |
| `world/water_flow.rs` | 4 | Flow simulation |
| `inventory/inventory.rs` | 19 | Add, remove, sort, lock, swap |
| `inventory/item.rs` | 13 | Stacks, tags, rarity, merge |
| `player/mod.rs` | 16 | AABB, physics, input handling |
| `player/persistence.rs` | 2 | Save/load roundtrip |
| `tests/world_integration.rs` | 17 | WorldManager lifecycle |

## Benchmarks

```bash
cargo bench                           # All benchmarks
cargo bench --bench meshing_bench     # Meshing only
cargo bench --bench generation_bench  # Generation only
cargo bench --bench lighting_bench    # Lighting only
```

### Reference Times (RTX 2080 Ti)

| Benchmark | Time |
|-----------|------|
| meshing/solid_chunk | ~31µs |
| meshing/sparse_20pct | ~170µs |
| meshing/checkerboard | ~775µs |
| meshing/lod_sparse/Full | ~157µs |
| meshing/lod_sparse/Half | ~58µs |
| meshing/lod_sparse/Quarter | ~6.5µs |

## Adding Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_behavior() {
        // Arrange
        let input = create_test_input();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

## Test Fixtures

Helper constructors in `Chunk`:
- `Chunk::new_solid(block)` - All blocks same type
- `Chunk::new_sparse(seed, occupancy)` - Random fill
- `Chunk::new_checkerboard()` - Alternating pattern (worst case for meshing)

## What's Tested vs Not

**Well tested**: Meshing, lighting, serialization, inventory, physics, generation
**Not unit-testable**: GPU shaders, renderer (need visual regression tests)
**Low priority**: Audio system
