# Testing & Benchmarking

## Running Tests

```bash
cargo test                     # All 24 tests
cargo test chunk::tests        # Chunk/block tests only
cargo test lighting::tests     # Lighting tests only
cargo test chunk_io::tests     # Serialization tests only
```

## Test Coverage

### chunk.rs (12 tests)
- `test_block_is_solid` - Air/Water/Torch not solid, others solid
- `test_block_is_transparent` - Air/Water/Leaves/Torch transparent
- `test_block_light_emission` - Torch=14, others=0
- `test_block_is_light_transparent` - Same as transparent
- `test_block_texture_indices` - Face textures for Grass/Wood/Stone
- `test_block_from_u8` - Deserialization, invalid→Air
- `test_chunk_pos_from_world` - Coordinate conversion
- `test_chunk_pos_world_origin` - Round-trip identity
- `test_chunk_light_packing` - Block/sky light isolation
- `test_chunk_in_bounds` - Boundary validation
- `test_new_solid` - Fixture for benchmarks
- `test_new_checkerboard` - Fixture for benchmarks

### lighting.rs (4 tests)
- `test_world_to_chunk_local` - Positive/negative/boundary coords
- `test_propagate_single_source` - Torch level 14 decay
- `test_propagate_stops_at_solid` - Light blocked by stone
- `test_light_removal` - Cleanup after torch removed

### chunk_io.rs (4 tests)
- `test_header_roundtrip` - Header serialize/deserialize
- `test_chunk_roundtrip` - Full chunk encode/decode
- `test_chunk_preserves_light` - Light data survives serialization
- `test_all_block_types_roundtrip` - All 16 block types

## Benchmarks

```bash
cargo bench                           # All benchmarks
cargo bench --bench meshing_bench     # Meshing only
cargo bench --bench generation_bench  # Generation only
cargo bench --bench lighting_bench    # Lighting only
```

### Benchmark Results (Reference)

| Benchmark | Time |
|-----------|------|
| meshing/solid_chunk | ~31µs |
| meshing/sparse_20pct | ~170µs |
| meshing/checkerboard | ~775µs |
| meshing/lod_sparse/Full | ~157µs |
| meshing/lod_sparse/Half | ~58µs |
| meshing/lod_sparse/Quarter | ~6.5µs |

### Adding Tests

**Unit test pattern:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // Arrange
        let input = ...;

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

**Benchmark pattern:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_benchmarks(c: &mut Criterion) {
    c.bench_function("name", |b| {
        b.iter(|| function_to_benchmark(black_box(input)));
    });
}

criterion_group!(benches, my_benchmarks);
criterion_main!(benches);
```

## Test Fixtures

Helper constructors in `Chunk`:
- `Chunk::new_solid(block)` - All blocks same type
- `Chunk::new_sparse(seed, occupancy)` - Random fill
- `Chunk::new_checkerboard()` - Alternating pattern (worst case)
