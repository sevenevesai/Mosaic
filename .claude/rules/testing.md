# Testing & Benchmarking

175 unit/integration tests across core systems.

## Commands

```bash
cargo test                     # All 175 tests
cargo test world::             # World module (108 tests)
cargo test inventory::         # Inventory module (32 tests)
cargo test player::            # Player module (18 tests)
cargo test meshing             # Meshing tests (22 tests)
cargo test lighting            # Lighting tests (12 tests)
cargo bench                    # All benchmarks
```

## Coverage

Well tested: meshing, lighting, serialization, inventory, physics, generation.
Not unit-testable: GPU shaders, renderer (need visual regression tests).

## Test Fixtures

- `Chunk::new_solid(block)` - All same type
- `Chunk::new_sparse(seed, occupancy)` - Random fill
- `Chunk::new_checkerboard()` - Worst case for meshing

## Benchmark Reference (RTX 2080 Ti)

| Benchmark | Time |
|-----------|------|
| meshing/solid_chunk | ~31us |
| meshing/sparse_20pct | ~170us |
| meshing/checkerboard | ~775us |
| meshing/lod_sparse/Full | ~157us |
| meshing/lod_sparse/Half | ~58us |
| meshing/lod_sparse/Quarter | ~6.5us |
