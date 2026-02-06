# Testing & Benchmarking

192 unit/integration tests across core systems.

## Commands

```bash
cargo test                     # All 192 tests
cargo test world::             # World module (113 tests incl. hardness, preferred_tool)
cargo test inventory::         # Inventory module (44 tests incl. tools, durability, persistence)
cargo test player::            # Player module (18 tests)
cargo test meshing             # Meshing tests (22 tests)
cargo test lighting            # Lighting tests (12 tests)
cargo test test_tool           # Tool-specific tests (12 tests)
cargo test test_block_hardness # Block hardness tests (4 tests)
cargo bench                    # All benchmarks
cargo bench shadow             # Shadow DDA benchmarks (algorithm comparison, quality presets)
```

## Coverage

Well tested: meshing, lighting, serialization, inventory (incl. tools), physics, generation, block hardness.
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
| shadow/algorithm/naive_112steps (500 rays) | ~940us |
| shadow/algorithm/dda_128steps (500 rays) | ~325us |
| shadow/algorithm/dda_96steps (500 rays) | ~248us |
| shadow/algorithm/dda_48steps (500 rays) | ~131us |
