---
paths:
  - "tests/**"
  - "benches/**"
---

# Tests and benchmarks

209 `#[test]` functions: 116 under `src/world` (19 meshing, 11 lighting, 8 storage), 44
inventory, 24 player, 8 renderer, 17 in `tests/world_integration.rs`.

## Commands

```bash
cargo test                     # everything
cargo test world::             # world module
cargo test meshing             # meshing only
cargo bench                    # all benchmarks
cargo bench shadow             # shadow DDA: algorithm comparison and quality presets
```

## Coverage boundary

Meshing, lighting, serialization, inventory (tools, durability, persistence), physics,
generation and block hardness are covered. GPU shaders and the renderer are not unit-testable
here; changes there need a visual check in the running game, so do not read a green
`cargo test` as evidence that a shader change is correct.

## Fixtures

`Chunk::new_solid(block)` for a uniform chunk, `Chunk::new_sparse(seed, occupancy)` for random
fill, `Chunk::new_checkerboard()` for the meshing worst case — no run is longer than one
block, so greedy merging degenerates.

## Recorded benchmark times (RTX 2080 Ti)

Reference points for spotting a regression, not thresholds; they were measured on a past run
and are not re-verified here.

| Benchmark | Time |
|---|---|
| meshing/solid_chunk | ~31 us |
| meshing/sparse_20pct | ~170 us |
| meshing/checkerboard | ~775 us |
| meshing/lod_sparse Full / Half / Quarter | ~157 / ~58 / ~6.5 us |
| shadow/algorithm naive_112steps (500 rays) | ~940 us |
| shadow/algorithm dda_128 / dda_96 / dda_48 | ~325 / ~248 / ~131 us |
