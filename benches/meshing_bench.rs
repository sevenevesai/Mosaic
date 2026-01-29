//! Benchmarks for chunk meshing algorithms

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;

use neolith::world::{Block, Chunk, ChunkPos, LodLevel, build_chunk_mesh, build_chunk_mesh_lod};
use neolith::constants::CHUNK_SIZE;

/// Create a chunk filled with a single block type
fn create_solid_chunk(block: Block) -> Arc<Chunk> {
    let mut chunk = Chunk::new();
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.set_block(x, y, z, block);
            }
        }
    }
    Arc::new(chunk)
}

/// Create a sparse chunk with random blocks
fn create_sparse_chunk(seed: u64, occupancy: f32) -> Arc<Chunk> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut chunk = Chunk::new();
    let threshold = (occupancy * u64::MAX as f32) as u64;

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let mut hasher = DefaultHasher::new();
                (seed, x, y, z).hash(&mut hasher);
                let hash = hasher.finish();

                if hash < threshold {
                    chunk.set_block(x, y, z, Block::Stone);
                }
            }
        }
    }
    Arc::new(chunk)
}

/// Create a checkerboard pattern (worst case for greedy meshing)
fn create_checkerboard_chunk() -> Arc<Chunk> {
    let mut chunk = Chunk::new();
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                if (x + y + z) % 2 == 0 {
                    chunk.set_block(x, y, z, Block::Stone);
                }
            }
        }
    }
    Arc::new(chunk)
}

/// Create a realistic terrain surface chunk (solid below, air above, some variation)
/// This simulates a typical surface terrain chunk with 60-70% fill
fn create_terrain_surface_chunk(seed: u64) -> Arc<Chunk> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut chunk = Chunk::new();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            // Compute height variation for this column (8-12 blocks high)
            let mut hasher = DefaultHasher::new();
            (seed, x, z).hash(&mut hasher);
            let height = 8 + (hasher.finish() % 5) as usize;

            for y in 0..height.min(CHUNK_SIZE) {
                // Mix of stone, dirt, and grass
                let mut hasher2 = DefaultHasher::new();
                (seed, x, y, z).hash(&mut hasher2);
                let block = if y == height - 1 {
                    Block::Grass
                } else if y >= height.saturating_sub(4) {
                    Block::Dirt
                } else {
                    Block::Stone
                };
                chunk.set_block(x, y, z, block);
            }
        }
    }
    Arc::new(chunk)
}

fn meshing_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("meshing");

    let pos = ChunkPos { x: 0, y: 0, z: 0 };
    let no_neighbors: [Option<Arc<Chunk>>; 6] = [None, None, None, None, None, None];

    // Benchmark empty chunk (all air - tests early exit optimization)
    // This should be extremely fast due to solid_count == 0 early exit
    let empty_chunk = Arc::new(Chunk::new());
    group.bench_function("empty_air", |b| {
        b.iter(|| {
            build_chunk_mesh(black_box(pos), black_box(&empty_chunk), black_box(&no_neighbors))
        });
    });

    // Benchmark solid chunk (minimal internal faces, all external)
    let solid_chunk = create_solid_chunk(Block::Stone);
    group.bench_function("solid_chunk", |b| {
        b.iter(|| {
            build_chunk_mesh(black_box(pos), black_box(&solid_chunk), black_box(&no_neighbors))
        });
    });

    // Benchmark sparse chunk (20% fill - typical terrain surface)
    let sparse_chunk = create_sparse_chunk(42, 0.2);
    group.bench_function("sparse_20pct", |b| {
        b.iter(|| {
            build_chunk_mesh(black_box(pos), black_box(&sparse_chunk), black_box(&no_neighbors))
        });
    });

    // Benchmark checkerboard (worst case - no greedy merging possible)
    let checker_chunk = create_checkerboard_chunk();
    group.bench_function("checkerboard", |b| {
        b.iter(|| {
            build_chunk_mesh(black_box(pos), black_box(&checker_chunk), black_box(&no_neighbors))
        });
    });

    // Benchmark realistic terrain surface (typical gameplay scenario)
    let terrain_chunk = create_terrain_surface_chunk(42);
    group.bench_function("terrain_surface", |b| {
        b.iter(|| {
            build_chunk_mesh(black_box(pos), black_box(&terrain_chunk), black_box(&no_neighbors))
        });
    });

    // Benchmark different LOD levels on sparse chunk
    for lod in [LodLevel::Full, LodLevel::Half, LodLevel::Quarter] {
        group.bench_with_input(
            BenchmarkId::new("lod_sparse", format!("{:?}", lod)),
            &lod,
            |b, &lod| {
                b.iter(|| {
                    build_chunk_mesh_lod(
                        black_box(pos),
                        black_box(&sparse_chunk),
                        black_box(&no_neighbors),
                        black_box(lod),
                    )
                });
            },
        );
    }

    // Benchmark different LOD levels on terrain surface chunk
    for lod in [LodLevel::Full, LodLevel::Half, LodLevel::Quarter] {
        group.bench_with_input(
            BenchmarkId::new("lod_terrain", format!("{:?}", lod)),
            &lod,
            |b, &lod| {
                b.iter(|| {
                    build_chunk_mesh_lod(
                        black_box(pos),
                        black_box(&terrain_chunk),
                        black_box(&no_neighbors),
                        black_box(lod),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, meshing_benchmarks);
criterion_main!(benches);
