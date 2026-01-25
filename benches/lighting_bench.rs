//! Benchmarks for lighting system

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;

use ahash::AHashMap;
use parking_lot::RwLock;

use neolith::world::{Block, Chunk, ChunkPos, LightEngine, apply_light_updates};
use neolith::constants::CHUNK_SIZE;

/// Create a 3x3x3 grid of empty chunks centered at origin
fn create_chunk_grid() -> Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>> {
    let mut map = AHashMap::new();

    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let pos = ChunkPos { x, y, z };
                map.insert(pos, Arc::new(Chunk::new()));
            }
        }
    }

    Arc::new(RwLock::new(map))
}

/// Create a grid with some solid blocks to test light blocking
fn create_partially_solid_grid() -> Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>> {
    let mut map = AHashMap::new();

    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let mut chunk = Chunk::new();

                // Add some walls in the center chunk
                if x == 0 && y == 0 && z == 0 {
                    for bz in 0..CHUNK_SIZE {
                        for by in 0..CHUNK_SIZE {
                            // Vertical wall at x=4
                            chunk.set_block(4, by, bz, Block::Stone);
                        }
                    }
                }

                let pos = ChunkPos { x, y, z };
                map.insert(pos, Arc::new(chunk));
            }
        }
    }

    Arc::new(RwLock::new(map))
}

fn lighting_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("lighting");

    // Benchmark single torch propagation in empty space
    group.bench_function("propagate_torch_empty", |b| {
        let chunks = create_chunk_grid();
        let mut engine = LightEngine::new();

        b.iter(|| {
            let updates = engine.propagate_light(
                black_box(8), black_box(8), black_box(8),
                black_box(14),
                black_box(&chunks),
            );
            engine.take_dirty_chunks(); // Reset for next iteration
            updates
        });
    });

    // Benchmark propagation with obstacles
    group.bench_function("propagate_torch_obstacles", |b| {
        let chunks = create_partially_solid_grid();
        let mut engine = LightEngine::new();

        b.iter(|| {
            let updates = engine.propagate_light(
                black_box(8), black_box(8), black_box(8),
                black_box(14),
                black_box(&chunks),
            );
            engine.take_dirty_chunks();
            updates
        });
    });

    // Benchmark light removal
    group.bench_function("remove_light", |b| {
        b.iter_batched(
            || {
                // Setup: create grid with light already propagated
                let chunks = create_chunk_grid();
                let mut engine = LightEngine::new();
                let updates = engine.propagate_light(8, 8, 8, 14, &chunks);
                apply_light_updates(updates, &chunks);
                engine.take_dirty_chunks();
                (chunks, engine)
            },
            |(chunks, mut engine)| {
                // Benchmark the removal
                let updates = engine.remove_light(
                    black_box(8), black_box(8), black_box(8),
                    black_box(&chunks),
                );
                engine.take_dirty_chunks();
                updates
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark cross-chunk propagation (light source at chunk boundary)
    group.bench_function("propagate_cross_chunk", |b| {
        let chunks = create_chunk_grid();
        let mut engine = LightEngine::new();

        b.iter(|| {
            // Place light at edge of chunk (will cross into neighbors)
            let updates = engine.propagate_light(
                black_box(0), black_box(8), black_box(8),
                black_box(14),
                black_box(&chunks),
            );
            engine.take_dirty_chunks();
            updates
        });
    });

    group.finish();
}

criterion_group!(benches, lighting_benchmarks);
criterion_main!(benches);
