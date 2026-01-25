//! Benchmarks for terrain generation

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use neolith::world::{ChunkPos, generate_chunk_with_seed};

fn generation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation");

    // Configure for slower operations
    group.sample_size(50);

    // Benchmark full chunk generation at different positions
    group.bench_function("chunk_at_origin", |b| {
        let pos = ChunkPos { x: 0, y: 2, z: 0 }; // Y=2 is surface level
        b.iter(|| {
            generate_chunk_with_seed(black_box(pos), black_box(12345))
        });
    });

    // Underground chunk (caves, ores)
    group.bench_function("chunk_underground", |b| {
        let pos = ChunkPos { x: 0, y: 0, z: 0 }; // Y=0 is deep underground
        b.iter(|| {
            generate_chunk_with_seed(black_box(pos), black_box(12345))
        });
    });

    // High altitude chunk (mostly air)
    group.bench_function("chunk_sky", |b| {
        let pos = ChunkPos { x: 0, y: 10, z: 0 }; // Y=10 is above terrain
        b.iter(|| {
            generate_chunk_with_seed(black_box(pos), black_box(12345))
        });
    });

    // Different seeds (affects terrain features)
    for seed in [1, 42, 12345, 999999] {
        group.bench_function(format!("seed_{}", seed), |b| {
            let pos = ChunkPos { x: 0, y: 2, z: 0 };
            b.iter(|| {
                generate_chunk_with_seed(black_box(pos), black_box(seed))
            });
        });
    }

    // Distant chunk (different terrain features)
    group.bench_function("chunk_distant", |b| {
        let pos = ChunkPos { x: 100, y: 2, z: 100 };
        b.iter(|| {
            generate_chunk_with_seed(black_box(pos), black_box(12345))
        });
    });

    group.finish();
}

criterion_group!(benches, generation_benchmarks);
criterion_main!(benches);
