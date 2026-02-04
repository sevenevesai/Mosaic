//! Detailed profiling benchmark for chunk generation
//!
//! Breaks down generation time by stage to identify optimization targets.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Instant;

use mosaic::world::{ChunkPos, generate_chunk_with_seed};

/// Profile breakdown of chunk generation stages
fn profile_generation_breakdown() {
    println!("\n=== Generation Stage Breakdown ===\n");

    let seed = 12345u64;
    let iterations = 100;

    // Warm up the thread-local cache
    let _ = generate_chunk_with_seed(ChunkPos { x: 0, y: 2, z: 0 }, seed);

    // Profile surface chunks (most complex)
    println!("Surface Chunks (y=2) - {} iterations:", iterations);
    let positions: Vec<ChunkPos> = (0..iterations)
        .map(|i| ChunkPos { x: i as i32, y: 2, z: 0 })
        .collect();

    let start = Instant::now();
    for pos in &positions {
        let _ = generate_chunk_with_seed(*pos, seed);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Average: {:.3}ms per chunk", avg_ms);
    println!("  Total: {:.1}ms for {} chunks", elapsed.as_secs_f64() * 1000.0, iterations);

    // Profile sky chunks (early exit)
    println!("\nSky Chunks (y=10) - {} iterations:", iterations);
    let positions: Vec<ChunkPos> = (0..iterations)
        .map(|i| ChunkPos { x: i as i32, y: 10, z: 0 })
        .collect();

    let start = Instant::now();
    for pos in &positions {
        let _ = generate_chunk_with_seed(*pos, seed);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Average: {:.3}ms per chunk", avg_ms);
    println!("  Total: {:.1}ms for {} chunks", elapsed.as_secs_f64() * 1000.0, iterations);

    // Underground chunks
    println!("\nUnderground Chunks (y=0) - {} iterations:", iterations);
    let positions: Vec<ChunkPos> = (0..iterations)
        .map(|i| ChunkPos { x: i as i32, y: 0, z: 0 })
        .collect();

    let start = Instant::now();
    for pos in &positions {
        let _ = generate_chunk_with_seed(*pos, seed);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Average: {:.3}ms per chunk", avg_ms);

    // Deep underground (mostly stone)
    println!("\nDeep Chunks (y=-2) - {} iterations:", iterations);
    let positions: Vec<ChunkPos> = (0..iterations)
        .map(|i| ChunkPos { x: i as i32, y: -2, z: 0 })
        .collect();

    let start = Instant::now();
    for pos in &positions {
        let _ = generate_chunk_with_seed(*pos, seed);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Average: {:.3}ms per chunk", avg_ms);

    println!();
}

fn generation_profile_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_profile");
    group.sample_size(50);

    let seed = 12345u64;

    // Surface chunk - the main optimization target
    group.bench_function("surface_y2", |b| {
        b.iter(|| {
            generate_chunk_with_seed(black_box(ChunkPos { x: 0, y: 2, z: 0 }), black_box(seed))
        });
    });

    // Varying X positions to ensure different terrain
    group.bench_function("surface_varied", |b| {
        let mut x = 0i32;
        b.iter(|| {
            x = x.wrapping_add(17); // Prime number for good distribution
            generate_chunk_with_seed(black_box(ChunkPos { x, y: 2, z: 0 }), black_box(seed))
        });
    });

    group.finish();
}

criterion_group!(benches, generation_profile_benchmarks);
criterion_main!(benches);

// Run: cargo bench --bench generation_profile_bench -- --nocapture
// Or uncomment below for quick stats:
// fn main() { profile_generation_breakdown(); }
