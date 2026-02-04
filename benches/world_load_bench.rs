//! Comprehensive benchmarks for world loading and chunk generation
//!
//! Tests the full chunk pipeline: generation, meshing, and parallel throughput
//! to identify bottlenecks in world loading performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use mosaic::world::{
    Block, Chunk, ChunkPos, LodLevel,
    generate_chunk_with_seed, build_chunk_mesh_lod,
};
use mosaic::constants::CHUNK_SIZE;

/// Generate chunks in a sphere pattern around origin (simulates initial world load)
fn generate_world_region(radius: i32, seed: u64) -> Vec<(ChunkPos, Chunk)> {
    let mut chunks = Vec::new();

    // Generate chunks in render distance sphere
    // Y range is typically 0-8 for surface terrain
    for y in 0..8 {
        for z in -radius..=radius {
            for x in -radius..=radius {
                // Skip corners for sphere-ish shape
                if x * x + z * z > radius * radius {
                    continue;
                }
                let pos = ChunkPos { x, y, z };
                let chunk = generate_chunk_with_seed(pos, seed);
                chunks.push((pos, chunk));
            }
        }
    }

    chunks
}

/// Generate chunks in parallel using Rayon (simulates 8 worker threads)
fn generate_world_region_parallel(radius: i32, seed: u64) -> Vec<(ChunkPos, Chunk)> {
    let mut positions = Vec::new();

    for y in 0..8 {
        for z in -radius..=radius {
            for x in -radius..=radius {
                if x * x + z * z > radius * radius {
                    continue;
                }
                positions.push(ChunkPos { x, y, z });
            }
        }
    }

    positions
        .into_par_iter()
        .map(|pos| {
            let chunk = generate_chunk_with_seed(pos, seed);
            (pos, chunk)
        })
        .collect()
}

/// Generate and mesh chunks in parallel (full pipeline)
fn generate_and_mesh_region_parallel(radius: i32, seed: u64) -> usize {
    let mut positions = Vec::new();

    for y in 0..8 {
        for z in -radius..=radius {
            for x in -radius..=radius {
                if x * x + z * z > radius * radius {
                    continue;
                }
                positions.push(ChunkPos { x, y, z });
            }
        }
    }

    // First pass: generate all chunks in parallel
    let chunks: Vec<(ChunkPos, Arc<Chunk>)> = positions
        .par_iter()
        .map(|&pos| {
            let chunk = generate_chunk_with_seed(pos, seed);
            (pos, Arc::new(chunk))
        })
        .collect();

    // Second pass: mesh all chunks in parallel (with no neighbors for benchmark simplicity)
    let no_neighbors: [Option<Arc<Chunk>>; 6] = [None, None, None, None, None, None];

    let total_vertices: usize = chunks
        .par_iter()
        .map(|(pos, chunk)| {
            let mesh = build_chunk_mesh_lod(*pos, chunk, &no_neighbors, LodLevel::Full);
            mesh.vertices.len()
        })
        .sum();

    total_vertices
}

/// Benchmark different chunk types to understand performance variation
fn bench_chunk_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_types");
    group.sample_size(50);

    let seed = 12345u64;

    // Surface chunks (most common, most complex)
    group.bench_function("surface_y2", |b| {
        b.iter(|| {
            generate_chunk_with_seed(black_box(ChunkPos { x: 0, y: 2, z: 0 }), black_box(seed))
        });
    });

    // Underground chunks (caves, ores)
    group.bench_function("underground_y0", |b| {
        b.iter(|| {
            generate_chunk_with_seed(black_box(ChunkPos { x: 0, y: 0, z: 0 }), black_box(seed))
        });
    });

    // Deep underground (mostly stone)
    group.bench_function("deep_y-2", |b| {
        b.iter(|| {
            generate_chunk_with_seed(black_box(ChunkPos { x: 0, y: -2, z: 0 }), black_box(seed))
        });
    });

    // Sky chunks (should be fast - early exit)
    group.bench_function("sky_y10", |b| {
        b.iter(|| {
            generate_chunk_with_seed(black_box(ChunkPos { x: 0, y: 10, z: 0 }), black_box(seed))
        });
    });

    // Various terrain at distance (different biomes)
    for dist in [10, 50, 100] {
        group.bench_function(format!("distant_x{}", dist), |b| {
            b.iter(|| {
                generate_chunk_with_seed(black_box(ChunkPos { x: dist, y: 2, z: 0 }), black_box(seed))
            });
        });
    }

    group.finish();
}

/// Benchmark parallel generation throughput (simulates world load)
fn bench_parallel_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_throughput");
    group.sample_size(10); // Fewer samples for slow operations

    let seed = 12345u64;

    // Small region (initial visible area) - radius 5 = ~600 chunks
    group.throughput(Throughput::Elements(count_chunks_in_radius(5) as u64));
    group.bench_function("radius_5_parallel", |b| {
        b.iter(|| {
            generate_world_region_parallel(black_box(5), black_box(seed))
        });
    });

    // Medium region (render distance 10) - ~2500 chunks
    group.throughput(Throughput::Elements(count_chunks_in_radius(10) as u64));
    group.bench_function("radius_10_parallel", |b| {
        b.iter(|| {
            generate_world_region_parallel(black_box(10), black_box(seed))
        });
    });

    // Compare sequential vs parallel for radius 5
    group.throughput(Throughput::Elements(count_chunks_in_radius(5) as u64));
    group.bench_function("radius_5_sequential", |b| {
        b.iter(|| {
            generate_world_region(black_box(5), black_box(seed))
        });
    });

    group.finish();
}

/// Benchmark full pipeline (generate + mesh)
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);

    let seed = 12345u64;

    // Small region with full pipeline
    let chunk_count = count_chunks_in_radius(5);
    group.throughput(Throughput::Elements(chunk_count as u64));
    group.bench_function("gen_mesh_radius_5", |b| {
        b.iter(|| {
            generate_and_mesh_region_parallel(black_box(5), black_box(seed))
        });
    });

    group.finish();
}

/// Quick stats printout for development
fn print_baseline_stats() {
    println!("\n=== World Generation Baseline Stats ===\n");

    let seed = 12345u64;

    // Single chunk timings
    println!("Single Chunk Generation:");
    for (name, pos) in [
        ("Surface (y=2)", ChunkPos { x: 0, y: 2, z: 0 }),
        ("Underground (y=0)", ChunkPos { x: 0, y: 0, z: 0 }),
        ("Sky (y=10)", ChunkPos { x: 0, y: 10, z: 0 }),
        ("Distant (x=100)", ChunkPos { x: 100, y: 2, z: 0 }),
    ] {
        let start = Instant::now();
        let chunk = generate_chunk_with_seed(pos, seed);
        let elapsed = start.elapsed();
        let solid_count = count_solid_blocks(&chunk);
        println!("  {}: {:.2}ms ({} solid blocks)", name, elapsed.as_secs_f32() * 1000.0, solid_count);
    }

    // Parallel throughput
    println!("\nParallel Generation Throughput:");
    for radius in [5, 10, 15] {
        let chunk_count = count_chunks_in_radius(radius);
        let start = Instant::now();
        let chunks = generate_world_region_parallel(radius, seed);
        let elapsed = start.elapsed();
        let chunks_per_sec = chunks.len() as f32 / elapsed.as_secs_f32();
        println!(
            "  Radius {}: {} chunks in {:.1}ms ({:.0} chunks/sec)",
            radius,
            chunks.len(),
            elapsed.as_secs_f32() * 1000.0,
            chunks_per_sec
        );
    }

    // Full pipeline (gen + mesh)
    println!("\nFull Pipeline (Gen + Mesh):");
    for radius in [5, 10] {
        let chunk_count = count_chunks_in_radius(radius);
        let start = Instant::now();
        let total_verts = generate_and_mesh_region_parallel(radius, seed);
        let elapsed = start.elapsed();
        println!(
            "  Radius {}: {} chunks, {} verts in {:.1}ms",
            radius,
            chunk_count,
            total_verts,
            elapsed.as_secs_f32() * 1000.0
        );
    }

    println!();
}

fn count_chunks_in_radius(radius: i32) -> usize {
    let mut count = 0;
    for y in 0..8 {
        for z in -radius..=radius {
            for x in -radius..=radius {
                if x * x + z * z <= radius * radius {
                    count += 1;
                }
            }
        }
    }
    count
}

fn count_solid_blocks(chunk: &Chunk) -> usize {
    let mut count = 0;
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                if chunk.get_block(x, y, z).is_solid() {
                    count += 1;
                }
            }
        }
    }
    count
}

criterion_group!(
    benches,
    bench_chunk_types,
    bench_parallel_throughput,
    bench_full_pipeline,
);
criterion_main!(benches);

// Uncomment to run quick stats check:
// fn main() { print_baseline_stats(); }
