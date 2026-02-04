//! Benchmarks for cloud generation system
//!
//! Tests the voxel grid generation and mesh building that happens on the background thread.
//! Target: optimize the ~160ms Ultra quality generation time.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Instant;

use mosaic::renderer::{
    CloudGrid, CloudMesh, CloudQuality,
    CLOUD_GRID_SIZE, CLOUD_VOXEL_SCALE, CLOUD_BASE_Y,
};

/// Simulate the full cloud generation pipeline as done in the background thread
fn generate_clouds_full(quality: CloudQuality, coverage: f32, center_x: f32, center_z: f32) -> (usize, usize) {
    let voxel_scale = quality.voxel_scale();
    let grid_size = CLOUD_GRID_SIZE / (voxel_scale / CLOUD_VOXEL_SCALE).max(1);
    let thickness = quality.cloud_thickness();

    // Create grid (same as background thread)
    let mut grid = CloudGrid::new(grid_size, thickness, grid_size, voxel_scale);

    // Generate voxels (the slow part)
    grid.generate(center_x, center_z, coverage, 12345);

    // Generate mesh
    let mesh = CloudMesh::from_grid(&grid, CLOUD_BASE_Y);

    (mesh.vertices.len(), mesh.indices.len())
}

/// Benchmark just the voxel generation (noise evaluation)
fn bench_voxel_generation(quality: CloudQuality, coverage: f32, center_x: f32, center_z: f32) {
    let voxel_scale = quality.voxel_scale();
    let grid_size = CLOUD_GRID_SIZE / (voxel_scale / CLOUD_VOXEL_SCALE).max(1);
    let thickness = quality.cloud_thickness();

    let mut grid = CloudGrid::new(grid_size, thickness, grid_size, voxel_scale);
    grid.generate(center_x, center_z, coverage, 12345);
}

/// Benchmark just the mesh generation (greedy meshing)
fn bench_mesh_generation(quality: CloudQuality, coverage: f32, center_x: f32, center_z: f32) -> CloudMesh {
    let voxel_scale = quality.voxel_scale();
    let grid_size = CLOUD_GRID_SIZE / (voxel_scale / CLOUD_VOXEL_SCALE).max(1);
    let thickness = quality.cloud_thickness();

    let mut grid = CloudGrid::new(grid_size, thickness, grid_size, voxel_scale);
    grid.generate(center_x, center_z, coverage, 12345);

    CloudMesh::from_grid(&grid, CLOUD_BASE_Y)
}

fn cloud_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("clouds");

    // Configure for longer measurements since these are slow operations
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(10));

    let center_x = 500.0;
    let center_z = 500.0;
    let coverage = 0.5; // 50% coverage (default)

    // === FULL PIPELINE BENCHMARKS (what we're optimizing) ===

    // Low quality - baseline (should be ~10ms)
    group.bench_function("full_pipeline/low", |b| {
        b.iter(|| {
            generate_clouds_full(
                black_box(CloudQuality::Low),
                black_box(coverage),
                black_box(center_x),
                black_box(center_z),
            )
        });
    });

    // Medium quality (~75ms)
    group.bench_function("full_pipeline/medium", |b| {
        b.iter(|| {
            generate_clouds_full(
                black_box(CloudQuality::Medium),
                black_box(coverage),
                black_box(center_x),
                black_box(center_z),
            )
        });
    });

    // High quality (~80ms)
    group.bench_function("full_pipeline/high", |b| {
        b.iter(|| {
            generate_clouds_full(
                black_box(CloudQuality::High),
                black_box(coverage),
                black_box(center_x),
                black_box(center_z),
            )
        });
    });

    // Ultra quality - TARGET (~160ms, want to reduce)
    group.bench_function("full_pipeline/ultra", |b| {
        b.iter(|| {
            generate_clouds_full(
                black_box(CloudQuality::Ultra),
                black_box(coverage),
                black_box(center_x),
                black_box(center_z),
            )
        });
    });

    // === VOXEL GENERATION ONLY (noise evaluation) ===

    group.bench_function("voxel_gen/ultra", |b| {
        b.iter(|| {
            bench_voxel_generation(
                black_box(CloudQuality::Ultra),
                black_box(coverage),
                black_box(center_x),
                black_box(center_z),
            )
        });
    });

    // === COVERAGE VARIATIONS ===

    for cov in [0.25, 0.5, 0.75, 1.0] {
        group.bench_with_input(
            BenchmarkId::new("coverage_ultra", format!("{:.0}pct", cov * 100.0)),
            &cov,
            |b, &cov| {
                b.iter(|| {
                    generate_clouds_full(
                        black_box(CloudQuality::Ultra),
                        black_box(cov),
                        black_box(center_x),
                        black_box(center_z),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Quick sanity check - run once and print stats
fn print_baseline_stats() {
    println!("\n=== Cloud Generation Baseline Stats ===\n");

    let center_x = 500.0;
    let center_z = 500.0;
    let coverage = 0.5;

    for quality in [CloudQuality::Low, CloudQuality::Medium, CloudQuality::High, CloudQuality::Ultra] {
        let start = Instant::now();
        let (verts, indices) = generate_clouds_full(quality, coverage, center_x, center_z);
        let elapsed = start.elapsed();

        println!(
            "{:?}: {:.1}ms - {} verts, {} quads",
            quality,
            elapsed.as_secs_f32() * 1000.0,
            verts,
            indices / 6
        );
    }
    println!();
}

criterion_group!(benches, cloud_benchmarks);
criterion_main!(benches);

// Uncomment to run quick stats check:
// fn main() { print_baseline_stats(); }
