//! Benchmarks for shadow DDA ray tracing algorithms
//!
//! Simulates voxel shadow tracing on CPU to allow rapid iteration on
//! step sizes, algorithms, and quality presets without running the game.
//!
//! Run: `cargo bench shadow`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

/// Voxel volume size (matches GPU volume)
const VOLUME_SIZE: usize = 256;
const HALF_SIZE: f32 = 128.0;

/// A 256^3 voxel volume stored as flat array of opacity values (0.0 = air, 1.0 = solid)
struct VoxelVolume {
    /// Flat array: index = z * 256*256 + y * 256 + x
    data: Vec<f32>,
}

impl VoxelVolume {
    fn new() -> Self {
        Self {
            data: vec![0.0; VOLUME_SIZE * VOLUME_SIZE * VOLUME_SIZE],
        }
    }

    #[inline]
    fn set(&mut self, x: usize, y: usize, z: usize, opacity: f32) {
        if x < VOLUME_SIZE && y < VOLUME_SIZE && z < VOLUME_SIZE {
            self.data[z * VOLUME_SIZE * VOLUME_SIZE + y * VOLUME_SIZE + x] = opacity;
        }
    }

    /// Sample with integer coordinates (no interpolation - matches textureLoad)
    #[inline]
    fn load(&self, x: i32, y: i32, z: i32) -> f32 {
        if x < 0 || y < 0 || z < 0 {
            return 0.0;
        }
        let (ux, uy, uz) = (x as usize, y as usize, z as usize);
        if ux >= VOLUME_SIZE || uy >= VOLUME_SIZE || uz >= VOLUME_SIZE {
            return 0.0;
        }
        self.data[uz * VOLUME_SIZE * VOLUME_SIZE + uy * VOLUME_SIZE + ux]
    }

    /// Sample with linear interpolation (matches textureSampleLevel with Linear filter)
    #[inline]
    fn sample_linear(&self, u: f32, v: f32, w: f32) -> f32 {
        // Convert UV [0,1] to voxel coords
        let x = u * VOLUME_SIZE as f32 - 0.5;
        let y = v * VOLUME_SIZE as f32 - 0.5;
        let z = w * VOLUME_SIZE as f32 - 0.5;

        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let z0 = z.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let fz = z - z0 as f32;

        // Trilinear interpolation
        let c000 = self.load(x0, y0, z0);
        let c100 = self.load(x0 + 1, y0, z0);
        let c010 = self.load(x0, y0 + 1, z0);
        let c110 = self.load(x0 + 1, y0 + 1, z0);
        let c001 = self.load(x0, y0, z0 + 1);
        let c101 = self.load(x0 + 1, y0, z0 + 1);
        let c011 = self.load(x0, y0 + 1, z0 + 1);
        let c111 = self.load(x0 + 1, y0 + 1, z0 + 1);

        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;

        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;

        c0 * (1.0 - fz) + c1 * fz
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCENARIO BUILDERS
// ═══════════════════════════════════════════════════════════════════════════

/// Flat ground at y=64 with a vertical wall at x=140
fn create_wall_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    // Ground plane
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    // Vertical wall at x=140, height 64-80
    for z in 100..160 {
        for y in 64..80 {
            vol.set(140, y, z, 1.0);
        }
    }
    vol
}

/// Dense tree canopy - scattered blocks above ground
fn create_canopy_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    // Ground
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    // Tree trunks and canopy (simple pseudo-random pattern)
    for tz in (80..200).step_by(8) {
        for tx in (80..200).step_by(8) {
            // Trunk
            for y in 64..72 {
                vol.set(tx, y, tz, 1.0);
            }
            // Canopy (3x3x2 block)
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    for y in 72..74 {
                        let cx = (tx as i32 + dx) as usize;
                        let cz = (tz as i32 + dz) as usize;
                        if cx < VOLUME_SIZE && cz < VOLUME_SIZE {
                            vol.set(cx, y, cz, 1.0);
                        }
                    }
                }
            }
        }
    }
    vol
}

/// Cave - ground with roof overhead
fn create_cave_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    // Floor
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..60 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    // Ceiling
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 68..VOLUME_SIZE {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    vol
}

/// Single isolated pillar - worst case for gap detection
fn create_pillar_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    // Ground
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    // Single 1x1 pillar at center
    for y in 64..80 {
        vol.set(128, y, 128, 1.0);
    }
    vol
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM: CURRENT NAIVE STEPPER (matches pipeline.rs ray_march_shadow_hard)
// ═══════════════════════════════════════════════════════════════════════════

/// Current naive adaptive-step shadow tracer (baseline)
/// Mirrors the WGSL ray_march_shadow_hard() from pipeline.rs
fn trace_naive(vol: &VoxelVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let volume_center = [HALF_SIZE; 3];
    let mut accumulated_opacity: f32 = 0.0;
    let mut solid_hits = 0i32;

    let mut pos = [
        world_pos[0] + light_dir[0] * 0.6,
        world_pos[1] + light_dir[1] * 0.6,
        world_pos[2] + light_dir[2] * 0.6,
    ];

    for i in 0..max_steps {
        let current_step = if i < 8 {
            0.35
        } else if i < 24 {
            0.5
        } else if i < 56 {
            1.0
        } else {
            2.0
        };

        pos[0] += light_dir[0] * current_step;
        pos[1] += light_dir[1] * current_step;
        pos[2] += light_dir[2] * current_step;

        // Check distance from center
        let dx = pos[0] - volume_center[0];
        let dy = pos[1] - volume_center[1];
        let dz = pos[2] - volume_center[2];
        let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if center_dist > 120.0 {
            break;
        }

        // Edge fade
        let edge_fade = 1.0 - smoothstep(100.0, 120.0, center_dist);

        // Sample with linear interpolation (current behavior)
        let u = (pos[0] - volume_center[0] + HALF_SIZE) / VOLUME_SIZE as f32;
        let v = (pos[1] - volume_center[1] + HALF_SIZE) / VOLUME_SIZE as f32;
        let w = (pos[2] - volume_center[2] + HALF_SIZE) / VOLUME_SIZE as f32;

        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || w < 0.0 || w > 1.0 {
            break;
        }

        let opacity = vol.sample_linear(u, v, w) * edge_fade;

        if opacity > 0.5 {
            solid_hits += 1;
            accumulated_opacity += 0.4;
            if solid_hits >= 3 {
                break;
            }
        } else if opacity > 0.1 {
            accumulated_opacity += opacity * 0.2;
        }

        if accumulated_opacity >= 1.1 {
            break;
        }
    }

    if solid_hits > 0 {
        let strength = smoothstep(0.3, 0.7, accumulated_opacity.clamp(0.0, 1.0));
        lerp(1.0, 0.25, strength)
    } else if accumulated_opacity > 0.0 {
        1.0 - (accumulated_opacity * 0.5).clamp(0.0, 0.5)
    } else {
        1.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM: PROPER AMANATIDES & WOO DDA
// ═══════════════════════════════════════════════════════════════════════════

/// True DDA shadow tracer (Amanatides & Woo 1987)
/// Steps exactly one voxel boundary per iteration - never skips a voxel.
fn trace_dda(vol: &VoxelVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let volume_center = [HALF_SIZE; 3];

    // Convert world position to voxel-space coordinates
    // Offset slightly along normal + light to escape source voxel
    let start = [
        world_pos[0] + light_dir[0] * 0.5,
        world_pos[1] + light_dir[1] * 0.5,
        world_pos[2] + light_dir[2] * 0.5,
    ];

    // Current integer voxel position
    let mut voxel = [
        start[0].floor() as i32,
        start[1].floor() as i32,
        start[2].floor() as i32,
    ];

    // Step direction per axis: +1 or -1
    let step = [
        if light_dir[0] >= 0.0 { 1i32 } else { -1 },
        if light_dir[1] >= 0.0 { 1i32 } else { -1 },
        if light_dir[2] >= 0.0 { 1i32 } else { -1 },
    ];

    // tDelta: how far along the ray (in t) to cross one voxel in each axis
    let t_delta = [
        if light_dir[0].abs() > 1e-8 { (1.0 / light_dir[0]).abs() } else { f32::MAX },
        if light_dir[1].abs() > 1e-8 { (1.0 / light_dir[1]).abs() } else { f32::MAX },
        if light_dir[2].abs() > 1e-8 { (1.0 / light_dir[2]).abs() } else { f32::MAX },
    ];

    // tMax: distance along ray to first boundary crossing per axis
    let t_max_init = [
        if light_dir[0] >= 0.0 {
            ((voxel[0] as f32 + 1.0) - start[0]) * t_delta[0]
        } else {
            (start[0] - voxel[0] as f32) * t_delta[0]
        },
        if light_dir[1] >= 0.0 {
            ((voxel[1] as f32 + 1.0) - start[1]) * t_delta[1]
        } else {
            (start[1] - voxel[1] as f32) * t_delta[1]
        },
        if light_dir[2] >= 0.0 {
            ((voxel[2] as f32 + 1.0) - start[2]) * t_delta[2]
        } else {
            (start[2] - voxel[2] as f32) * t_delta[2]
        },
    ];
    let mut t_max = t_max_init;

    let mut accumulated_opacity: f32 = 0.0;
    let mut solid_hits = 0i32;

    for _i in 0..max_steps {
        // Check bounds (distance from volume center)
        let dx = voxel[0] as f32 + 0.5 - volume_center[0];
        let dy = voxel[1] as f32 + 0.5 - volume_center[1];
        let dz = voxel[2] as f32 + 0.5 - volume_center[2];
        let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if center_dist > 120.0 {
            break;
        }

        // Edge fade
        let edge_fade = 1.0 - smoothstep(100.0, 120.0, center_dist);

        // Sample at integer voxel position (no interpolation - no edge leaks)
        let opacity = vol.load(voxel[0], voxel[1], voxel[2]) * edge_fade;

        if opacity > 0.5 {
            solid_hits += 1;
            accumulated_opacity += 0.4;
            if solid_hits >= 3 {
                break;
            }
        } else if opacity > 0.1 {
            accumulated_opacity += opacity * 0.2;
        }

        if accumulated_opacity >= 1.1 {
            break;
        }

        // Step to next voxel boundary (Amanatides & Woo)
        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                voxel[0] += step[0];
                t_max[0] += t_delta[0];
            } else {
                voxel[2] += step[2];
                t_max[2] += t_delta[2];
            }
        } else {
            if t_max[1] < t_max[2] {
                voxel[1] += step[1];
                t_max[1] += t_delta[1];
            } else {
                voxel[2] += step[2];
                t_max[2] += t_delta[2];
            }
        }
    }

    // Same shadow evaluation as naive
    if solid_hits > 0 {
        let strength = smoothstep(0.3, 0.7, accumulated_opacity.clamp(0.0, 1.0));
        lerp(1.0, 0.25, strength)
    } else if accumulated_opacity > 0.0 {
        1.0 - (accumulated_opacity * 0.5).clamp(0.0, 0.5)
    } else {
        1.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BRANCHLESS DDA - Mirrors the GPU shader implementation
// Uses comparison masks instead of nested if/else for axis selection.
// On GPU this reduces warp divergence; on CPU it tests the same logic path.
// ═══════════════════════════════════════════════════════════════════════════

fn trace_dda_branchless(vol: &VoxelVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let start = [
        world_pos[0] + light_dir[0] * 0.5,
        world_pos[1] + light_dir[1] * 0.5,
        world_pos[2] + light_dir[2] * 0.5,
    ];

    let mut voxel = [
        start[0].floor() as i32,
        start[1].floor() as i32,
        start[2].floor() as i32,
    ];

    let step = [
        if light_dir[0] >= 0.0 { 1i32 } else { -1 },
        if light_dir[1] >= 0.0 { 1i32 } else { -1 },
        if light_dir[2] >= 0.0 { 1i32 } else { -1 },
    ];

    let t_delta = [
        if light_dir[0].abs() > 1e-8 { (1.0 / light_dir[0]).abs() } else { f32::MAX },
        if light_dir[1].abs() > 1e-8 { (1.0 / light_dir[1]).abs() } else { f32::MAX },
        if light_dir[2].abs() > 1e-8 { (1.0 / light_dir[2]).abs() } else { f32::MAX },
    ];

    // Branchless t_max init (matches shader select() pattern)
    let frac = [
        start[0] - start[0].floor(),
        start[1] - start[1].floor(),
        start[2] - start[2].floor(),
    ];
    let mut t_max = [
        if light_dir[0] >= 0.0 { (1.0 - frac[0]) * t_delta[0] } else { frac[0] * t_delta[0] },
        if light_dir[1] >= 0.0 { (1.0 - frac[1]) * t_delta[1] } else { frac[1] * t_delta[1] },
        if light_dir[2] >= 0.0 { (1.0 - frac[2]) * t_delta[2] } else { frac[2] * t_delta[2] },
    ];

    let mut accumulated_opacity: f32 = 0.0;
    let mut solid_hits = 0i32;

    for _i in 0..max_steps {
        let dx = voxel[0] as f32 + 0.5 - HALF_SIZE;
        let dy = voxel[1] as f32 + 0.5 - HALF_SIZE;
        let dz = voxel[2] as f32 + 0.5 - HALF_SIZE;
        let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if center_dist > 120.0 {
            break;
        }

        let edge_fade = 1.0 - smoothstep(100.0, 120.0, center_dist);
        let opacity = vol.load(voxel[0], voxel[1], voxel[2]) * edge_fade;

        if opacity > 0.5 {
            solid_hits += 1;
            accumulated_opacity += 0.4;
            if solid_hits >= 3 {
                break;
            }
        } else if opacity > 0.1 {
            accumulated_opacity += opacity * 0.2;
        }

        if accumulated_opacity >= 1.1 {
            break;
        }

        // Branchless axis selection using comparison masks
        let x_le_y = t_max[0] <= t_max[1];
        let x_le_z = t_max[0] <= t_max[2];
        let y_le_z = t_max[1] <= t_max[2];

        let step_x = x_le_y && x_le_z;
        let step_y = !x_le_y && y_le_z;
        let step_z = !step_x && !step_y;

        // Conditional add via bool-to-int multiplication
        voxel[0] += step[0] * step_x as i32;
        voxel[1] += step[1] * step_y as i32;
        voxel[2] += step[2] * step_z as i32;
        t_max[0] += t_delta[0] * step_x as i32 as f32;
        t_max[1] += t_delta[1] * step_y as i32 as f32;
        t_max[2] += t_delta[2] * step_z as i32 as f32;
    }

    if solid_hits > 0 {
        let strength = smoothstep(0.3, 0.7, accumulated_opacity.clamp(0.0, 1.0));
        lerp(1.0, 0.25, strength)
    } else if accumulated_opacity > 0.0 {
        1.0 - (accumulated_opacity * 0.5).clamp(0.0, 0.5)
    } else {
        1.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-8 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Generate test rays: surface positions looking toward sun at various angles
fn generate_test_rays(count: usize) -> Vec<([f32; 3], [f32; 3])> {
    let mut rays = Vec::with_capacity(count);

    // Sun directions at various times of day
    let sun_dirs = [
        normalize([0.5, 0.8, 0.3]),   // High noon-ish
        normalize([0.7, 0.3, 0.1]),   // Low sun (dawn/dusk) - gap-prone
        normalize([-0.4, 0.6, 0.5]),  // Afternoon
        normalize([0.1, 0.9, 0.1]),   // Nearly overhead
        normalize([0.8, 0.15, 0.2]),  // Very low angle - most gap-prone
    ];

    let rays_per_dir = count / sun_dirs.len();

    for sun_dir in &sun_dirs {
        for i in 0..rays_per_dir {
            // Spread positions across the ground surface (y=64)
            let x = 80.0 + (i % 100) as f32 * 0.8;
            let z = 80.0 + (i / 100) as f32 * 0.8;
            rays.push(([x, 64.0, z], *sun_dir));
        }
    }

    rays
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn shadow_benchmarks(c: &mut Criterion) {
    let wall_vol = create_wall_scenario();
    let canopy_vol = create_canopy_scenario();
    let cave_vol = create_cave_scenario();
    let pillar_vol = create_pillar_scenario();
    let rays = generate_test_rays(500);

    // --- Algorithm comparison (wall scenario) ---
    {
        let mut group = c.benchmark_group("shadow/algorithm");

        group.bench_function("naive_112steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_naive(black_box(&wall_vol), *pos, *dir, 112);
                }
                total
            });
        });

        group.bench_function("dda_128steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda(black_box(&wall_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("dda_96steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda(black_box(&wall_vol), *pos, *dir, 96);
                }
                total
            });
        });

        group.bench_function("dda_64steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda(black_box(&wall_vol), *pos, *dir, 64);
                }
                total
            });
        });

        group.bench_function("dda_48steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda(black_box(&wall_vol), *pos, *dir, 48);
                }
                total
            });
        });

        group.bench_function("dda_branchless_128steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda_branchless(black_box(&wall_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("dda_branchless_96steps", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda_branchless(black_box(&wall_vol), *pos, *dir, 96);
                }
                total
            });
        });

        group.finish();
    }

    // --- Quality presets across scenarios ---
    {
        let scenarios: Vec<(&str, &VoxelVolume)> = vec![
            ("wall", &wall_vol),
            ("canopy", &canopy_vol),
            ("cave", &cave_vol),
            ("pillar", &pillar_vol),
        ];

        let presets: Vec<(&str, i32)> = vec![
            ("low_48", 48),
            ("medium_96", 96),
            ("high_128", 128),
        ];

        for (scenario_name, vol) in &scenarios {
            let mut group = c.benchmark_group(format!("shadow/{}", scenario_name));

            for (preset_name, max_steps) in &presets {
                group.bench_with_input(
                    BenchmarkId::new("dda", preset_name),
                    max_steps,
                    |b, &steps| {
                        b.iter(|| {
                            let mut total = 0.0f32;
                            for (pos, dir) in &rays {
                                total += trace_dda(black_box(vol), *pos, *dir, steps);
                            }
                            total
                        });
                    },
                );
            }

            // Also benchmark naive for comparison
            group.bench_function("naive_112", |b| {
                b.iter(|| {
                    let mut total = 0.0f32;
                    for (pos, dir) in &rays {
                        total += trace_naive(black_box(vol), *pos, *dir, 112);
                    }
                    total
                });
            });

            group.finish();
        }
    }

    // --- Shadow correctness validation (not timed, but useful for iteration) ---
    // Test that DDA catches shadows the naive tracer misses
    {
        let mut group = c.benchmark_group("shadow/gap_detection");

        // Low-angle rays through single pillar - this is where gaps appear
        let low_sun = normalize([0.8, 0.15, 0.2]);
        let pillar_rays: Vec<_> = (0..100)
            .map(|i| {
                let z = 120.0 + i as f32 * 0.2;
                ([100.0f32, 64.0, z], low_sun)
            })
            .collect();

        group.bench_function("naive_pillar_low_angle", |b| {
            b.iter(|| {
                let mut shadowed = 0;
                for (pos, dir) in &pillar_rays {
                    let s = trace_naive(black_box(&pillar_vol), *pos, *dir, 112);
                    if s < 0.9 { shadowed += 1; }
                }
                shadowed
            });
        });

        group.bench_function("dda_pillar_low_angle", |b| {
            b.iter(|| {
                let mut shadowed = 0;
                for (pos, dir) in &pillar_rays {
                    let s = trace_dda(black_box(&pillar_vol), *pos, *dir, 128);
                    if s < 0.9 { shadowed += 1; }
                }
                shadowed
            });
        });

        group.finish();
    }
}

/// Print shadow accuracy comparison (run with `cargo bench shadow -- --nocapture`)
fn print_shadow_stats() {
    let pillar_vol = create_pillar_scenario();
    let low_sun = normalize([0.8, 0.15, 0.2]);

    println!("\n=== Shadow Gap Detection Test ===");
    println!("Scenario: Single pillar, low-angle sun");
    println!("{:<20} {:<15} {:<15}", "Algorithm", "Shadowed/100", "Shadow Value");
    println!("{:-<50}", "");

    let mut naive_shadowed = 0;
    let mut dda_shadowed = 0;

    for i in 0..100 {
        let z = 120.0 + i as f32 * 0.2;
        let pos = [100.0f32, 64.0, z];

        let naive_s = trace_naive(&pillar_vol, pos, low_sun, 112);
        let dda_s = trace_dda(&pillar_vol, pos, low_sun, 128);

        if naive_s < 0.9 { naive_shadowed += 1; }
        if dda_s < 0.9 { dda_shadowed += 1; }
    }

    println!("{:<20} {:<15} {:<15}", "Naive (112 steps)", naive_shadowed, "-");
    println!("{:<20} {:<15} {:<15}", "DDA (128 steps)", dda_shadowed, "-");
    println!("\nMore shadowed pixels = better shadow coverage (fewer gaps)");
}

criterion_group!(benches, shadow_benchmarks);
criterion_main!(benches);
