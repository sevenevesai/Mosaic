//! Engine-accurate shadow DDA benchmarks
//!
//! Ports the EXACT shader logic from pipeline.rs to CPU for benchmarking.
//! Tests three variants:
//!   1. Baseline DDA-128 (current production, no bitmask)
//!   2. Bitmask DDA with max_safe cap (current engine implementation)
//!   3. Bitmask DDA unconstrained (original benchmark winner, no cross-axis cap)
//!
//! Run: `cargo bench shadow_engine`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const VOLUME_SIZE: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════
// SHARED TYPES
// ═══════════════════════════════════════════════════════════════════════════

struct VoxelVolume {
    data: Vec<f32>,
}

impl VoxelVolume {
    fn new() -> Self {
        Self { data: vec![0.0; VOLUME_SIZE * VOLUME_SIZE * VOLUME_SIZE] }
    }

    #[inline(always)]
    fn set(&mut self, x: usize, y: usize, z: usize, opacity: f32) {
        if x < VOLUME_SIZE && y < VOLUME_SIZE && z < VOLUME_SIZE {
            self.data[z * VOLUME_SIZE * VOLUME_SIZE + y * VOLUME_SIZE + x] = opacity;
        }
    }

    #[inline(always)]
    fn load(&self, x: i32, y: i32, z: i32) -> f32 {
        if x < 0 || y < 0 || z < 0 { return 0.0; }
        let (ux, uy, uz) = (x as usize, y as usize, z as usize);
        if ux >= VOLUME_SIZE || uy >= VOLUME_SIZE || uz >= VOLUME_SIZE { return 0.0; }
        unsafe { *self.data.get_unchecked(uz * VOLUME_SIZE * VOLUME_SIZE + uy * VOLUME_SIZE + ux) }
    }
}

struct TriAxisBitmask {
    /// rows_x[z * 256 + y] = 256-bit mask along X axis (stored as 8 u32s to match GPU)
    rows_x: Vec<[u32; 8]>,
    /// rows_y[z * 256 + x] = 256-bit mask along Y axis
    rows_y: Vec<[u32; 8]>,
    /// rows_z[y * 256 + x] = 256-bit mask along Z axis
    rows_z: Vec<[u32; 8]>,
}

impl TriAxisBitmask {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let n = VOLUME_SIZE;
        let mut rows_x = vec![[0u32; 8]; n * n];
        let mut rows_y = vec![[0u32; 8]; n * n];
        let mut rows_z = vec![[0u32; 8]; n * n];

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if vol.data[z * n * n + y * n + x] > 0.5 {
                        // Match GPU indexing exactly:
                        // X-rows: index = (y * 256 + z) * 8 + (x / 32), bit = x % 32
                        rows_x[y * n + z][x / 32] |= 1u32 << (x % 32);
                        // Y-rows: index = (x * 256 + z) * 8 + (y / 32), bit = y % 32
                        rows_y[x * n + z][y / 32] |= 1u32 << (y % 32);
                        // Z-rows: index = (x * 256 + y) * 8 + (z / 32), bit = z % 32
                        rows_z[x * n + y][z / 32] |= 1u32 << (z % 32);
                    }
                }
            }
        }
        Self { rows_x, rows_y, rows_z }
    }

    /// Find next set bit at or after `start` in positive direction (matches GPU find_pos)
    #[inline]
    fn find_pos(row: &[u32; 8], start: i32) -> i32 {
        if start >= 256 || start < 0 { return -1; }
        let s = start as u32;
        let word_idx = (s / 32) as usize;
        let bit_idx = s % 32;
        let masked = row[word_idx] & (0xFFFFFFFFu32 << bit_idx);
        if masked != 0 { return (word_idx as i32) * 32 + masked.trailing_zeros() as i32; }
        for w in (word_idx + 1)..8 {
            if row[w] != 0 { return (w as i32) * 32 + row[w].trailing_zeros() as i32; }
        }
        -1
    }

    /// Find next set bit at or before `start` in negative direction (matches GPU find_neg)
    #[inline]
    fn find_neg(row: &[u32; 8], start: i32) -> i32 {
        if start >= 256 || start < 0 { return -1; }
        let s = start as u32;
        let word_idx = (s / 32) as usize;
        let bit_idx = s % 32;
        let masked = row[word_idx] & (0xFFFFFFFFu32 >> (31 - bit_idx));
        if masked != 0 { return (word_idx as i32) * 32 + 31 - masked.leading_zeros() as i32; }
        if word_idx == 0 { return -1; }
        for w in (0..word_idx).rev() {
            if row[w] != 0 { return (w as i32) * 32 + 31 - row[w].leading_zeros() as i32; }
        }
        -1
    }

    #[inline]
    fn get_row_x(&self, y: usize, z: usize) -> &[u32; 8] { &self.rows_x[y * 256 + z] }
    #[inline]
    fn get_row_y(&self, x: usize, z: usize) -> &[u32; 8] { &self.rows_y[x * 256 + z] }
    #[inline]
    fn get_row_z(&self, x: usize, y: usize) -> &[u32; 8] { &self.rows_z[x * 256 + y] }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCENARIOS (same as existing benches)
// ═══════════════════════════════════════════════════════════════════════════

fn create_wall_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    for z in 100..160 { for y in 64..80 { vol.set(140, y, z, 1.0); } }
    vol
}

fn create_canopy_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    for tz in (80..200).step_by(8) {
        for tx in (80..200).step_by(8) {
            for y in 64..72 { vol.set(tx, y, tz, 1.0); }
            for dz in -1i32..=1 { for dx in -1i32..=1 { for y in 72..74 {
                let cx = (tx as i32 + dx) as usize;
                let cz = (tz as i32 + dz) as usize;
                if cx < VOLUME_SIZE && cz < VOLUME_SIZE { vol.set(cx, y, cz, 1.0); }
            }}}
        }
    }
    vol
}

fn create_open_sky_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    vol
}

fn create_dense_city_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    for bz in (70..200).step_by(6) {
        for bx in (70..200).step_by(6) {
            let height = 64 + ((bx * 7 + bz * 13) % 40);
            for x in bx..bx.min(bx + 4).min(VOLUME_SIZE) {
                for z in bz..bz.min(bz + 4).min(VOLUME_SIZE) {
                    for y in 64..height.min(VOLUME_SIZE) { vol.set(x, y, z, 1.0); }
                }
            }
        }
    }
    vol
}

fn create_sparse_forest_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    for tz in (80..200).step_by(5) { for tx in (80..200).step_by(5) { for y in 64..80 { vol.set(tx, y, tz, 1.0); } } }
    vol
}

fn create_terrain_caves_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    for z in 80..200 {
        for x in 80..200 {
            let hill_h = 64 + ((((x as f32 * 0.05).sin() * 8.0 + (z as f32 * 0.07).cos() * 6.0) as i32).max(0) as usize);
            for y in 64..hill_h.min(VOLUME_SIZE) { vol.set(x, y, z, 1.0); }
        }
    }
    for z in 100..150 { for x in 110..140 { for y in 72..78 { vol.set(x, y, z, 1.0); } } }
    vol
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len < 1e-8 { return [0.0, 1.0, 0.0]; }
    [v[0]/len, v[1]/len, v[2]/len]
}

fn generate_test_rays(count: usize) -> Vec<([f32; 3], [f32; 3])> {
    let sun_dirs = [
        normalize([0.5, 0.8, 0.3]),
        normalize([0.7, 0.3, 0.1]),
        normalize([-0.4, 0.6, 0.5]),
        normalize([0.1, 0.9, 0.1]),
        normalize([0.8, 0.15, 0.2]),
    ];
    let mut rays = Vec::with_capacity(count);
    let rays_per_dir = count / sun_dirs.len();
    for sun_dir in &sun_dirs {
        for i in 0..rays_per_dir {
            let x = 80.0 + (i % 100) as f32 * 0.8;
            let z = 80.0 + (i / 100) as f32 * 0.8;
            rays.push(([x, 64.0, z], *sun_dir));
        }
    }
    rays
}

fn generate_diverse_rays(count: usize) -> Vec<([f32; 3], [f32; 3])> {
    let sun_dirs = [
        normalize([0.5, 0.8, 0.3]),
        normalize([0.7, 0.3, 0.1]),
        normalize([-0.4, 0.6, 0.5]),
        normalize([0.1, 0.9, 0.1]),
        normalize([0.8, 0.15, 0.2]),
        normalize([-0.6, 0.4, -0.3]),
        normalize([0.3, 0.5, 0.7]),
        normalize([-0.2, 0.8, -0.4]),
    ];
    let mut rays = Vec::with_capacity(count);
    let rays_per_dir = count / sun_dirs.len();
    for sun_dir in &sun_dirs {
        for i in 0..rays_per_dir {
            let x = 80.0 + (i % 120) as f32 * 0.7;
            let z = 80.0 + (i / 120) as f32 * 0.7;
            let y = 64.0 + (i % 3) as f32 * 4.0;
            rays.push(([x, y, z], *sun_dir));
        }
    }
    rays
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 1: BASELINE DDA-128 (exact match of current production shader)
// ═══════════════════════════════════════════════════════════════════════════

fn trace_baseline_dda(vol: &VoxelVolume, pos: [f32; 3], dir: [f32; 3], max_steps: i32) -> f32 {
    let s = [pos[0]+dir[0]*0.3, pos[1]+dir[1]*0.3, pos[2]+dir[2]*0.3];
    let mut v = [s[0].floor() as i32, s[1].floor() as i32, s[2].floor() as i32];
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];
    let td = [
        if dir[0].abs()>1e-8{(1.0/dir[0]).abs()}else{f32::MAX},
        if dir[1].abs()>1e-8{(1.0/dir[1]).abs()}else{f32::MAX},
        if dir[2].abs()>1e-8{(1.0/dir[2]).abs()}else{f32::MAX},
    ];
    let f = [s[0]-s[0].floor(), s[1]-s[1].floor(), s[2]-s[2].floor()];
    let mut tm = [
        if dir[0]>=0.0{(1.0-f[0])*td[0]}else{f[0]*td[0]},
        if dir[1]>=0.0{(1.0-f[1])*td[1]}else{f[1]*td[1]},
        if dir[2]>=0.0{(1.0-f[2])*td[2]}else{f[2]*td[2]},
    ];
    for _ in 0..max_steps {
        if v[0]<0||v[1]<0||v[2]<0||v[0]>=256||v[1]>=256||v[2]>=256 { break; }
        let cd = (v[0] as f32-127.5).abs().max((v[1] as f32-127.5).abs()).max((v[2] as f32-127.5).abs());
        if cd > 120.0 { break; }
        if vol.load(v[0],v[1],v[2]) > 0.5 { return 0.2; }
        if tm[0]<tm[1] { if tm[0]<tm[2]{v[0]+=st[0];tm[0]+=td[0];}else{v[2]+=st[2];tm[2]+=td[2];}}
        else { if tm[1]<tm[2]{v[1]+=st[1];tm[1]+=td[1];}else{v[2]+=st[2];tm[2]+=td[2];}}
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 2: BITMASK DDA WITH MAX_SAFE CAP (exact match of current engine shader)
// ═══════════════════════════════════════════════════════════════════════════
// This is the EXACT logic from pipeline.rs dda_trace_shadow() after our fixes.
// - Check current voxel with textureLoad first
// - Only query bitmask on the stepping axis
// - Cap skip by max_safe = floor((next_cross_t - t_max) / t_delta)
// - Only invoke bitmask when max_safe > 1

fn trace_bitmask_capped(vol: &VoxelVolume, bm: &TriAxisBitmask, pos: [f32; 3], dir: [f32; 3], max_steps: i32) -> f32 {
    let s = [pos[0]+dir[0]*0.3, pos[1]+dir[1]*0.3, pos[2]+dir[2]*0.3];
    let mut v = [s[0].floor() as i32, s[1].floor() as i32, s[2].floor() as i32];
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];
    let td = [
        if dir[0].abs()>1e-8{(1.0/dir[0]).abs()}else{f32::MAX},
        if dir[1].abs()>1e-8{(1.0/dir[1]).abs()}else{f32::MAX},
        if dir[2].abs()>1e-8{(1.0/dir[2]).abs()}else{f32::MAX},
    ];
    let f = [s[0]-s[0].floor(), s[1]-s[1].floor(), s[2]-s[2].floor()];
    let mut tm = [
        if dir[0]>=0.0{(1.0-f[0])*td[0]}else{f[0]*td[0]},
        if dir[1]>=0.0{(1.0-f[1])*td[1]}else{f[1]*td[1]},
        if dir[2]>=0.0{(1.0-f[2])*td[2]}else{f[2]*td[2]},
    ];

    for _ in 0..max_steps {
        if v[0]<0||v[1]<0||v[2]<0||v[0]>=256||v[1]>=256||v[2]>=256 { break; }
        let cd = (v[0] as f32-127.5).abs().max((v[1] as f32-127.5).abs()).max((v[2] as f32-127.5).abs());
        if cd > 120.0 { break; }

        // Always check current voxel first (matches engine)
        if vol.load(v[0],v[1],v[2]) > 0.5 { return 0.2; }

        // DDA step with bitmask acceleration, capped by max_safe
        if tm[0] < tm[1] && tm[0] < tm[2] {
            // X axis
            let max_safe = ((tm[1].min(tm[2]) - tm[0]) / td[0]).floor() as i32;
            if max_safe > 1 {
                let search_from = v[0] + st[0];
                let row = bm.get_row_x(v[1] as usize, v[2] as usize);
                let next_solid = if st[0] > 0 {
                    TriAxisBitmask::find_pos(row, search_from)
                } else {
                    TriAxisBitmask::find_neg(row, search_from)
                };
                let mut advance = max_safe;
                if next_solid >= 0 { advance = advance.min((next_solid - v[0]).abs()); }
                advance = advance.max(1);
                v[0] += st[0] * advance;
                tm[0] += td[0] * advance as f32;
            } else {
                v[0] += st[0]; tm[0] += td[0];
            }
        } else if tm[1] < tm[2] {
            // Y axis
            let max_safe = ((tm[0].min(tm[2]) - tm[1]) / td[1]).floor() as i32;
            if max_safe > 1 {
                let search_from = v[1] + st[1];
                let row = bm.get_row_y(v[0] as usize, v[2] as usize);
                let next_solid = if st[1] > 0 {
                    TriAxisBitmask::find_pos(row, search_from)
                } else {
                    TriAxisBitmask::find_neg(row, search_from)
                };
                let mut advance = max_safe;
                if next_solid >= 0 { advance = advance.min((next_solid - v[1]).abs()); }
                advance = advance.max(1);
                v[1] += st[1] * advance;
                tm[1] += td[1] * advance as f32;
            } else {
                v[1] += st[1]; tm[1] += td[1];
            }
        } else {
            // Z axis
            let max_safe = ((tm[0].min(tm[1]) - tm[2]) / td[2]).floor() as i32;
            if max_safe > 1 {
                let search_from = v[2] + st[2];
                let row = bm.get_row_z(v[0] as usize, v[1] as usize);
                let next_solid = if st[2] > 0 {
                    TriAxisBitmask::find_pos(row, search_from)
                } else {
                    TriAxisBitmask::find_neg(row, search_from)
                };
                let mut advance = max_safe;
                if next_solid >= 0 { advance = advance.min((next_solid - v[2]).abs()); }
                advance = advance.max(1);
                v[2] += st[2] * advance;
                tm[2] += td[2] * advance as f32;
            } else {
                v[2] += st[2]; tm[2] += td[2];
            }
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 3: BITMASK DDA UNCONSTRAINED (original benchmark winner - no max_safe)
// ═══════════════════════════════════════════════════════════════════════════
// This is the algorithm that showed 23x speedup in round 2 benches.
// It skips freely on the stepping axis with NO cross-axis safety cap.
// May miss some cross-axis solids but dramatically reduces iterations.

fn trace_bitmask_unconstrained(vol: &VoxelVolume, bm: &TriAxisBitmask, pos: [f32; 3], dir: [f32; 3], _max_steps: i32) -> f32 {
    let s = [pos[0]+dir[0]*0.3, pos[1]+dir[1]*0.3, pos[2]+dir[2]*0.3];
    let mut v = [s[0].floor() as i32, s[1].floor() as i32, s[2].floor() as i32];
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];
    let td = [
        if dir[0].abs()>1e-8{(1.0/dir[0]).abs()}else{f32::MAX},
        if dir[1].abs()>1e-8{(1.0/dir[1]).abs()}else{f32::MAX},
        if dir[2].abs()>1e-8{(1.0/dir[2]).abs()}else{f32::MAX},
    ];
    let f = [s[0]-s[0].floor(), s[1]-s[1].floor(), s[2]-s[2].floor()];
    let mut tm = [
        if dir[0]>=0.0{(1.0-f[0])*td[0]}else{f[0]*td[0]},
        if dir[1]>=0.0{(1.0-f[1])*td[1]}else{f[1]*td[1]},
        if dir[2]>=0.0{(1.0-f[2])*td[2]}else{f[2]*td[2]},
    ];

    for _ in 0..256 {
        if v[0]<0||v[1]<0||v[2]<0||v[0]>=256||v[1]>=256||v[2]>=256 { break; }
        let cd = (v[0] as f32-127.5).abs().max((v[1] as f32-127.5).abs()).max((v[2] as f32-127.5).abs());
        if cd > 120.0 { break; }

        // Check current voxel
        if vol.load(v[0],v[1],v[2]) > 0.5 { return 0.2; }

        // Determine step axis and skip freely (no cross-axis cap)
        let axis = if tm[0] <= tm[1] && tm[0] <= tm[2] { 0 }
                   else if tm[1] <= tm[2] { 1 }
                   else { 2 };

        let (row, cur_pos): (&[u32; 8], i32) = match axis {
            0 => (bm.get_row_x(v[1] as usize, v[2] as usize), v[0]),
            1 => (bm.get_row_y(v[0] as usize, v[2] as usize), v[1]),
            _ => (bm.get_row_z(v[0] as usize, v[1] as usize), v[2]),
        };

        let search_from = cur_pos + st[axis];
        let next_solid = if st[axis] > 0 {
            TriAxisBitmask::find_pos(row, search_from)
        } else {
            TriAxisBitmask::find_neg(row, search_from)
        };

        if let Some(ns) = (next_solid >= 0).then_some(next_solid) {
            let skip = (ns - cur_pos).abs();
            if skip > 1 {
                let advance = skip - 1;
                v[axis] += st[axis] * advance;
                tm[axis] += td[axis] * advance as f32;
                continue;
            }
        } else {
            break; // No solid in entire row along step direction
        }

        v[axis] += st[axis];
        tm[axis] += td[axis];
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn engine_shadow_benchmarks(c: &mut Criterion) {
    let wall_vol = create_wall_scenario();
    let canopy_vol = create_canopy_scenario();
    let open_vol = create_open_sky_scenario();
    let city_vol = create_dense_city_scenario();
    let forest_vol = create_sparse_forest_scenario();
    let caves_vol = create_terrain_caves_scenario();

    let rays = generate_test_rays(500);
    let diverse_rays = generate_diverse_rays(1000);

    println!("\nBuilding bitmask structures (u32x8 to match GPU)...");
    let wall_bm = TriAxisBitmask::from_volume(&wall_vol);
    let canopy_bm = TriAxisBitmask::from_volume(&canopy_vol);
    let open_bm = TriAxisBitmask::from_volume(&open_vol);
    let city_bm = TriAxisBitmask::from_volume(&city_vol);
    let forest_bm = TriAxisBitmask::from_volume(&forest_vol);
    let caves_bm = TriAxisBitmask::from_volume(&caves_vol);
    println!("Done.\n");

    let scenarios: Vec<(&str, &VoxelVolume, &TriAxisBitmask)> = vec![
        ("wall", &wall_vol, &wall_bm),
        ("open_sky", &open_vol, &open_bm),
        ("canopy", &canopy_vol, &canopy_bm),
        ("forest", &forest_vol, &forest_bm),
        ("city", &city_vol, &city_bm),
        ("caves", &caves_vol, &caves_bm),
    ];

    // Per-scenario benchmarks
    for (name, vol, bm) in &scenarios {
        let mut g = c.benchmark_group(format!("engine/{}", name));
        g.sample_size(50);

        g.bench_function("baseline_dda_128", |b| b.iter(|| {
            let mut t = 0.0f32;
            for (p, d) in &rays { t += trace_baseline_dda(black_box(vol), *p, *d, 128); }
            t
        }));

        g.bench_function("bitmask_capped", |b| b.iter(|| {
            let mut t = 0.0f32;
            for (p, d) in &rays { t += trace_bitmask_capped(black_box(vol), black_box(bm), *p, *d, 128); }
            t
        }));

        g.bench_function("bitmask_unconstrained", |b| b.iter(|| {
            let mut t = 0.0f32;
            for (p, d) in &rays { t += trace_bitmask_unconstrained(black_box(vol), black_box(bm), *p, *d, 128); }
            t
        }));

        g.finish();
    }

    // Diverse ray throughput
    {
        let mut g = c.benchmark_group("engine/diverse_1000");
        g.sample_size(30);

        g.bench_function("baseline_dda_128", |b| b.iter(|| {
            let mut t = 0.0f32;
            for (p, d) in &diverse_rays { t += trace_baseline_dda(black_box(&wall_vol), *p, *d, 128); }
            t
        }));
        g.bench_function("bitmask_capped", |b| b.iter(|| {
            let mut t = 0.0f32;
            for (p, d) in &diverse_rays { t += trace_bitmask_capped(black_box(&wall_vol), black_box(&wall_bm), *p, *d, 128); }
            t
        }));
        g.bench_function("bitmask_unconstrained", |b| b.iter(|| {
            let mut t = 0.0f32;
            for (p, d) in &diverse_rays { t += trace_bitmask_unconstrained(black_box(&wall_vol), black_box(&wall_bm), *p, *d, 128); }
            t
        }));

        g.finish();
    }

    // Correctness validation
    {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  ENGINE BENCH CORRECTNESS (shadow hit counts / 500)");
        println!("═══════════════════════════════════════════════════════");
        println!("  {:12} | {:>8} {:>8} {:>8}", "scenario", "baseline", "capped", "uncon");
        println!("  {:-<52}", "");

        let all_scenarios: Vec<(&str, &VoxelVolume, &TriAxisBitmask)> = vec![
            ("wall", &wall_vol, &wall_bm),
            ("open_sky", &open_vol, &open_bm),
            ("canopy", &canopy_vol, &canopy_bm),
            ("forest", &forest_vol, &forest_bm),
            ("city", &city_vol, &city_bm),
            ("caves", &caves_vol, &caves_bm),
        ];

        for (name, vol, bm) in &all_scenarios {
            let mut base = 0; let mut capped = 0; let mut uncon = 0;
            for (p, d) in &rays {
                if trace_baseline_dda(vol, *p, *d, 128) < 0.9 { base += 1; }
                if trace_bitmask_capped(vol, bm, *p, *d, 128) < 0.9 { capped += 1; }
                if trace_bitmask_unconstrained(vol, bm, *p, *d, 128) < 0.9 { uncon += 1; }
            }
            println!("  {:12} | {:>8} {:>8} {:>8}", name, base, capped, uncon);
        }
        println!("═══════════════════════════════════════════════════════\n");
    }
}

criterion_group!(benches, engine_shadow_benchmarks);
criterion_main!(benches);
