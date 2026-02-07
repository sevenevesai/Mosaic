//! Round 2: Advanced shadow optimization benchmarks
//!
//! Builds on Round 1 findings:
//!   - Bitmask DDA: 14.6x faster (winner)
//!   - Hierarchical DDA: 5.9x faster (runner-up)
//!   - Jittered DDA: 1.7x free optimization
//!
//! New experiments combining winners + novel approaches:
//!
//! 1. Tri-axis bitmask DDA - use Y/Z row masks too (not just X), skip on whichever axis steps
//! 2. Hierarchical + Bitmask hybrid - coarse mip skip, then bitmask for fine refinement
//! 3. Block-occupancy bitmask - 8^3 occupancy blocks as u8 + row masks within occupied blocks
//! 4. Bitmask DDA + jitter - combine both winners at reduced step count
//! 5. Branchless DDA - eliminate if/else in step selection via min-component trick
//! 6. 2-level bitmask - coarse 64-bit (4x4 blocks) + fine 256-bit rows
//! 7. SIMD-style 4-ray batch - process 4 rays sharing traversal setup
//! 8. Column-major bitmask - reorganize memory for Y-dominant (sun-typical) rays
//!
//! Run: `cargo bench shadow_experimental2`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

const VOLUME_SIZE: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════
// SHARED VOLUME TYPE
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

// ═══════════════════════════════════════════════════════════════════════════
// TRI-AXIS BITMASK VOLUME
// ═══════════════════════════════════════════════════════════════════════════
//
// Stores row bitmasks for all 3 axes so we can skip on whichever axis
// the DDA is currently stepping on — not just X.

struct TriAxisBitmask {
    /// rows_x[z * 256 + y] = 256-bit mask along X axis
    rows_x: Vec<[u64; 4]>,
    /// rows_y[z * 256 + x] = 256-bit mask along Y axis
    rows_y: Vec<[u64; 4]>,
    /// rows_z[y * 256 + x] = 256-bit mask along Z axis
    rows_z: Vec<[u64; 4]>,
    data: Vec<f32>,
}

impl TriAxisBitmask {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let n = VOLUME_SIZE;
        let mut rows_x = vec![[0u64; 4]; n * n];
        let mut rows_y = vec![[0u64; 4]; n * n];
        let mut rows_z = vec![[0u64; 4]; n * n];

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if vol.data[z * n * n + y * n + x] > 0.5 {
                        rows_x[z * n + y][x / 64] |= 1u64 << (x % 64);
                        rows_y[z * n + x][y / 64] |= 1u64 << (y % 64);
                        rows_z[y * n + x][z / 64] |= 1u64 << (z % 64);
                    }
                }
            }
        }
        Self { rows_x, rows_y, rows_z, data: vol.data.clone() }
    }

    /// Find next solid along +axis direction from start position
    #[inline]
    fn next_solid_pos(row: &[u64; 4], start: usize) -> Option<usize> {
        if start >= 256 { return None; }
        let word_idx = start / 64;
        let bit_idx = start % 64;

        if word_idx < 4 {
            let masked = row[word_idx] & (!0u64 << bit_idx);
            if masked != 0 {
                return Some(word_idx * 64 + masked.trailing_zeros() as usize);
            }
        }
        for w in (word_idx + 1)..4 {
            if row[w] != 0 {
                return Some(w * 64 + row[w].trailing_zeros() as usize);
            }
        }
        None
    }

    /// Find next solid along -axis direction from start position
    #[inline]
    fn next_solid_neg(row: &[u64; 4], start: usize) -> Option<usize> {
        if start >= 256 { return None; }
        let word_idx = start / 64;
        let bit_idx = start % 64;

        if word_idx < 4 && bit_idx < 64 {
            let mask = if bit_idx == 63 { !0u64 } else { (1u64 << (bit_idx + 1)) - 1 };
            let masked = row[word_idx] & mask;
            if masked != 0 {
                return Some(word_idx * 64 + 63 - masked.leading_zeros() as usize);
            }
        }
        if word_idx > 0 {
            for w in (0..word_idx).rev() {
                if row[w] != 0 {
                    return Some(w * 64 + 63 - row[w].leading_zeros() as usize);
                }
            }
        }
        None
    }

    #[inline]
    fn get_row_x(&self, y: usize, z: usize) -> &[u64; 4] {
        &self.rows_x[z * VOLUME_SIZE + y]
    }

    #[inline]
    fn get_row_y(&self, x: usize, z: usize) -> &[u64; 4] {
        &self.rows_y[z * VOLUME_SIZE + x]
    }

    #[inline]
    fn get_row_z(&self, x: usize, y: usize) -> &[u64; 4] {
        &self.rows_z[y * VOLUME_SIZE + x]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BLOCK OCCUPANCY + BITMASK (2-LEVEL ACCELERATION)
// ═══════════════════════════════════════════════════════════════════════════
//
// Level 1: 32^3 block grid, each bit says "any solid in this 8^3 block?"
// Level 2: Per-row bitmasks within occupied blocks only
// The block level is a single u32 per row of 32 blocks = instant skip of 8-voxel regions.

struct BlockBitmask {
    /// 32^3 occupancy: block_occ[bz * 32 * 32 + by * 32 + bx] = true if any solid
    block_occ: Vec<bool>,
    /// Block-row masks: for each (bz, by), a u32 where bit bx = occupied
    block_rows_x: Vec<u32>,
    /// Fine X-axis masks (same as round 1)
    rows_x: Vec<[u64; 4]>,
    /// Fine Y-axis masks
    rows_y: Vec<[u64; 4]>,
    /// Fine Z-axis masks
    rows_z: Vec<[u64; 4]>,
    data: Vec<f32>,
}

impl BlockBitmask {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let n = VOLUME_SIZE;
        let bn = 32; // block count per axis
        let bs = 8;  // block size

        let mut block_occ = vec![false; bn * bn * bn];
        let mut block_rows_x = vec![0u32; bn * bn]; // [bz * 32 + by]
        let mut rows_x = vec![[0u64; 4]; n * n];
        let mut rows_y = vec![[0u64; 4]; n * n];
        let mut rows_z = vec![[0u64; 4]; n * n];

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if vol.data[z * n * n + y * n + x] > 0.5 {
                        rows_x[z * n + y][x / 64] |= 1u64 << (x % 64);
                        rows_y[z * n + x][y / 64] |= 1u64 << (y % 64);
                        rows_z[y * n + x][z / 64] |= 1u64 << (z % 64);

                        let bx = x / bs;
                        let by = y / bs;
                        let bz = z / bs;
                        block_occ[bz * bn * bn + by * bn + bx] = true;
                        block_rows_x[bz * bn + by] |= 1u32 << bx;
                    }
                }
            }
        }
        Self { block_occ, block_rows_x, rows_x, rows_y, rows_z, data: vol.data.clone() }
    }

    /// Find next occupied block along X from block_x, at block (by, bz)
    #[inline]
    fn next_occ_block_x(&self, by: usize, bz: usize, start_bx: usize) -> Option<usize> {
        if by >= 32 || bz >= 32 || start_bx >= 32 { return None; }
        let row = self.block_rows_x[bz * 32 + by];
        let masked = row & (!0u32 << start_bx);
        if masked != 0 {
            Some(masked.trailing_zeros() as usize)
        } else {
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HIERARCHICAL VOLUME (from round 1, needed for hybrid)
// ═══════════════════════════════════════════════════════════════════════════

struct HierarchicalVolume {
    mip0: Vec<f32>,
    mip1: Vec<f32>,
    mip2: Vec<f32>,
    mip3: Vec<f32>,
}

impl HierarchicalVolume {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let mip0 = vol.data.clone();
        let mut mip1 = vec![0.0f32; 128 * 128 * 128];
        for z in 0..128usize {
            for y in 0..128usize {
                for x in 0..128usize {
                    let mut m = 0.0f32;
                    for dz in 0..2u32 { for dy in 0..2u32 { for dx in 0..2u32 {
                        let v = mip0[(z*2+dz as usize)*256*256 + (y*2+dy as usize)*256 + x*2+dx as usize];
                        if v > m { m = v; }
                    }}}
                    mip1[z * 128 * 128 + y * 128 + x] = m;
                }
            }
        }
        let mut mip2 = vec![0.0f32; 64 * 64 * 64];
        for z in 0..64usize {
            for y in 0..64usize {
                for x in 0..64usize {
                    let mut m = 0.0f32;
                    for dz in 0..2u32 { for dy in 0..2u32 { for dx in 0..2u32 {
                        let v = mip1[(z*2+dz as usize)*128*128 + (y*2+dy as usize)*128 + x*2+dx as usize];
                        if v > m { m = v; }
                    }}}
                    mip2[z * 64 * 64 + y * 64 + x] = m;
                }
            }
        }
        let mut mip3 = vec![0.0f32; 32 * 32 * 32];
        for z in 0..32usize {
            for y in 0..32usize {
                for x in 0..32usize {
                    let mut m = 0.0f32;
                    for dz in 0..2u32 { for dy in 0..2u32 { for dx in 0..2u32 {
                        let v = mip2[(z*2+dz as usize)*64*64 + (y*2+dy as usize)*64 + x*2+dx as usize];
                        if v > m { m = v; }
                    }}}
                    mip3[z * 32 * 32 + y * 32 + x] = m;
                }
            }
        }
        Self { mip0, mip1, mip2, mip3 }
    }

    #[inline(always)]
    fn load_mip(&self, level: u32, x: i32, y: i32, z: i32) -> f32 {
        let size = (256 >> level) as i32;
        if x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size { return 0.0; }
        let s = size as usize;
        let idx = z as usize * s * s + y as usize * s + x as usize;
        match level { 0 => self.mip0[idx], 1 => self.mip1[idx], 2 => self.mip2[idx], 3 => self.mip3[idx], _ => 0.0 }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCENARIOS
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

fn create_sparse_forest_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    for tz in (80..200).step_by(5) { for tx in (80..200).step_by(5) { for y in 64..80 { vol.set(tx, y, tz, 1.0); } } }
    vol
}

fn create_open_sky_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    vol
}

/// Dense city: many solid columns of varying height (GPU-realistic worst case)
fn create_dense_city_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    // Buildings: 4x4 base, varying heights, every 6 blocks
    for bz in (70..200).step_by(6) {
        for bx in (70..200).step_by(6) {
            let height = 64 + ((bx * 7 + bz * 13) % 40); // pseudo-random height 64-104
            for x in bx..bx.min(bx + 4).min(VOLUME_SIZE) {
                for z in bz..bz.min(bz + 4).min(VOLUME_SIZE) {
                    for y in 64..height.min(VOLUME_SIZE) {
                        vol.set(x, y, z, 1.0);
                    }
                }
            }
        }
    }
    vol
}

/// Terrain with overhangs and cave mouths (complex occlusion topology)
fn create_terrain_caves_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE { for x in 0..VOLUME_SIZE { for y in 0..64 { vol.set(x, y, z, 1.0); } } }
    // Hills
    for z in 80..200 {
        for x in 80..200 {
            let hill_h = 64 + ((((x as f32 * 0.05).sin() * 8.0 + (z as f32 * 0.07).cos() * 6.0) as i32).max(0) as usize);
            for y in 64..hill_h.min(VOLUME_SIZE) {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    // Overhangs / cave mouths
    for z in 100..150 {
        for x in 110..140 {
            for y in 72..78 {
                vol.set(x, y, z, 1.0); // ceiling
            }
            // Leave 68-72 open as cave
        }
    }
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
        normalize([0.5, 0.8, 0.3]),     // High noon
        normalize([0.7, 0.3, 0.1]),     // Low sun (dawn/dusk)
        normalize([-0.4, 0.6, 0.5]),    // Afternoon
        normalize([0.1, 0.9, 0.1]),     // Nearly overhead
        normalize([0.8, 0.15, 0.2]),    // Very low angle
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

/// Wider variety of sun angles for more realistic workload
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
            let y = 64.0 + (i % 3) as f32 * 4.0; // vary Y too
            rays.push(([x, y, z], *sun_dir));
        }
    }
    rays
}

// ═══════════════════════════════════════════════════════════════════════════
// BASELINE: DDA-128 (production code)
// ═══════════════════════════════════════════════════════════════════════════

fn trace_dda_baseline(vol: &VoxelVolume, pos: [f32; 3], dir: [f32; 3], max_steps: i32) -> f32 {
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
// ALG 1: TRI-AXIS BITMASK DDA
// ═══════════════════════════════════════════════════════════════════════════
//
// Round 1 bitmask only skipped on X-axis steps. This version skips on ALL
// three axes — whichever axis the DDA is about to step on gets a bitmask
// lookup. This should help for Y-dominant rays (nearly overhead sun) and
// Z-dominant rays (east/west sun) enormously.

fn trace_triaxis_bitmask(bm: &TriAxisBitmask, pos: [f32; 3], dir: [f32; 3], _max_steps: i32) -> f32 {
    let s = [pos[0]+dir[0]*0.3, pos[1]+dir[1]*0.3, pos[2]+dir[2]*0.3];
    let mut v = [s[0].floor() as i32, s[1].floor() as i32, s[2].floor() as i32];
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];
    let ad = [dir[0].abs(), dir[1].abs(), dir[2].abs()];
    let td = [
        if ad[0]>1e-8{1.0/ad[0]}else{f32::MAX},
        if ad[1]>1e-8{1.0/ad[1]}else{f32::MAX},
        if ad[2]>1e-8{1.0/ad[2]}else{f32::MAX},
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
        let idx = v[2] as usize * 256*256 + v[1] as usize * 256 + v[0] as usize;
        if idx < bm.data.len() && bm.data[idx] > 0.5 { return 0.2; }

        // Determine which axis to step next
        let step_axis = if tm[0] <= tm[1] && tm[0] <= tm[2] { 0 }
                        else if tm[1] <= tm[2] { 1 }
                        else { 2 };

        // Try bitmask skip on the stepping axis
        let vy = v[1] as usize;
        let vx = v[0] as usize;
        let vz = v[2] as usize;

        match step_axis {
            0 => {
                // X axis step — lookup X-row at (y, z)
                if vy < 256 && vz < 256 {
                    let row = bm.get_row_x(vy, vz);
                    let next_pos = if st[0] > 0 {
                        if vx + 1 < 256 { TriAxisBitmask::next_solid_pos(row, vx + 1) } else { None }
                    } else {
                        if vx > 0 { TriAxisBitmask::next_solid_neg(row, vx - 1) } else { None }
                    };

                    if let Some(nx) = next_pos {
                        let skip = (nx as i32 - v[0]).unsigned_abs() as i32;
                        if skip > 1 {
                            let advance = (skip - 1) as f32;
                            v[0] += st[0] * (skip - 1);
                            tm[0] += td[0] * advance;
                            continue;
                        }
                    } else {
                        break; // No solid in entire row
                    }
                }
                v[0] += st[0]; tm[0] += td[0];
            }
            1 => {
                // Y axis step — lookup Y-row at (x, z)
                if vx < 256 && vz < 256 {
                    let row = bm.get_row_y(vx, vz);
                    let next_pos = if st[1] > 0 {
                        if vy + 1 < 256 { TriAxisBitmask::next_solid_pos(row, vy + 1) } else { None }
                    } else {
                        if vy > 0 { TriAxisBitmask::next_solid_neg(row, vy - 1) } else { None }
                    };

                    if let Some(ny) = next_pos {
                        let skip = (ny as i32 - v[1]).unsigned_abs() as i32;
                        if skip > 1 {
                            let advance = (skip - 1) as f32;
                            v[1] += st[1] * (skip - 1);
                            tm[1] += td[1] * advance;
                            continue;
                        }
                    } else {
                        break;
                    }
                }
                v[1] += st[1]; tm[1] += td[1];
            }
            _ => {
                // Z axis step — lookup Z-row at (x, y)
                if vx < 256 && vy < 256 {
                    let row = bm.get_row_z(vx, vy);
                    let next_pos = if st[2] > 0 {
                        if vz + 1 < 256 { TriAxisBitmask::next_solid_pos(row, vz + 1) } else { None }
                    } else {
                        if vz > 0 { TriAxisBitmask::next_solid_neg(row, vz - 1) } else { None }
                    };

                    if let Some(nz) = next_pos {
                        let skip = (nz as i32 - v[2]).unsigned_abs() as i32;
                        if skip > 1 {
                            let advance = (skip - 1) as f32;
                            v[2] += st[2] * (skip - 1);
                            tm[2] += td[2] * advance;
                            continue;
                        }
                    } else {
                        break;
                    }
                }
                v[2] += st[2]; tm[2] += td[2];
            }
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 2: HIERARCHICAL + BITMASK HYBRID
// ═══════════════════════════════════════════════════════════════════════════
//
// Coarse mip3 (32^3) DDA to skip empty 8^3 regions, then bitmask DDA for
// fine traversal within occupied regions. Best of both: O(1) skip on coarse,
// O(1) skip on fine.

fn trace_hier_bitmask(hier: &HierarchicalVolume, bm: &TriAxisBitmask, pos: [f32; 3], dir: [f32; 3], max_steps: i32) -> f32 {
    let s = [pos[0]+dir[0]*0.3, pos[1]+dir[1]*0.3, pos[2]+dir[2]*0.3];
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];

    // Coarse DDA at 32^3 (step size = 8 voxels)
    let scale = 8.0;
    let inv_scale = 1.0 / scale;
    let cs = [s[0]*inv_scale, s[1]*inv_scale, s[2]*inv_scale];
    let mut cv = [cs[0].floor() as i32, cs[1].floor() as i32, cs[2].floor() as i32];

    let ctd = [
        if dir[0].abs()>1e-8{(1.0/dir[0]).abs()*scale}else{f32::MAX},
        if dir[1].abs()>1e-8{(1.0/dir[1]).abs()*scale}else{f32::MAX},
        if dir[2].abs()>1e-8{(1.0/dir[2]).abs()*scale}else{f32::MAX},
    ];
    let cf = [cs[0]-cs[0].floor(), cs[1]-cs[1].floor(), cs[2]-cs[2].floor()];
    let mut ctm = [
        if dir[0]>=0.0{(1.0-cf[0])*ctd[0]}else{cf[0]*ctd[0]},
        if dir[1]>=0.0{(1.0-cf[1])*ctd[1]}else{cf[1]*ctd[1]},
        if dir[2]>=0.0{(1.0-cf[2])*ctd[2]}else{cf[2]*ctd[2]},
    ];

    let coarse_steps = max_steps / 6 + 8;
    for _ in 0..coarse_steps {
        if cv[0]<0||cv[1]<0||cv[2]<0||cv[0]>=32||cv[1]>=32||cv[2]>=32 { break; }

        let coarse_val = hier.load_mip(3, cv[0], cv[1], cv[2]);
        if coarse_val > 0.5 {
            // Occupied coarse block — refine with bitmask DDA within 8^3 region
            let base = [cv[0]*8, cv[1]*8, cv[2]*8];
            let entry = [
                (s[0] + dir[0] * (ctm[0] - ctd[0]).max(0.0)).floor() as i32,
                (s[1] + dir[1] * (ctm[1] - ctd[1]).max(0.0)).floor() as i32,
                (s[2] + dir[2] * (ctm[2] - ctd[2]).max(0.0)).floor() as i32,
            ];
            let mut fv = [
                entry[0].max(base[0]).min(base[0]+7),
                entry[1].max(base[1]).min(base[1]+7),
                entry[2].max(base[2]).min(base[2]+7),
            ];

            let ftd = [
                if dir[0].abs()>1e-8{(1.0/dir[0]).abs()}else{f32::MAX},
                if dir[1].abs()>1e-8{(1.0/dir[1]).abs()}else{f32::MAX},
                if dir[2].abs()>1e-8{(1.0/dir[2]).abs()}else{f32::MAX},
            ];
            let ff = [
                (entry[0] as f32 - s[0]).abs().fract(),
                (entry[1] as f32 - s[1]).abs().fract(),
                (entry[2] as f32 - s[2]).abs().fract(),
            ];
            let mut ftm = [
                if dir[0]>=0.0{(1.0-ff[0])*ftd[0]}else{ff[0]*ftd[0]},
                if dir[1]>=0.0{(1.0-ff[1])*ftd[1]}else{ff[1]*ftd[1]},
                if dir[2]>=0.0{(1.0-ff[2])*ftd[2]}else{ff[2]*ftd[2]},
            ];

            // Up to 14 fine steps (diagonal of 8^3 + margin)
            for _ in 0..14 {
                if fv[0]<0||fv[1]<0||fv[2]<0||fv[0]>=256||fv[1]>=256||fv[2]>=256 { break; }
                // Exit if we left the coarse block
                if fv[0] < base[0] || fv[0] > base[0]+7
                    || fv[1] < base[1] || fv[1] > base[1]+7
                    || fv[2] < base[2] || fv[2] > base[2]+7 { break; }

                let idx = fv[2] as usize * 256*256 + fv[1] as usize * 256 + fv[0] as usize;
                if idx < bm.data.len() && bm.data[idx] > 0.5 { return 0.2; }

                // Determine step axis and try bitmask skip within block
                let axis = if ftm[0]<=ftm[1]&&ftm[0]<=ftm[2]{0} else if ftm[1]<=ftm[2]{1} else {2};
                match axis {
                    0 => {
                        let vy = fv[1] as usize; let vz = fv[2] as usize;
                        if vy < 256 && vz < 256 {
                            let row = bm.get_row_x(vy, vz);
                            let next = if st[0]>0 {
                                if (fv[0]+1) as usize <= (base[0]+7) as usize {
                                    TriAxisBitmask::next_solid_pos(row, (fv[0]+1) as usize)
                                } else { None }
                            } else {
                                if fv[0] > base[0] {
                                    TriAxisBitmask::next_solid_neg(row, (fv[0]-1) as usize)
                                } else { None }
                            };
                            if let Some(nx) = next {
                                // Only skip within this block
                                let bounded = if st[0]>0 { nx.min((base[0]+7) as usize) } else { nx.max(base[0] as usize) };
                                let skip = (bounded as i32 - fv[0]).unsigned_abs() as i32;
                                if skip > 1 {
                                    fv[0] += st[0] * (skip-1);
                                    ftm[0] += ftd[0] * (skip-1) as f32;
                                    continue;
                                }
                            }
                        }
                        fv[0]+=st[0]; ftm[0]+=ftd[0];
                    }
                    1 => { fv[1]+=st[1]; ftm[1]+=ftd[1]; }
                    _ => { fv[2]+=st[2]; ftm[2]+=ftd[2]; }
                }
            }
        }

        // Step coarse DDA
        if ctm[0]<ctm[1] { if ctm[0]<ctm[2]{cv[0]+=st[0];ctm[0]+=ctd[0];}else{cv[2]+=st[2];ctm[2]+=ctd[2];}}
        else { if ctm[1]<ctm[2]{cv[1]+=st[1];ctm[1]+=ctd[1];}else{cv[2]+=st[2];ctm[2]+=ctd[2];}}
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 3: BLOCK-OCCUPANCY + FINE BITMASK (2-LEVEL)
// ═══════════════════════════════════════════════════════════════════════════
//
// Level 1: 32-bit block-row scan (32 blocks of 8^3 per row)
// Level 2: 256-bit fine row within occupied blocks
// No mip chain needed — just two integer operations per skip.

fn trace_block_bitmask(bb: &BlockBitmask, pos: [f32; 3], dir: [f32; 3], _max_steps: i32) -> f32 {
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

        let idx = v[2] as usize*256*256 + v[1] as usize*256 + v[0] as usize;
        if idx < bb.data.len() && bb.data[idx] > 0.5 { return 0.2; }

        let axis = if tm[0]<=tm[1]&&tm[0]<=tm[2]{0} else if tm[1]<=tm[2]{1} else {2};

        if axis == 0 {
            // X step — try block-level skip first
            let by = (v[1] as usize) / 8;
            let bz = (v[2] as usize) / 8;
            let cur_bx = (v[0] as usize) / 8;

            if st[0] > 0 {
                let next_bx_start = cur_bx + 1;
                if let Some(next_occ_bx) = bb.next_occ_block_x(by, bz, next_bx_start) {
                    let target_voxel = next_occ_bx * 8; // Start of next occupied block
                    let skip = target_voxel as i32 - v[0];
                    if skip > 2 {
                        // Jump to start of next occupied block
                        let advance = (skip - 1) as f32;
                        v[0] += skip - 1;
                        tm[0] += td[0] * advance;
                        continue;
                    }
                    // Within or adjacent to occupied block — try fine bitmask
                    let vy = v[1] as usize;
                    let vz = v[2] as usize;
                    if vy < 256 && vz < 256 {
                        let row = &bb.rows_x[vz * 256 + vy];
                        if let Some(nx) = {
                            let start_x = (v[0]+1) as usize;
                            if start_x < 256 {
                                let word_idx = start_x / 64;
                                let bit_idx = start_x % 64;
                                let mut result = None;
                                if word_idx < 4 {
                                    let masked = row[word_idx] & (!0u64 << bit_idx);
                                    if masked != 0 { result = Some(word_idx*64 + masked.trailing_zeros() as usize); }
                                }
                                if result.is_none() {
                                    for w in (word_idx+1)..4 {
                                        if row[w] != 0 { result = Some(w*64 + row[w].trailing_zeros() as usize); break; }
                                    }
                                }
                                result
                            } else { None }
                        } {
                            let skip = nx as i32 - v[0];
                            if skip > 1 {
                                v[0] += skip - 1;
                                tm[0] += td[0] * (skip - 1) as f32;
                                continue;
                            }
                        } else {
                            break; // No solid in entire row
                        }
                    }
                } else {
                    break; // No occupied blocks ahead
                }
            }
            v[0]+=st[0]; tm[0]+=td[0];
        } else if axis == 1 {
            // Y step — try fine bitmask
            let vx = v[0] as usize;
            let vz = v[2] as usize;
            if vx < 256 && vz < 256 {
                let row = &bb.rows_y[vz * 256 + vx];
                let next = if st[1] > 0 {
                    let sy = (v[1]+1) as usize;
                    if sy < 256 {
                        let wi = sy/64; let bi = sy%64;
                        let mut r = None;
                        if wi<4 { let m=row[wi]&(!0u64<<bi); if m!=0{r=Some(wi*64+m.trailing_zeros() as usize);}}
                        if r.is_none(){for w in (wi+1)..4{if row[w]!=0{r=Some(w*64+row[w].trailing_zeros() as usize);break;}}}
                        r
                    } else { None }
                } else {
                    if v[1] > 0 {
                        let sy = (v[1]-1) as usize;
                        let wi = sy/64; let bi = sy%64;
                        let mask = if bi==63{!0u64}else{(1u64<<(bi+1))-1};
                        let mut r = None;
                        if wi<4{let m=row[wi]&mask; if m!=0{r=Some(wi*64+63-m.leading_zeros() as usize);}}
                        if r.is_none()&&wi>0{for w in (0..wi).rev(){if row[w]!=0{r=Some(w*64+63-row[w].leading_zeros() as usize);break;}}}
                        r
                    } else { None }
                };
                if let Some(ny) = next {
                    let skip = (ny as i32 - v[1]).unsigned_abs() as i32;
                    if skip > 1 { v[1]+=st[1]*(skip-1); tm[1]+=td[1]*(skip-1)as f32; continue; }
                } else { break; }
            }
            v[1]+=st[1]; tm[1]+=td[1];
        } else {
            // Z step — try fine bitmask
            let vx = v[0] as usize;
            let vy = v[1] as usize;
            if vx < 256 && vy < 256 {
                let row = &bb.rows_z[vy * 256 + vx];
                let next = if st[2] > 0 {
                    let sz = (v[2]+1) as usize;
                    if sz < 256 {
                        let wi = sz/64; let bi = sz%64;
                        let mut r = None;
                        if wi<4{let m=row[wi]&(!0u64<<bi);if m!=0{r=Some(wi*64+m.trailing_zeros() as usize);}}
                        if r.is_none(){for w in (wi+1)..4{if row[w]!=0{r=Some(w*64+row[w].trailing_zeros() as usize);break;}}}
                        r
                    } else { None }
                } else {
                    if v[2] > 0 {
                        let sz = (v[2]-1) as usize;
                        let wi = sz/64; let bi = sz%64;
                        let mask = if bi==63{!0u64}else{(1u64<<(bi+1))-1};
                        let mut r = None;
                        if wi<4{let m=row[wi]&mask;if m!=0{r=Some(wi*64+63-m.leading_zeros() as usize);}}
                        if r.is_none()&&wi>0{for w in (0..wi).rev(){if row[w]!=0{r=Some(w*64+63-row[w].leading_zeros() as usize);break;}}}
                        r
                    } else { None }
                };
                if let Some(nz) = next {
                    let skip = (nz as i32 - v[2]).unsigned_abs() as i32;
                    if skip > 1 { v[2]+=st[2]*(skip-1); tm[2]+=td[2]*(skip-1)as f32; continue; }
                } else { break; }
            }
            v[2]+=st[2]; tm[2]+=td[2];
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 4: BRANCHLESS DDA (eliminate branch mispredictions in step selection)
// ═══════════════════════════════════════════════════════════════════════════
//
// GPU shaders hate branches. This version uses min-component selection
// with mask arithmetic instead of if/else. On CPU this tests whether
// branch elimination helps; on GPU this maps to select() intrinsics.

fn trace_branchless_dda(vol: &VoxelVolume, pos: [f32; 3], dir: [f32; 3], max_steps: i32) -> f32 {
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

        // Branchless: find axis with minimum t_max using comparison masks
        let x_le_y = (tm[0] <= tm[1]) as i32;
        let x_le_z = (tm[0] <= tm[2]) as i32;
        let y_le_z = (tm[1] <= tm[2]) as i32;

        // axis 0 steps if x <= y AND x <= z
        // axis 1 steps if y < x AND y <= z
        // axis 2 steps if z < x AND z < y
        let step_x = x_le_y & x_le_z;
        let step_y = (1 - x_le_y) & y_le_z;
        let step_z = 1 - step_x - step_y;

        v[0] += st[0] * step_x;
        v[1] += st[1] * step_y;
        v[2] += st[2] * step_z;
        tm[0] += td[0] * step_x as f32;
        tm[1] += td[1] * step_y as f32;
        tm[2] += td[2] * step_z as f32;
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 5: TRI-AXIS BITMASK + JITTER (combine both winners at low step count)
// ═══════════════════════════════════════════════════════════════════════════

fn trace_triaxis_bitmask_jittered(bm: &TriAxisBitmask, pos: [f32; 3], dir: [f32; 3], _max_steps: i32, jitter: f32) -> f32 {
    let jitter_offset = jitter * 0.9;
    let s = [pos[0]+dir[0]*(0.3+jitter_offset), pos[1]+dir[1]*(0.3+jitter_offset), pos[2]+dir[2]*(0.3+jitter_offset)];
    let mut v = [s[0].floor() as i32, s[1].floor() as i32, s[2].floor() as i32];
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];
    let ad = [dir[0].abs(), dir[1].abs(), dir[2].abs()];
    let td = [
        if ad[0]>1e-8{1.0/ad[0]}else{f32::MAX},
        if ad[1]>1e-8{1.0/ad[1]}else{f32::MAX},
        if ad[2]>1e-8{1.0/ad[2]}else{f32::MAX},
    ];
    let f = [s[0]-s[0].floor(), s[1]-s[1].floor(), s[2]-s[2].floor()];
    let mut tm = [
        if dir[0]>=0.0{(1.0-f[0])*td[0]}else{f[0]*td[0]},
        if dir[1]>=0.0{(1.0-f[1])*td[1]}else{f[1]*td[1]},
        if dir[2]>=0.0{(1.0-f[2])*td[2]}else{f[2]*td[2]},
    ];

    // With bitmask skip we need far fewer iterations
    for _ in 0..128 {
        if v[0]<0||v[1]<0||v[2]<0||v[0]>=256||v[1]>=256||v[2]>=256 { break; }
        let cd = (v[0] as f32-127.5).abs().max((v[1] as f32-127.5).abs()).max((v[2] as f32-127.5).abs());
        if cd > 120.0 { break; }

        let idx = v[2] as usize*256*256 + v[1] as usize*256 + v[0] as usize;
        if idx < bm.data.len() && bm.data[idx] > 0.5 { return 0.2; }

        let axis = if tm[0]<=tm[1]&&tm[0]<=tm[2]{0} else if tm[1]<=tm[2]{1} else {2};
        let vy = v[1] as usize; let vx = v[0] as usize; let vz = v[2] as usize;

        // Same tri-axis bitmask skip logic, just with jittered start
        let (row, cur, step_dir): (&[u64;4], usize, i32) = match axis {
            0 => { if vy<256&&vz<256 { (bm.get_row_x(vy, vz), vx, st[0]) } else { v[0]+=st[0];tm[0]+=td[0];continue; } }
            1 => { if vx<256&&vz<256 { (bm.get_row_y(vx, vz), vy, st[1]) } else { v[1]+=st[1];tm[1]+=td[1];continue; } }
            _ => { if vx<256&&vy<256 { (bm.get_row_z(vx, vy), vz, st[2]) } else { v[2]+=st[2];tm[2]+=td[2];continue; } }
        };

        let next = if step_dir > 0 {
            if cur + 1 < 256 { TriAxisBitmask::next_solid_pos(row, cur + 1) } else { None }
        } else {
            if cur > 0 { TriAxisBitmask::next_solid_neg(row, cur - 1) } else { None }
        };

        if let Some(n) = next {
            let skip = (n as i32 - v[axis] as i32).unsigned_abs() as i32;
            if skip > 1 {
                let advance = (skip - 1) as f32;
                v[axis] += st[axis] * (skip - 1);
                tm[axis] += td[axis] * advance;
                continue;
            }
        } else {
            break;
        }

        v[axis] += st[axis];
        tm[axis] += td[axis];
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALG 6: 4-RAY BATCH (amortize setup cost, test coherent rays together)
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulates GPU warp coherence: 4 rays with same direction, nearby origins.
// Shared DDA setup, then parallel traversal. On GPU this maps to SIMD lanes.

fn trace_4ray_batch(vol: &VoxelVolume, rays: &[([f32;3],[f32;3])], start_idx: usize) -> [f32; 4] {
    let mut results = [1.0f32; 4];
    let dir = rays[start_idx].1; // All 4 use same direction

    // Shared DDA constants
    let st = [if dir[0]>=0.0{1i32}else{-1}, if dir[1]>=0.0{1}else{-1}, if dir[2]>=0.0{1}else{-1}];
    let td = [
        if dir[0].abs()>1e-8{(1.0/dir[0]).abs()}else{f32::MAX},
        if dir[1].abs()>1e-8{(1.0/dir[1]).abs()}else{f32::MAX},
        if dir[2].abs()>1e-8{(1.0/dir[2]).abs()}else{f32::MAX},
    ];

    // Per-ray state
    let mut vs = [[0i32; 3]; 4];
    let mut tms = [[0.0f32; 3]; 4];
    let mut active = [true; 4];

    for r in 0..4 {
        let pos = rays[start_idx + r].0;
        let s = [pos[0]+dir[0]*0.3, pos[1]+dir[1]*0.3, pos[2]+dir[2]*0.3];
        vs[r] = [s[0].floor() as i32, s[1].floor() as i32, s[2].floor() as i32];
        let f = [s[0]-s[0].floor(), s[1]-s[1].floor(), s[2]-s[2].floor()];
        tms[r] = [
            if dir[0]>=0.0{(1.0-f[0])*td[0]}else{f[0]*td[0]},
            if dir[1]>=0.0{(1.0-f[1])*td[1]}else{f[1]*td[1]},
            if dir[2]>=0.0{(1.0-f[2])*td[2]}else{f[2]*td[2]},
        ];
    }

    for _ in 0..128 {
        let mut any_active = false;
        for r in 0..4 {
            if !active[r] { continue; }
            let v = &mut vs[r];
            let tm = &mut tms[r];

            if v[0]<0||v[1]<0||v[2]<0||v[0]>=256||v[1]>=256||v[2]>=256 { active[r]=false; continue; }
            let cd = (v[0] as f32-127.5).abs().max((v[1] as f32-127.5).abs()).max((v[2] as f32-127.5).abs());
            if cd > 120.0 { active[r]=false; continue; }

            if vol.load(v[0],v[1],v[2]) > 0.5 { results[r] = 0.2; active[r]=false; continue; }

            if tm[0]<tm[1] { if tm[0]<tm[2]{v[0]+=st[0];tm[0]+=td[0];}else{v[2]+=st[2];tm[2]+=td[2];}}
            else { if tm[1]<tm[2]{v[1]+=st[1];tm[1]+=td[1];}else{v[2]+=st[2];tm[2]+=td[2];}}
            any_active = true;
        }
        if !any_active { break; }
    }
    results
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn round2_benchmarks(c: &mut Criterion) {
    let wall_vol = create_wall_scenario();
    let canopy_vol = create_canopy_scenario();
    let forest_vol = create_sparse_forest_scenario();
    let open_vol = create_open_sky_scenario();
    let city_vol = create_dense_city_scenario();
    let caves_vol = create_terrain_caves_scenario();

    let rays = generate_test_rays(500);
    let diverse_rays = generate_diverse_rays(1000);

    println!("\nBuilding round 2 acceleration structures...");
    let wall_tri = TriAxisBitmask::from_volume(&wall_vol);
    let canopy_tri = TriAxisBitmask::from_volume(&canopy_vol);
    let forest_tri = TriAxisBitmask::from_volume(&forest_vol);
    let open_tri = TriAxisBitmask::from_volume(&open_vol);
    let city_tri = TriAxisBitmask::from_volume(&city_vol);
    let caves_tri = TriAxisBitmask::from_volume(&caves_vol);

    let wall_hier = HierarchicalVolume::from_volume(&wall_vol);
    let forest_hier = HierarchicalVolume::from_volume(&forest_vol);
    let city_hier = HierarchicalVolume::from_volume(&city_vol);

    let wall_block = BlockBitmask::from_volume(&wall_vol);
    let forest_block = BlockBitmask::from_volume(&forest_vol);
    let city_block = BlockBitmask::from_volume(&city_vol);
    let caves_block = BlockBitmask::from_volume(&caves_vol);
    println!("Round 2 structures built.\n");

    // ═════════ WALL SCENARIO ═════════
    {
        let mut g = c.benchmark_group("r2/wall");
        g.sample_size(50);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_dda_baseline(black_box(&wall_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_triaxis_bitmask(black_box(&wall_tri),*p,*d,128);}t }));
        g.bench_function("hier_bitmask_hybrid", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_hier_bitmask(black_box(&wall_hier),black_box(&wall_tri),*p,*d,128);}t }));
        g.bench_function("block_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_block_bitmask(black_box(&wall_block),*p,*d,128);}t }));
        g.bench_function("branchless_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_branchless_dda(black_box(&wall_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask_jit", |b| b.iter(|| { let mut t=0.0f32; for(i,(p,d))in rays.iter().enumerate(){t+=trace_triaxis_bitmask_jittered(black_box(&wall_tri),*p,*d,128,(i as f32*0.618)%1.0);}t }));
        g.finish();
    }

    // ═════════ OPEN SKY ═════════
    {
        let mut g = c.benchmark_group("r2/open_sky");
        g.sample_size(50);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_dda_baseline(black_box(&open_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_triaxis_bitmask(black_box(&open_tri),*p,*d,128);}t }));
        g.bench_function("block_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_block_bitmask(black_box(&wall_block),*p,*d,128);}t }));
        g.finish();
    }

    // ═════════ CANOPY ═════════
    {
        let mut g = c.benchmark_group("r2/canopy");
        g.sample_size(50);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_dda_baseline(black_box(&canopy_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_triaxis_bitmask(black_box(&canopy_tri),*p,*d,128);}t }));
        g.finish();
    }

    // ═════════ SPARSE FOREST ═════════
    {
        let mut g = c.benchmark_group("r2/sparse_forest");
        g.sample_size(50);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_dda_baseline(black_box(&forest_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_triaxis_bitmask(black_box(&forest_tri),*p,*d,128);}t }));
        g.bench_function("hier_bitmask_hybrid", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_hier_bitmask(black_box(&forest_hier),black_box(&forest_tri),*p,*d,128);}t }));
        g.bench_function("block_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_block_bitmask(black_box(&forest_block),*p,*d,128);}t }));
        g.finish();
    }

    // ═════════ DENSE CITY (NEW) ═════════
    {
        let mut g = c.benchmark_group("r2/dense_city");
        g.sample_size(50);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_dda_baseline(black_box(&city_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_triaxis_bitmask(black_box(&city_tri),*p,*d,128);}t }));
        g.bench_function("hier_bitmask_hybrid", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_hier_bitmask(black_box(&city_hier),black_box(&city_tri),*p,*d,128);}t }));
        g.bench_function("block_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_block_bitmask(black_box(&city_block),*p,*d,128);}t }));
        g.bench_function("branchless_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_branchless_dda(black_box(&city_vol),*p,*d,128);}t }));
        g.finish();
    }

    // ═════════ TERRAIN CAVES (NEW) ═════════
    {
        let mut g = c.benchmark_group("r2/terrain_caves");
        g.sample_size(50);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_dda_baseline(black_box(&caves_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_triaxis_bitmask(black_box(&caves_tri),*p,*d,128);}t }));
        g.bench_function("block_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&rays{t+=trace_block_bitmask(black_box(&caves_block),*p,*d,128);}t }));
        g.finish();
    }

    // ═════════ 4-RAY BATCH THROUGHPUT ═════════
    {
        let mut g = c.benchmark_group("r2/batch_throughput");
        g.sample_size(30);
        // Need rays grouped by 4 with same direction
        let grouped_rays = generate_test_rays(1000);
        g.bench_function("single_ray_x1000", |b| b.iter(|| {
            let mut t=0.0f32;
            for(p,d) in &grouped_rays { t+=trace_dda_baseline(black_box(&wall_vol),*p,*d,128); }
            t
        }));
        g.bench_function("4ray_batch_x1000", |b| b.iter(|| {
            let mut t=0.0f32;
            for i in (0..grouped_rays.len()).step_by(4) {
                if i+3 < grouped_rays.len() {
                    let r = trace_4ray_batch(black_box(&wall_vol), &grouped_rays, i);
                    t += r[0]+r[1]+r[2]+r[3];
                }
            }
            t
        }));
        g.finish();
    }

    // ═════════ DIVERSE RAY THROUGHPUT (realistic workload) ═════════
    {
        let mut g = c.benchmark_group("r2/diverse_1000");
        g.sample_size(30);
        g.bench_function("baseline_dda_128", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&diverse_rays{t+=trace_dda_baseline(black_box(&wall_vol),*p,*d,128);}t }));
        g.bench_function("triaxis_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&diverse_rays{t+=trace_triaxis_bitmask(black_box(&wall_tri),*p,*d,128);}t }));
        g.bench_function("block_bitmask", |b| b.iter(|| { let mut t=0.0f32; for(p,d)in&diverse_rays{t+=trace_block_bitmask(black_box(&wall_block),*p,*d,128);}t }));
        g.finish();
    }

    // ═════════ CORRECTNESS VALIDATION ═════════
    {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  ROUND 2 CORRECTNESS VALIDATION (shadow hit counts)");
        println!("═══════════════════════════════════════════════════════");

        let scenarios: Vec<(&str, &VoxelVolume)> = vec![
            ("wall", &wall_vol), ("canopy", &canopy_vol), ("forest", &forest_vol),
            ("open", &open_vol), ("city", &city_vol), ("caves", &caves_vol),
        ];

        for (name, vol) in &scenarios {
            let tri = TriAxisBitmask::from_volume(vol);
            let bb = BlockBitmask::from_volume(vol);

            let mut base = 0; let mut triax = 0; let mut block = 0; let mut bless = 0;
            for (p,d) in &rays {
                if trace_dda_baseline(vol,*p,*d,128) < 0.9 { base+=1; }
                if trace_triaxis_bitmask(&tri,*p,*d,128) < 0.9 { triax+=1; }
                if trace_block_bitmask(&bb,*p,*d,128) < 0.9 { block+=1; }
                if trace_branchless_dda(vol,*p,*d,128) < 0.9 { bless+=1; }
            }
            println!("  {:12} | base={:3} triax={:3} block={:3} brless={:3}", name, base, triax, block, bless);
        }
        println!("═══════════════════════════════════════════════════════\n");
    }
}

criterion_group!(benches, round2_benchmarks);
criterion_main!(benches);
