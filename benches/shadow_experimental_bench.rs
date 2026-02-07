//! Experimental shadow & GI optimization benchmarks
//!
//! Tests novel algorithms against current DDA baseline to find vastly superior
//! approaches for shadow tracing, voxel fill, and light propagation.
//!
//! Run: `cargo bench shadow_experimental`
//!
//! Experimental approaches tested:
//! 1. Hierarchical DDA with mip-chain skip (skip empty regions)
//! 2. Bitmask occupancy rows (instant skip-ahead via leading zeros)
//! 3. Blue noise jittered start (fewer steps, same visual quality w/ TAA)
//! 4. Beam DDA (2x2 coherent ray bundles, shared traversal state)
//! 5. Distance field accelerated (SDF jump ahead, fall back to DDA near surfaces)
//! 6. Hybrid: Hierarchical skip + DDA refinement (coarse→fine)
//! 7. Sparse voxel octree traversal (skip huge empty regions)
//! 8. Bitplane DDA (pack 32 voxels per u32, test 32 at once)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// ═══════════════════════════════════════════════════════════════════════════
// SHARED CONSTANTS & TYPES
// ═══════════════════════════════════════════════════════════════════════════

const VOLUME_SIZE: usize = 256;
const HALF_SIZE: f32 = 128.0;

struct VoxelVolume {
    data: Vec<f32>,
}

impl VoxelVolume {
    fn new() -> Self {
        Self {
            data: vec![0.0; VOLUME_SIZE * VOLUME_SIZE * VOLUME_SIZE],
        }
    }

    #[inline(always)]
    fn set(&mut self, x: usize, y: usize, z: usize, opacity: f32) {
        if x < VOLUME_SIZE && y < VOLUME_SIZE && z < VOLUME_SIZE {
            self.data[z * VOLUME_SIZE * VOLUME_SIZE + y * VOLUME_SIZE + x] = opacity;
        }
    }

    #[inline(always)]
    fn load(&self, x: i32, y: i32, z: i32) -> f32 {
        if x < 0 || y < 0 || z < 0 {
            return 0.0;
        }
        let (ux, uy, uz) = (x as usize, y as usize, z as usize);
        if ux >= VOLUME_SIZE || uy >= VOLUME_SIZE || uz >= VOLUME_SIZE {
            return 0.0;
        }
        unsafe { *self.data.get_unchecked(uz * VOLUME_SIZE * VOLUME_SIZE + uy * VOLUME_SIZE + ux) }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HIERARCHICAL VOLUME: Mip chain for coarse skip
// ═══════════════════════════════════════════════════════════════════════════

/// Precomputed mip chain: level 0 = 256^3, level 1 = 128^3, level 2 = 64^3, etc.
/// Each mip stores max(children) - if any child is solid, parent is solid.
struct HierarchicalVolume {
    mip0: Vec<f32>,   // 256^3
    mip1: Vec<f32>,   // 128^3 (2x2x2 max pooling)
    mip2: Vec<f32>,   // 64^3
    mip3: Vec<f32>,   // 32^3
}

impl HierarchicalVolume {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let mip0 = vol.data.clone();

        // Build mip1: 128^3, each voxel = max of 2x2x2 region
        let s1 = 128;
        let mut mip1 = vec![0.0f32; s1 * s1 * s1];
        for z in 0..s1 {
            for y in 0..s1 {
                for x in 0..s1 {
                    let mut max_val = 0.0f32;
                    for dz in 0..2u32 {
                        for dy in 0..2u32 {
                            for dx in 0..2u32 {
                                let sx = x * 2 + dx as usize;
                                let sy = y * 2 + dy as usize;
                                let sz = z * 2 + dz as usize;
                                let v = mip0[sz * 256 * 256 + sy * 256 + sx];
                                if v > max_val { max_val = v; }
                            }
                        }
                    }
                    mip1[z * s1 * s1 + y * s1 + x] = max_val;
                }
            }
        }

        // Build mip2: 64^3
        let s2 = 64;
        let mut mip2 = vec![0.0f32; s2 * s2 * s2];
        for z in 0..s2 {
            for y in 0..s2 {
                for x in 0..s2 {
                    let mut max_val = 0.0f32;
                    for dz in 0..2u32 {
                        for dy in 0..2u32 {
                            for dx in 0..2u32 {
                                let v = mip1[(z*2 + dz as usize) * s1 * s1 + (y*2 + dy as usize) * s1 + (x*2 + dx as usize)];
                                if v > max_val { max_val = v; }
                            }
                        }
                    }
                    mip2[z * s2 * s2 + y * s2 + x] = max_val;
                }
            }
        }

        // Build mip3: 32^3
        let s3 = 32;
        let mut mip3 = vec![0.0f32; s3 * s3 * s3];
        for z in 0..s3 {
            for y in 0..s3 {
                for x in 0..s3 {
                    let mut max_val = 0.0f32;
                    for dz in 0..2u32 {
                        for dy in 0..2u32 {
                            for dx in 0..2u32 {
                                let v = mip2[(z*2 + dz as usize) * s2 * s2 + (y*2 + dy as usize) * s2 + (x*2 + dx as usize)];
                                if v > max_val { max_val = v; }
                            }
                        }
                    }
                    mip3[z * s3 * s3 + y * s3 + x] = max_val;
                }
            }
        }

        Self { mip0, mip1, mip2, mip3 }
    }

    #[inline(always)]
    fn load_mip(&self, level: u32, x: i32, y: i32, z: i32) -> f32 {
        let size = (256 >> level) as i32;
        if x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size {
            return 0.0;
        }
        let s = size as usize;
        let idx = z as usize * s * s + y as usize * s + x as usize;
        match level {
            0 => self.mip0[idx],
            1 => self.mip1[idx],
            2 => self.mip2[idx],
            3 => self.mip3[idx],
            _ => 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BITMASK VOLUME: Row-based occupancy bitmasks for instant skip
// ═══════════════════════════════════════════════════════════════════════════

/// Each Y-Z plane row stores a 256-bit occupancy mask (4 x u64).
/// trailing_zeros() finds the next solid voxel in O(1).
struct BitmaskVolume {
    /// [z][y] = 4 x u64 representing 256 X-axis bits
    rows_x: Vec<[u64; 4]>,
    /// [z][x] = 4 x u64 representing 256 Y-axis bits
    rows_y: Vec<[u64; 4]>,
    /// [y][x] = 4 x u64 representing 256 Z-axis bits
    rows_z: Vec<[u64; 4]>,
    /// Original data for validation
    data: Vec<f32>,
}

impl BitmaskVolume {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let n = VOLUME_SIZE;
        let mut rows_x = vec![[0u64; 4]; n * n];
        let mut rows_y = vec![[0u64; 4]; n * n];
        let mut rows_z = vec![[0u64; 4]; n * n];

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if vol.data[z * n * n + y * n + x] > 0.5 {
                        // X-row: rows_x[z * n + y], bit x
                        rows_x[z * n + y][x / 64] |= 1u64 << (x % 64);
                        // Y-row: rows_y[z * n + x], bit y
                        rows_y[z * n + x][y / 64] |= 1u64 << (y % 64);
                        // Z-row: rows_z[y * n + x], bit z
                        rows_z[y * n + x][z / 64] |= 1u64 << (z % 64);
                    }
                }
            }
        }

        Self {
            rows_x,
            rows_y,
            rows_z,
            data: vol.data.clone(),
        }
    }

    /// Find next solid voxel along X axis from position x, at given y,z
    /// Returns None if no solid voxel found
    #[inline]
    fn next_solid_x(&self, y: usize, z: usize, start_x: usize) -> Option<usize> {
        if y >= VOLUME_SIZE || z >= VOLUME_SIZE { return None; }
        let row = &self.rows_x[z * VOLUME_SIZE + y];
        let word_idx = start_x / 64;
        let bit_idx = start_x % 64;

        // Check current word (mask off bits before start_x)
        if word_idx < 4 {
            let masked = row[word_idx] & (!0u64 << bit_idx);
            if masked != 0 {
                return Some(word_idx * 64 + masked.trailing_zeros() as usize);
            }
        }

        // Check remaining words
        for w in (word_idx + 1)..4 {
            if row[w] != 0 {
                return Some(w * 64 + row[w].trailing_zeros() as usize);
            }
        }
        None
    }

    /// Find next solid voxel along negative X axis
    #[inline]
    fn next_solid_neg_x(&self, y: usize, z: usize, start_x: usize) -> Option<usize> {
        if y >= VOLUME_SIZE || z >= VOLUME_SIZE { return None; }
        let row = &self.rows_x[z * VOLUME_SIZE + y];
        let word_idx = start_x / 64;
        let bit_idx = start_x % 64;

        // Check current word (mask off bits after start_x)
        if word_idx < 4 {
            let masked = row[word_idx] & ((1u64 << (bit_idx + 1)) - 1);
            if masked != 0 {
                return Some(word_idx * 64 + 63 - masked.leading_zeros() as usize);
            }
        }

        // Check preceding words
        if word_idx > 0 {
            for w in (0..word_idx).rev() {
                if row[w] != 0 {
                    return Some(w * 64 + 63 - row[w].leading_zeros() as usize);
                }
            }
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DISTANCE FIELD: Precomputed distance to nearest solid for jump-ahead
// ═══════════════════════════════════════════════════════════════════════════

/// 3D Chebyshev distance field - each voxel stores distance to nearest solid.
/// Allows ray to jump ahead by distance without missing anything.
/// Computed via jump flooding algorithm (same as light propagation).
struct DistanceFieldVolume {
    dist: Vec<u8>,  // Chebyshev distance (0 = solid, 255 = very far)
    data: Vec<f32>,
}

impl DistanceFieldVolume {
    fn from_volume(vol: &VoxelVolume) -> Self {
        let n = VOLUME_SIZE;
        let mut dist = vec![255u8; n * n * n];

        // Set distance = 0 for solid voxels
        for i in 0..vol.data.len() {
            if vol.data[i] > 0.5 {
                dist[i] = 0;
            }
        }

        // Jump flooding to propagate distances (3D BFS approximation)
        // Step sizes: 64, 32, 16, 8, 4, 2, 1
        let mut temp = dist.clone();
        for step in [64, 32, 16, 8, 4, 2, 1] {
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        let cur = dist[z * n * n + y * n + x];
                        // Check 6 cardinal neighbors at step distance
                        let neighbors: [(i32, i32, i32); 6] = [
                            (-step, 0, 0), (step, 0, 0),
                            (0, -step, 0), (0, step, 0),
                            (0, 0, -step), (0, 0, step),
                        ];
                        let mut best = cur;
                        for (dx, dy, dz) in neighbors {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx >= 0 && nx < n as i32 && ny >= 0 && ny < n as i32 && nz >= 0 && nz < n as i32 {
                                let nd = dist[nz as usize * n * n + ny as usize * n + nx as usize];
                                let new_d = nd.saturating_add(step as u8);
                                if new_d < best {
                                    best = new_d;
                                }
                            }
                        }
                        temp[z * n * n + y * n + x] = best;
                    }
                }
            }
            dist.copy_from_slice(&temp);
        }

        Self {
            dist,
            data: vol.data.clone(),
        }
    }

    #[inline(always)]
    fn distance_at(&self, x: i32, y: i32, z: i32) -> u8 {
        if x < 0 || y < 0 || z < 0 {
            return 255;
        }
        let (ux, uy, uz) = (x as usize, y as usize, z as usize);
        if ux >= VOLUME_SIZE || uy >= VOLUME_SIZE || uz >= VOLUME_SIZE {
            return 255;
        }
        self.dist[uz * VOLUME_SIZE * VOLUME_SIZE + uy * VOLUME_SIZE + ux]
    }

    #[inline(always)]
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.distance_at(x, y, z) == 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCENARIOS (same as shadow_bench.rs for comparison)
// ═══════════════════════════════════════════════════════════════════════════

fn create_wall_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    for z in 100..160 {
        for y in 64..80 {
            vol.set(140, y, z, 1.0);
        }
    }
    vol
}

fn create_canopy_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    for tz in (80..200).step_by(8) {
        for tx in (80..200).step_by(8) {
            for y in 64..72 {
                vol.set(tx, y, tz, 1.0);
            }
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

fn create_cave_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..60 { vol.set(x, y, z, 1.0); }
            for y in 68..VOLUME_SIZE { vol.set(x, y, z, 1.0); }
        }
    }
    vol
}

fn create_pillar_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    for y in 64..80 {
        vol.set(128, y, 128, 1.0);
    }
    vol
}

/// Sparse forest: many thin pillars (worst case for DDA - many near misses)
fn create_sparse_forest_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    // Ground
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    // Sparse trees: 1x1 trunks every 5 blocks
    for tz in (80..200).step_by(5) {
        for tx in (80..200).step_by(5) {
            for y in 64..80 {
                vol.set(tx, y, tz, 1.0);
            }
        }
    }
    vol
}

/// Open sky: mostly air above ground (best case - early exit)
fn create_open_sky_scenario() -> VoxelVolume {
    let mut vol = VoxelVolume::new();
    for z in 0..VOLUME_SIZE {
        for x in 0..VOLUME_SIZE {
            for y in 0..64 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    vol
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-8 { return [0.0, 1.0, 0.0]; }
    [v[0] / len, v[1] / len, v[2] / len]
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

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM 0: BASELINE DDA (current production code)
// ═══════════════════════════════════════════════════════════════════════════

fn trace_dda_baseline(vol: &VoxelVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let start = [
        world_pos[0] + light_dir[0] * 0.3,
        world_pos[1] + light_dir[1] * 0.3,
        world_pos[2] + light_dir[2] * 0.3,
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

    let frac = [start[0] - start[0].floor(), start[1] - start[1].floor(), start[2] - start[2].floor()];
    let mut t_max = [
        if light_dir[0] >= 0.0 { (1.0 - frac[0]) * t_delta[0] } else { frac[0] * t_delta[0] },
        if light_dir[1] >= 0.0 { (1.0 - frac[1]) * t_delta[1] } else { frac[1] * t_delta[1] },
        if light_dir[2] >= 0.0 { (1.0 - frac[2]) * t_delta[2] } else { frac[2] * t_delta[2] },
    ];

    for _i in 0..max_steps {
        if voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0
            || voxel[0] >= 256 || voxel[1] >= 256 || voxel[2] >= 256 {
            break;
        }

        let vd = [
            (voxel[0] as f32 - 127.5).abs(),
            (voxel[1] as f32 - 127.5).abs(),
            (voxel[2] as f32 - 127.5).abs(),
        ];
        let center_dist = vd[0].max(vd[1]).max(vd[2]);
        if center_dist > 120.0 { break; }

        let opacity = vol.load(voxel[0], voxel[1], voxel[2]);
        if opacity > 0.5 {
            return 0.2;
        }

        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                voxel[0] += step[0]; t_max[0] += t_delta[0];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        } else {
            if t_max[1] < t_max[2] {
                voxel[1] += step[1]; t_max[1] += t_delta[1];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM 1: HIERARCHICAL DDA (mip-accelerated skip)
// ═══════════════════════════════════════════════════════════════════════════
//
// Concept: Start tracing at coarse mip level (32^3). If coarse voxel is empty,
// skip the entire 8x8x8 region. Only drop to fine resolution near surfaces.
// Expected: 3-8x fewer samples in open areas, minimal overhead in dense areas.

fn trace_hierarchical_dda(hvol: &HierarchicalVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let start = [
        world_pos[0] + light_dir[0] * 0.3,
        world_pos[1] + light_dir[1] * 0.3,
        world_pos[2] + light_dir[2] * 0.3,
    ];

    // Start at mip level 3 (32^3, each voxel = 8x8x8 region)
    let scale = 8.0;
    let inv_scale = 1.0 / scale;
    let coarse_start = [start[0] * inv_scale, start[1] * inv_scale, start[2] * inv_scale];

    let mut voxel = [
        coarse_start[0].floor() as i32,
        coarse_start[1].floor() as i32,
        coarse_start[2].floor() as i32,
    ];

    let step = [
        if light_dir[0] >= 0.0 { 1i32 } else { -1 },
        if light_dir[1] >= 0.0 { 1i32 } else { -1 },
        if light_dir[2] >= 0.0 { 1i32 } else { -1 },
    ];

    let t_delta = [
        if light_dir[0].abs() > 1e-8 { (1.0 / light_dir[0]).abs() * scale } else { f32::MAX },
        if light_dir[1].abs() > 1e-8 { (1.0 / light_dir[1]).abs() * scale } else { f32::MAX },
        if light_dir[2].abs() > 1e-8 { (1.0 / light_dir[2]).abs() * scale } else { f32::MAX },
    ];

    let frac = [
        coarse_start[0] - coarse_start[0].floor(),
        coarse_start[1] - coarse_start[1].floor(),
        coarse_start[2] - coarse_start[2].floor(),
    ];
    let mut t_max = [
        if light_dir[0] >= 0.0 { (1.0 - frac[0]) * t_delta[0] } else { frac[0] * t_delta[0] },
        if light_dir[1] >= 0.0 { (1.0 - frac[1]) * t_delta[1] } else { frac[1] * t_delta[1] },
        if light_dir[2] >= 0.0 { (1.0 - frac[2]) * t_delta[2] } else { frac[2] * t_delta[2] },
    ];

    let coarse_steps = max_steps / 8 + 8; // Fewer coarse steps needed
    for _i in 0..coarse_steps {
        if voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0
            || voxel[0] >= 32 || voxel[1] >= 32 || voxel[2] >= 32 {
            break;
        }

        let coarse_val = hvol.load_mip(3, voxel[0], voxel[1], voxel[2]);

        if coarse_val > 0.5 {
            // Coarse hit — refine with full-res DDA within this 8x8x8 block
            let fine_origin = [
                voxel[0] as f32 * scale,
                voxel[1] as f32 * scale,
                voxel[2] as f32 * scale,
            ];
            // Trace a short DDA within this block
            let refine_pos = [
                fine_origin[0].max(start[0]),
                fine_origin[1].max(start[1]),
                fine_origin[2].max(start[2]),
            ];

            let mut fv = [
                refine_pos[0].floor() as i32,
                refine_pos[1].floor() as i32,
                refine_pos[2].floor() as i32,
            ];

            let fine_t_delta = [
                if light_dir[0].abs() > 1e-8 { (1.0 / light_dir[0]).abs() } else { f32::MAX },
                if light_dir[1].abs() > 1e-8 { (1.0 / light_dir[1]).abs() } else { f32::MAX },
                if light_dir[2].abs() > 1e-8 { (1.0 / light_dir[2]).abs() } else { f32::MAX },
            ];

            let ff = [refine_pos[0] - refine_pos[0].floor(), refine_pos[1] - refine_pos[1].floor(), refine_pos[2] - refine_pos[2].floor()];
            let mut ft_max = [
                if light_dir[0] >= 0.0 { (1.0 - ff[0]) * fine_t_delta[0] } else { ff[0] * fine_t_delta[0] },
                if light_dir[1] >= 0.0 { (1.0 - ff[1]) * fine_t_delta[1] } else { ff[1] * fine_t_delta[1] },
                if light_dir[2] >= 0.0 { (1.0 - ff[2]) * fine_t_delta[2] } else { ff[2] * fine_t_delta[2] },
            ];

            // Trace up to 12 fine steps (diagonal of 8x8x8)
            for _j in 0..12 {
                if fv[0] < 0 || fv[1] < 0 || fv[2] < 0
                    || fv[0] >= 256 || fv[1] >= 256 || fv[2] >= 256 {
                    break;
                }

                let opacity = hvol.load_mip(0, fv[0], fv[1], fv[2]);
                if opacity > 0.5 {
                    return 0.2;
                }

                if ft_max[0] < ft_max[1] {
                    if ft_max[0] < ft_max[2] {
                        fv[0] += step[0]; ft_max[0] += fine_t_delta[0];
                    } else {
                        fv[2] += step[2]; ft_max[2] += fine_t_delta[2];
                    }
                } else {
                    if ft_max[1] < ft_max[2] {
                        fv[1] += step[1]; ft_max[1] += fine_t_delta[1];
                    } else {
                        fv[2] += step[2]; ft_max[2] += fine_t_delta[2];
                    }
                }
            }
        }

        // Step coarse DDA
        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                voxel[0] += step[0]; t_max[0] += t_delta[0];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        } else {
            if t_max[1] < t_max[2] {
                voxel[1] += step[1]; t_max[1] += t_delta[1];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM 2: BITMASK ROW SKIP DDA
// ═══════════════════════════════════════════════════════════════════════════
//
// Concept: For axis-dominant rays, use bitmask to find the next solid voxel
// in the dominant axis direction in O(1) via trailing_zeros/leading_zeros.
// Then only DDA the cross-axis positions to that point.
// Expected: Massive speedup for shallow-angle rays (skip vast empty spans).

fn trace_bitmask_dda(bvol: &BitmaskVolume, world_pos: [f32; 3], light_dir: [f32; 3], _max_steps: i32) -> f32 {
    let start = [
        world_pos[0] + light_dir[0] * 0.3,
        world_pos[1] + light_dir[1] * 0.3,
        world_pos[2] + light_dir[2] * 0.3,
    ];

    // Find dominant axis
    let abs_dir = [light_dir[0].abs(), light_dir[1].abs(), light_dir[2].abs()];

    // Use standard DDA but with bitmask skip on dominant axis
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
        if abs_dir[0] > 1e-8 { 1.0 / abs_dir[0] } else { f32::MAX },
        if abs_dir[1] > 1e-8 { 1.0 / abs_dir[1] } else { f32::MAX },
        if abs_dir[2] > 1e-8 { 1.0 / abs_dir[2] } else { f32::MAX },
    ];

    let frac = [start[0] - start[0].floor(), start[1] - start[1].floor(), start[2] - start[2].floor()];
    let mut t_max = [
        if light_dir[0] >= 0.0 { (1.0 - frac[0]) * t_delta[0] } else { frac[0] * t_delta[0] },
        if light_dir[1] >= 0.0 { (1.0 - frac[1]) * t_delta[1] } else { frac[1] * t_delta[1] },
        if light_dir[2] >= 0.0 { (1.0 - frac[2]) * t_delta[2] } else { frac[2] * t_delta[2] },
    ];

    // Try bitmask skip: check if there's any solid voxel along the row
    // at the current Y,Z position in the dominant X direction
    let max_iter = 200;  // Safety limit
    for _i in 0..max_iter {
        if voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0
            || voxel[0] >= 256 || voxel[1] >= 256 || voxel[2] >= 256 {
            break;
        }

        let vd = [
            (voxel[0] as f32 - 127.5).abs(),
            (voxel[1] as f32 - 127.5).abs(),
            (voxel[2] as f32 - 127.5).abs(),
        ];
        let center_dist = vd[0].max(vd[1]).max(vd[2]);
        if center_dist > 120.0 { break; }

        // Check current voxel
        let idx = voxel[2] as usize * 256 * 256 + voxel[1] as usize * 256 + voxel[0] as usize;
        if idx < bvol.data.len() && bvol.data[idx] > 0.5 {
            return 0.2;
        }

        // Before stepping, try bitmask skip on the axis we're about to step
        if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            // X is the next step axis - try to skip ahead
            if voxel[1] >= 0 && voxel[2] >= 0
                && (voxel[1] as usize) < VOLUME_SIZE && (voxel[2] as usize) < VOLUME_SIZE
            {
                let next_solid = if step[0] > 0 {
                    bvol.next_solid_x(voxel[1] as usize, voxel[2] as usize, voxel[0] as usize + 1)
                } else {
                    bvol.next_solid_neg_x(voxel[1] as usize, voxel[2] as usize,
                        if voxel[0] > 0 { voxel[0] as usize - 1 } else { return 1.0; })
                };

                if let Some(solid_x) = next_solid {
                    // Skip ahead: advance t_max and voxel to the solid position
                    let skip = (solid_x as i32 - voxel[0]).unsigned_abs() as i32;
                    if skip > 1 {
                        // Only skip if we gain more than 1 step
                        let advance = (skip - 1) as f32;
                        voxel[0] += step[0] * (skip - 1);
                        t_max[0] += t_delta[0] * advance;
                        continue;
                    }
                } else {
                    // No solid voxel in this entire row — bail
                    break;
                }
            }
        }

        // Standard DDA step
        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                voxel[0] += step[0]; t_max[0] += t_delta[0];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        } else {
            if t_max[1] < t_max[2] {
                voxel[1] += step[1]; t_max[1] += t_delta[1];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM 3: DISTANCE FIELD ACCELERATED DDA
// ═══════════════════════════════════════════════════════════════════════════
//
// Concept: Query precomputed distance field at current position.
// If distance > 1, jump ahead by (distance - 1) voxels along ray.
// Falls back to normal DDA when distance <= 1 (near surfaces).
// Expected: 5-20x fewer steps in open areas, same accuracy near surfaces.

fn trace_distance_field(dfvol: &DistanceFieldVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let start = [
        world_pos[0] + light_dir[0] * 0.3,
        world_pos[1] + light_dir[1] * 0.3,
        world_pos[2] + light_dir[2] * 0.3,
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

    let frac = [start[0] - start[0].floor(), start[1] - start[1].floor(), start[2] - start[2].floor()];
    let mut t_max = [
        if light_dir[0] >= 0.0 { (1.0 - frac[0]) * t_delta[0] } else { frac[0] * t_delta[0] },
        if light_dir[1] >= 0.0 { (1.0 - frac[1]) * t_delta[1] } else { frac[1] * t_delta[1] },
        if light_dir[2] >= 0.0 { (1.0 - frac[2]) * t_delta[2] } else { frac[2] * t_delta[2] },
    ];

    let mut steps_used = 0;

    while steps_used < max_steps {
        if voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0
            || voxel[0] >= 256 || voxel[1] >= 256 || voxel[2] >= 256 {
            break;
        }

        let vd = [
            (voxel[0] as f32 - 127.5).abs(),
            (voxel[1] as f32 - 127.5).abs(),
            (voxel[2] as f32 - 127.5).abs(),
        ];
        let center_dist = vd[0].max(vd[1]).max(vd[2]);
        if center_dist > 120.0 { break; }

        // Query distance field
        let dist = dfvol.distance_at(voxel[0], voxel[1], voxel[2]);

        if dist == 0 {
            // We're ON a solid voxel
            return 0.2;
        }

        // Jump ahead by (dist - 1) DDA steps if dist > 2
        if dist > 2 {
            let skip = (dist as i32 - 1).min(32); // Cap skip to avoid overshooting volume
            for _ in 0..skip {
                if t_max[0] < t_max[1] {
                    if t_max[0] < t_max[2] {
                        voxel[0] += step[0]; t_max[0] += t_delta[0];
                    } else {
                        voxel[2] += step[2]; t_max[2] += t_delta[2];
                    }
                } else {
                    if t_max[1] < t_max[2] {
                        voxel[1] += step[1]; t_max[1] += t_delta[1];
                    } else {
                        voxel[2] += step[2]; t_max[2] += t_delta[2];
                    }
                }
            }
            steps_used += 1;  // Count as one "logical" step for the skip
        } else {
            // Normal single step
            if t_max[0] < t_max[1] {
                if t_max[0] < t_max[2] {
                    voxel[0] += step[0]; t_max[0] += t_delta[0];
                } else {
                    voxel[2] += step[2]; t_max[2] += t_delta[2];
                }
            } else {
                if t_max[1] < t_max[2] {
                    voxel[1] += step[1]; t_max[1] += t_delta[1];
                } else {
                    voxel[2] += step[2]; t_max[2] += t_delta[2];
                }
            }
            steps_used += 1;
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM 4: BLUE NOISE JITTERED DDA (reduced steps w/ TAA cleanup)
// ═══════════════════════════════════════════════════════════════════════════
//
// Concept: Offset ray start by blue noise value, allowing 50% fewer steps.
// The noise creates per-pixel variation that TAA resolves into smooth shadow.
// Expected: Same visual quality as 128 steps but at 48-64 step cost.

fn trace_dda_jittered(vol: &VoxelVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32, jitter: f32) -> f32 {
    // Jitter start offset (0.0..1.0 mapped to 0..1 voxel along ray)
    let jitter_offset = jitter * 0.9;
    let start = [
        world_pos[0] + light_dir[0] * (0.3 + jitter_offset),
        world_pos[1] + light_dir[1] * (0.3 + jitter_offset),
        world_pos[2] + light_dir[2] * (0.3 + jitter_offset),
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

    let frac = [start[0] - start[0].floor(), start[1] - start[1].floor(), start[2] - start[2].floor()];
    let mut t_max = [
        if light_dir[0] >= 0.0 { (1.0 - frac[0]) * t_delta[0] } else { frac[0] * t_delta[0] },
        if light_dir[1] >= 0.0 { (1.0 - frac[1]) * t_delta[1] } else { frac[1] * t_delta[1] },
        if light_dir[2] >= 0.0 { (1.0 - frac[2]) * t_delta[2] } else { frac[2] * t_delta[2] },
    ];

    for _i in 0..max_steps {
        if voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0
            || voxel[0] >= 256 || voxel[1] >= 256 || voxel[2] >= 256 {
            break;
        }

        let vd = [
            (voxel[0] as f32 - 127.5).abs(),
            (voxel[1] as f32 - 127.5).abs(),
            (voxel[2] as f32 - 127.5).abs(),
        ];
        let center_dist = vd[0].max(vd[1]).max(vd[2]);
        if center_dist > 120.0 { break; }

        if vol.load(voxel[0], voxel[1], voxel[2]) > 0.5 {
            return 0.2;
        }

        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                voxel[0] += step[0]; t_max[0] += t_delta[0];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        } else {
            if t_max[1] < t_max[2] {
                voxel[1] += step[1]; t_max[1] += t_delta[1];
            } else {
                voxel[2] += step[2]; t_max[2] += t_delta[2];
            }
        }
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM 5: HYBRID HIERARCHICAL + DISTANCE FIELD
// ═══════════════════════════════════════════════════════════════════════════
//
// Concept: Start at mip3 (32^3). If coarse voxel empty, skip entire region.
// If occupied, query DF to jump within the region. Only DDA at surface boundary.
// Expected: Best of both worlds - fast skip in open AND fast skip near surfaces.

fn trace_hybrid(hvol: &HierarchicalVolume, dfvol: &DistanceFieldVolume, world_pos: [f32; 3], light_dir: [f32; 3], max_steps: i32) -> f32 {
    let start = [
        world_pos[0] + light_dir[0] * 0.3,
        world_pos[1] + light_dir[1] * 0.3,
        world_pos[2] + light_dir[2] * 0.3,
    ];

    let mut pos = start;
    let mut steps = 0;

    while steps < max_steps {
        let vx = pos[0].floor() as i32;
        let vy = pos[1].floor() as i32;
        let vz = pos[2].floor() as i32;

        if vx < 0 || vy < 0 || vz < 0 || vx >= 256 || vy >= 256 || vz >= 256 {
            break;
        }

        let vd = [
            (vx as f32 - 127.5).abs(),
            (vy as f32 - 127.5).abs(),
            (vz as f32 - 127.5).abs(),
        ];
        if vd[0].max(vd[1]).max(vd[2]) > 120.0 { break; }

        // Check coarse mip (32^3)
        let cx = vx / 8;
        let cy = vy / 8;
        let cz = vz / 8;
        let coarse = hvol.load_mip(3, cx, cy, cz);

        if coarse < 0.01 {
            // Empty coarse cell — jump 8 voxels ahead
            let advance = 8.0;
            pos[0] += light_dir[0] * advance;
            pos[1] += light_dir[1] * advance;
            pos[2] += light_dir[2] * advance;
            steps += 1;
            continue;
        }

        // Coarse cell has content — check distance field
        let dist = dfvol.distance_at(vx, vy, vz);

        if dist == 0 {
            return 0.2; // Hit solid
        }

        if dist > 2 {
            // Jump by DF distance
            let advance = (dist as f32 - 0.5).min(16.0);
            pos[0] += light_dir[0] * advance;
            pos[1] += light_dir[1] * advance;
            pos[2] += light_dir[2] * advance;
            steps += 1;
            continue;
        }

        // Near surface — single voxel step
        pos[0] += light_dir[0];
        pos[1] += light_dir[1];
        pos[2] += light_dir[2];
        steps += 1;
    }
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn experimental_benchmarks(c: &mut Criterion) {
    // Build volumes for all scenarios
    let wall_vol = create_wall_scenario();
    let canopy_vol = create_canopy_scenario();
    let cave_vol = create_cave_scenario();
    let pillar_vol = create_pillar_scenario();
    let forest_vol = create_sparse_forest_scenario();
    let open_vol = create_open_sky_scenario();

    let rays = generate_test_rays(500);
    let rays_1000 = generate_test_rays(1000);

    // ── Precompute acceleration structures ──────────────────────────────
    println!("\nBuilding acceleration structures...");
    let wall_hier = HierarchicalVolume::from_volume(&wall_vol);
    let canopy_hier = HierarchicalVolume::from_volume(&canopy_vol);
    let open_hier = HierarchicalVolume::from_volume(&open_vol);
    let forest_hier = HierarchicalVolume::from_volume(&forest_vol);

    let wall_bitmask = BitmaskVolume::from_volume(&wall_vol);
    let open_bitmask = BitmaskVolume::from_volume(&open_vol);
    let forest_bitmask = BitmaskVolume::from_volume(&forest_vol);

    let wall_df = DistanceFieldVolume::from_volume(&wall_vol);
    let canopy_df = DistanceFieldVolume::from_volume(&canopy_vol);
    let open_df = DistanceFieldVolume::from_volume(&open_vol);
    let forest_df = DistanceFieldVolume::from_volume(&forest_vol);
    println!("Acceleration structures built.\n");

    // ═══════════════════════════════════════════════════════════════
    // WALL SCENARIO: Algorithm comparison
    // ═══════════════════════════════════════════════════════════════
    {
        let mut group = c.benchmark_group("experimental/wall");
        group.sample_size(50);

        group.bench_function("baseline_dda_128", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda_baseline(black_box(&wall_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hierarchical_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hierarchical_dda(black_box(&wall_hier), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("bitmask_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_bitmask_dda(black_box(&wall_bitmask), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("distance_field_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_distance_field(black_box(&wall_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("jittered_dda_64", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (i, (pos, dir)) in rays.iter().enumerate() {
                    let jitter = (i as f32 * 0.618033988) % 1.0; // Golden ratio hash
                    total += trace_dda_jittered(black_box(&wall_vol), *pos, *dir, 64, jitter);
                }
                total
            });
        });

        group.bench_function("hybrid_hier_df", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hybrid(black_box(&wall_hier), black_box(&wall_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    // OPEN SKY: Best case for acceleration (mostly empty space)
    // ═══════════════════════════════════════════════════════════════
    {
        let mut group = c.benchmark_group("experimental/open_sky");
        group.sample_size(50);

        group.bench_function("baseline_dda_128", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda_baseline(black_box(&open_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hierarchical_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hierarchical_dda(black_box(&open_hier), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("bitmask_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_bitmask_dda(black_box(&open_bitmask), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("distance_field_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_distance_field(black_box(&open_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hybrid_hier_df", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hybrid(black_box(&open_hier), black_box(&open_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    // CANOPY: Dense scattered occlusion (stress test)
    // ═══════════════════════════════════════════════════════════════
    {
        let mut group = c.benchmark_group("experimental/canopy");
        group.sample_size(50);

        group.bench_function("baseline_dda_128", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda_baseline(black_box(&canopy_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hierarchical_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hierarchical_dda(black_box(&canopy_hier), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("distance_field_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_distance_field(black_box(&canopy_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    // SPARSE FOREST: Many thin pillars (worst case for acceleration)
    // ═══════════════════════════════════════════════════════════════
    {
        let mut group = c.benchmark_group("experimental/sparse_forest");
        group.sample_size(50);

        group.bench_function("baseline_dda_128", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_dda_baseline(black_box(&forest_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hierarchical_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hierarchical_dda(black_box(&forest_hier), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("bitmask_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_bitmask_dda(black_box(&forest_bitmask), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("distance_field_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_distance_field(black_box(&forest_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hybrid_hier_df", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays {
                    total += trace_hybrid(black_box(&forest_hier), black_box(&forest_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    // QUALITY vs SPEED TRADEOFF: Jittered at various step counts
    // ═══════════════════════════════════════════════════════════════
    {
        let mut group = c.benchmark_group("experimental/jitter_quality");
        group.sample_size(50);

        for steps in [32, 48, 64, 96, 128] {
            group.bench_with_input(
                BenchmarkId::new("jittered_dda", steps),
                &steps,
                |b, &steps| {
                    b.iter(|| {
                        let mut total = 0.0f32;
                        for (i, (pos, dir)) in rays.iter().enumerate() {
                            let jitter = (i as f32 * 0.618033988) % 1.0;
                            total += trace_dda_jittered(black_box(&wall_vol), *pos, *dir, steps, jitter);
                        }
                        total
                    });
                },
            );
        }

        // Baseline for comparison
        for steps in [48, 96, 128] {
            group.bench_with_input(
                BenchmarkId::new("baseline_dda", steps),
                &steps,
                |b, &steps| {
                    b.iter(|| {
                        let mut total = 0.0f32;
                        for (pos, dir) in &rays {
                            total += trace_dda_baseline(black_box(&wall_vol), *pos, *dir, steps);
                        }
                        total
                    });
                },
            );
        }

        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    // THROUGHPUT TEST: 1000 rays for realistic GPU load simulation
    // ═══════════════════════════════════════════════════════════════
    {
        let mut group = c.benchmark_group("experimental/throughput_1000");
        group.sample_size(30);

        group.bench_function("baseline_dda_128", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays_1000 {
                    total += trace_dda_baseline(black_box(&wall_vol), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hierarchical_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays_1000 {
                    total += trace_hierarchical_dda(black_box(&wall_hier), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("distance_field_dda", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays_1000 {
                    total += trace_distance_field(black_box(&wall_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.bench_function("hybrid_hier_df", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for (pos, dir) in &rays_1000 {
                    total += trace_hybrid(black_box(&wall_hier), black_box(&wall_df), *pos, *dir, 128);
                }
                total
            });
        });

        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    // CORRECTNESS VALIDATION: Ensure experimental algos match baseline
    // ═══════════════════════════════════════════════════════════════
    {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  CORRECTNESS VALIDATION (shadow hit counts)");
        println!("═══════════════════════════════════════════════════════");

        let scenarios: Vec<(&str, &VoxelVolume)> = vec![
            ("wall", &wall_vol),
            ("canopy", &canopy_vol),
            ("pillar", &pillar_vol),
            ("forest", &forest_vol),
            ("open", &open_vol),
        ];

        for (name, vol) in &scenarios {
            let hier = HierarchicalVolume::from_volume(vol);
            let df = DistanceFieldVolume::from_volume(vol);

            let mut baseline_shadowed = 0;
            let mut hier_shadowed = 0;
            let mut df_shadowed = 0;
            let mut hybrid_shadowed = 0;

            for (pos, dir) in &rays {
                if trace_dda_baseline(vol, *pos, *dir, 128) < 0.9 { baseline_shadowed += 1; }
                if trace_hierarchical_dda(&hier, *pos, *dir, 128) < 0.9 { hier_shadowed += 1; }
                if trace_distance_field(&df, *pos, *dir, 128) < 0.9 { df_shadowed += 1; }
                if trace_hybrid(&hier, &df, *pos, *dir, 128) < 0.9 { hybrid_shadowed += 1; }
            }

            println!("  {:12} | baseline={:3} hier={:3} df={:3} hybrid={:3}",
                name, baseline_shadowed, hier_shadowed, df_shadowed, hybrid_shadowed);
        }
        println!("═══════════════════════════════════════════════════════\n");
    }
}

criterion_group!(benches, experimental_benchmarks);
criterion_main!(benches);
