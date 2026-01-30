//! Natural stone formations generation
//!
//! SDF-based shapes for arches, hoodoos, pillars, and rock formations.
//! Uses deterministic noise for chunk-independent placement.

use noise::{NoiseFn, Perlin};

use crate::world::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32};

use super::biome::BiomeId;

/// Formation types that can be generated
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FormationType {
    /// Natural stone arch
    Arch,
    /// Tall narrow rock pillar (hoodoo)
    Hoodoo,
    /// Sea stack (coastal pillar)
    SeaStack,
    /// Boulder cluster
    Boulders,
}

/// Parameters for arch generation
#[derive(Debug, Clone)]
pub struct ArchParams {
    /// Center position in world coordinates
    pub center: [f32; 3],
    /// Horizontal span (width) of the arch
    pub span: f32,
    /// Height of the arch apex
    pub height: f32,
    /// Thickness of the arch ring
    pub thickness: f32,
    /// Arc angle in radians (PI/2 to PI)
    pub arc_angle: f32,
    /// Rotation around Y axis
    pub rotation: f32,
}

/// Parameters for pillar/hoodoo generation
#[derive(Debug, Clone)]
pub struct PillarParams {
    /// Base position in world coordinates
    pub base: [f32; 3],
    /// Total height
    pub height: f32,
    /// Radius at base
    pub base_radius: f32,
    /// Radius at top
    pub top_radius: f32,
    /// Whether to add a mushroom cap
    pub has_cap: bool,
}

/// Stone formation noise and placement
pub struct FormationNoise {
    /// Placement noise for formations
    placement_noise: Perlin,
    /// Detail noise for surface variation
    detail_noise: Perlin,
    /// Seed for deterministic hashing
    seed: u32,
}

impl FormationNoise {
    pub fn new(seed: u32) -> Self {
        Self {
            placement_noise: Perlin::new(seed.wrapping_add(70)),
            detail_noise: Perlin::new(seed.wrapping_add(71)),
            seed,
        }
    }

    /// Check if an arch should be placed at this grid cell
    /// Uses ~200 block spacing with noise threshold
    pub fn should_place_arch(&self, cell_x: i32, cell_z: i32, biome_id: BiomeId) -> bool {
        // Only place in appropriate biomes
        if !matches!(biome_id, BiomeId::Desert | BiomeId::Mountains | BiomeId::CavernousHighlands) {
            return false;
        }

        let hash = self.cell_hash(cell_x, cell_z, 0);
        let noise_val = self.placement_noise.get([
            cell_x as f64 * 0.1,
            cell_z as f64 * 0.1,
        ]);

        // ~5% chance per cell, boosted by noise
        (hash % 100) < 5 && noise_val > 0.3
    }

    /// Check if a hoodoo should be placed at this grid cell
    /// Uses ~80 block spacing
    pub fn should_place_hoodoo(&self, cell_x: i32, cell_z: i32, biome_id: BiomeId) -> bool {
        // Only in desert/badlands type biomes
        if !matches!(biome_id, BiomeId::Desert | BiomeId::Mountains) {
            return false;
        }

        let hash = self.cell_hash(cell_x, cell_z, 1);
        let noise_val = self.placement_noise.get([
            cell_x as f64 * 0.15 + 100.0,
            cell_z as f64 * 0.15 + 100.0,
        ]);

        // ~12% chance, boosted by noise
        (hash % 100) < 12 && noise_val > 0.2
    }

    /// Check if boulders should be placed at this grid cell
    pub fn should_place_boulders(&self, cell_x: i32, cell_z: i32, biome_id: BiomeId) -> bool {
        // Boulders in mountains, highlands, tundra
        if !matches!(biome_id, BiomeId::Mountains | BiomeId::CavernousHighlands | BiomeId::Tundra) {
            return false;
        }

        let hash = self.cell_hash(cell_x, cell_z, 2);
        let noise_val = self.placement_noise.get([
            cell_x as f64 * 0.2 + 200.0,
            cell_z as f64 * 0.2 + 200.0,
        ]);

        (hash % 100) < 20 && noise_val > 0.1
    }

    /// Get arch parameters for a grid cell
    pub fn get_arch_params(&self, cell_x: i32, cell_z: i32, base_y: f32) -> ArchParams {
        let hash = self.cell_hash(cell_x, cell_z, 10);

        // Jitter position within cell
        let jitter_x = ((hash >> 8) & 0xFF) as f32 / 255.0 * 40.0 - 20.0;
        let jitter_z = ((hash >> 16) & 0xFF) as f32 / 255.0 * 40.0 - 20.0;

        let center_x = (cell_x * ARCH_SPACING) as f32 + jitter_x;
        let center_z = (cell_z * ARCH_SPACING) as f32 + jitter_z;

        // Randomize dimensions
        let span = 20.0 + ((hash >> 4) & 0x1F) as f32 * 1.5;  // 20-68 blocks
        let height = 15.0 + ((hash >> 9) & 0x1F) as f32;      // 15-46 blocks
        let thickness = 3.0 + ((hash >> 14) & 0x7) as f32;    // 3-10 blocks
        let arc_angle = std::f32::consts::FRAC_PI_2 + ((hash >> 17) & 0xFF) as f32 / 255.0 * std::f32::consts::FRAC_PI_2;
        let rotation = ((hash >> 25) & 0x7F) as f32 / 127.0 * std::f32::consts::PI;

        ArchParams {
            center: [center_x, base_y, center_z],
            span,
            height,
            thickness,
            arc_angle,
            rotation,
        }
    }

    /// Get pillar/hoodoo parameters for a grid cell
    pub fn get_pillar_params(&self, cell_x: i32, cell_z: i32, base_y: f32) -> PillarParams {
        let hash = self.cell_hash(cell_x, cell_z, 20);

        let jitter_x = ((hash >> 8) & 0xFF) as f32 / 255.0 * 20.0 - 10.0;
        let jitter_z = ((hash >> 16) & 0xFF) as f32 / 255.0 * 20.0 - 10.0;

        let base_x = (cell_x * HOODOO_SPACING) as f32 + jitter_x;
        let base_z = (cell_z * HOODOO_SPACING) as f32 + jitter_z;

        let height = 8.0 + ((hash >> 4) & 0x3F) as f32 * 0.6;  // 8-46 blocks
        let base_radius = 3.0 + ((hash >> 10) & 0x7) as f32 * 0.7;  // 3-8 blocks
        let top_radius = 1.5 + ((hash >> 13) & 0x7) as f32 * 0.4;   // 1.5-4.3 blocks
        let has_cap = (hash >> 20) & 1 == 1;

        PillarParams {
            base: [base_x, base_y, base_z],
            height,
            base_radius,
            top_radius,
            has_cap,
        }
    }

    /// Deterministic hash for grid cells
    fn cell_hash(&self, cell_x: i32, cell_z: i32, variant: u32) -> u32 {
        let mut h = self.seed.wrapping_add(variant);
        h = h.wrapping_mul(0x45d9f3b).wrapping_add(cell_x as u32);
        h = h.wrapping_mul(0x45d9f3b).wrapping_add(cell_z as u32);
        h ^ (h >> 16)
    }

    /// Get surface noise for natural irregularity
    pub fn get_surface_noise(&self, x: f32, y: f32, z: f32) -> f32 {
        self.detail_noise.get([
            x as f64 * 0.15,
            y as f64 * 0.15,
            z as f64 * 0.15,
        ]) as f32 * 1.5
    }
}

/// Grid spacing for different formation types (in blocks)
const ARCH_SPACING: i32 = 200;
const HOODOO_SPACING: i32 = 80;
const BOULDER_SPACING: i32 = 40;

/// Signed distance function for a capped torus (arch shape)
fn arch_sdf(x: f32, y: f32, z: f32, params: &ArchParams) -> f32 {
    // Transform to arch-local coordinates
    let dx = x - params.center[0];
    let dy = y - params.center[1] - params.height;
    let dz = z - params.center[2];

    // Apply rotation around Y axis
    let cos_r = params.rotation.cos();
    let sin_r = params.rotation.sin();
    let rx = dx * cos_r - dz * sin_r;
    let rz = dx * sin_r + dz * cos_r;

    // Capped torus SDF
    let sc_sin = (params.arc_angle * 0.5).sin();
    let sc_cos = (params.arc_angle * 0.5).cos();
    let major_r = params.span * 0.5;

    let px = rx.abs();
    let k = if sc_cos * px > sc_sin * dy {
        px * sc_sin + dy * sc_cos
    } else {
        (px * px + dy * dy).sqrt()
    };

    let dist_sq = rx * rx + dy * dy + rz * rz + major_r * major_r - 2.0 * major_r * k;
    if dist_sq < 0.0 {
        -params.thickness
    } else {
        dist_sq.sqrt() - params.thickness
    }
}

/// Signed distance function for a tapered pillar/hoodoo
fn pillar_sdf(x: f32, y: f32, z: f32, params: &PillarParams) -> f32 {
    let dx = x - params.base[0];
    let dy = y - params.base[1];
    let dz = z - params.base[2];

    // Vertical progress through pillar
    let t = (dy / params.height).clamp(0.0, 1.0);

    // Hoodoo profile: narrow stem, possible wider cap
    let radius = if params.has_cap && t > 0.85 {
        // Mushroom cap at top
        let cap_t = (t - 0.85) / 0.15;
        let cap_t_smooth = cap_t * cap_t * (3.0 - 2.0 * cap_t); // smoothstep
        lerp(params.top_radius * 0.6, params.base_radius * 0.8, cap_t_smooth)
    } else if t < 0.85 {
        // Narrow stem with slight taper
        lerp(params.base_radius * 0.7, params.top_radius * 0.6, t / 0.85)
    } else {
        params.top_radius * 0.6
    };

    let horizontal_dist = (dx * dx + dz * dz).sqrt();

    // Return distance (negative = inside)
    if dy < 0.0 || dy > params.height {
        // Above or below pillar
        f32::MAX
    } else {
        horizontal_dist - radius
    }
}

/// Signed distance function for a boulder
fn boulder_sdf(x: f32, y: f32, z: f32, center: [f32; 3], radius: f32, squash: f32) -> f32 {
    let dx = x - center[0];
    let dy = (y - center[1]) / squash; // Squash vertically
    let dz = z - center[2];

    (dx * dx + dy * dy + dz * dz).sqrt() - radius
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Formation generator for adding formations to chunks
pub struct FormationGenerator<'a> {
    noise: &'a FormationNoise,
}

impl<'a> FormationGenerator<'a> {
    pub fn new(noise: &'a FormationNoise) -> Self {
        Self { noise }
    }

    /// Add formations to a chunk
    pub fn add_formations<F, B>(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        get_terrain_height: F,
        get_biome_id: B,
    )
    where
        F: Fn(i32, i32) -> i32,
        B: Fn(i32, i32) -> BiomeId,
    {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let _world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Check for arches in nearby cells (arches are large, check wider area)
        let arch_cell_x_start = (world_x - 60) / ARCH_SPACING;
        let arch_cell_x_end = (world_x + CHUNK_SIZE_I32 + 60) / ARCH_SPACING;
        let arch_cell_z_start = (world_z - 60) / ARCH_SPACING;
        let arch_cell_z_end = (world_z + CHUNK_SIZE_I32 + 60) / ARCH_SPACING;

        for cell_z in arch_cell_z_start..=arch_cell_z_end {
            for cell_x in arch_cell_x_start..=arch_cell_x_end {
                let sample_x = cell_x * ARCH_SPACING;
                let sample_z = cell_z * ARCH_SPACING;
                let biome_id = get_biome_id(sample_x, sample_z);

                if self.noise.should_place_arch(cell_x, cell_z, biome_id) {
                    let base_y = get_terrain_height(sample_x, sample_z) as f32;
                    let params = self.noise.get_arch_params(cell_x, cell_z, base_y);
                    self.carve_arch(chunk, pos, &params);
                }
            }
        }

        // Check for hoodoos/pillars
        let hoodoo_cell_x_start = (world_x - 15) / HOODOO_SPACING;
        let hoodoo_cell_x_end = (world_x + CHUNK_SIZE_I32 + 15) / HOODOO_SPACING;
        let hoodoo_cell_z_start = (world_z - 15) / HOODOO_SPACING;
        let hoodoo_cell_z_end = (world_z + CHUNK_SIZE_I32 + 15) / HOODOO_SPACING;

        for cell_z in hoodoo_cell_z_start..=hoodoo_cell_z_end {
            for cell_x in hoodoo_cell_x_start..=hoodoo_cell_x_end {
                let sample_x = cell_x * HOODOO_SPACING;
                let sample_z = cell_z * HOODOO_SPACING;
                let biome_id = get_biome_id(sample_x, sample_z);

                if self.noise.should_place_hoodoo(cell_x, cell_z, biome_id) {
                    let base_y = get_terrain_height(sample_x, sample_z) as f32;
                    let params = self.noise.get_pillar_params(cell_x, cell_z, base_y);
                    self.build_pillar(chunk, pos, &params, biome_id);
                }
            }
        }

        // Check for boulder clusters
        let boulder_cell_x_start = (world_x - 10) / BOULDER_SPACING;
        let boulder_cell_x_end = (world_x + CHUNK_SIZE_I32 + 10) / BOULDER_SPACING;
        let boulder_cell_z_start = (world_z - 10) / BOULDER_SPACING;
        let boulder_cell_z_end = (world_z + CHUNK_SIZE_I32 + 10) / BOULDER_SPACING;

        for cell_z in boulder_cell_z_start..=boulder_cell_z_end {
            for cell_x in boulder_cell_x_start..=boulder_cell_x_end {
                let sample_x = cell_x * BOULDER_SPACING;
                let sample_z = cell_z * BOULDER_SPACING;
                let biome_id = get_biome_id(sample_x, sample_z);

                if self.noise.should_place_boulders(cell_x, cell_z, biome_id) {
                    let base_y = get_terrain_height(sample_x, sample_z) as f32;
                    self.build_boulders(chunk, pos, cell_x, cell_z, base_y, biome_id);
                }
            }
        }
    }

    /// Carve an arch shape (removes blocks to form arch)
    fn carve_arch(&self, chunk: &mut Chunk, pos: ChunkPos, params: &ArchParams) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Bounding box for the arch
        let half_span = params.span * 0.5 + params.thickness + 5.0;
        let arch_height = params.height + params.thickness + 5.0;

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    let wx = (world_x + x as i32) as f32;
                    let wy = (world_y + y as i32) as f32;
                    let wz = (world_z + z as i32) as f32;

                    // Quick bounds check
                    let dx = (wx - params.center[0]).abs();
                    let dz = (wz - params.center[2]).abs();
                    if dx > half_span || dz > half_span {
                        continue;
                    }
                    if wy < params.center[1] - 2.0 || wy > params.center[1] + arch_height {
                        continue;
                    }

                    let sdf = arch_sdf(wx, wy, wz, params);
                    let noise_offset = self.noise.get_surface_noise(wx, wy, wz);

                    // Carve if inside arch opening (add some noise for natural look)
                    if sdf + noise_offset < -1.0 {
                        let current = chunk.get_block(x, y, z);
                        if current.is_solid() && current != Block::Bedrock {
                            chunk.set_block(x, y, z, Block::Air);
                        }
                    }
                }
            }
        }
    }

    /// Build a pillar/hoodoo (adds blocks)
    fn build_pillar(&self, chunk: &mut Chunk, pos: ChunkPos, params: &PillarParams, biome_id: BiomeId) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        let block = match biome_id {
            BiomeId::Desert => Block::Sandstone,
            BiomeId::Mountains => Block::Stone,
            _ => Block::Stone,
        };

        let max_radius = params.base_radius.max(params.top_radius) + 2.0;

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    let wx = (world_x + x as i32) as f32;
                    let wy = (world_y + y as i32) as f32;
                    let wz = (world_z + z as i32) as f32;

                    // Quick bounds check
                    let dx = (wx - params.base[0]).abs();
                    let dz = (wz - params.base[2]).abs();
                    if dx > max_radius || dz > max_radius {
                        continue;
                    }
                    if wy < params.base[1] || wy > params.base[1] + params.height + 3.0 {
                        continue;
                    }

                    let sdf = pillar_sdf(wx, wy, wz, params);
                    let noise_offset = self.noise.get_surface_noise(wx, wy, wz) * 0.5;

                    // Build if inside pillar
                    if sdf + noise_offset < 0.0 {
                        let current = chunk.get_block(x, y, z);
                        if current == Block::Air {
                            chunk.set_block(x, y, z, block);
                        }
                    }
                }
            }
        }
    }

    /// Build a cluster of boulders
    fn build_boulders(&self, chunk: &mut Chunk, pos: ChunkPos, cell_x: i32, cell_z: i32, base_y: f32, biome_id: BiomeId) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        let hash = self.noise.cell_hash(cell_x, cell_z, 30);
        let num_boulders = 2 + (hash % 4) as i32; // 2-5 boulders

        let block = match biome_id {
            BiomeId::Tundra => Block::Stone,
            BiomeId::Mountains => Block::Stone,
            _ => Block::Stone,
        };

        let base_x = (cell_x * BOULDER_SPACING) as f32;
        let base_z = (cell_z * BOULDER_SPACING) as f32;

        for i in 0..num_boulders {
            let boulder_hash = hash.wrapping_mul(0x45d9f3b).wrapping_add(i as u32);

            let offset_x = ((boulder_hash >> 4) & 0xFF) as f32 / 255.0 * 16.0 - 8.0;
            let offset_z = ((boulder_hash >> 12) & 0xFF) as f32 / 255.0 * 16.0 - 8.0;
            let radius = 1.5 + ((boulder_hash >> 20) & 0xF) as f32 * 0.3; // 1.5-6 blocks
            let squash = 0.6 + ((boulder_hash >> 24) & 0xF) as f32 / 15.0 * 0.4; // 0.6-1.0

            let center = [
                base_x + offset_x,
                base_y + radius * squash * 0.5,
                base_z + offset_z,
            ];

            // Render boulder in chunk
            let extent = (radius + 2.0) as i32;
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    for y in 0..CHUNK_SIZE {
                        let wx = (world_x + x as i32) as f32;
                        let wy = (world_y + y as i32) as f32;
                        let wz = (world_z + z as i32) as f32;

                        // Quick bounds
                        if (wx - center[0]).abs() > extent as f32 {
                            continue;
                        }
                        if (wz - center[2]).abs() > extent as f32 {
                            continue;
                        }

                        let sdf = boulder_sdf(wx, wy, wz, center, radius, squash);
                        let noise_offset = self.noise.get_surface_noise(wx, wy, wz) * 0.3;

                        if sdf + noise_offset < 0.0 {
                            let current = chunk.get_block(x, y, z);
                            if current == Block::Air {
                                chunk.set_block(x, y, z, block);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formation_noise_creation() {
        let _noise = FormationNoise::new(12345);
    }

    #[test]
    fn test_arch_sdf() {
        let params = ArchParams {
            center: [0.0, 0.0, 0.0],
            span: 20.0,
            height: 15.0,
            thickness: 4.0,
            arc_angle: std::f32::consts::PI,
            rotation: 0.0,
        };

        // Point far outside should have positive SDF
        let far = arch_sdf(100.0, 100.0, 100.0, &params);
        assert!(far > 0.0, "Far point should be outside arch");

        // Point on the arch ring itself should be near zero or negative
        // The arch ring is at height and at radius span/2 from center
        let on_ring = arch_sdf(10.0, 15.0, 0.0, &params);
        assert!(on_ring < 5.0, "Point near arch ring should be close to surface: {}", on_ring);
    }

    #[test]
    fn test_pillar_sdf() {
        let params = PillarParams {
            base: [0.0, 0.0, 0.0],
            height: 20.0,
            base_radius: 4.0,
            top_radius: 2.0,
            has_cap: false,
        };

        // Point inside pillar
        let inside = pillar_sdf(0.0, 10.0, 0.0, &params);
        assert!(inside < 0.0, "Center of pillar should be inside");

        // Point outside pillar
        let outside = pillar_sdf(10.0, 10.0, 0.0, &params);
        assert!(outside > 0.0, "Far from pillar should be outside");
    }

    #[test]
    fn test_placement_deterministic() {
        let noise = FormationNoise::new(12345);

        let result1 = noise.should_place_hoodoo(10, 20, BiomeId::Desert);
        let result2 = noise.should_place_hoodoo(10, 20, BiomeId::Desert);
        assert_eq!(result1, result2, "Placement should be deterministic");
    }
}
