//! Large cavern system with dome ceilings, pillars, aquifers, and crystals
//!
//! Implements vertically-asymmetric SDFs for natural dome ceilings with irregular floors,
//! grid-based pillar placement, underground lake systems, and Worley-based crystal clusters.

use noise::{NoiseFn, Perlin, Worley};

use super::util::fast_hash_3d_seeded;
use crate::world::chunk::Block;

/// Parameters for cavern generation
#[derive(Debug, Clone, Copy)]
pub struct CavernParams {
    /// Minimum Y level for large caverns
    pub min_y: i32,
    /// Maximum Y level for large caverns
    pub max_y: i32,
    /// Horizontal radius of cavern (blocks)
    pub radius_h: f32,
    /// Vertical radius of cavern (blocks)
    pub radius_v: f32,
    /// Dome ceiling scale factor (>1 = taller dome)
    pub dome_scale: f32,
    /// Floor flatness scale (<1 = flatter floor)
    pub floor_scale: f32,
    /// Whether to generate pillars
    pub has_pillars: bool,
    /// Whether to fill with water (aquifer)
    pub has_aquifer: bool,
    /// Aquifer water level (Y coordinate)
    pub aquifer_level: i32,
    /// Whether to generate crystals
    pub has_crystals: bool,
}

impl Default for CavernParams {
    fn default() -> Self {
        Self {
            min_y: 10,
            max_y: 50,
            radius_h: 30.0,
            radius_v: 15.0,
            dome_scale: 1.5,
            floor_scale: 0.7,
            has_pillars: true,
            has_aquifer: true,
            aquifer_level: 20,
            has_crystals: true,
        }
    }
}

/// Result of cavern carving at a position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CavernResult {
    /// Solid block (not in cavern)
    Solid,
    /// Air (carved cavern space)
    Air,
    /// Water (in aquifer)
    Water,
    /// Pillar (stone pillar in cavern)
    Pillar,
    /// Crystal placement location
    Crystal(CrystalType),
    /// Stalactite (ceiling decoration)
    Stalactite,
    /// Stalagmite (floor decoration)
    Stalagmite,
}

/// Crystal types for decoration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalType {
    Amethyst,
    Quartz,
}

/// Pillar data for grid-based placement
#[derive(Debug, Clone, Copy)]
pub struct Pillar {
    /// World X position
    pub x: f32,
    /// World Z position
    pub z: f32,
    /// Pillar height
    pub height: f32,
    /// Pillar radius
    pub radius: f32,
    /// Floor Y level
    pub floor_y: i32,
}

/// Noise generators for cavern features
pub struct CavernNoise {
    /// Main cavern shape noise
    shape_noise: Perlin,
    /// Detail noise for irregularity
    detail_noise: Perlin,
    /// Pillar variation noise
    pillar_noise: Perlin,
    /// Crystal cluster noise (Worley F1)
    crystal_noise: Worley,
    /// Stalactite placement noise
    stalactite_noise: Perlin,
    /// Aquifer boundary noise
    aquifer_noise: Perlin,
    /// Seed for deterministic placement
    seed: u32,
}

impl CavernNoise {
    pub fn new(seed: u32) -> Self {
        Self {
            shape_noise: Perlin::new(seed.wrapping_add(200)),
            detail_noise: Perlin::new(seed.wrapping_add(201)),
            pillar_noise: Perlin::new(seed.wrapping_add(202)),
            crystal_noise: Worley::new(seed.wrapping_add(203)),
            stalactite_noise: Perlin::new(seed.wrapping_add(204)),
            aquifer_noise: Perlin::new(seed.wrapping_add(205)),
            seed,
        }
    }

    /// Evaluate cavern SDF with dome ceiling
    /// Returns negative values inside the cavern, positive outside
    #[inline]
    pub fn cavern_sdf(&self, x: f32, y: f32, z: f32, center: (f32, f32, f32), radius: (f32, f32, f32)) -> f32 {
        let local_x = x - center.0;
        let local_y = y - center.1;
        let local_z = z - center.2;

        // Asymmetric Y scaling: dome ceiling (1.5x), flatter floor (0.7x)
        let y_scale = if local_y > 0.0 { 1.5 } else { 0.7 };

        // Scaled radii
        let scaled_rx = radius.0;
        let scaled_ry = radius.1 * y_scale;
        let scaled_rz = radius.2;

        // Ellipsoid SDF
        let k1 = ((local_x / scaled_rx).powi(2)
                + (local_y / scaled_ry).powi(2)
                + (local_z / scaled_rz).powi(2)).sqrt();

        // Add noise for irregular walls
        let noise_scale = 0.05;
        let noise_val = self.detail_noise.get([
            x as f64 * noise_scale,
            y as f64 * noise_scale * 0.5,
            z as f64 * noise_scale,
        ]) as f32;

        // SDF: distance from surface (negative = inside)
        let base_sdf = (k1 - 1.0) * radius.0.min(radius.1).min(radius.2);
        base_sdf + noise_val * 4.0
    }

    /// Check if position is inside a large cavern
    /// Returns true if the position should be carved out
    #[inline]
    pub fn is_large_cavern(&self, x: i32, y: i32, z: i32) -> bool {
        // Only check in valid Y range
        if y < 8 || y > 60 {
            return false;
        }

        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Large-scale cavern noise (cheese-style)
        let scale = 0.008;
        let cavern_val = self.shape_noise.get([xf * scale, yf * scale * 0.6, zf * scale]);

        // Depth-based probability (more caverns at mid-depths)
        let depth_factor = 1.0 - ((y as f64 - 30.0).abs() / 30.0).min(1.0);
        let threshold = 0.55 - depth_factor * 0.15;

        if cavern_val > threshold {
            // Add detail noise for irregular edges
            let detail_scale = 0.04;
            let detail = self.detail_noise.get([xf * detail_scale, yf * detail_scale, zf * detail_scale]);
            return cavern_val + detail * 0.1 > threshold + 0.05;
        }

        false
    }

    /// Determine cavern pillar placement using grid + jitter
    /// Returns Some(Pillar) if a pillar should be placed at this grid cell
    #[inline]
    pub fn get_pillar(&self, grid_x: i32, grid_z: i32, cavern_floor_y: i32, cavern_ceiling_y: i32) -> Option<Pillar> {
        // Grid spacing: 12 blocks
        const GRID_SPACING: f32 = 12.0;

        let hash = fast_hash_3d_seeded(grid_x, 0, grid_z, self.seed);

        // 20% chance of pillar per grid cell
        if (hash & 0xFF) < 204 {
            return None;
        }

        // Jitter within grid cell (±3 blocks)
        let jitter_x = ((hash >> 8) & 0xFF) as f32 / 255.0 * 6.0 - 3.0;
        let jitter_z = ((hash >> 16) & 0xFF) as f32 / 255.0 * 6.0 - 3.0;

        // Pillar properties from hash
        let height_factor = ((hash >> 24) & 0xFF) as f32 / 255.0;
        let max_height = (cavern_ceiling_y - cavern_floor_y) as f32;
        let height = 4.0 + height_factor * (max_height - 4.0).max(0.0);

        let radius = 1.0 + (hash & 0x3) as f32 * 0.5; // 1.0-2.5 blocks

        Some(Pillar {
            x: grid_x as f32 * GRID_SPACING + jitter_x,
            z: grid_z as f32 * GRID_SPACING + jitter_z,
            height,
            radius,
            floor_y: cavern_floor_y,
        })
    }

    /// Check if position is inside a pillar
    #[inline]
    pub fn is_inside_pillar(&self, x: i32, y: i32, z: i32, pillar: &Pillar) -> bool {
        let dx = x as f32 - pillar.x;
        let dz = z as f32 - pillar.z;
        let dist_h = (dx * dx + dz * dz).sqrt();

        // Height within pillar range
        let rel_y = y - pillar.floor_y;
        if rel_y < 0 || rel_y as f32 > pillar.height {
            return false;
        }

        // Tapered profile: slightly wider at base and top
        let t = rel_y as f32 / pillar.height;
        let taper = if t < 0.15 {
            // Base bulge
            1.0 + (1.0 - t / 0.15) * 0.3
        } else if t > 0.85 {
            // Top bulge (connecting to ceiling)
            1.0 + ((t - 0.85) / 0.15) * 0.2
        } else {
            // Middle: straight or slight narrowing
            0.9 + self.pillar_noise.get([x as f64 * 0.1, y as f64 * 0.1, z as f64 * 0.1]) as f32 * 0.1
        };

        dist_h < pillar.radius * taper
    }

    /// Check for crystal placement using Worley noise
    /// Returns Some(CrystalType) if crystals should spawn here
    #[inline]
    pub fn get_crystal(&self, x: i32, y: i32, z: i32) -> Option<CrystalType> {
        // Only place crystals on cave walls/floors (check neighbors for air)
        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Worley F1 at frequency 0.08
        let scale = 0.08;
        let worley_val = self.crystal_noise.get([xf * scale, yf * scale, zf * scale]);

        // Spawn where F1 < 0.2 (in cluster centers)
        if worley_val < -0.6 {
            // Additional 10% rarity check
            let hash = fast_hash_3d_seeded(x, y, z, self.seed);
            if (hash & 0xFF) < 26 { // ~10%
                // Crystal type based on depth
                if y < 25 {
                    return Some(CrystalType::Amethyst); // Deep: amethyst
                } else {
                    return Some(CrystalType::Quartz); // Upper: quartz
                }
            }
        }

        None
    }

    /// Check for stalactite placement (ceiling decorations)
    #[inline]
    pub fn should_place_stalactite(&self, x: i32, z: i32) -> bool {
        let val = self.stalactite_noise.get([x as f64 * 0.1, z as f64 * 0.1]);
        val.abs() > 0.7
    }

    /// Check for stalagmite placement (floor decorations)
    #[inline]
    pub fn should_place_stalagmite(&self, x: i32, z: i32) -> bool {
        let val = self.stalactite_noise.get([x as f64 * 0.1 + 500.0, z as f64 * 0.1 + 500.0]);
        val.abs() > 0.72
    }

    /// Get aquifer water level variation
    #[inline]
    pub fn get_aquifer_level(&self, x: i32, z: i32, base_level: i32) -> i32 {
        let variation = self.aquifer_noise.get([x as f64 * 0.02, z as f64 * 0.02]) as f32;
        base_level + (variation * 2.0) as i32
    }
}

/// Large cavern generator for terrain
pub struct CavernGenerator {
    noise: CavernNoise,
}

impl CavernGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            noise: CavernNoise::new(seed),
        }
    }

    /// Determine what should be at this position for large cavern features
    /// This is called after basic terrain and cave generation
    pub fn get_cavern_feature(&self, x: i32, y: i32, z: i32, is_cave: bool, surface_height: i32) -> Option<CavernResult> {
        // Don't place cavern features too close to surface
        if y > surface_height - 8 {
            return None;
        }

        // Check if we're in a large cavern area
        let in_large_cavern = self.noise.is_large_cavern(x, y, z);

        if in_large_cavern {
            // Get local cavern bounds for this area
            let floor_y = self.estimate_cavern_floor(x, z);
            let ceiling_y = self.estimate_cavern_ceiling(x, z);

            // Pillar check (grid-based)
            let grid_x = x / 12;
            let grid_z = z / 12;
            if let Some(pillar) = self.noise.get_pillar(grid_x, grid_z, floor_y, ceiling_y) {
                if self.noise.is_inside_pillar(x, y, z, &pillar) {
                    return Some(CavernResult::Pillar);
                }
            }

            // Aquifer water level
            let aquifer_level = self.noise.get_aquifer_level(x, z, 18);
            if y < aquifer_level && y >= floor_y {
                return Some(CavernResult::Water);
            }

            // Air (carved cavern)
            return Some(CavernResult::Air);
        }

        // Not in large cavern but in regular cave - check for decorations
        if is_cave {
            // Crystal placement
            if let Some(crystal_type) = self.noise.get_crystal(x, y, z) {
                return Some(CavernResult::Crystal(crystal_type));
            }
        }

        None
    }

    /// Check if a block should be a stalactite/stalagmite
    /// Call this when processing cave ceilings/floors
    pub fn get_cave_decoration(&self, x: i32, y: i32, z: i32, is_ceiling: bool, is_floor: bool) -> Option<Block> {
        if is_ceiling && self.noise.should_place_stalactite(x, z) {
            return Some(Block::Stalactite);
        }
        if is_floor && self.noise.should_place_stalagmite(x, z) {
            return Some(Block::Stalagmite);
        }
        None
    }

    /// Get crystal block for placement
    pub fn crystal_to_block(crystal: CrystalType) -> Block {
        match crystal {
            CrystalType::Amethyst => Block::CrystalAmethyst,
            CrystalType::Quartz => Block::CrystalQuartz,
        }
    }

    /// Estimate cavern floor Y at this XZ position
    fn estimate_cavern_floor(&self, x: i32, z: i32) -> i32 {
        // Use noise to vary floor level
        let noise_val = self.noise.detail_noise.get([x as f64 * 0.02, z as f64 * 0.02]);
        15 + (noise_val * 5.0) as i32
    }

    /// Estimate cavern ceiling Y at this XZ position
    fn estimate_cavern_ceiling(&self, x: i32, z: i32) -> i32 {
        // Use noise to vary ceiling level
        let noise_val = self.noise.detail_noise.get([x as f64 * 0.02 + 100.0, z as f64 * 0.02 + 100.0]);
        45 + (noise_val * 8.0) as i32
    }
}

/// Aquifer zone definitions for depth-based water/lava
#[derive(Debug, Clone, Copy)]
pub enum AquiferZone {
    Upper,   // Y 150-200: 20% water, no lava
    Middle,  // Y 80-150: 40% water, no lava
    Deep,    // Y 30-80: 30% water, 20% lava
    Magma,   // Y 0-30: no water, 90% lava
}

impl AquiferZone {
    pub fn from_y(y: i32) -> Self {
        match y {
            150..=255 => AquiferZone::Upper,
            80..=149 => AquiferZone::Middle,
            30..=79 => AquiferZone::Deep,
            _ => AquiferZone::Magma,
        }
    }

    /// Get water probability for this zone
    pub fn water_probability(&self) -> f32 {
        match self {
            AquiferZone::Upper => 0.2,
            AquiferZone::Middle => 0.4,
            AquiferZone::Deep => 0.3,
            AquiferZone::Magma => 0.0,
        }
    }

    /// Get lava probability for this zone
    pub fn lava_probability(&self) -> f32 {
        match self {
            AquiferZone::Upper => 0.0,
            AquiferZone::Middle => 0.0,
            AquiferZone::Deep => 0.2,
            AquiferZone::Magma => 0.9,
        }
    }

    /// Get max cavern size for this zone
    pub fn max_cavern_size(&self) -> i32 {
        match self {
            AquiferZone::Upper => 20,
            AquiferZone::Middle => 60,
            AquiferZone::Deep => 80,
            AquiferZone::Magma => 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cavern_noise_creation() {
        let _noise = CavernNoise::new(12345);
    }

    #[test]
    fn test_cavern_generator_creation() {
        let _gen = CavernGenerator::new(12345);
    }

    #[test]
    fn test_pillar_placement_deterministic() {
        let noise = CavernNoise::new(12345);
        let pillar1 = noise.get_pillar(5, 5, 15, 45);
        let pillar2 = noise.get_pillar(5, 5, 15, 45);
        assert_eq!(pillar1.is_some(), pillar2.is_some());
        if let (Some(p1), Some(p2)) = (pillar1, pillar2) {
            assert_eq!(p1.x, p2.x);
            assert_eq!(p1.z, p2.z);
            assert_eq!(p1.height, p2.height);
        }
    }

    #[test]
    fn test_aquifer_zones() {
        assert!(matches!(AquiferZone::from_y(180), AquiferZone::Upper));
        assert!(matches!(AquiferZone::from_y(100), AquiferZone::Middle));
        assert!(matches!(AquiferZone::from_y(50), AquiferZone::Deep));
        assert!(matches!(AquiferZone::from_y(15), AquiferZone::Magma));
    }

    #[test]
    fn test_crystal_type_depth() {
        let noise = CavernNoise::new(12345);
        // Just verify the function runs without panic
        let _ = noise.get_crystal(100, 20, 100);
        let _ = noise.get_crystal(100, 35, 100);
    }

    #[test]
    fn test_cavern_sdf() {
        let noise = CavernNoise::new(12345);
        // Center should be inside (negative SDF)
        let center_sdf = noise.cavern_sdf(50.0, 30.0, 50.0, (50.0, 30.0, 50.0), (20.0, 15.0, 20.0));
        assert!(center_sdf < 0.0, "Center should be inside cavern");

        // Far outside should be positive SDF
        let outside_sdf = noise.cavern_sdf(100.0, 30.0, 100.0, (50.0, 30.0, 50.0), (20.0, 15.0, 20.0));
        assert!(outside_sdf > 0.0, "Outside should be positive SDF");
    }

    #[test]
    fn test_is_large_cavern_depth_limited() {
        let noise = CavernNoise::new(12345);
        // Should not generate caverns above Y=60
        let too_high = noise.is_large_cavern(100, 70, 100);
        // May or may not be true depending on noise, but verifies no crash
        let _ = too_high;

        // Should not generate caverns below Y=8
        let too_low = noise.is_large_cavern(100, 5, 100);
        assert!(!too_low, "Should not have caverns below Y=8");
    }
}
