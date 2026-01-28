//! Procedural terrain generation
//!
//! Uses layered noise for realistic terrain with caves, ores, biomes, and trees.
//! Features multiple tree types, rocky outcrops, cliffs, arches, and floating islands.
//!
//! Optimized with column-based caching to minimize redundant noise calculations.

use noise::{NoiseFn, Perlin, Fbm, MultiFractal, RidgedMulti};

use super::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32, WATER_SEA_LEVEL, WATER_FULL_HEIGHT, BLOCK_SCALE};

/// Fast integer hash for deterministic pseudo-randomness without noise overhead
#[inline]
fn fast_hash(x: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393);
    h = h.wrapping_add((z as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

/// Fast 3D hash
#[inline]
fn fast_hash_3d(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393);
    h = h.wrapping_add((y as u32).wrapping_mul(668265263));
    h = h.wrapping_add((z as u32).wrapping_mul(1440670291));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

/// Cached data for a single terrain column (x, z)
/// Pre-computed once per column to avoid redundant noise calculations
#[derive(Clone, Copy)]
struct ColumnCache {
    height: i32,
    is_mountain: bool,
    cliff_factor: f64,
    is_rocky_outcrop: bool,
    biome: Biome,
    is_beach: bool,
}

/// Biome types for terrain variety
#[derive(Debug, Clone, Copy, PartialEq)]
enum Biome {
    /// Default grasslands - rolling hills, caves lead underground, no surface openings
    Grasslands,
    /// Cavernous highlands - hilly green terrain with many surface caverns
    CavernousHighlands,
}

/// Tree types for variety in forests
#[derive(Debug, Clone, Copy, PartialEq)]
enum TreeType {
    /// Small spherical canopy tree (original style)
    SmallOak,
    /// Large spreading oak with thick trunk
    LargeOak,
    /// Tall narrow pine with conical canopy
    TallPine,
    /// Short bushy shrub
    BushyShrub,
}

impl TreeType {
    /// Get the log block type for this tree
    fn log_block(&self) -> Block {
        match self {
            TreeType::SmallOak => Block::OakLog,
            TreeType::LargeOak => Block::OakLog,
            TreeType::TallPine => Block::Wood,      // Keep original for pines
            TreeType::BushyShrub => Block::BirchLog, // Birch for shrubs (variety)
        }
    }
}

/// Generate a chunk at the given position with the specified seed
pub fn generate_chunk_with_seed(pos: ChunkPos, seed: u64) -> Chunk {
    let mut chunk = Chunk::new();
    let generator = TerrainGenerator::new(seed as u32);

    let world_x = pos.x * CHUNK_SIZE_I32;
    let world_y = pos.y * CHUNK_SIZE_I32;
    let world_z = pos.z * CHUNK_SIZE_I32;

    // Pre-compute column cache for all 16x16 columns
    // This avoids redundant noise calculations for height, cliffs, etc.
    let mut columns = [[ColumnCache {
        height: 0,
        is_mountain: false,
        cliff_factor: 0.0,
        is_rocky_outcrop: false,
        biome: Biome::Grasslands,
        is_beach: false,
    }; CHUNK_SIZE]; CHUNK_SIZE];

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let wx = world_x + x as i32;
            let wz = world_z + z as i32;
            columns[z][x] = generator.compute_column_cache(wx, wz);
        }
    }

    // Generate blocks using cached column data
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let col = &columns[z][x];
            let wx = world_x + x as i32;
            let wz = world_z + z as i32;

            for y in 0..CHUNK_SIZE {
                let wy = world_y + y as i32;
                let block = generator.get_block_cached(wx, wy, wz, col);
                chunk.set_block(x, y, z, block);
            }
        }
    }

    // Add trees (need to check neighboring positions)
    generator.add_trees(&mut chunk, pos);

    // Generate water heightfield data
    generator.generate_water_heightfield(&mut chunk, pos);

    chunk
}

/// Generate a chunk at the given position (uses default seed for backwards compatibility)
#[allow(dead_code)]
pub fn generate_chunk(pos: ChunkPos) -> Chunk {
    generate_chunk_with_seed(pos, 12345)
}

/// Terrain generator using layered noise
struct TerrainGenerator {
    // Base terrain
    height_noise: Fbm<Perlin>,
    detail_noise: Perlin,

    // Mountain system
    mountain_noise: RidgedMulti<Perlin>,  // Ridged for sharp peaks
    ridge_noise: Perlin,                   // Ridge direction/placement

    // Cave system (expanded for better caves)
    cave_worm_noise: Perlin,       // Worm-like tunnel caves
    cave_worm_noise2: Perlin,      // Second layer for 3D worms
    cave_chamber_noise: Perlin,    // Large chambers
    cave_cavern_noise: Perlin,     // Large open caverns
    cave_width_noise: Perlin,      // Varies tunnel width

    // Ores
    ore_noise: Perlin,

    // Trees and biomes
    tree_noise: Perlin,
    forest_noise: Perlin,

    // Structures
    overhang_noise: Perlin,        // Overhangs and cliff shelves
    outcrop_noise: Perlin,         // Rocky outcrops

    // Biomes
    biome_noise: Perlin,           // Large-scale biome regions
}

impl TerrainGenerator {
    fn new(seed: u32) -> Self {
        Self {
            // Base terrain - 6 octaves for natural variation
            height_noise: Fbm::new(seed)
                .set_octaves(6)
                .set_frequency(0.004)
                .set_lacunarity(2.0)
                .set_persistence(0.5),
            detail_noise: Perlin::new(seed.wrapping_add(1)),

            // Mountain system
            mountain_noise: RidgedMulti::new(seed.wrapping_add(10))
                .set_octaves(5)
                .set_frequency(0.003)
                .set_lacunarity(2.2),
            ridge_noise: Perlin::new(seed.wrapping_add(11)),

            // Cave system (expanded for better caves)
            cave_worm_noise: Perlin::new(seed.wrapping_add(20)),
            cave_worm_noise2: Perlin::new(seed.wrapping_add(21)),
            cave_chamber_noise: Perlin::new(seed.wrapping_add(22)),
            cave_cavern_noise: Perlin::new(seed.wrapping_add(23)),
            cave_width_noise: Perlin::new(seed.wrapping_add(24)),

            // Ores
            ore_noise: Perlin::new(seed.wrapping_add(30)),

            // Trees and biomes
            tree_noise: Perlin::new(seed.wrapping_add(40)),
            forest_noise: Perlin::new(seed.wrapping_add(41)),

            // Structures
            overhang_noise: Perlin::new(seed.wrapping_add(50)),
            outcrop_noise: Perlin::new(seed.wrapping_add(52)),

            // Biomes
            biome_noise: Perlin::new(seed.wrapping_add(60)),
        }
    }

    /// Compute and cache all column-based data for a single (x, z) position
    fn compute_column_cache(&self, x: i32, z: i32) -> ColumnCache {
        let height = self.get_terrain_height(x, z);
        let is_mountain = self.get_mountain_mask(x, z) > 0.3;

        // Simplified cliff detection - sample fewer points
        let h_ns = (self.get_terrain_height(x, z - 2) - self.get_terrain_height(x, z + 2)).abs();
        let h_ew = (self.get_terrain_height(x + 2, z) - self.get_terrain_height(x - 2, z)).abs();
        let cliff_factor = ((h_ns.max(h_ew)) as f64 / 8.0).min(1.0);

        let is_rocky_outcrop = self.is_rocky_outcrop(x, z);

        // Determine biome based on large-scale noise
        let biome = self.get_biome(x, z);

        // Beach near water level
        let is_beach = height < WATER_SEA_LEVEL + 2 && height >= WATER_SEA_LEVEL - 3;

        ColumnCache {
            height,
            is_mountain,
            cliff_factor,
            is_rocky_outcrop,
            biome,
            is_beach,
        }
    }

    /// Determine which biome this position belongs to
    fn get_biome(&self, x: i32, z: i32) -> Biome {
        let xf = x as f64;
        let zf = z as f64;

        // Very low frequency noise for large biome regions
        let biome_value = self.biome_noise.get([xf * 0.002, zf * 0.002]);

        // Secondary noise for biome edge variation
        let detail = self.biome_noise.get([xf * 0.008 + 100.0, zf * 0.008 + 100.0]) * 0.15;

        let combined = biome_value + detail;

        // Cavernous Highlands in ~25% of the world (where biome noise is high)
        if combined > 0.35 {
            Biome::CavernousHighlands
        } else {
            Biome::Grasslands
        }
    }

    /// Get block using pre-computed column cache (fast path)
    fn get_block_cached(&self, x: i32, y: i32, z: i32, col: &ColumnCache) -> Block {
        // Bedrock at bottom
        if y <= 0 {
            return Block::Bedrock;
        }

        // Above terrain surface
        if y > col.height {
            if y < WATER_SEA_LEVEL {
                return Block::Water;
            }
            return Block::Air;
        }

        // Check for caves - biome determines cave style
        if y > 5 && self.is_cave_biome_aware(x, y, z, col.height, col.biome) {
            // Underwater caves fill with water
            if y < WATER_SEA_LEVEL && col.height < WATER_SEA_LEVEL + 4 {
                return Block::Water;
            }
            return Block::Air;
        }

        // Determine block type based on depth
        let depth = col.height - y;

        // Surface layer
        if depth == 0 {
            if col.is_beach {
                return Block::Sand;
            }
            if col.is_mountain && col.height > 85 {
                return self.get_stone_variant(x, y, z);
            }
            if col.cliff_factor > 0.6 {
                return self.get_stone_variant(x, y, z);
            }
            if col.is_rocky_outcrop {
                return self.get_stone_variant(x, y, z);
            }
            return Block::Grass;
        }

        // Near surface (1-3 blocks deep)
        if depth < 4 {
            if col.is_beach {
                return Block::Sand;
            }
            if col.cliff_factor > 0.5 || (col.is_mountain && col.height > 75) {
                return self.get_stone_variant(x, y, z);
            }
            return Block::Dirt;
        }

        // Underground
        self.get_ore_or_stone_fast(x, y, z)
    }

    #[allow(dead_code)]
    fn get_block(&self, x: i32, y: i32, z: i32) -> Block {
        let col = self.compute_column_cache(x, z);
        self.get_block_cached(x, y, z, &col)
    }

    /// Calculate terrain height with mountains, ridges, and detail
    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        let xf = x as f64;
        let zf = z as f64;

        // Base continental terrain
        let base = self.height_noise.get([xf, zf]) * 25.0;

        // Small detail variation
        let detail = self.detail_noise.get([xf * 0.08, zf * 0.08]) * 4.0;

        // Mountain regions using ridged noise for dramatic peaks
        let mountain_mask = self.get_mountain_mask(x, z);
        let mountain_height = if mountain_mask > 0.0 {
            let ridge = self.mountain_noise.get([xf, zf]);

            // Multiple ridge layers for complex mountain shapes
            let ridge2 = self.ridge_noise.get([xf * 0.015, zf * 0.015]).abs();

            // Combine for dramatic peaks
            let combined = (ridge * 0.7 + ridge2 * 0.3).max(0.0);
            combined * mountain_mask * 100.0
        } else {
            0.0
        };

        // Gentle hills in non-mountain areas
        let hills = if mountain_mask < 0.2 {
            let hill_noise = self.detail_noise.get([xf * 0.02, zf * 0.02]);
            (hill_noise * 15.0).max(0.0) * (1.0 - mountain_mask * 5.0).max(0.0)
        } else {
            0.0
        };

        // Overhang influence (raises terrain where overhangs form)
        let overhang_boost = self.get_overhang_boost(x, z);

        (42.0 + base + detail + mountain_height + hills + overhang_boost) as i32
    }

    /// Get mountain region mask (0.0 = no mountains, 1.0 = full mountains)
    fn get_mountain_mask(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        // Large-scale mountain regions
        let region = self.height_noise.get([xf * 0.001, zf * 0.001]);

        // Mountains form where noise is high
        if region > 0.25 {
            ((region - 0.25) / 0.5).min(1.0)
        } else {
            0.0
        }
    }

    /// Get cliff steepness factor (0.0 = flat, 1.0 = vertical cliff)
    fn get_cliff_factor(&self, x: i32, z: i32) -> f64 {
        // Sample terrain height at nearby points to detect slopes
        let h_north = self.get_raw_height(x, z - 2);
        let h_south = self.get_raw_height(x, z + 2);
        let h_east = self.get_raw_height(x + 2, z);
        let h_west = self.get_raw_height(x - 2, z);

        // Calculate maximum slope
        let slope_ns = (h_north - h_south).abs();
        let slope_ew = (h_east - h_west).abs();
        let max_slope = slope_ns.max(slope_ew);

        // Normalize to 0-1 range (slopes > 8 blocks per 4 = cliff)
        (max_slope / 8.0).min(1.0)
    }

    /// Get raw height without clamping (for slope calculations)
    fn get_raw_height(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        let base = self.height_noise.get([xf, zf]) * 25.0;
        let mountain_mask = self.get_mountain_mask(x, z);
        let mountain_height = if mountain_mask > 0.0 {
            let ridge = self.mountain_noise.get([xf, zf]);
            ridge.max(0.0) * mountain_mask * 100.0
        } else {
            0.0
        };

        42.0 + base + mountain_height
    }

    /// Get overhang height boost (creates overhanging cliffs)
    fn get_overhang_boost(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        // Only create overhangs in cliff/mountain regions
        let cliff = self.get_cliff_factor(x, z);
        if cliff < 0.4 {
            return 0.0;
        }

        // Overhang noise
        let overhang = self.overhang_noise.get([xf * 0.05, zf * 0.05]);

        if overhang > 0.5 {
            (overhang - 0.5) * 10.0 * cliff
        } else {
            0.0
        }
    }

    /// Biome-aware cave detection
    /// - Grasslands: caves stay underground, tunnel-based, rarely open to surface
    /// - CavernousHighlands: many surface caverns, large open spaces
    #[inline]
    fn is_cave_biome_aware(&self, x: i32, y: i32, z: i32, surface_height: i32, biome: Biome) -> bool {
        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Depth below surface - used to prevent surface openings in Grasslands
        let depth_below_surface = surface_height - y;

        match biome {
            Biome::Grasslands => self.is_cave_grasslands(xf, yf, zf, y, depth_below_surface),
            Biome::CavernousHighlands => self.is_cave_cavernous(xf, yf, zf, y, depth_below_surface),
        }
    }

    /// Grasslands caves: stay underground, tunnel entrances, no surface openings
    #[inline]
    fn is_cave_grasslands(&self, xf: f64, yf: f64, zf: f64, y: i32, depth: i32) -> bool {
        // No caves within 5 blocks of surface (prevents surface openings)
        if depth < 5 {
            return false;
        }

        // Primary worm cave system - narrower tunnels
        let worm_scale = 0.04;
        let worm1 = self.cave_worm_noise.get([xf * worm_scale, yf * worm_scale * 0.7, zf * worm_scale]);
        let worm2 = self.cave_worm_noise2.get([xf * worm_scale + 100.0, yf * worm_scale * 0.7, zf * worm_scale + 100.0]);

        // Narrower threshold for Grasslands (0.07 vs 0.09)
        if worm1.abs() < 0.07 && worm2.abs() < 0.07 {
            return true;
        }

        // Small chambers deep underground only
        if y > 8 && y < 35 && depth > 10 {
            let chamber = self.cave_chamber_noise.get([xf * 0.03, yf * 0.025, zf * 0.03]);
            // Higher threshold = rarer chambers (0.68)
            if chamber > 0.68 {
                return true;
            }
        }

        // Rare large caverns - only very deep underground
        if y > 5 && y < 25 && depth > 20 {
            let cavern = self.cave_cavern_noise.get([xf * 0.015, yf * 0.01, zf * 0.015]);
            // Very high threshold = rare (0.72)
            if cavern > 0.72 {
                return true;
            }
        }

        false
    }

    /// Cavernous Highlands caves: many surface openings, large caverns
    #[inline]
    fn is_cave_cavernous(&self, xf: f64, yf: f64, zf: f64, y: i32, _depth: i32) -> bool {
        // Variable tunnel width
        let width_mod = self.cave_width_noise.get([xf * 0.02, yf * 0.015, zf * 0.02]);
        let threshold = 0.08 + width_mod.abs() * 0.06;

        // Primary worm caves with variable width
        let worm_scale = 0.04;
        let worm1 = self.cave_worm_noise.get([xf * worm_scale, yf * worm_scale * 0.7, zf * worm_scale]);
        let worm2 = self.cave_worm_noise2.get([xf * worm_scale + 100.0, yf * worm_scale * 0.7, zf * worm_scale + 100.0]);

        if worm1.abs() < threshold && worm2.abs() < threshold {
            return true;
        }

        // Secondary larger tunnel network
        let worm_scale2 = 0.025;
        let worm3 = self.cave_worm_noise.get([xf * worm_scale2 + 500.0, yf * worm_scale2 * 0.6, zf * worm_scale2 + 500.0]);
        let worm4 = self.cave_worm_noise2.get([xf * worm_scale2 + 600.0, yf * worm_scale2 * 0.6, zf * worm_scale2 + 600.0]);
        if worm3.abs() < 0.07 && worm4.abs() < 0.07 {
            return true;
        }

        // Large surface caverns (the signature feature)
        if y > 8 && y < 55 {
            let cavern = self.cave_cavern_noise.get([xf * 0.012, yf * 0.008, zf * 0.012]);
            if cavern > 0.5 {
                let detail = self.cave_chamber_noise.get([xf * 0.06, yf * 0.04, zf * 0.06]);
                if cavern + detail * 0.15 > 0.55 {
                    return true;
                }
            }
        }

        // Medium chambers throughout
        if y > 10 && y < 45 {
            let chamber = self.cave_chamber_noise.get([xf * 0.025, yf * 0.02, zf * 0.025]);
            if chamber > 0.55 {
                return true;
            }
        }

        // Deep caverns
        if y > 5 && y < 30 {
            let deep = self.cave_cavern_noise.get([xf * 0.018 + 200.0, yf * 0.01, zf * 0.018 + 200.0]);
            if deep > 0.48 {
                return true;
            }
        }

        false
    }

    /// Check if position is a rocky outcrop
    fn is_rocky_outcrop(&self, x: i32, z: i32) -> bool {
        let xf = x as f64;
        let zf = z as f64;

        // Rocky outcrops appear in certain areas
        let outcrop = self.outcrop_noise.get([xf * 0.02, zf * 0.02]);

        if outcrop > 0.6 {
            // Fine detail for individual rocks
            let detail = self.outcrop_noise.get([xf * 0.15, zf * 0.15]);
            detail > 0.3
        } else {
            false
        }
    }

    /// Get stone block (simplified - no random variants)
    #[inline]
    fn get_stone_variant(&self, _x: i32, _y: i32, _z: i32) -> Block {
        Block::Stone
    }

    /// Balanced ore generation with depth-based distribution
    /// Each position can only have ONE ore type - determined by hash
    #[inline]
    fn get_ore_or_stone_fast(&self, x: i32, y: i32, z: i32) -> Block {
        let hash = fast_hash_3d(x, y, z);

        // Noise for vein clustering
        let ore_noise = self.ore_noise.get([
            x as f64 * 0.1,
            y as f64 * 0.1,
            z as f64 * 0.1,
        ]);

        // Coal uses lower frequency noise for chunkier veins (4-8 blocks typically)
        let coal_noise = self.ore_noise.get([
            x as f64 * 0.06,
            y as f64 * 0.06,
            z as f64 * 0.06,
        ]);

        // Use hash to pick ONE ore type - smaller slots = rarer
        let ore_slot = hash % 1000;

        // Coal first - uses separate noise for chunky veins
        if y >= 20 && y < 90 && coal_noise > 0.5 && ore_slot < 350 {
            return Block::CoalOre;
        }

        // Must be in an ore-favorable zone for other ores
        if ore_noise < 0.55 {
            return Block::Stone;
        }

        // Deep ores (y < 16) - very rare
        if y < 16 {
            if ore_slot < 3 && y < 10 { return Block::UraniumOre; }      // 0.3%
            if ore_slot < 10 && y < 14 { return Block::TitaniumOre; }    // 0.7%
            if ore_slot < 25 { return Block::DiamondOre; }               // 1.5%
            if ore_slot < 45 { return Block::SapphireOre; }              // 2%
        }

        // Mid-deep ores (y < 32)
        if y < 32 {
            if ore_slot < 15 { return Block::RedstoneOre; }              // 1.5%
            if ore_slot < 25 { return Block::EmeraldOre; }               // 1% (rare)
            if ore_slot < 45 && y < 28 { return Block::AmethystOre; }    // 2%
            if ore_slot < 75 { return Block::GoldOre; }                  // 3%
            if ore_slot < 105 { return Block::LeadOre; }                 // 3%
        }

        // Mid ores (y < 50)
        if y < 50 {
            if ore_slot < 40 { return Block::SilverOre; }                // 4%
            if ore_slot < 80 { return Block::NickelOre; }                // 4%
            if ore_slot < 130 { return Block::QuartzOre; }               // 5%
            if ore_slot < 200 { return Block::IronOre; }                 // 7%
        }

        // Upper ores (y < 70)
        if y < 70 {
            if ore_slot < 80 { return Block::CopperOre; }                // 8%
            if ore_slot < 160 { return Block::TinOre; }                  // 8%
        }

        Block::Stone
    }

    #[allow(dead_code)]
    fn get_ore_or_stone(&self, x: i32, y: i32, z: i32) -> Block {
        self.get_ore_or_stone_fast(x, y, z)
    }

    /// Determine tree type based on position and noise
    fn get_tree_type(&self, x: i32, z: i32) -> TreeType {
        let xf = x as f64;
        let zf = z as f64;

        // Use tree noise for variety
        let type_noise = self.tree_noise.get([xf * 0.1 + 200.0, zf * 0.1 + 200.0]);

        // Check for mountain/high elevation - prefer pines
        let height = self.get_terrain_height(x, z);
        if height > 65 {
            return TreeType::TallPine;
        }

        // Hash for additional randomness
        let hash = ((x.wrapping_mul(73856093)) ^ (z.wrapping_mul(19349663))) as u32;
        let hash_val = (hash % 100) as f64 / 100.0;

        if type_noise > 0.4 && hash_val > 0.7 {
            TreeType::LargeOak
        } else if type_noise < -0.3 {
            TreeType::TallPine
        } else if type_noise < 0.0 && hash_val < 0.3 {
            TreeType::BushyShrub
        } else {
            TreeType::SmallOak
        }
    }

    /// Check if a tree should be placed at this world position
    fn should_place_tree(&self, x: i32, z: i32) -> bool {
        // Forest biome noise - low frequency creates large forest regions
        let forest_value = self.forest_noise.get([x as f64 * 0.008, z as f64 * 0.008]);

        // Only place trees in forest biomes (where forest noise is high)
        if forest_value < 0.2 {
            return false;  // Not a forest area
        }

        // Tree density increases toward forest center
        let forest_density = ((forest_value - 0.2) / 0.8).min(1.0);

        // Local tree placement noise - higher frequency for individual tree spots
        let tree_value = self.tree_noise.get([x as f64 * 0.5, z as f64 * 0.5]);

        // Use a hash for final randomization
        let hash = ((x.wrapping_mul(73856093)) ^ (z.wrapping_mul(19349663))) as u32;
        let hash_value = (hash % 1000) as f64 / 1000.0;

        // Dense forests: ~15-25% tree coverage in forest centers
        // Sparse edges: ~5% coverage at forest edges
        let density_threshold = 0.75 + (1.0 - forest_density) * 0.20;

        tree_value > 0.0 && hash_value > density_threshold
    }

    /// Check if ground is solid at and around a position (no caves)
    /// Returns true if the surface is stable for tree placement
    fn is_ground_stable_for_tree(&self, x: i32, surface_y: i32, z: i32) -> bool {
        let biome = self.get_biome(x, z);

        // Check the tree base position and immediate surroundings
        // Need solid ground at surface and a few blocks below
        for dy in 0..3 {
            let check_y = surface_y - dy;
            if check_y <= 0 {
                continue;
            }
            // Check if this would be a cave
            if self.is_cave_biome_aware(x, check_y, z, surface_y, biome) {
                return false; // Cave at or near surface
            }
        }

        // Also check the 4 cardinal neighbors at surface level
        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let nz = z + dz;
            let neighbor_height = self.get_terrain_height(nx, nz);
            let neighbor_biome = self.get_biome(nx, nz);

            // Check if neighbor surface is a cave
            if self.is_cave_biome_aware(nx, neighbor_height, nz, neighbor_height, neighbor_biome) {
                return false; // Adjacent ground is hollow
            }

            // Check for large height differences (cliff edges)
            if (neighbor_height - surface_y).abs() > 2 {
                return false; // Too steep for tree
            }
        }

        true
    }

    /// Get tree height based on type and position
    fn get_tree_height(&self, x: i32, z: i32, tree_type: TreeType) -> i32 {
        let variation = self.tree_noise.get([x as f64 * 0.5 + 100.0, z as f64 * 0.5 + 100.0]);
        let var_int = (variation * 2.0).abs() as i32;

        match tree_type {
            TreeType::SmallOak => 5 + var_int,      // 5-7 blocks
            TreeType::LargeOak => 8 + var_int * 2,  // 8-12 blocks
            TreeType::TallPine => 10 + var_int * 2, // 10-14 blocks
            TreeType::BushyShrub => 2 + (var_int / 2), // 2-3 blocks
        }
    }

    /// Add trees to a chunk
    fn add_trees(&self, chunk: &mut Chunk, pos: ChunkPos) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Tree canopy can extend up to 4 blocks in each direction for large oaks
        let margin = 5;

        for z in -margin..(CHUNK_SIZE as i32 + margin) {
            for x in -margin..(CHUNK_SIZE as i32 + margin) {
                let wx = world_x + x;
                let wz = world_z + z;

                if !self.should_place_tree(wx, wz) {
                    continue;
                }

                let surface_y = self.get_terrain_height(wx, wz);

                // Only place trees on suitable terrain
                if surface_y < 35 || surface_y > 90 {
                    continue; // Skip underwater and extreme peaks
                }

                // Skip steep cliffs
                if self.get_cliff_factor(wx, wz) > 0.5 {
                    continue;
                }

                // IMPORTANT: Check that ground is solid (no caves beneath)
                // This prevents floating trees over caverns
                if !self.is_ground_stable_for_tree(wx, surface_y, wz) {
                    continue;
                }

                let tree_type = self.get_tree_type(wx, wz);
                let tree_height = self.get_tree_height(wx, wz, tree_type);
                let trunk_base = surface_y + 1;

                self.generate_tree_in_chunk(chunk, pos, wx, trunk_base, wz, tree_height, tree_type);
            }
        }
    }

    /// Generate a tree's blocks within a specific chunk
    fn generate_tree_in_chunk(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        tree_x: i32,
        tree_base_y: i32,
        tree_z: i32,
        height: i32,
        tree_type: TreeType,
    ) {
        match tree_type {
            TreeType::SmallOak => self.generate_small_oak(chunk, pos, tree_x, tree_base_y, tree_z, height),
            TreeType::LargeOak => self.generate_large_oak(chunk, pos, tree_x, tree_base_y, tree_z, height),
            TreeType::TallPine => self.generate_tall_pine(chunk, pos, tree_x, tree_base_y, tree_z, height),
            TreeType::BushyShrub => self.generate_bushy_shrub(chunk, pos, tree_x, tree_base_y, tree_z, height),
        }
    }

    /// Generate a small oak tree (original style)
    fn generate_small_oak(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        tree_x: i32,
        tree_base_y: i32,
        tree_z: i32,
        height: i32,
    ) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Generate trunk (stop 2 blocks below top so it's hidden in canopy)
        let trunk_height = height - 2;
        for dy in 0..trunk_height {
            let wy = tree_base_y + dy;
            self.try_set_block(chunk, tree_x - world_x, wy - world_y, tree_z - world_z, Block::OakLog, false);
        }

        // Generate leaves (spherical canopy)
        let canopy_center = tree_base_y + height - 2;
        let leaf_start = canopy_center - 1;
        let leaf_end = canopy_center + 2;

        for dy in leaf_start..=leaf_end {
            let y_offset = dy - canopy_center;
            let current_radius: i32 = if y_offset.abs() >= 2 {
                1  // Top and bottom are narrow
            } else {
                2  // Middle is full
            };

            for dx in -current_radius..=current_radius {
                for dz in -current_radius..=current_radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > current_radius * current_radius + 1 {
                        continue;
                    }

                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;

                    self.try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Generate a large spreading oak
    fn generate_large_oak(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        tree_x: i32,
        tree_base_y: i32,
        tree_z: i32,
        height: i32,
    ) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Trunk stops inside canopy (height - 3)
        let trunk_top = height - 3;
        for dy in 0..trunk_top {
            let wy = tree_base_y + dy;

            // Trunk gets thinner toward top
            if dy < height / 3 {
                // 2x2 trunk at base
                for dx in 0..=1 {
                    for dz in 0..=1 {
                        self.try_set_block(chunk, tree_x + dx - world_x, wy - world_y, tree_z + dz - world_z, Block::OakLog, false);
                    }
                }
            } else {
                // 1x1 trunk higher up
                self.try_set_block(chunk, tree_x - world_x, wy - world_y, tree_z - world_z, Block::OakLog, false);
            }
        }

        // Large spreading canopy
        let canopy_center_y = tree_base_y + height - 3;
        let canopy_radius: i32 = 4;

        for dy in (canopy_center_y - 2)..=(canopy_center_y + 2) {
            let y_dist = (dy - canopy_center_y).abs();
            let layer_radius = if y_dist == 0 {
                canopy_radius
            } else if y_dist == 1 {
                canopy_radius - 1
            } else {
                2
            };

            for dx in -layer_radius..=layer_radius {
                for dz in -layer_radius..=layer_radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > layer_radius * layer_radius + 2 {
                        continue;
                    }

                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;

                    self.try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Generate a tall pine tree
    fn generate_tall_pine(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        tree_x: i32,
        tree_base_y: i32,
        tree_z: i32,
        height: i32,
    ) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Trunk stops before the pointed top (height - 2)
        let trunk_top = height - 2;
        for dy in 0..trunk_top {
            let wy = tree_base_y + dy;
            self.try_set_block(chunk, tree_x - world_x, wy - world_y, tree_z - world_z, Block::Wood, false);
        }

        // Conical canopy - layered tiers like a real pine
        let canopy_start = tree_base_y + (height / 3);
        let canopy_end = tree_base_y + height;

        for dy in canopy_start..=canopy_end {
            // Radius decreases as we go up, with tiers
            let progress = (dy - canopy_start) as f32 / (canopy_end - canopy_start) as f32;
            let base_radius = ((1.0 - progress) * 3.5) as i32;

            // Create tiered effect - every other layer is slightly smaller
            let tier_offset = if (dy - canopy_start) % 2 == 0 { 0 } else { 1 };
            let layer_radius = (base_radius - tier_offset).max(0);

            if layer_radius == 0 {
                // Top point is leaves
                let lx = tree_x - world_x;
                let ly = dy - world_y;
                let lz = tree_z - world_z;
                self.try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                continue;
            }

            for dx in -layer_radius..=layer_radius {
                for dz in -layer_radius..=layer_radius {
                    // Diamond shape for pine layers
                    if dx.abs() + dz.abs() > layer_radius {
                        continue;
                    }

                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;

                    self.try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Generate a short bushy shrub
    fn generate_bushy_shrub(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        tree_x: i32,
        tree_base_y: i32,
        tree_z: i32,
        height: i32,
    ) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Short trunk (1-2 blocks)
        let trunk_height = height.min(2);
        for dy in 0..trunk_height {
            let wy = tree_base_y + dy;
            self.try_set_block(chunk, tree_x - world_x, wy - world_y, tree_z - world_z, Block::BirchLog, false);
        }

        // Dense bushy leaves
        let bush_center_y = tree_base_y + trunk_height;
        let bush_radius: i32 = 2;

        for dy in (bush_center_y - 1)..=(bush_center_y + 1) {
            for dx in -bush_radius..=bush_radius {
                for dz in -bush_radius..=bush_radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > bush_radius * bush_radius {
                        continue;
                    }

                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;

                    self.try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Try to set a block if within chunk bounds
    fn try_set_block(&self, chunk: &mut Chunk, lx: i32, ly: i32, lz: i32, block: Block, skip_solid: bool) {
        if lx >= 0 && lx < CHUNK_SIZE_I32
            && ly >= 0 && ly < CHUNK_SIZE_I32
            && lz >= 0 && lz < CHUNK_SIZE_I32
        {
            if skip_solid {
                let current = chunk.get_block(lx as usize, ly as usize, lz as usize);
                if current != Block::Air {
                    return;
                }
            }

            // Canopy gaps: randomly skip ~25% of leaf blocks for natural look
            if block == Block::Leaves {
                let hash = fast_hash_3d(lx, ly, lz);
                // Skip leaves at edges more often (creates natural tapered edges)
                // and randomly skip some interior leaves for gaps
                if hash % 100 < 25 {
                    return;
                }
            }

            chunk.set_block(lx as usize, ly as usize, lz as usize, block);
        }
    }

    /// Generate water heightfield data for a chunk
    fn generate_water_heightfield(&self, chunk: &mut Chunk, pos: ChunkPos) {
        let world_y_base = pos.y * CHUNK_SIZE_I32;
        let world_x_base = pos.x * CHUNK_SIZE_I32;
        let world_z_base = pos.z * CHUNK_SIZE_I32;

        let chunk_y_min = world_y_base;
        let chunk_y_max = world_y_base + CHUNK_SIZE_I32;

        if chunk_y_max < WATER_SEA_LEVEL - 2 || chunk_y_min > WATER_SEA_LEVEL + 2 {
            return;
        }

        let mut has_water = false;

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let wx = world_x_base + x as i32;
                let wz = world_z_base + z as i32;

                let terrain_height = self.get_terrain_height(wx, wz);

                if terrain_height < WATER_SEA_LEVEL {
                    let water_surface_y = (WATER_SEA_LEVEL as f32 - 1.0 + WATER_FULL_HEIGHT) * BLOCK_SCALE;

                    let depth = (WATER_SEA_LEVEL - 1 - terrain_height).max(0);
                    let foam: f32 = if depth <= 1 {
                        0.8
                    } else if depth <= 3 {
                        0.4
                    } else {
                        0.0
                    };

                    let edge_foam = self.calculate_edge_foam(chunk, x, z);
                    let total_foam = foam.max(edge_foam);

                    chunk.set_water_surface(x, z, water_surface_y, total_foam);
                    has_water = true;
                }
            }
        }

        if has_water {
            if let Some(water_data) = chunk.water_data_mut() {
                water_data.recalculate_bounds();
            }
        }
    }

    fn calculate_edge_foam(&self, chunk: &Chunk, x: usize, z: usize) -> f32 {
        let mut near_edge = false;

        for y in 0..CHUNK_SIZE {
            let block = chunk.get_block(x, y, z);
            if block.is_solid() {
                for (dx, dz) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if nx >= 0 && nx < CHUNK_SIZE_I32 && nz >= 0 && nz < CHUNK_SIZE_I32 {
                        let neighbor = chunk.get_block(nx as usize, y, nz as usize);
                        if matches!(neighbor, Block::Water | Block::Air) {
                            near_edge = true;
                            break;
                        }
                    }
                }
            }
            if near_edge {
                break;
            }
        }

        if near_edge { 0.6 } else { 0.0 }
    }
}
