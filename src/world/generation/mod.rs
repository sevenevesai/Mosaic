//! Procedural terrain generation
//!
//! Uses layered noise for realistic terrain with caves, ores, biomes, and trees.
//! Features multiple tree types, rocky outcrops, cliffs, and biome-specific terrain.
//!
//! Optimized with column-based caching to minimize redundant noise calculations.

pub mod biome;
pub mod biome_data;
pub mod caverns;
pub mod caves;
pub mod formations;
pub mod ruins;
pub mod trees;
pub mod util;

use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

use super::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{
    BLOCK_SCALE, CHUNK_SIZE, CHUNK_SIZE_I32, WATER_FLOW_INITIAL_SCALE, WATER_FULL_HEIGHT,
    WATER_SEA_LEVEL,
};

use biome::BiomeRegistry;
use caverns::CavernGenerator;
use caves::CaveNoise;
use formations::{FormationGenerator, FormationNoise};
use ruins::RuinGenerator;
use trees::{TreeGenerator, TreeNoise};
use util::fast_hash_3d;

use std::cell::RefCell;

// Thread-local cached TerrainGenerator to avoid recreating noise objects per-chunk
// Each worker thread maintains its own generator instance, reused across all chunk generations
thread_local! {
    static CACHED_GENERATOR: RefCell<Option<(u32, TerrainGenerator)>> = const { RefCell::new(None) };
}

/// Cached data for a single terrain column (x, z)
/// Pre-computed once per column to avoid redundant noise calculations
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct ColumnCache {
    height: i32,
    is_mountain: bool,
    cliff_factor: f64,
    is_rocky_outcrop: bool,
    biome_id: biome::BiomeId,
    is_beach: bool,
}

/// Generate a chunk at the given position with the specified seed
pub fn generate_chunk_with_seed(pos: ChunkPos, seed: u64) -> Chunk {
    // Use thread-local cached generator to avoid recreating noise objects per-chunk
    // This provides significant speedup since noise initialization is expensive
    CACHED_GENERATOR.with(|cache| {
        let mut cache = cache.borrow_mut();
        let seed32 = seed as u32;

        // Create generator if not cached or seed changed
        if cache.as_ref().map(|(s, _)| *s) != Some(seed32) {
            *cache = Some((seed32, TerrainGenerator::new(seed32)));
        }

        let generator = &cache.as_ref().unwrap().1;
        generate_chunk_with_generator(pos, generator)
    })
}

/// Internal function that generates a chunk using a pre-existing generator
fn generate_chunk_with_generator(pos: ChunkPos, generator: &TerrainGenerator) -> Chunk {
    let mut chunk = Chunk::new();

    let world_x = pos.x * CHUNK_SIZE_I32;
    let world_y = pos.y * CHUNK_SIZE_I32;
    let world_z = pos.z * CHUNK_SIZE_I32;

    // Pre-compute column cache for all 16x16 columns
    let mut columns = [[ColumnCache {
        height: 0,
        is_mountain: false,
        cliff_factor: 0.0,
        is_rocky_outcrop: false,
        biome_id: biome::BiomeId::Grasslands,
        is_beach: false,
    }; CHUNK_SIZE]; CHUNK_SIZE];

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let wx = world_x + x as i32;
            let wz = world_z + z as i32;
            columns[z][x] = generator.compute_column_cache(wx, wz);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EARLY EXIT: Skip block generation for chunks entirely above terrain
    // If chunk bottom is above max terrain height + max tree height, it's all air
    // ═══════════════════════════════════════════════════════════════════════════
    let max_terrain = columns.iter().flatten().map(|c| c.height).max().unwrap_or(0);
    const MAX_TREE_HEIGHT: i32 = 20; // Tallest trees + some margin
    const MAX_STRUCTURE_HEIGHT: i32 = 30; // Formations, ruins, etc.

    if world_y > max_terrain + MAX_TREE_HEIGHT + MAX_STRUCTURE_HEIGHT
        && world_y >= WATER_SEA_LEVEL
    {
        // Chunk is entirely above terrain and water - return empty (all air)
        // Sky light calculation will set full sky light for all blocks
        calculate_sky_light(&mut chunk, pos, &columns);
        return chunk;
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

    // Add stone formations (arches, hoodoos, boulders)
    generator.add_formations(&mut chunk, pos);

    // Add ruins and ancient structures
    generator.add_ruins(&mut chunk, pos);

    // Add trees
    generator.add_trees(&mut chunk, pos);

    // Generate water heightfield data
    generator.generate_water_heightfield(&mut chunk, pos);

    // Calculate sky light for this chunk
    calculate_sky_light(&mut chunk, pos, &columns);

    chunk
}

/// Calculate sky light for a chunk based on actual block opacity
/// Sky light propagates downward, attenuating through transparent blocks and blocked by solid blocks
fn calculate_sky_light(chunk: &mut Chunk, _pos: ChunkPos, _columns: &[[ColumnCache; CHUNK_SIZE]; CHUNK_SIZE]) {
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            // Process from top to bottom of chunk
            // Start with full sky light at the top
            let mut current_sky_light: u8 = 15;

            for y in (0..CHUNK_SIZE).rev() {
                let block = chunk.get_block(x, y, z);

                if block.is_solid() {
                    // Solid blocks always have 0 sky light and block all light below
                    chunk.set_sky_light(x, y, z, 0);
                    current_sky_light = 0;
                } else if block.is_light_transparent() {
                    // Transparent blocks (air, water, etc.) receive current sky light
                    chunk.set_sky_light(x, y, z, current_sky_light);
                    // Sky light attenuates by 1 when passing through any transparent block
                    // except air which doesn't attenuate sky light
                    if block != super::chunk::Block::Air {
                        current_sky_light = current_sky_light.saturating_sub(1);
                    }
                } else {
                    // Non-solid, non-transparent (shouldn't happen, but handle it)
                    chunk.set_sky_light(x, y, z, current_sky_light);
                }
            }
        }
    }
}

/// Generate a chunk at the given position (uses default seed for backwards compatibility)
#[allow(dead_code)]
pub fn generate_chunk(pos: ChunkPos) -> Chunk {
    generate_chunk_with_seed(pos, 12345)
}

/// Terrain generator using layered noise
pub struct TerrainGenerator {
    // Base terrain
    height_noise: Fbm<Perlin>,
    detail_noise: Perlin,

    // Mountain system
    mountain_noise: RidgedMulti<Perlin>,
    ridge_noise: Perlin,

    // Cave system
    cave_noise: CaveNoise,

    // Large caverns (dome ceilings, pillars, crystals, aquifers)
    #[allow(dead_code)]
    cavern_generator: CavernGenerator,

    // Tree system
    tree_noise: TreeNoise,

    // Stone formations (arches, hoodoos, boulders)
    formation_noise: FormationNoise,

    // Ruins and ancient structures
    ruin_generator: RuinGenerator,

    // Ores
    ore_noise: Perlin,

    // Structures
    overhang_noise: Perlin,
    outcrop_noise: Perlin,

    // Biome registry
    biome_registry: BiomeRegistry,
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

            // Cave system
            cave_noise: CaveNoise::new(seed),

            // Large caverns
            cavern_generator: CavernGenerator::new(seed),

            // Tree system
            tree_noise: TreeNoise::new(seed),

            // Stone formations
            formation_noise: FormationNoise::new(seed),

            // Ruins
            ruin_generator: RuinGenerator::new(seed),

            // Ores
            ore_noise: Perlin::new(seed.wrapping_add(30)),

            // Structures
            overhang_noise: Perlin::new(seed.wrapping_add(50)),
            outcrop_noise: Perlin::new(seed.wrapping_add(52)),

            // Biome registry
            biome_registry: BiomeRegistry::new(seed),
        }
    }

    /// Compute and cache all column-based data for a single (x, z) position
    fn compute_column_cache(&self, x: i32, z: i32) -> ColumnCache {
        let height = self.get_terrain_height(x, z);
        let is_mountain = self.get_mountain_mask(x, z) > 0.3;

        // Simplified cliff detection
        let h_ns =
            (self.get_terrain_height(x, z - 2) - self.get_terrain_height(x, z + 2)).abs();
        let h_ew =
            (self.get_terrain_height(x + 2, z) - self.get_terrain_height(x - 2, z)).abs();
        let cliff_factor = ((h_ns.max(h_ew)) as f64 / 8.0).min(1.0);

        let is_rocky_outcrop = self.is_rocky_outcrop(x, z);
        let biome_id = self.biome_registry.get_biome_id(x, z);
        let is_beach = height < WATER_SEA_LEVEL + 2 && height >= WATER_SEA_LEVEL - 3;

        ColumnCache {
            height,
            is_mountain,
            cliff_factor,
            is_rocky_outcrop,
            biome_id,
            is_beach,
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

        // Get biome for surface blocks and cave checks
        let biome = self.biome_registry.get_biome(x, z);

        // Check for caves using biome-specific logic
        let is_cave = y > 5 && biome.is_cave(x, y, z, col.height, &self.cave_noise);

        if is_cave {
            // Underwater caves fill with water
            if y < WATER_SEA_LEVEL && col.height < WATER_SEA_LEVEL + 4 {
                return Block::Water;
            }
            return Block::Air;
        }

        // Determine block type based on depth
        let depth = col.height - y;
        let surface_params = biome.surface_params();

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
            // Use biome-specific surface block (can vary by position)
            return biome.get_surface_block(x, y, z);
        }

        // Near surface (1-3 blocks deep)
        if depth < 4 {
            if col.is_beach {
                return Block::Sand;
            }
            if col.cliff_factor > 0.5 || (col.is_mountain && col.height > 75) {
                return self.get_stone_variant(x, y, z);
            }
            // Use biome-specific subsurface block
            return surface_params.subsurface_block;
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

        // Mountain regions using ridged noise
        let mountain_mask = self.get_mountain_mask(x, z);
        let mountain_height = if mountain_mask > 0.0 {
            let ridge = self.mountain_noise.get([xf, zf]);
            let ridge2 = self.ridge_noise.get([xf * 0.015, zf * 0.015]).abs();
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

        // Overhang influence
        let overhang_boost = self.get_overhang_boost(x, z);

        (42.0 + base + detail + mountain_height + hills + overhang_boost) as i32
    }

    /// Get mountain region mask (0.0 = no mountains, 1.0 = full mountains)
    fn get_mountain_mask(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        let region = self.height_noise.get([xf * 0.001, zf * 0.001]);

        if region > 0.25 {
            ((region - 0.25) / 0.5).min(1.0)
        } else {
            0.0
        }
    }

    /// Get cliff steepness factor
    fn get_cliff_factor(&self, x: i32, z: i32) -> f64 {
        let h_north = self.get_raw_height(x, z - 2);
        let h_south = self.get_raw_height(x, z + 2);
        let h_east = self.get_raw_height(x + 2, z);
        let h_west = self.get_raw_height(x - 2, z);

        let slope_ns = (h_north - h_south).abs();
        let slope_ew = (h_east - h_west).abs();
        let max_slope = slope_ns.max(slope_ew);

        (max_slope / 8.0).min(1.0)
    }

    /// Get raw height without clamping
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

    /// Get overhang height boost
    fn get_overhang_boost(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        let cliff = self.get_cliff_factor(x, z);
        if cliff < 0.4 {
            return 0.0;
        }

        let overhang = self.overhang_noise.get([xf * 0.05, zf * 0.05]);

        if overhang > 0.5 {
            (overhang - 0.5) * 10.0 * cliff
        } else {
            0.0
        }
    }

    /// Check if position is a rocky outcrop
    fn is_rocky_outcrop(&self, x: i32, z: i32) -> bool {
        let xf = x as f64;
        let zf = z as f64;

        let outcrop = self.outcrop_noise.get([xf * 0.02, zf * 0.02]);

        if outcrop > 0.6 {
            let detail = self.outcrop_noise.get([xf * 0.15, zf * 0.15]);
            detail > 0.3
        } else {
            false
        }
    }

    /// Get stone block variant
    #[inline]
    fn get_stone_variant(&self, _x: i32, _y: i32, _z: i32) -> Block {
        Block::Stone
    }

    /// Balanced ore generation with depth-based distribution
    #[inline]
    fn get_ore_or_stone_fast(&self, x: i32, y: i32, z: i32) -> Block {
        let hash = fast_hash_3d(x, y, z);

        let ore_noise = self.ore_noise.get([x as f64 * 0.1, y as f64 * 0.1, z as f64 * 0.1]);

        // Coal uses lower frequency noise for chunkier veins
        let coal_noise = self.ore_noise.get([x as f64 * 0.06, y as f64 * 0.06, z as f64 * 0.06]);

        let ore_slot = hash % 1000;

        // Coal first
        if y >= 20 && y < 90 && coal_noise > 0.5 && ore_slot < 350 {
            return Block::CoalOre;
        }

        if ore_noise < 0.55 {
            return Block::Stone;
        }

        // Deep ores (y < 16)
        if y < 16 {
            if ore_slot < 3 && y < 10 {
                return Block::UraniumOre;
            }
            if ore_slot < 10 && y < 14 {
                return Block::TitaniumOre;
            }
            if ore_slot < 25 {
                return Block::DiamondOre;
            }
            if ore_slot < 45 {
                return Block::SapphireOre;
            }
        }

        // Mid-deep ores (y < 32)
        if y < 32 {
            if ore_slot < 15 {
                return Block::RedstoneOre;
            }
            if ore_slot < 25 {
                return Block::EmeraldOre;
            }
            if ore_slot < 45 && y < 28 {
                return Block::AmethystOre;
            }
            if ore_slot < 75 {
                return Block::GoldOre;
            }
            if ore_slot < 105 {
                return Block::LeadOre;
            }
        }

        // Mid ores (y < 50)
        if y < 50 {
            if ore_slot < 40 {
                return Block::SilverOre;
            }
            if ore_slot < 80 {
                return Block::NickelOre;
            }
            if ore_slot < 130 {
                return Block::QuartzOre;
            }
            if ore_slot < 200 {
                return Block::IronOre;
            }
        }

        // Upper ores (y < 70)
        if y < 70 {
            if ore_slot < 80 {
                return Block::CopperOre;
            }
            if ore_slot < 160 {
                return Block::TinOre;
            }
        }

        Block::Stone
    }

    /// Check if ground is solid for tree placement
    fn is_ground_stable_for_tree(&self, x: i32, surface_y: i32, z: i32) -> bool {
        let biome = self.biome_registry.get_biome(x, z);

        // Check tree base and immediate surroundings
        for dy in 0..3 {
            let check_y = surface_y - dy;
            if check_y <= 0 {
                continue;
            }
            if biome.is_cave(x, check_y, z, surface_y, &self.cave_noise) {
                return false;
            }
        }

        // Check cardinal neighbors
        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let nz = z + dz;
            let neighbor_height = self.get_terrain_height(nx, nz);
            let neighbor_biome = self.biome_registry.get_biome(nx, nz);

            if neighbor_biome.is_cave(nx, neighbor_height, nz, neighbor_height, &self.cave_noise) {
                return false;
            }

            if (neighbor_height - surface_y).abs() > 2 {
                return false;
            }
        }

        true
    }

    /// Add stone formations to a chunk (arches, hoodoos, boulders)
    fn add_formations(&self, chunk: &mut Chunk, pos: ChunkPos) {
        let formation_generator = FormationGenerator::new(&self.formation_noise);

        formation_generator.add_formations(
            chunk,
            pos,
            |x, z| self.get_terrain_height(x, z),
            |x, z| self.biome_registry.get_biome_id(x, z),
        );
    }

    /// Add ruins and ancient structures to a chunk
    fn add_ruins(&self, chunk: &mut Chunk, pos: ChunkPos) {
        self.ruin_generator.add_ruins(
            chunk,
            pos,
            |x, z| self.get_terrain_height(x, z),
        );
    }

    /// Add trees to a chunk
    fn add_trees(&self, chunk: &mut Chunk, pos: ChunkPos) {
        let tree_generator = TreeGenerator::new(&self.tree_noise);

        tree_generator.add_trees(
            chunk,
            pos,
            |x, z| self.get_terrain_height(x, z),
            |x, z| self.get_cliff_factor(x, z),
            |x, y, z| self.is_ground_stable_for_tree(x, y, z),
            |x, z| self.biome_registry.get_biome(x, z).surface_params().tree_density,
            |x, z| self.biome_registry.get_biome_id(x, z),
        );
    }

    /// Generate water heightfield data
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
                    let water_surface_y =
                        (WATER_SEA_LEVEL as f32 - 1.0 + WATER_FULL_HEIGHT) * BLOCK_SCALE;

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

                    // Calculate terrain gradient for flow direction
                    let h_xp = self.get_terrain_height(wx + 1, wz) as f32;
                    let h_xm = self.get_terrain_height(wx - 1, wz) as f32;
                    let h_zp = self.get_terrain_height(wx, wz + 1) as f32;
                    let h_zm = self.get_terrain_height(wx, wz - 1) as f32;

                    let gradient_x = (h_xp - h_xm) / 2.0;
                    let gradient_z = (h_zp - h_zm) / 2.0;

                    let flow_velocity_x = -gradient_x * WATER_FLOW_INITIAL_SCALE;
                    let flow_velocity_z = -gradient_z * WATER_FLOW_INITIAL_SCALE;

                    let is_source = depth >= 3;

                    chunk.set_water_surface_with_flow(
                        x,
                        z,
                        water_surface_y,
                        total_foam,
                        flow_velocity_x,
                        flow_velocity_z,
                        is_source,
                    );
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

        if near_edge {
            0.6
        } else {
            0.0
        }
    }
}
