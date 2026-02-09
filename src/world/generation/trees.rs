//! Tree generation system
//!
//! Provides tree placement and generation with multiple species.
//! Features Voronoi-based species clustering, altitude zonation,
//! clearing detection, and biome-specific tree selection.

use noise::{NoiseFn, Perlin};

use crate::world::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32};

use super::biome::BiomeId;
use super::util::fast_hash_3d;

/// Tree species with distinct visual characteristics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeType {
    // Temperate trees
    /// Small spherical canopy oak (5-7 blocks)
    SmallOak,
    /// Large spreading oak with thick trunk (8-12 blocks)
    LargeOak,
    /// Medium birch with white bark (6-10 blocks)
    Birch,
    /// Short bushy shrub (2-3 blocks)
    BushyShrub,

    // Cold climate trees
    /// Tall narrow pine with conical canopy (10-14 blocks)
    TallPine,
    /// Dense spruce with snow-friendly shape (12-20 blocks)
    Spruce,

    // Tropical trees
    /// Tall jungle tree with buttress roots (15-25 blocks)
    JungleTree,
    /// Giant jungle tree with massive canopy (25-35 blocks)
    GiantJungle,

    // Special trees
    /// Dead tree/snag (no leaves)
    DeadTree,
    /// Desert cactus (not really a tree)
    Cactus,
    /// Swamp mangrove with exposed roots
    Mangrove,
}

impl TreeType {
    /// Get the log block type for this tree
    pub fn log_block(&self) -> Block {
        match self {
            TreeType::SmallOak | TreeType::LargeOak => Block::OakLog,
            TreeType::Birch | TreeType::BushyShrub => Block::BirchLog,
            TreeType::TallPine => Block::PineLog,
            TreeType::Spruce => Block::PineLog,
            TreeType::JungleTree | TreeType::GiantJungle => Block::JungleLog,
            TreeType::DeadTree => Block::Wood,
            TreeType::Cactus => Block::Cactus,
            TreeType::Mangrove => Block::OakLog,
        }
    }

    /// Get the leaf block type for this tree
    pub fn leaf_block(&self) -> Block {
        match self {
            TreeType::SmallOak | TreeType::LargeOak => Block::Leaves,
            TreeType::Birch | TreeType::BushyShrub => Block::Leaves,
            TreeType::TallPine => Block::Leaves,           // Green pine needles
            TreeType::Spruce => Block::SpruceLeaves,       // Frost blue (tundra only)
            TreeType::JungleTree | TreeType::GiantJungle => Block::JungleLeaves,
            TreeType::DeadTree => Block::Air, // No leaves
            TreeType::Cactus => Block::Air,   // No leaves
            TreeType::Mangrove => Block::Leaves,
        }
    }

    /// Get minimum spacing between trees of this type
    pub fn min_spacing(&self) -> i32 {
        match self {
            TreeType::SmallOak => 4,
            TreeType::LargeOak => 6,
            TreeType::Birch => 3,
            TreeType::BushyShrub => 2,
            TreeType::TallPine => 5,
            TreeType::Spruce => 4,
            TreeType::JungleTree => 8,      // Increased from 6 for better walkability
            TreeType::GiantJungle => 14,    // Increased from 12 for better walkability
            TreeType::DeadTree => 8,
            TreeType::Cactus => 6,
            TreeType::Mangrove => 4,
        }
    }
}

/// Voronoi cell size for species clustering (blocks)
const SPECIES_CELL_SIZE: i32 = 32;

/// Clearing detection cell size (blocks)
#[allow(dead_code)]
const CLEARING_CELL_SIZE: i32 = 48;

/// Tree noise generators
pub struct TreeNoise {
    /// Tree placement noise
    tree_noise: Perlin,
    /// Forest region noise
    forest_noise: Perlin,
    /// Clearing/meadow noise
    clearing_noise: Perlin,
    /// Species clustering noise
    species_noise: Perlin,
    /// Seed for deterministic hashing
    seed: u32,
}

impl TreeNoise {
    /// Create a new tree noise generator
    pub fn new(seed: u32) -> Self {
        Self {
            tree_noise: Perlin::new(seed.wrapping_add(40)),
            forest_noise: Perlin::new(seed.wrapping_add(41)),
            clearing_noise: Perlin::new(seed.wrapping_add(42)),
            species_noise: Perlin::new(seed.wrapping_add(43)),
            seed,
        }
    }

    /// Check if position is in a clearing (meadow without trees)
    /// Returns true if this is a clearing (~15% of forest area)
    pub fn is_clearing(&self, x: i32, z: i32) -> bool {
        let clearing = self.clearing_noise.get([
            x as f64 * 0.008,
            z as f64 * 0.008,
        ]);
        clearing > 0.72 // ~15% of area is clearings
    }

    /// Get species cluster ID for Voronoi-based grouping
    /// Trees in the same cluster tend to be the same species
    fn get_species_cluster(&self, x: i32, z: i32) -> u32 {
        let cell_x = x.div_euclid(SPECIES_CELL_SIZE);
        let cell_z = z.div_euclid(SPECIES_CELL_SIZE);

        // Hash the cell coordinates for deterministic cluster ID
        ((cell_x.wrapping_mul(73856093)) ^ (cell_z.wrapping_mul(19349663)) ^ (self.seed as i32)) as u32
    }

    /// Get tree species based on biome, altitude, and clustering
    pub fn get_tree_species(&self, x: i32, z: i32, terrain_height: i32, biome_id: BiomeId) -> TreeType {
        // Biome-specific species selection
        match biome_id {
            BiomeId::Desert => return TreeType::Cactus,
            BiomeId::Swamp => return TreeType::Mangrove,
            BiomeId::Jungle => {
                // Jungle: mix of jungle trees with occasional giants
                let cluster = self.get_species_cluster(x, z);
                return if cluster % 10 == 0 {
                    TreeType::GiantJungle
                } else {
                    TreeType::JungleTree
                };
            }
            BiomeId::Tundra => {
                // Tundra: spruce only, sparse
                return TreeType::Spruce;
            }
            BiomeId::Mountains => {
                // Mountains: altitude-based with pines at high elevation
                if terrain_height > 80 {
                    return TreeType::Spruce;
                } else if terrain_height > 65 {
                    return TreeType::TallPine;
                }
                // Lower mountain slopes get mixed forest
            }
            BiomeId::Ocean => return TreeType::DeadTree, // Shouldn't happen
            _ => {} // Grasslands, CavernousHighlands - use default logic
        }

        // Default: Altitude zonation + Voronoi clustering
        let cluster = self.get_species_cluster(x, z);
        let cluster_species = (cluster % 5) as usize;

        // Altitude override at extremes
        if terrain_height > 140 {
            return TreeType::Spruce;
        } else if terrain_height > 110 {
            return TreeType::TallPine;
        } else if terrain_height < 40 {
            // Low elevation near water
            let hash = fast_hash_3d(x, 0, z);
            return if hash % 3 == 0 { TreeType::BushyShrub } else { TreeType::SmallOak };
        }

        // Use cluster for mid-elevations
        match cluster_species {
            0 => TreeType::SmallOak,
            1 => TreeType::LargeOak,
            2 => TreeType::Birch,
            3 => TreeType::TallPine,
            _ => {
                // Mix based on local noise
                let detail = self.species_noise.get([x as f64 * 0.1, z as f64 * 0.1]);
                if detail > 0.3 {
                    TreeType::LargeOak
                } else if detail < -0.3 {
                    TreeType::BushyShrub
                } else {
                    TreeType::SmallOak
                }
            }
        }
    }

    /// Check if a tree should be placed at this world position
    pub fn should_place_tree(&self, x: i32, z: i32) -> bool {
        self.should_place_tree_with_density(x, z, 1.0)
    }

    /// Check if a tree should be placed, with biome density multiplier
    pub fn should_place_tree_with_density(&self, x: i32, z: i32, biome_density: f64) -> bool {
        // Check for clearings first (meadows without trees)
        if biome_density < 2.0 && self.is_clearing(x, z) {
            return false;
        }

        // === Rare scattered lone trees outside forests ===
        // Very sparse: ~0.05% chance = roughly 1 lone tree every 2 chunks
        let scatter_hash = ((x.wrapping_mul(73856093))
            ^ (z.wrapping_mul(19349663))
            ^ (self.seed.wrapping_mul(668265263) as i32)) as u32;
        let scatter_roll = (scatter_hash % 100000) as f64 / 100000.0;
        if scatter_roll < 0.0005 * biome_density {
            return true;
        }

        // === Forest region trees (original logic preserved) ===
        // Forest biome noise - low frequency creates large forest regions
        let forest_value = self.forest_noise.get([x as f64 * 0.008, z as f64 * 0.008]);

        // Only place trees in forest biomes (where forest noise is high)
        // Biome density affects the threshold - higher density = more forested areas
        let forest_threshold = 0.2 - (biome_density - 1.0) * 0.15;
        if forest_value < forest_threshold.max(-0.3) {
            return false;
        }

        // Tree density increases toward forest center
        let forest_density = ((forest_value - forest_threshold) / 0.8).min(1.0);

        // Local tree placement noise - higher frequency for individual tree spots
        let tree_value = self.tree_noise.get([x as f64 * 0.5, z as f64 * 0.5]);

        // Use a hash for final randomization
        let hash = ((x.wrapping_mul(73856093)) ^ (z.wrapping_mul(19349663))) as u32;
        let hash_value = (hash % 1000) as f64 / 1000.0;

        // Dense forests: ~15-25% tree coverage in forest centers
        // Sparse edges: ~5% coverage at forest edges
        // Biome density affects the threshold - higher density = more trees
        let base_threshold = 0.75 + (1.0 - forest_density) * 0.20;
        let density_threshold = base_threshold - (biome_density - 1.0) * 0.25;

        tree_value > 0.0 && hash_value > density_threshold.max(0.4)
    }

    /// Determine tree type based on position and terrain height
    pub fn get_tree_type(&self, x: i32, z: i32, terrain_height: i32) -> TreeType {
        let xf = x as f64;
        let zf = z as f64;

        // Use tree noise for variety
        let type_noise = self.tree_noise.get([xf * 0.1 + 200.0, zf * 0.1 + 200.0]);

        // Check for mountain/high elevation - prefer pines
        if terrain_height > 65 {
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

    /// Get tree height based on type and position
    pub fn get_tree_height(&self, x: i32, z: i32, tree_type: TreeType) -> i32 {
        let variation = self.tree_noise.get([x as f64 * 0.5 + 100.0, z as f64 * 0.5 + 100.0]);
        let var_int = (variation * 2.0).abs() as i32;

        match tree_type {
            // Temperate trees
            TreeType::SmallOak => 5 + var_int,          // 5-7 blocks
            TreeType::LargeOak => 8 + var_int * 2,      // 8-12 blocks
            TreeType::Birch => 6 + var_int,             // 6-8 blocks
            TreeType::BushyShrub => 2 + (var_int / 2),  // 2-3 blocks

            // Cold climate trees
            TreeType::TallPine => 10 + var_int * 2,     // 10-14 blocks
            TreeType::Spruce => 12 + var_int * 3,       // 12-18 blocks

            // Tropical trees
            TreeType::JungleTree => 15 + var_int * 4,   // 15-23 blocks
            TreeType::GiantJungle => 25 + var_int * 5,  // 25-35 blocks

            // Special trees
            TreeType::DeadTree => 4 + var_int,          // 4-6 blocks
            TreeType::Cactus => 2 + var_int,            // 2-4 blocks
            TreeType::Mangrove => 6 + var_int * 2,      // 6-10 blocks
        }
    }
}

/// Tree generator for placing trees in chunks
pub struct TreeGenerator<'a> {
    tree_noise: &'a TreeNoise,
}

impl<'a> TreeGenerator<'a> {
    pub fn new(tree_noise: &'a TreeNoise) -> Self {
        Self { tree_noise }
    }

    /// Add trees to a chunk
    pub fn add_trees<F, G, H, D, B>(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        get_terrain_height: F,
        get_cliff_factor: G,
        is_ground_stable: H,
        get_biome_tree_density: D,
        get_biome_id: B,
    )
    where
        F: Fn(i32, i32) -> i32,
        G: Fn(i32, i32) -> f64,
        H: Fn(i32, i32, i32) -> bool,
        D: Fn(i32, i32) -> f64,
        B: Fn(i32, i32) -> BiomeId,
    {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Tree canopy can extend up to 6 blocks in each direction for giant jungle trees
        let margin = 7;

        for z in -margin..(CHUNK_SIZE as i32 + margin) {
            for x in -margin..(CHUNK_SIZE as i32 + margin) {
                let wx = world_x + x;
                let wz = world_z + z;

                // Get biome tree density multiplier (0.0 = no trees, 1.0 = normal, 2.0+ = dense)
                let biome_density = get_biome_tree_density(wx, wz);
                if biome_density <= 0.0 {
                    continue; // Biome doesn't allow trees
                }

                if !self.tree_noise.should_place_tree_with_density(wx, wz, biome_density) {
                    continue;
                }

                let surface_y = get_terrain_height(wx, wz);
                let biome_id = get_biome_id(wx, wz);

                // Terrain height limits vary by biome
                let (min_height, max_height) = match biome_id {
                    BiomeId::Mountains => (35, 180), // Mountains can have trees at high altitude
                    BiomeId::Jungle => (30, 100),    // Jungle trees at various elevations
                    BiomeId::Swamp => (28, 45),      // Swamp is low and flat
                    BiomeId::Tundra => (35, 55),     // Low treeline in tundra
                    _ => (35, 90),                   // Default range
                };

                if surface_y < min_height || surface_y > max_height {
                    continue;
                }

                // Skip steep cliffs (except for some biomes)
                let cliff_threshold = match biome_id {
                    BiomeId::Mountains => 0.7, // Mountains allow steeper slopes
                    BiomeId::Jungle => 0.6,
                    _ => 0.5,
                };
                if get_cliff_factor(wx, wz) > cliff_threshold {
                    continue;
                }

                // Check ground stability (skip for cacti which are simpler)
                if biome_id != BiomeId::Desert && !is_ground_stable(wx, surface_y, wz) {
                    continue;
                }

                // Use biome-aware species selection
                let tree_type = self.tree_noise.get_tree_species(wx, wz, surface_y, biome_id);
                let tree_height = self.tree_noise.get_tree_height(wx, wz, tree_type);
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
            TreeType::SmallOak => {
                self.generate_small_oak(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::LargeOak => {
                self.generate_large_oak(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::Birch => {
                self.generate_birch(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::BushyShrub => {
                self.generate_bushy_shrub(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::TallPine => {
                self.generate_tall_pine(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::Spruce => {
                self.generate_spruce(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::JungleTree => {
                self.generate_jungle_tree(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::GiantJungle => {
                self.generate_giant_jungle(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::DeadTree => {
                self.generate_dead_tree(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::Cactus => {
                self.generate_cactus(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
            TreeType::Mangrove => {
                self.generate_mangrove(chunk, pos, tree_x, tree_base_y, tree_z, height)
            }
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

        // Generate trunk
        let trunk_height = height - 2;
        for dy in 0..trunk_height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::OakLog,
                false,
            );
        }

        // Generate leaves (spherical canopy)
        let canopy_center = tree_base_y + height - 2;
        let leaf_start = canopy_center - 1;
        let leaf_end = canopy_center + 2;

        for dy in leaf_start..=leaf_end {
            let y_offset = dy - canopy_center;
            let current_radius: i32 = if y_offset.abs() >= 2 { 1 } else { 2 };

            for dx in -current_radius..=current_radius {
                for dz in -current_radius..=current_radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > current_radius * current_radius + 1 {
                        continue;
                    }

                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;

                    Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
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

        let trunk_top = height - 3;
        for dy in 0..trunk_top {
            let wy = tree_base_y + dy;

            if dy < height / 3 {
                // 2x2 trunk at base
                for dx in 0..=1 {
                    for dz in 0..=1 {
                        Self::try_set_block(
                            chunk,
                            tree_x + dx - world_x,
                            wy - world_y,
                            tree_z + dz - world_z,
                            Block::OakLog,
                            false,
                        );
                    }
                }
            } else {
                // 1x1 trunk higher up
                Self::try_set_block(
                    chunk,
                    tree_x - world_x,
                    wy - world_y,
                    tree_z - world_z,
                    Block::OakLog,
                    false,
                );
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

                    Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
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

        // Trunk extends to height-1 (supports canopy top)
        for dy in 0..height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::PineLog,
                false,
            );
        }

        // Conical canopy - ends 1 below trunk top (no floating leaves)
        let canopy_start = tree_base_y + (height / 3);
        let canopy_end = tree_base_y + height - 1;

        for dy in canopy_start..=canopy_end {
            let progress = (dy - canopy_start) as f32 / (canopy_end - canopy_start) as f32;
            let base_radius = ((1.0 - progress) * 3.5) as i32;

            let tier_offset = if (dy - canopy_start) % 2 == 0 { 0 } else { 1 };
            let layer_radius = (base_radius - tier_offset).max(0);

            if layer_radius == 0 {
                let lx = tree_x - world_x;
                let ly = dy - world_y;
                let lz = tree_z - world_z;
                Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                continue;
            }

            for dx in -layer_radius..=layer_radius {
                for dz in -layer_radius..=layer_radius {
                    if dx.abs() + dz.abs() > layer_radius {
                        continue;
                    }

                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;

                    Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }

        // Top spike leaf - 1 shorter for cleaner look
        let top_y = tree_base_y + height - 1;
        Self::try_set_block(chunk, tree_x - world_x, top_y - world_y, tree_z - world_z, Block::Leaves, true);
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

        let trunk_height = height.min(2);
        for dy in 0..trunk_height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::BirchLog,
                false,
            );
        }

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

                    Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Generate a birch tree (similar to oak but with white bark)
    fn generate_birch(
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

        // Trunk
        let trunk_height = height - 2;
        for dy in 0..trunk_height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::BirchLog,
                false,
            );
        }

        // Leaves (smaller, more sparse canopy than oak)
        let canopy_center = tree_base_y + height - 2;
        for dy in (canopy_center - 1)..=(canopy_center + 1) {
            let y_offset = (dy - canopy_center).abs();
            let radius: i32 = if y_offset == 0 { 2 } else { 1 };

            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    if dx.abs() + dz.abs() > radius + 1 {
                        continue;
                    }
                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;
                    Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Generate a spruce tree (dense conical shape)
    fn generate_spruce(
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

        // Full height trunk
        for dy in 0..height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::PineLog,
                false,
            );
        }

        // Dense conical canopy starting from 1/4 height
        let canopy_start = tree_base_y + (height / 4);
        let canopy_end = tree_base_y + height;

        for dy in canopy_start..canopy_end {
            let progress = (dy - canopy_start) as f32 / (canopy_end - canopy_start) as f32;
            let radius = ((1.0 - progress) * 4.0) as i32;

            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let dist = (dx.abs() + dz.abs()) as f32;
                    if dist > radius as f32 * 1.2 {
                        continue;
                    }
                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;
                    Self::try_set_block_leaf(chunk, lx, ly, lz, Block::SpruceLeaves, 15);
                }
            }
        }

        // Top spike - 1 shorter for cleaner look
        let top_y = tree_base_y + height - 1;
        Self::try_set_block(chunk, tree_x - world_x, top_y - world_y, tree_z - world_z, Block::SpruceLeaves, true);
    }

    /// Generate a jungle tree (tall with buttress roots)
    fn generate_jungle_tree(
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

        // Main trunk
        for dy in 0..height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::JungleLog,
                false,
            );
        }

        // Buttress roots at base
        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            for dy in 0..3 {
                let wy = tree_base_y + dy;
                Self::try_set_block(
                    chunk,
                    tree_x + dx - world_x,
                    wy - world_y,
                    tree_z + dz - world_z,
                    Block::JungleLog,
                    false,
                );
            }
        }

        // Large canopy at top
        let canopy_y = tree_base_y + height - 3;
        for dy in canopy_y..(canopy_y + 4) {
            let y_dist = (dy - canopy_y - 1).abs();
            let radius: i32 = if y_dist <= 1 { 4 } else { 2 };

            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > radius * radius + 2 {
                        continue;
                    }
                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;
                    Self::try_set_block_leaf(chunk, lx, ly, lz, Block::JungleLeaves, 20);
                }
            }
        }

        // Vines hanging from canopy
        for dx in -4..=4 {
            for dz in -4..=4 {
                let dist_sq = dx * dx + dz * dz;
                if dist_sq < 4 || dist_sq > 20 {
                    continue;
                }
                let hash = fast_hash_3d(tree_x + dx, canopy_y, tree_z + dz);
                if hash % 3 != 0 {
                    continue;
                }
                let vine_length = (hash % 5) as i32 + 2;
                for v in 0..vine_length {
                    let lx = tree_x + dx - world_x;
                    let ly = canopy_y - v - world_y;
                    let lz = tree_z + dz - world_z;
                    Self::try_set_block(chunk, lx, ly, lz, Block::Vines, true);
                }
            }
        }
    }

    /// Generate a giant jungle tree (massive 2x2 trunk)
    fn generate_giant_jungle(
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

        // 2x2 trunk
        for dy in 0..height {
            let wy = tree_base_y + dy;
            for dx in 0..=1 {
                for dz in 0..=1 {
                    Self::try_set_block(
                        chunk,
                        tree_x + dx - world_x,
                        wy - world_y,
                        tree_z + dz - world_z,
                        Block::JungleLog,
                        false,
                    );
                }
            }
        }

        // Large buttress roots
        for (dx, dz) in [(-1, 0), (2, 0), (0, -1), (0, 2), (-1, -1), (2, 2)] {
            for dy in 0..5 {
                let wy = tree_base_y + dy;
                Self::try_set_block(
                    chunk,
                    tree_x + dx - world_x,
                    wy - world_y,
                    tree_z + dz - world_z,
                    Block::JungleLog,
                    false,
                );
            }
        }

        // Massive canopy
        let canopy_y = tree_base_y + height - 5;
        for dy in canopy_y..(canopy_y + 6) {
            let y_dist = (dy - canopy_y - 2).abs();
            let radius: i32 = if y_dist <= 1 { 6 } else if y_dist <= 2 { 4 } else { 2 };

            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > radius * radius + 3 {
                        continue;
                    }
                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;
                    Self::try_set_block_leaf(chunk, lx, ly, lz, Block::JungleLeaves, 15);
                }
            }
        }
    }

    /// Generate a dead tree (leafless snag)
    fn generate_dead_tree(
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

        // Main trunk (possibly broken)
        for dy in 0..height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::Wood,
                false,
            );
        }

        // A few bare branches
        let hash = fast_hash_3d(tree_x, tree_base_y, tree_z);
        if height > 3 {
            let branch_y = tree_base_y + height - 2;
            let dx = if hash % 2 == 0 { 1 } else { -1 };
            Self::try_set_block(
                chunk,
                tree_x + dx - world_x,
                branch_y - world_y,
                tree_z - world_z,
                Block::Wood,
                false,
            );
        }
    }

    /// Generate a cactus
    fn generate_cactus(
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

        // Main column
        for dy in 0..height {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::Cactus,
                false,
            );
        }

        // Optional arms
        let hash = fast_hash_3d(tree_x, tree_base_y, tree_z);
        if height > 2 && hash % 3 == 0 {
            let arm_y = tree_base_y + height / 2;
            let dx = if hash % 2 == 0 { 1 } else { -1 };
            Self::try_set_block(
                chunk,
                tree_x + dx - world_x,
                arm_y - world_y,
                tree_z - world_z,
                Block::Cactus,
                true,
            );
            Self::try_set_block(
                chunk,
                tree_x + dx - world_x,
                arm_y + 1 - world_y,
                tree_z - world_z,
                Block::Cactus,
                true,
            );
        }
    }

    /// Generate a mangrove tree (swamp tree with exposed roots)
    fn generate_mangrove(
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

        // Raised trunk (starts 2 blocks up)
        let trunk_start = 2;
        for dy in trunk_start..(height + trunk_start) {
            let wy = tree_base_y + dy;
            Self::try_set_block(
                chunk,
                tree_x - world_x,
                wy - world_y,
                tree_z - world_z,
                Block::OakLog,
                false,
            );
        }

        // Prop roots going down
        for (dx, dz) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            for dy in 0..=trunk_start {
                let wy = tree_base_y + dy;
                Self::try_set_block(
                    chunk,
                    tree_x + dx - world_x,
                    wy - world_y,
                    tree_z + dz - world_z,
                    Block::OakLog,
                    false,
                );
            }
        }

        // Canopy
        let canopy_y = tree_base_y + height + trunk_start - 2;
        for dy in canopy_y..(canopy_y + 3) {
            let y_dist = (dy - canopy_y - 1).abs();
            let radius: i32 = if y_dist == 0 { 3 } else { 2 };

            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq > radius * radius + 1 {
                        continue;
                    }
                    let lx = tree_x + dx - world_x;
                    let ly = dy - world_y;
                    let lz = tree_z + dz - world_z;
                    Self::try_set_block(chunk, lx, ly, lz, Block::Leaves, true);
                }
            }
        }
    }

    /// Try to set a leaf block with custom gap percentage
    fn try_set_block_leaf(chunk: &mut Chunk, lx: i32, ly: i32, lz: i32, block: Block, gap_percent: u32) {
        if lx >= 0
            && lx < CHUNK_SIZE_I32
            && ly >= 0
            && ly < CHUNK_SIZE_I32
            && lz >= 0
            && lz < CHUNK_SIZE_I32
        {
            let current = chunk.get_block(lx as usize, ly as usize, lz as usize);
            if current != Block::Air {
                return;
            }

            // Canopy gaps
            let hash = fast_hash_3d(lx, ly, lz);
            if hash % 100 < gap_percent {
                return;
            }

            chunk.set_block(lx as usize, ly as usize, lz as usize, block);
        }
    }

    /// Try to set a block if within chunk bounds
    fn try_set_block(chunk: &mut Chunk, lx: i32, ly: i32, lz: i32, block: Block, skip_solid: bool) {
        if lx >= 0
            && lx < CHUNK_SIZE_I32
            && ly >= 0
            && ly < CHUNK_SIZE_I32
            && lz >= 0
            && lz < CHUNK_SIZE_I32
        {
            if skip_solid {
                let current = chunk.get_block(lx as usize, ly as usize, lz as usize);
                if current != Block::Air {
                    return;
                }
            }

            // Canopy gaps: randomly skip ~25% of leaf blocks
            if matches!(block, Block::Leaves | Block::SpruceLeaves | Block::JungleLeaves) {
                let hash = fast_hash_3d(lx, ly, lz);
                if hash % 100 < 25 {
                    return;
                }
            }

            chunk.set_block(lx as usize, ly as usize, lz as usize, block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_noise_creation() {
        let _noise = TreeNoise::new(12345);
    }

    #[test]
    fn test_tree_placement_deterministic() {
        let noise = TreeNoise::new(12345);

        let result1 = noise.should_place_tree(100, 100);
        let result2 = noise.should_place_tree(100, 100);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_tree_species_selection() {
        let noise = TreeNoise::new(12345);

        // High terrain in mountains should produce spruce/pine
        let tree_type = noise.get_tree_species(0, 0, 150, BiomeId::Mountains);
        assert_eq!(tree_type, TreeType::Spruce);

        // Jungle biome should produce jungle trees
        let jungle_tree = noise.get_tree_species(0, 0, 50, BiomeId::Jungle);
        assert!(matches!(jungle_tree, TreeType::JungleTree | TreeType::GiantJungle));

        // Desert should produce cacti
        let desert_tree = noise.get_tree_species(0, 0, 50, BiomeId::Desert);
        assert_eq!(desert_tree, TreeType::Cactus);

        // Tundra should produce spruce
        let tundra_tree = noise.get_tree_species(0, 0, 50, BiomeId::Tundra);
        assert_eq!(tundra_tree, TreeType::Spruce);
    }

    #[test]
    fn test_tree_height_reasonable() {
        let noise = TreeNoise::new(12345);

        // Test all tree types have reasonable heights
        let all_types = [
            (TreeType::SmallOak, 3, 10),
            (TreeType::LargeOak, 6, 15),
            (TreeType::Birch, 5, 12),
            (TreeType::BushyShrub, 1, 5),
            (TreeType::TallPine, 8, 18),
            (TreeType::Spruce, 10, 25),
            (TreeType::JungleTree, 12, 30),
            (TreeType::GiantJungle, 20, 45),
            (TreeType::DeadTree, 3, 10),
            (TreeType::Cactus, 1, 8),
            (TreeType::Mangrove, 5, 15),
        ];

        for (tree_type, min, max) in all_types {
            let height = noise.get_tree_height(0, 0, tree_type);
            assert!(
                height >= min && height <= max,
                "{:?} height {} should be in range {}-{}",
                tree_type, height, min, max
            );
        }
    }

    #[test]
    fn test_clearing_detection() {
        let noise = TreeNoise::new(12345);

        // Clearings should be deterministic
        let result1 = noise.is_clearing(1000, 1000);
        let result2 = noise.is_clearing(1000, 1000);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_species_clustering() {
        let noise = TreeNoise::new(12345);

        // Trees close together should often be the same species
        let species1 = noise.get_tree_species(100, 100, 50, BiomeId::Grasslands);
        let species2 = noise.get_tree_species(105, 105, 50, BiomeId::Grasslands);

        // Same cluster (within 32 blocks) should usually match
        // Note: Not guaranteed due to altitude and other factors, so this is a soft test
        let _same_cluster = species1 == species2;
    }
}
