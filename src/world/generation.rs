//! Procedural terrain generation
//!
//! Uses layered noise for realistic terrain with caves, ores, biomes, and trees.

use noise::{NoiseFn, Perlin, Fbm, MultiFractal};

use super::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32};

/// Generate a chunk at the given position
pub fn generate_chunk(pos: ChunkPos) -> Chunk {
    let mut chunk = Chunk::new();
    let generator = TerrainGenerator::new(12345); // Fixed seed for reproducibility

    let world_x = pos.x * CHUNK_SIZE_I32;
    let world_y = pos.y * CHUNK_SIZE_I32;
    let world_z = pos.z * CHUNK_SIZE_I32;

    // First pass: generate base terrain
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let wx = world_x + x as i32;
                let wy = world_y + y as i32;
                let wz = world_z + z as i32;

                let block = generator.get_block(wx, wy, wz);
                chunk.set_block(x, y, z, block);
            }
        }
    }

    // Second pass: add trees (need to check neighboring positions)
    generator.add_trees(&mut chunk, pos);

    chunk
}

/// Terrain generator using layered noise
struct TerrainGenerator {
    height_noise: Fbm<Perlin>,
    cave_noise: Perlin,
    ore_noise: Perlin,
    detail_noise: Perlin,
    tree_noise: Perlin,
    forest_noise: Perlin,  // For forest biome clustering
}

impl TerrainGenerator {
    fn new(seed: u32) -> Self {
        Self {
            height_noise: Fbm::new(seed)
                .set_octaves(6)
                .set_frequency(0.005)
                .set_lacunarity(2.0)
                .set_persistence(0.5),
            cave_noise: Perlin::new(seed.wrapping_add(1)),
            ore_noise: Perlin::new(seed.wrapping_add(2)),
            detail_noise: Perlin::new(seed.wrapping_add(3)),
            tree_noise: Perlin::new(seed.wrapping_add(4)),
            forest_noise: Perlin::new(seed.wrapping_add(5)),
        }
    }

    fn get_block(&self, x: i32, y: i32, z: i32) -> Block {
        // Base terrain height using fractal noise
        let base_height = self.get_terrain_height(x, z);

        // Bedrock at bottom
        if y <= 0 {
            return Block::Bedrock;
        }

        // Above terrain surface
        if y > base_height {
            if y <= 32 && y > 30 {
                // Water level
                return Block::Water;
            }
            return Block::Air;
        }

        // Check for caves
        if self.is_cave(x, y, z) && y > 5 {
            return Block::Air;
        }

        // Determine block type based on depth
        let depth = base_height - y;

        if depth == 0 {
            // Surface block
            if base_height < 33 {
                Block::Sand // Beach
            } else {
                Block::Grass
            }
        } else if depth < 4 {
            // Near surface
            if base_height < 33 {
                Block::Sand
            } else {
                Block::Dirt
            }
        } else if depth < 60 {
            // Underground - check for ores
            self.get_ore_or_stone(x, y, z)
        } else {
            // Deep underground
            if self.ore_noise.get([x as f64 * 0.1, y as f64 * 0.1, z as f64 * 0.1]) > 0.7 {
                Block::DiamondOre
            } else {
                Block::Stone
            }
        }
    }

    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        // Multi-layer height calculation
        let base = self.height_noise.get([x as f64, z as f64]) * 30.0;

        // Add detail variation
        let detail = self.detail_noise.get([x as f64 * 0.1, z as f64 * 0.1]) * 5.0;

        // Mountains in some areas
        let mountain = self.height_noise.get([x as f64 * 0.002, z as f64 * 0.002]);
        let mountain_height = if mountain > 0.3 {
            (mountain - 0.3) * 80.0
        } else {
            0.0
        };

        (45.0 + base + detail + mountain_height) as i32
    }

    fn is_cave(&self, x: i32, y: i32, z: i32) -> bool {
        // 3D noise for cave systems
        let scale = 0.05;
        let value = self.cave_noise.get([
            x as f64 * scale,
            y as f64 * scale * 0.5, // Stretched vertically
            z as f64 * scale,
        ]);

        // Threshold for cave formation
        value > 0.6
    }

    fn get_ore_or_stone(&self, x: i32, y: i32, z: i32) -> Block {
        let ore_value = self.ore_noise.get([
            x as f64 * 0.15,
            y as f64 * 0.15,
            z as f64 * 0.15,
        ]);

        // Ore distribution based on depth and noise
        if ore_value > 0.85 {
            // Rare ores
            if y < 16 {
                Block::GoldOre
            } else {
                Block::IronOre
            }
        } else if ore_value > 0.75 {
            // Common ores
            if y < 50 {
                Block::IronOre
            } else {
                Block::CoalOre
            }
        } else if ore_value > 0.7 {
            Block::CoalOre
        } else if ore_value < -0.8 && y < 20 {
            Block::RedstoneOre
        } else {
            Block::Stone
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

    /// Get tree height variation based on position
    fn get_tree_height(&self, x: i32, z: i32) -> i32 {
        let variation = self.tree_noise.get([x as f64 * 0.5 + 100.0, z as f64 * 0.5 + 100.0]);
        5 + (variation * 3.0).abs() as i32  // Height 5-8 blocks
    }

    /// Add trees to a chunk
    /// Checks an expanded area to capture trees from neighboring chunks that extend into this one
    fn add_trees(&self, chunk: &mut Chunk, pos: ChunkPos) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Tree canopy can extend up to 3 blocks in each direction
        // Check expanded area to capture trees rooted in neighboring chunks
        let margin = 4;

        for z in -margin..(CHUNK_SIZE as i32 + margin) {
            for x in -margin..(CHUNK_SIZE as i32 + margin) {
                let wx = world_x + x;
                let wz = world_z + z;

                // Only place trees where noise says so
                if !self.should_place_tree(wx, wz) {
                    continue;
                }

                // Find the surface height at this position
                let surface_y = self.get_terrain_height(wx, wz);

                // Only place trees on grass (not sand, water, etc.)
                if surface_y < 35 {
                    continue; // Skip beaches and underwater
                }

                // Get tree parameters
                let tree_height = self.get_tree_height(wx, wz);
                let trunk_base = surface_y + 1;

                // Generate tree parts that fall within this chunk
                // (the function checks bounds internally)
                self.generate_tree_in_chunk(chunk, pos, wx, trunk_base, wz, tree_height);
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
    ) {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Generate trunk
        for dy in 0..height {
            let wy = tree_base_y + dy;
            let local_x = tree_x - world_x;
            let local_y = wy - world_y;
            let local_z = tree_z - world_z;

            if local_x >= 0 && local_x < CHUNK_SIZE_I32
                && local_y >= 0 && local_y < CHUNK_SIZE_I32
                && local_z >= 0 && local_z < CHUNK_SIZE_I32
            {
                chunk.set_block(local_x as usize, local_y as usize, local_z as usize, Block::Wood);
            }
        }

        // Generate leaves (full spherical canopy)
        let leaf_start = tree_base_y + height - 4;  // Start leaves lower
        let leaf_end = tree_base_y + height + 1;
        let leaf_radius: i32 = 3;  // Larger radius for fuller canopy

        for dy in leaf_start..=leaf_end {
            // Radius varies by height in canopy - fuller in middle
            let y_offset = dy - (tree_base_y + height - 2);
            let current_radius: i32 = if y_offset >= 2 {
                1  // Top is narrow
            } else if y_offset <= -2 {
                2  // Bottom is medium
            } else {
                leaf_radius  // Middle is full
            };

            for dx in -current_radius..=current_radius {
                for dz in -current_radius..=current_radius {
                    // Skip corners for rounder shape (only on outer layers)
                    let dist_sq = dx * dx + dz * dz;
                    let max_dist_sq = current_radius * current_radius + 1;
                    if dist_sq > max_dist_sq {
                        continue;
                    }

                    let wx = tree_x + dx;
                    let wy = dy;
                    let wz = tree_z + dz;

                    let local_x = wx - world_x;
                    let local_y = wy - world_y;
                    let local_z = wz - world_z;

                    if local_x >= 0 && local_x < CHUNK_SIZE_I32
                        && local_y >= 0 && local_y < CHUNK_SIZE_I32
                        && local_z >= 0 && local_z < CHUNK_SIZE_I32
                    {
                        // Don't overwrite trunk
                        let current = chunk.get_block(
                            local_x as usize,
                            local_y as usize,
                            local_z as usize,
                        );
                        if current == Block::Air {
                            chunk.set_block(
                                local_x as usize,
                                local_y as usize,
                                local_z as usize,
                                Block::Leaves,
                            );
                        }
                    }
                }
            }
        }
    }
}
