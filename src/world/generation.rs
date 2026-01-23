//! Procedural terrain generation
//!
//! Uses layered noise for realistic terrain with caves, ores, and biomes.

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

    chunk
}

/// Terrain generator using layered noise
struct TerrainGenerator {
    height_noise: Fbm<Perlin>,
    cave_noise: Perlin,
    ore_noise: Perlin,
    detail_noise: Perlin,
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
}
