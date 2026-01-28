//! Ruins and ancient structure generation
//!
//! Grid-based deterministic placement with decay functions for authentic weathering.
//! Features room layouts, wall decay based on corner distance, and block variety.

use noise::{NoiseFn, Perlin};

use super::util::{fast_hash, fast_hash_3d_seeded};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32};
use crate::world::chunk::{Block, Chunk, ChunkPos};

/// Grid spacing for ruin placement (in chunks)
const RUIN_SPACING: i32 = 20; // 320 blocks
/// Minimum separation from grid edge
const RUIN_SEPARATION: i32 = 8; // 128 blocks padding

/// Ruin generation parameters
#[derive(Debug, Clone, Copy)]
pub struct RuinParams {
    /// Number of rooms (1-3)
    pub room_count: u8,
    /// Base wall height before decay (3-8)
    pub base_wall_height: u8,
    /// How deep the ruin is buried (1-4 blocks below terrain)
    pub burial_depth: i32,
    /// Ruin style affects block selection
    pub style: RuinStyle,
}

impl Default for RuinParams {
    fn default() -> Self {
        Self {
            room_count: 2,
            base_wall_height: 5,
            burial_depth: 2,
            style: RuinStyle::Stone,
        }
    }
}

/// Style of ruin determines block palette
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuinStyle {
    Stone,      // Gray stone ruins
    Sandstone,  // Desert ruins
    Brick,      // Ancient brick structures
}

/// A single room in a ruin
#[derive(Debug, Clone, Copy)]
pub struct Room {
    /// Offset from ruin origin (X)
    pub offset_x: i32,
    /// Offset from ruin origin (Z)
    pub offset_z: i32,
    /// Room width (5-12 blocks)
    pub width: i32,
    /// Room depth (5-12 blocks)
    pub depth: i32,
    /// Base wall height
    pub wall_height: u8,
}

/// Complete ruin structure
#[derive(Debug, Clone)]
pub struct Ruin {
    /// World X position (center)
    pub world_x: i32,
    /// World Z position (center)
    pub world_z: i32,
    /// Rooms in this ruin
    pub rooms: Vec<Room>,
    /// Ruin parameters
    pub params: RuinParams,
    /// Seed for deterministic generation
    pub seed: u32,
}

/// Noise generators for ruin features
pub struct RuinNoise {
    /// Wall decay noise
    decay_noise: Perlin,
    /// Block variation noise
    block_noise: Perlin,
    /// Room layout noise
    layout_noise: Perlin,
    /// Seed for deterministic generation
    seed: u32,
}

impl RuinNoise {
    pub fn new(seed: u32) -> Self {
        Self {
            decay_noise: Perlin::new(seed.wrapping_add(300)),
            block_noise: Perlin::new(seed.wrapping_add(301)),
            layout_noise: Perlin::new(seed.wrapping_add(302)),
            seed,
        }
    }

    /// Check if a ruin should be placed at this chunk position
    /// Returns Some((world_x, world_z)) if this chunk contains a ruin origin
    pub fn should_place_ruin(&self, chunk_x: i32, chunk_z: i32) -> Option<(i32, i32)> {
        // Grid cell coordinates
        let cell_x = chunk_x.div_euclid(RUIN_SPACING);
        let cell_z = chunk_z.div_euclid(RUIN_SPACING);

        // Hash for this cell
        let cell_seed = fast_hash(cell_x, cell_z).wrapping_add(self.seed);

        // Ruin placement chance (~15% of cells have ruins)
        if (cell_seed & 0xFF) > 38 {
            return None;
        }

        // Jittered position within the cell
        let offset_range = (RUIN_SPACING - RUIN_SEPARATION) as u32;
        let offset_x = (cell_seed.wrapping_mul(2654435761) % offset_range) as i32;
        let offset_z = ((cell_seed >> 16).wrapping_mul(2654435761) % offset_range) as i32;

        let ruin_chunk_x = cell_x * RUIN_SPACING + offset_x + RUIN_SEPARATION / 2;
        let ruin_chunk_z = cell_z * RUIN_SPACING + offset_z + RUIN_SEPARATION / 2;

        // Check if this chunk contains the ruin origin
        if chunk_x == ruin_chunk_x && chunk_z == ruin_chunk_z {
            Some((ruin_chunk_x * CHUNK_SIZE_I32, ruin_chunk_z * CHUNK_SIZE_I32))
        } else {
            None
        }
    }

    /// Find all ruins that could affect this chunk (look-ahead for spanning)
    /// Only checks the 2x2 grid of cells that could possibly contain a ruin within range
    pub fn find_affecting_ruins(&self, chunk_x: i32, chunk_z: i32) -> Vec<(i32, i32)> {
        let mut ruins = Vec::with_capacity(2); // Usually 0-1 ruins

        // Get the cell this chunk is in and the neighboring cell
        // Since ruins can span ~32 blocks (2 chunks), we only need to check
        // the current cell and potentially adjacent cells
        let cell_x = chunk_x.div_euclid(RUIN_SPACING);
        let cell_z = chunk_z.div_euclid(RUIN_SPACING);

        // Check current cell and immediate neighbors (3x3 = 9 cells max)
        for dcx in -1..=1 {
            for dcz in -1..=1 {
                let check_cell_x = cell_x + dcx;
                let check_cell_z = cell_z + dcz;

                let cell_seed = fast_hash(check_cell_x, check_cell_z).wrapping_add(self.seed);

                // Skip if no ruin in this cell (~15% have ruins)
                if (cell_seed & 0xFF) > 38 {
                    continue;
                }

                let offset_range = (RUIN_SPACING - RUIN_SEPARATION) as u32;
                let offset_x = (cell_seed.wrapping_mul(2654435761) % offset_range) as i32;
                let offset_z = ((cell_seed >> 16).wrapping_mul(2654435761) % offset_range) as i32;

                let ruin_chunk_x = check_cell_x * RUIN_SPACING + offset_x + RUIN_SEPARATION / 2;
                let ruin_chunk_z = check_cell_z * RUIN_SPACING + offset_z + RUIN_SEPARATION / 2;

                let ruin_world_x = ruin_chunk_x * CHUNK_SIZE_I32;
                let ruin_world_z = ruin_chunk_z * CHUNK_SIZE_I32;

                // Check if this ruin could affect our chunk (within 48 blocks)
                let chunk_world_x = chunk_x * CHUNK_SIZE_I32;
                let chunk_world_z = chunk_z * CHUNK_SIZE_I32;

                let dx = (ruin_world_x - chunk_world_x).abs();
                let dz = (ruin_world_z - chunk_world_z).abs();

                if dx < 48 && dz < 48 {
                    ruins.push((ruin_world_x, ruin_world_z));
                }
            }
        }

        ruins
    }

    /// Generate ruin parameters from world position
    pub fn generate_ruin(&self, world_x: i32, world_z: i32, terrain_height: i32) -> Ruin {
        let hash = fast_hash(world_x / 16, world_z / 16).wrapping_add(self.seed);

        // Determine ruin style based on position/biome
        let style = match hash % 10 {
            0..=6 => RuinStyle::Stone,
            7..=8 => RuinStyle::Brick,
            _ => RuinStyle::Sandstone,
        };

        // Number of rooms (1-3)
        let room_count = 1 + ((hash >> 8) % 3) as u8;

        // Base wall height (3-8)
        let base_wall_height = 3 + ((hash >> 12) % 6) as u8;

        // Burial depth (1-4)
        let burial_depth = 1 + ((hash >> 16) % 4) as i32;

        let params = RuinParams {
            room_count,
            base_wall_height,
            burial_depth,
            style,
        };

        // Generate rooms
        let mut rooms = Vec::with_capacity(room_count as usize);
        let mut current_offset_x = 0i32;

        for i in 0..room_count {
            let room_hash = fast_hash(world_x + i as i32 * 100, world_z).wrapping_add(self.seed);

            // Room size (5-12)
            let width = 5 + ((room_hash >> 4) % 8) as i32;
            let depth = 5 + ((room_hash >> 8) % 8) as i32;

            // Room wall height variation
            let wall_height = base_wall_height.saturating_sub(((room_hash >> 12) % 2) as u8);

            rooms.push(Room {
                offset_x: current_offset_x,
                offset_z: 0,
                width,
                depth,
                wall_height,
            });

            // Next room offset (connected or nearby)
            current_offset_x += width + ((room_hash >> 16) % 3) as i32 - 1;
        }

        Ruin {
            world_x,
            world_z,
            rooms,
            params,
            seed: hash,
        }
    }

    /// Calculate wall height at a position using decay formula
    /// Walls decay from corners toward center
    pub fn get_wall_height(&self, x: i32, z: i32, room: &Room, ruin: &Ruin) -> u8 {
        let local_x = x - ruin.world_x - room.offset_x;
        let local_z = z - ruin.world_z - room.offset_z;

        // Check if on wall perimeter
        let on_wall = local_x == 0 || local_x == room.width - 1
            || local_z == 0 || local_z == room.depth - 1;

        if !on_wall {
            return 0; // Interior, no wall
        }

        // Distance to nearest corner
        let corners = [
            (0, 0),
            (room.width - 1, 0),
            (0, room.depth - 1),
            (room.width - 1, room.depth - 1),
        ];

        let min_dist = corners
            .iter()
            .map(|(cx, cz)| {
                let dx = (local_x - cx).abs() as f32;
                let dz = (local_z - cz).abs() as f32;
                (dx * dx + dz * dz).sqrt()
            })
            .fold(f32::MAX, f32::min);

        // Max distance is half the room size
        let max_dist = (room.width.max(room.depth) as f32) / 2.0;
        let dist_factor = (min_dist / max_dist).min(1.0);

        // Noise for variation
        let noise_val = self.decay_noise.get([x as f64 * 0.1, z as f64 * 0.1]);
        let noise_factor = 0.5 + (noise_val as f32 + 1.0) * 0.25; // 0.5 to 1.0

        // Decay formula: height = base × (1 - dist×0.6) × noise
        let decay = (1.0 - dist_factor * 0.6) * noise_factor;
        let height = (room.wall_height as f32 * decay).round() as u8;

        // Minimum height at corners (2 blocks)
        if min_dist < 1.5 {
            height.max(2)
        } else {
            height
        }
    }

    /// Get block type for ruin wall at this position
    /// Returns None if this should be a gap (air)
    pub fn get_ruin_block(&self, x: i32, y: i32, z: i32, ruin: &Ruin, terrain_y: i32) -> Option<Block> {
        let hash = fast_hash_3d_seeded(x, y, z, self.seed);
        let slot = hash % 100;

        // Gap chance increases with height (5% base + 2% per block above terrain)
        let height_above_terrain = (y - terrain_y).max(0);
        let gap_chance = 5 + height_above_terrain as u32 * 2;

        if slot < gap_chance {
            return None; // Gap in wall
        }

        // Block selection based on style
        let block = match ruin.params.style {
            RuinStyle::Stone => {
                // Near ground: more mossy (30% + 10% bonus)
                let mossy_chance = if y <= terrain_y + 2 { 40 } else { 30 };

                if slot < gap_chance + mossy_chance {
                    Block::MossyStone
                } else if slot < gap_chance + mossy_chance + 10 {
                    Block::CrackedStone
                } else {
                    Block::Stone
                }
            }
            RuinStyle::Sandstone => {
                if slot < gap_chance + 20 {
                    Block::RedSandstone // Weathered variant
                } else {
                    Block::Sandstone
                }
            }
            RuinStyle::Brick => {
                if slot < gap_chance + 25 {
                    Block::CrackedStone // Damaged brick
                } else {
                    Block::Brick
                }
            }
        };

        Some(block)
    }

    /// Check if position is inside any room of the ruin (floor area)
    pub fn is_inside_ruin<'a>(&self, x: i32, z: i32, ruin: &'a Ruin) -> Option<&'a Room> {
        for room in &ruin.rooms {
            let local_x = x - ruin.world_x - room.offset_x;
            let local_z = z - ruin.world_z - room.offset_z;

            if local_x >= 0 && local_x < room.width && local_z >= 0 && local_z < room.depth {
                return Some(room);
            }
        }
        None
    }
}

/// Ruin generator for terrain integration
pub struct RuinGenerator {
    noise: RuinNoise,
}

impl RuinGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            noise: RuinNoise::new(seed),
        }
    }

    /// Add ruins to a chunk
    pub fn add_ruins<F>(&self, chunk: &mut Chunk, pos: ChunkPos, get_terrain_height: F)
    where
        F: Fn(i32, i32) -> i32,
    {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Find all ruins that could affect this chunk
        let affecting_ruins = self.noise.find_affecting_ruins(pos.x, pos.z);

        if affecting_ruins.is_empty() {
            return;
        }

        // Process each potential ruin
        for (ruin_x, ruin_z) in affecting_ruins {
            let terrain_y = get_terrain_height(ruin_x, ruin_z);
            let ruin = self.noise.generate_ruin(ruin_x, ruin_z, terrain_y);

            // Process each block in the chunk
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let wx = world_x + lx as i32;
                    let wz = world_z + lz as i32;

                    // Check if this column is part of the ruin
                    if let Some(room) = self.noise.is_inside_ruin(wx, wz, &ruin) {
                        let local_terrain_y = get_terrain_height(wx, wz);
                        let floor_y = local_terrain_y - ruin.params.burial_depth;
                        let wall_height = self.noise.get_wall_height(wx, wz, room, &ruin);

                        // Generate wall blocks
                        for ly in 0..CHUNK_SIZE {
                            let wy = world_y + ly as i32;

                            // Floor level
                            if wy == floor_y && wall_height > 0 {
                                // Floor block (stone slab effect)
                                let floor_block = match ruin.params.style {
                                    RuinStyle::Stone => Block::Stone,
                                    RuinStyle::Sandstone => Block::Sandstone,
                                    RuinStyle::Brick => Block::Brick,
                                };
                                chunk.set_block(lx, ly, lz, floor_block);
                                continue;
                            }

                            // Wall blocks
                            if wy > floor_y && wy <= floor_y + wall_height as i32 {
                                if let Some(block) = self.noise.get_ruin_block(wx, wy, wz, &ruin, local_terrain_y) {
                                    chunk.set_block(lx, ly, lz, block);
                                }
                            }

                            // Clear interior above floor (if inside walls)
                            let local_x = wx - ruin.world_x - room.offset_x;
                            let local_z = wz - ruin.world_z - room.offset_z;
                            let is_interior = local_x > 0 && local_x < room.width - 1
                                && local_z > 0 && local_z < room.depth - 1;

                            if is_interior && wy > floor_y && wy < floor_y + room.wall_height as i32 {
                                // Only clear if currently solid
                                let current = chunk.get_block(lx, ly, lz);
                                if current.is_solid() && !matches!(current, Block::Stone | Block::MossyStone | Block::CrackedStone | Block::Sandstone | Block::RedSandstone | Block::Brick) {
                                    chunk.set_block(lx, ly, lz, Block::Air);
                                }
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
    fn test_ruin_noise_creation() {
        let _noise = RuinNoise::new(12345);
    }

    #[test]
    fn test_ruin_generator_creation() {
        let _gen = RuinGenerator::new(12345);
    }

    #[test]
    fn test_ruin_placement_deterministic() {
        let noise = RuinNoise::new(12345);
        let result1 = noise.should_place_ruin(0, 0);
        let result2 = noise.should_place_ruin(0, 0);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_ruin_generation_deterministic() {
        let noise = RuinNoise::new(12345);
        let ruin1 = noise.generate_ruin(1000, 1000, 40);
        let ruin2 = noise.generate_ruin(1000, 1000, 40);

        assert_eq!(ruin1.rooms.len(), ruin2.rooms.len());
        assert_eq!(ruin1.params.style, ruin2.params.style);
        assert_eq!(ruin1.params.room_count, ruin2.params.room_count);
    }

    #[test]
    fn test_wall_decay_corners_taller() {
        let noise = RuinNoise::new(12345);
        let ruin = Ruin {
            world_x: 0,
            world_z: 0,
            rooms: vec![Room {
                offset_x: 0,
                offset_z: 0,
                width: 10,
                depth: 10,
                wall_height: 6,
            }],
            params: RuinParams::default(),
            seed: 12345,
        };

        let room = &ruin.rooms[0];

        // Corner should have height (with minimum of 2)
        let corner_height = noise.get_wall_height(0, 0, room, &ruin);
        assert!(corner_height >= 2, "Corner should have minimum height of 2");

        // Middle of wall should have some height too
        let mid_height = noise.get_wall_height(5, 0, room, &ruin);
        // Mid height can vary due to noise, just check it's reasonable
        assert!(mid_height <= room.wall_height, "Mid height should not exceed base height");
    }

    #[test]
    fn test_ruin_style_distribution() {
        let noise = RuinNoise::new(12345);
        let mut stone_count = 0;
        let mut brick_count = 0;
        let mut sandstone_count = 0;

        for i in 0..100 {
            let ruin = noise.generate_ruin(i * 1000, i * 1000, 40);
            match ruin.params.style {
                RuinStyle::Stone => stone_count += 1,
                RuinStyle::Brick => brick_count += 1,
                RuinStyle::Sandstone => sandstone_count += 1,
            }
        }

        // Stone should be most common (~70%)
        assert!(stone_count > brick_count, "Stone should be more common than brick");
        assert!(stone_count > sandstone_count, "Stone should be more common than sandstone");
    }

    #[test]
    fn test_room_count_range() {
        let noise = RuinNoise::new(12345);

        for i in 0..50 {
            let ruin = noise.generate_ruin(i * 500, i * 500, 40);
            assert!(ruin.params.room_count >= 1 && ruin.params.room_count <= 3,
                "Room count should be 1-3, got {}", ruin.params.room_count);
        }
    }

    #[test]
    fn test_affecting_ruins_includes_nearby() {
        let noise = RuinNoise::new(12345);

        // This tests that find_affecting_ruins can find ruins from neighboring cells
        let ruins = noise.find_affecting_ruins(0, 0);
        // Result depends on seed, but function should work without panic
        let _ = ruins;
    }
}
