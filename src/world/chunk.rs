//! Chunk data structures

use glam::Vec3;

use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_F32, CHUNK_SIZE_I32};

/// Position of a chunk in chunk coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub fn from_world_pos(pos: Vec3) -> Self {
        Self {
            x: (pos.x / CHUNK_SIZE_F32).floor() as i32,
            y: (pos.y / CHUNK_SIZE_F32).floor() as i32,
            z: (pos.z / CHUNK_SIZE_F32).floor() as i32,
        }
    }

    pub fn world_origin(&self) -> Vec3 {
        Vec3::new(
            self.x as f32 * CHUNK_SIZE_F32,
            self.y as f32 * CHUNK_SIZE_F32,
            self.z as f32 * CHUNK_SIZE_F32,
        )
    }

    pub fn offset(&self, dx: i32, dy: i32, dz: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            z: self.z + dz,
        }
    }
}

/// Block types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Block {
    Air = 0,
    Dirt = 1,
    Grass = 2,
    Stone = 3,
    Cobblestone = 4,
    Sand = 5,
    Water = 6,
    Wood = 7,
    Leaves = 8,
    CoalOre = 9,
    IronOre = 10,
    GoldOre = 11,
    DiamondOre = 12,
    RedstoneOre = 13,
    Bedrock = 14,
}

impl Block {
    pub fn is_solid(&self) -> bool {
        !matches!(self, Block::Air | Block::Water)
    }

    pub fn is_transparent(&self) -> bool {
        matches!(self, Block::Air | Block::Water | Block::Leaves)
    }

    /// Get texture indices for each face: [top, bottom, front, back, left, right]
    pub fn texture_indices(&self) -> [u32; 6] {
        match self {
            Block::Air => [0; 6],
            Block::Dirt => [0; 6],
            Block::Grass => [1, 0, 2, 2, 2, 2], // Top=grass, Bottom=dirt, Sides=grass_side
            Block::Stone => [3; 6],
            Block::Cobblestone => [4; 6],
            Block::Sand => [5; 6],
            Block::Water => [6; 6],
            Block::Wood => [8, 8, 7, 7, 7, 7], // Top/Bottom=cross-section, Sides=bark
            Block::Leaves => [9; 6],
            Block::CoalOre => [10; 6],
            Block::IronOre => [11; 6],
            Block::GoldOre => [12; 6],
            Block::DiamondOre => [13; 6],
            Block::RedstoneOre => [14; 6],
            Block::Bedrock => [15; 6],
        }
    }
}

impl From<u8> for Block {
    fn from(value: u8) -> Self {
        match value {
            0 => Block::Air,
            1 => Block::Dirt,
            2 => Block::Grass,
            3 => Block::Stone,
            4 => Block::Cobblestone,
            5 => Block::Sand,
            6 => Block::Water,
            7 => Block::Wood,
            8 => Block::Leaves,
            9 => Block::CoalOre,
            10 => Block::IronOre,
            11 => Block::GoldOre,
            12 => Block::DiamondOre,
            13 => Block::RedstoneOre,
            14 => Block::Bedrock,
            _ => Block::Air,
        }
    }
}

/// A 16x16x16 chunk of blocks
pub struct Chunk {
    /// Block data stored as flat array (x + y * SIZE + z * SIZE * SIZE)
    blocks: Box<[Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            blocks: Box::new([Block::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
        }
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }

    #[inline]
    pub fn get_block(&self, x: usize, y: usize, z: usize) -> Block {
        self.blocks[Self::index(x, y, z)]
    }

    #[inline]
    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: Block) {
        self.blocks[Self::index(x, y, z)] = block;
    }

    /// Check if a block position is within chunk bounds
    #[inline]
    pub fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && x < CHUNK_SIZE_I32
            && y >= 0
            && y < CHUNK_SIZE_I32
            && z >= 0
            && z < CHUNK_SIZE_I32
    }

    /// Get block with bounds checking, returns None if out of bounds
    pub fn get_block_checked(&self, x: i32, y: i32, z: i32) -> Option<Block> {
        if Self::in_bounds(x, y, z) {
            Some(self.get_block(x as usize, y as usize, z as usize))
        } else {
            None
        }
    }

    /// Iterator over all blocks with their positions
    pub fn iter_blocks(&self) -> impl Iterator<Item = (usize, usize, usize, Block)> + '_ {
        (0..CHUNK_SIZE).flat_map(move |z| {
            (0..CHUNK_SIZE).flat_map(move |y| {
                (0..CHUNK_SIZE).map(move |x| (x, y, z, self.get_block(x, y, z)))
            })
        })
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}
