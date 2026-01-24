//! Chunk data structures

use glam::Vec3;

use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_F32, CHUNK_SIZE_I32, BLOCK_SCALE};

/// Position of a chunk in chunk coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub fn from_world_pos(pos: Vec3) -> Self {
        let chunk_world_size = CHUNK_SIZE_F32 * BLOCK_SCALE;
        Self {
            x: (pos.x / chunk_world_size).floor() as i32,
            y: (pos.y / chunk_world_size).floor() as i32,
            z: (pos.z / chunk_world_size).floor() as i32,
        }
    }

    pub fn world_origin(&self) -> Vec3 {
        let chunk_world_size = CHUNK_SIZE_F32 * BLOCK_SCALE;
        Vec3::new(
            self.x as f32 * chunk_world_size,
            self.y as f32 * chunk_world_size,
            self.z as f32 * chunk_world_size,
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
    Torch = 15,
}

impl Block {
    pub fn is_solid(&self) -> bool {
        !matches!(self, Block::Air | Block::Water | Block::Torch)
    }

    pub fn is_transparent(&self) -> bool {
        matches!(self, Block::Air | Block::Water | Block::Leaves | Block::Torch)
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
            Block::Torch => [12; 6], // Torch uses gold ore texture (yellow) temporarily
        }
    }

    /// Get light emission level (0-15) for this block type
    /// Torches emit level 14, providing ~14 block illumination radius
    #[inline]
    pub fn light_emission(&self) -> u8 {
        match self {
            Block::Torch => 14,
            _ => 0,
        }
    }

    /// Check if this block allows light to pass through
    #[inline]
    pub fn is_light_transparent(&self) -> bool {
        matches!(self, Block::Air | Block::Water | Block::Leaves | Block::Torch)
    }

    /// Get texture index for a specific face direction
    /// face_idx: 0=top(+Y), 1=bottom(-Y), 2=front(+Z), 3=back(-Z), 4=left(-X), 5=right(+X)
    #[inline]
    pub fn face_texture(&self, face_idx: usize) -> u32 {
        self.texture_indices()[face_idx]
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
            15 => Block::Torch,
            _ => Block::Air,
        }
    }
}

/// A 16x16x16 chunk of blocks
pub struct Chunk {
    /// Block data stored as flat array (x + y * SIZE + z * SIZE * SIZE)
    blocks: Box<[Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]>,
    /// Light data: upper 4 bits = sky light, lower 4 bits = block light
    /// Each value ranges 0-15, packed into one byte per block
    light: Box<[u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            blocks: Box::new([Block::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
            light: Box::new([0u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
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

    // ============ LIGHT ACCESS METHODS ============

    /// Get raw light value (sky << 4 | block)
    #[inline]
    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.light[Self::index(x, y, z)]
    }

    /// Set raw light value (sky << 4 | block)
    #[inline]
    pub fn set_light(&mut self, x: usize, y: usize, z: usize, value: u8) {
        self.light[Self::index(x, y, z)] = value;
    }

    /// Get sky light level (0-15) from upper 4 bits
    #[inline]
    pub fn get_sky_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.light[Self::index(x, y, z)] >> 4
    }

    /// Set sky light level (0-15) in upper 4 bits
    #[inline]
    pub fn set_sky_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        let idx = Self::index(x, y, z);
        self.light[idx] = ((level & 0x0F) << 4) | (self.light[idx] & 0x0F);
    }

    /// Get block light level (0-15) from lower 4 bits
    #[inline]
    pub fn get_block_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.light[Self::index(x, y, z)] & 0x0F
    }

    /// Set block light level (0-15) in lower 4 bits
    #[inline]
    pub fn set_block_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        let idx = Self::index(x, y, z);
        self.light[idx] = (self.light[idx] & 0xF0) | (level & 0x0F);
    }

    /// Get combined light level (max of sky and block light)
    /// This is the effective brightness for rendering
    #[inline]
    pub fn get_combined_light(&self, x: usize, y: usize, z: usize) -> u8 {
        let light = self.light[Self::index(x, y, z)];
        (light >> 4).max(light & 0x0F)
    }

    /// Get light with bounds checking
    #[inline]
    pub fn get_light_checked(&self, x: i32, y: i32, z: i32) -> Option<u8> {
        if Self::in_bounds(x, y, z) {
            Some(self.get_light(x as usize, y as usize, z as usize))
        } else {
            None
        }
    }

    /// Get block light with bounds checking
    #[inline]
    pub fn get_block_light_checked(&self, x: i32, y: i32, z: i32) -> Option<u8> {
        if Self::in_bounds(x, y, z) {
            Some(self.get_block_light(x as usize, y as usize, z as usize))
        } else {
            None
        }
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}
