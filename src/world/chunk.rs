//! Chunk data structures

use glam::Vec3;

use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_F32, CHUNK_SIZE_I32, BLOCK_SCALE};

/// Water data for a single column in a chunk
/// Stores the water surface height for heightfield rendering
#[derive(Debug, Clone, Copy)]
pub struct WaterColumn {
    /// Water surface height in world Y coordinates
    /// This is the actual Y position of the water surface
    pub surface_y: f32,
    /// Whether this column has water at all
    pub has_water: bool,
    /// Foam factor (0.0-1.0) - higher near block edges
    pub foam: f32,
    /// Flow direction for animation (radians, 0 = +X direction)
    pub flow_angle: f32,
}

impl Default for WaterColumn {
    fn default() -> Self {
        Self {
            surface_y: 0.0,
            has_water: false,
            foam: 0.0,
            flow_angle: 0.0,
        }
    }
}

/// Water heightfield data for a chunk
/// Stored as a 2D grid (X-Z plane) since water surface is primarily horizontal
pub struct ChunkWaterData {
    /// 16x16 grid of water columns (X, Z indexed)
    columns: Box<[[WaterColumn; CHUNK_SIZE]; CHUNK_SIZE]>,
    /// Cached: does this chunk have any water?
    has_any_water: bool,
    /// Cached: minimum water Y in this chunk (for culling)
    min_water_y: f32,
    /// Cached: maximum water Y in this chunk (for culling)
    max_water_y: f32,
}

impl ChunkWaterData {
    pub fn new() -> Self {
        Self {
            columns: Box::new([[WaterColumn::default(); CHUNK_SIZE]; CHUNK_SIZE]),
            has_any_water: false,
            min_water_y: f32::MAX,
            max_water_y: f32::MIN,
        }
    }

    /// Get water column at local chunk coordinates
    #[inline]
    pub fn get_column(&self, x: usize, z: usize) -> &WaterColumn {
        &self.columns[z][x]
    }

    /// Get mutable water column at local chunk coordinates
    #[inline]
    pub fn get_column_mut(&mut self, x: usize, z: usize) -> &mut WaterColumn {
        &mut self.columns[z][x]
    }

    /// Set water surface height at a column
    pub fn set_water(&mut self, x: usize, z: usize, surface_y: f32, foam: f32) {
        let col = &mut self.columns[z][x];
        col.surface_y = surface_y;
        col.has_water = true;
        col.foam = foam;
        self.has_any_water = true;
        self.min_water_y = self.min_water_y.min(surface_y);
        self.max_water_y = self.max_water_y.max(surface_y);
    }

    /// Clear water at a column
    pub fn clear_water(&mut self, x: usize, z: usize) {
        self.columns[z][x].has_water = false;
        // Note: has_any_water cache not updated here for performance
        // Call recalculate_bounds() if needed
    }

    /// Check if this chunk has any water
    #[inline]
    pub fn has_water(&self) -> bool {
        self.has_any_water
    }

    /// Get the water surface height at a position, with bilinear interpolation
    /// Returns None if no water at this position
    pub fn sample_height(&self, local_x: f32, local_z: f32) -> Option<f32> {
        // Clamp to valid range
        let x = local_x.clamp(0.0, (CHUNK_SIZE - 1) as f32);
        let z = local_z.clamp(0.0, (CHUNK_SIZE - 1) as f32);

        let x0 = x.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(CHUNK_SIZE - 1);
        let z1 = (z0 + 1).min(CHUNK_SIZE - 1);

        let fx = x.fract();
        let fz = z.fract();

        // Sample four corners
        let c00 = &self.columns[z0][x0];
        let c10 = &self.columns[z0][x1];
        let c01 = &self.columns[z1][x0];
        let c11 = &self.columns[z1][x1];

        // Check if any corner has water
        if !c00.has_water && !c10.has_water && !c01.has_water && !c11.has_water {
            return None;
        }

        // Bilinear interpolation of heights
        // For corners without water, use the average of those with water
        let h00 = if c00.has_water { c00.surface_y } else { 0.0 };
        let h10 = if c10.has_water { c10.surface_y } else { 0.0 };
        let h01 = if c01.has_water { c01.surface_y } else { 0.0 };
        let h11 = if c11.has_water { c11.surface_y } else { 0.0 };

        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;
        let height = h0 * (1.0 - fz) + h1 * fz;

        Some(height)
    }

    /// Sample foam factor with bilinear interpolation
    pub fn sample_foam(&self, local_x: f32, local_z: f32) -> f32 {
        let x = local_x.clamp(0.0, (CHUNK_SIZE - 1) as f32);
        let z = local_z.clamp(0.0, (CHUNK_SIZE - 1) as f32);

        let x0 = x.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(CHUNK_SIZE - 1);
        let z1 = (z0 + 1).min(CHUNK_SIZE - 1);

        let fx = x.fract();
        let fz = z.fract();

        let f00 = self.columns[z0][x0].foam;
        let f10 = self.columns[z0][x1].foam;
        let f01 = self.columns[z1][x0].foam;
        let f11 = self.columns[z1][x1].foam;

        let f0 = f00 * (1.0 - fx) + f10 * fx;
        let f1 = f01 * (1.0 - fx) + f11 * fx;
        f0 * (1.0 - fz) + f1 * fz
    }

    /// Get Y bounds of water in this chunk
    #[inline]
    pub fn get_y_bounds(&self) -> (f32, f32) {
        (self.min_water_y, self.max_water_y)
    }

    /// Recalculate cached bounds (call after bulk modifications)
    pub fn recalculate_bounds(&mut self) {
        self.has_any_water = false;
        self.min_water_y = f32::MAX;
        self.max_water_y = f32::MIN;

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let col = &self.columns[z][x];
                if col.has_water {
                    self.has_any_water = true;
                    self.min_water_y = self.min_water_y.min(col.surface_y);
                    self.max_water_y = self.max_water_y.max(col.surface_y);
                }
            }
        }
    }

    /// Iterator over all columns with water
    pub fn iter_water_columns(&self) -> impl Iterator<Item = (usize, usize, &WaterColumn)> {
        (0..CHUNK_SIZE).flat_map(move |z| {
            (0..CHUNK_SIZE).filter_map(move |x| {
                let col = &self.columns[z][x];
                if col.has_water {
                    Some((x, z, col))
                } else {
                    None
                }
            })
        })
    }
}

impl Default for ChunkWaterData {
    fn default() -> Self {
        Self::new()
    }
}

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
            Block::Torch => [16; 6], // Torch has dedicated warm orange texture
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
    /// Optional water heightfield data - only allocated if chunk contains water
    water: Option<Box<ChunkWaterData>>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            blocks: Box::new([Block::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
            light: Box::new([0u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
            water: None,
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

    // ============ WATER ACCESS METHODS ============

    /// Check if this chunk has water data allocated
    #[inline]
    pub fn has_water_data(&self) -> bool {
        self.water.is_some()
    }

    /// Check if this chunk has any water
    #[inline]
    pub fn has_water(&self) -> bool {
        self.water.as_ref().map_or(false, |w| w.has_water())
    }

    /// Get reference to water data if present
    #[inline]
    pub fn water_data(&self) -> Option<&ChunkWaterData> {
        self.water.as_deref()
    }

    /// Get mutable reference to water data if present
    #[inline]
    pub fn water_data_mut(&mut self) -> Option<&mut ChunkWaterData> {
        self.water.as_deref_mut()
    }

    /// Ensure water data is allocated and return mutable reference
    pub fn ensure_water_data(&mut self) -> &mut ChunkWaterData {
        if self.water.is_none() {
            self.water = Some(Box::new(ChunkWaterData::new()));
        }
        self.water.as_deref_mut().unwrap()
    }

    /// Set water surface at a column (allocates water data if needed)
    pub fn set_water_surface(&mut self, x: usize, z: usize, surface_y: f32, foam: f32) {
        self.ensure_water_data().set_water(x, z, surface_y, foam);
    }

    /// Clear all water data from this chunk
    pub fn clear_water(&mut self) {
        self.water = None;
    }

    /// Get water height at a column (returns None if no water)
    #[inline]
    pub fn get_water_height(&self, x: usize, z: usize) -> Option<f32> {
        self.water.as_ref().and_then(|w| {
            let col = w.get_column(x, z);
            if col.has_water {
                Some(col.surface_y)
            } else {
                None
            }
        })
    }

    /// Sample water height with interpolation
    pub fn sample_water_height(&self, local_x: f32, local_z: f32) -> Option<f32> {
        self.water.as_ref().and_then(|w| w.sample_height(local_x, local_z))
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    /// Create a chunk filled with a single block type (for testing/benchmarking)
    pub fn new_solid(block: Block) -> Self {
        let mut chunk = Self::new();
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    chunk.set_block(x, y, z, block);
                }
            }
        }
        chunk
    }

    /// Create a chunk with random sparse blocks (for testing/benchmarking)
    pub fn new_sparse(seed: u64, occupancy: f32) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut chunk = Self::new();
        let threshold = (occupancy * u64::MAX as f32) as u64;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let mut hasher = DefaultHasher::new();
                    (seed, x, y, z).hash(&mut hasher);
                    let hash = hasher.finish();

                    if hash < threshold {
                        chunk.set_block(x, y, z, Block::Stone);
                    }
                }
            }
        }
        chunk
    }

    /// Create a checkerboard pattern chunk (worst case for greedy meshing)
    pub fn new_checkerboard() -> Self {
        let mut chunk = Self::new();
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    if (x + y + z) % 2 == 0 {
                        chunk.set_block(x, y, z, Block::Stone);
                    }
                }
            }
        }
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ Block Tests ============

    #[test]
    fn test_block_is_solid() {
        // Non-solid blocks
        assert!(!Block::Air.is_solid(), "Air should not be solid");
        assert!(!Block::Water.is_solid(), "Water should not be solid");
        assert!(!Block::Torch.is_solid(), "Torch should not be solid");

        // Solid blocks
        assert!(Block::Stone.is_solid(), "Stone should be solid");
        assert!(Block::Dirt.is_solid(), "Dirt should be solid");
        assert!(Block::Grass.is_solid(), "Grass should be solid");
        assert!(Block::Wood.is_solid(), "Wood should be solid");
        assert!(Block::Leaves.is_solid(), "Leaves should be solid");
        assert!(Block::Bedrock.is_solid(), "Bedrock should be solid");
    }

    #[test]
    fn test_block_is_transparent() {
        // Transparent blocks
        assert!(Block::Air.is_transparent(), "Air should be transparent");
        assert!(Block::Water.is_transparent(), "Water should be transparent");
        assert!(Block::Leaves.is_transparent(), "Leaves should be transparent");
        assert!(Block::Torch.is_transparent(), "Torch should be transparent");

        // Opaque blocks
        assert!(!Block::Stone.is_transparent(), "Stone should not be transparent");
        assert!(!Block::Dirt.is_transparent(), "Dirt should not be transparent");
        assert!(!Block::Grass.is_transparent(), "Grass should not be transparent");
    }

    #[test]
    fn test_block_light_emission() {
        // Torch emits light level 14
        assert_eq!(Block::Torch.light_emission(), 14, "Torch should emit level 14");

        // Other blocks don't emit light
        assert_eq!(Block::Air.light_emission(), 0);
        assert_eq!(Block::Stone.light_emission(), 0);
        assert_eq!(Block::Water.light_emission(), 0);
        assert_eq!(Block::Leaves.light_emission(), 0);
    }

    #[test]
    fn test_block_is_light_transparent() {
        // Light-transparent blocks
        assert!(Block::Air.is_light_transparent());
        assert!(Block::Water.is_light_transparent());
        assert!(Block::Leaves.is_light_transparent());
        assert!(Block::Torch.is_light_transparent());

        // Light-blocking blocks
        assert!(!Block::Stone.is_light_transparent());
        assert!(!Block::Dirt.is_light_transparent());
        assert!(!Block::Bedrock.is_light_transparent());
    }

    #[test]
    fn test_block_texture_indices() {
        // Grass: top=1, bottom=0, sides=2
        let grass = Block::Grass.texture_indices();
        assert_eq!(grass, [1, 0, 2, 2, 2, 2], "Grass texture indices incorrect");

        // Wood: top/bottom=8 (cross-section), sides=7 (bark)
        let wood = Block::Wood.texture_indices();
        assert_eq!(wood, [8, 8, 7, 7, 7, 7], "Wood texture indices incorrect");

        // Stone: all faces same
        let stone = Block::Stone.texture_indices();
        assert_eq!(stone, [3, 3, 3, 3, 3, 3], "Stone texture indices incorrect");
    }

    #[test]
    fn test_block_from_u8() {
        // Valid conversions
        assert_eq!(Block::from(0), Block::Air);
        assert_eq!(Block::from(1), Block::Dirt);
        assert_eq!(Block::from(2), Block::Grass);
        assert_eq!(Block::from(3), Block::Stone);
        assert_eq!(Block::from(15), Block::Torch);

        // Invalid values default to Air
        assert_eq!(Block::from(16), Block::Air);
        assert_eq!(Block::from(255), Block::Air);
    }

    // ============ ChunkPos Tests ============

    #[test]
    fn test_chunk_pos_from_world() {
        // Origin
        let pos = ChunkPos::from_world_pos(Vec3::new(0.0, 0.0, 0.0));
        assert_eq!((pos.x, pos.y, pos.z), (0, 0, 0));

        // Positive coordinates (assuming BLOCK_SCALE = 1.0 and CHUNK_SIZE = 16)
        let pos = ChunkPos::from_world_pos(Vec3::new(17.0, 0.0, 0.0));
        assert_eq!(pos.x, 1, "17.0 should be in chunk 1");

        // Negative coordinates
        let pos = ChunkPos::from_world_pos(Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!((pos.x, pos.y, pos.z), (-1, -1, -1), "Negative coords should be chunk -1");

        // Exact boundary
        let pos = ChunkPos::from_world_pos(Vec3::new(16.0, 0.0, 0.0));
        assert_eq!(pos.x, 1, "Exactly 16.0 should be in chunk 1");
    }

    #[test]
    fn test_chunk_pos_world_origin() {
        let pos = ChunkPos { x: 1, y: 2, z: 3 };
        let origin = pos.world_origin();

        // Convert back and verify
        let back = ChunkPos::from_world_pos(origin);
        assert_eq!((back.x, back.y, back.z), (1, 2, 3), "Round-trip should preserve chunk pos");
    }

    // ============ Chunk Light Tests ============

    #[test]
    fn test_chunk_light_packing() {
        let mut chunk = Chunk::new();

        // Set block light only
        chunk.set_block_light(0, 0, 0, 14);
        assert_eq!(chunk.get_block_light(0, 0, 0), 14);
        assert_eq!(chunk.get_sky_light(0, 0, 0), 0, "Sky light should be unaffected");

        // Set sky light only
        chunk.set_sky_light(1, 1, 1, 15);
        assert_eq!(chunk.get_sky_light(1, 1, 1), 15);
        assert_eq!(chunk.get_block_light(1, 1, 1), 0, "Block light should be unaffected");

        // Set both on same block
        chunk.set_block_light(2, 2, 2, 7);
        chunk.set_sky_light(2, 2, 2, 10);
        assert_eq!(chunk.get_block_light(2, 2, 2), 7);
        assert_eq!(chunk.get_sky_light(2, 2, 2), 10);
        assert_eq!(chunk.get_combined_light(2, 2, 2), 10, "Combined should be max");

        // Test combined light (block > sky)
        chunk.set_block_light(3, 3, 3, 12);
        chunk.set_sky_light(3, 3, 3, 5);
        assert_eq!(chunk.get_combined_light(3, 3, 3), 12, "Combined should be max");
    }

    #[test]
    fn test_chunk_in_bounds() {
        // Valid coordinates
        assert!(Chunk::in_bounds(0, 0, 0));
        assert!(Chunk::in_bounds(15, 15, 15));
        assert!(Chunk::in_bounds(8, 8, 8));

        // Invalid coordinates
        assert!(!Chunk::in_bounds(-1, 0, 0));
        assert!(!Chunk::in_bounds(0, -1, 0));
        assert!(!Chunk::in_bounds(0, 0, -1));
        assert!(!Chunk::in_bounds(16, 0, 0));
        assert!(!Chunk::in_bounds(0, 16, 0));
        assert!(!Chunk::in_bounds(0, 0, 16));
    }

    // ============ Test Fixture Tests ============

    #[test]
    fn test_new_solid() {
        let chunk = Chunk::new_solid(Block::Stone);

        // Check corners and center
        assert_eq!(chunk.get_block(0, 0, 0), Block::Stone);
        assert_eq!(chunk.get_block(15, 15, 15), Block::Stone);
        assert_eq!(chunk.get_block(8, 8, 8), Block::Stone);
    }

    #[test]
    fn test_new_checkerboard() {
        let chunk = Chunk::new_checkerboard();

        // Even sum = Stone, odd sum = Air
        assert_eq!(chunk.get_block(0, 0, 0), Block::Stone); // 0+0+0 = 0 (even)
        assert_eq!(chunk.get_block(1, 0, 0), Block::Air);   // 1+0+0 = 1 (odd)
        assert_eq!(chunk.get_block(1, 1, 0), Block::Stone); // 1+1+0 = 2 (even)
        assert_eq!(chunk.get_block(1, 1, 1), Block::Air);   // 1+1+1 = 3 (odd)
    }
}
