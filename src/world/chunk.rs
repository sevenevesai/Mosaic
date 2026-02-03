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
    /// Computed from flow velocity components
    pub flow_angle: f32,
    /// Flow velocity X component (blocks/sec, positive = +X direction)
    pub flow_velocity_x: f32,
    /// Flow velocity Z component (blocks/sec, positive = +Z direction)
    pub flow_velocity_z: f32,
    /// Whether this is an infinite water source (springs, sea level water)
    /// Source water maintains its level and doesn't drain
    pub is_source: bool,
}

impl Default for WaterColumn {
    fn default() -> Self {
        Self {
            surface_y: 0.0,
            has_water: false,
            foam: 0.0,
            flow_angle: 0.0,
            flow_velocity_x: 0.0,
            flow_velocity_z: 0.0,
            is_source: false,
        }
    }
}

impl WaterColumn {
    /// Calculate flow angle from velocity components
    /// Returns angle in radians (0 = +X, PI/2 = +Z)
    #[inline]
    pub fn calculate_flow_angle(&self) -> f32 {
        if self.flow_velocity_x.abs() < 0.001 && self.flow_velocity_z.abs() < 0.001 {
            // No significant flow, return current angle for stability
            self.flow_angle
        } else {
            self.flow_velocity_z.atan2(self.flow_velocity_x)
        }
    }

    /// Get flow speed (magnitude of velocity)
    #[inline]
    pub fn flow_speed(&self) -> f32 {
        (self.flow_velocity_x * self.flow_velocity_x +
         self.flow_velocity_z * self.flow_velocity_z).sqrt()
    }

    /// Update flow angle from current velocity components
    #[inline]
    pub fn update_flow_angle(&mut self) {
        self.flow_angle = self.calculate_flow_angle();
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

    /// Set water with flow data and source flag
    /// This is the full version that sets all water column properties
    pub fn set_water_with_flow(
        &mut self,
        x: usize,
        z: usize,
        surface_y: f32,
        foam: f32,
        flow_velocity_x: f32,
        flow_velocity_z: f32,
        is_source: bool,
    ) {
        let col = &mut self.columns[z][x];
        col.surface_y = surface_y;
        col.has_water = true;
        col.foam = foam;
        col.flow_velocity_x = flow_velocity_x;
        col.flow_velocity_z = flow_velocity_z;
        col.is_source = is_source;
        col.update_flow_angle();
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

// ============ PROP DATA ============

/// A single prop placement within a chunk
#[derive(Debug, Clone, Copy)]
pub struct PropPlacement {
    /// Prop type identifier (matches PropId from renderer::prop_registry)
    pub prop_id: u8,
    /// Local X position within chunk (0.0 - 16.0, sub-block precision)
    pub local_x: f32,
    /// Local Y position within chunk (0.0 - 16.0)
    pub local_y: f32,
    /// Local Z position within chunk (0.0 - 16.0)
    pub local_z: f32,
    /// Rotation around Y axis (radians, 0 - 2π)
    pub rotation: f32,
    /// Scale multiplier (typically 0.8 - 1.2)
    pub scale: f32,
}

impl PropPlacement {
    pub fn new(prop_id: u8, x: f32, y: f32, z: f32, rotation: f32, scale: f32) -> Self {
        Self {
            prop_id,
            local_x: x,
            local_y: y,
            local_z: z,
            rotation,
            scale,
        }
    }

    /// Convert to world position given chunk world origin
    #[inline]
    pub fn world_position(&self, chunk_origin: Vec3) -> Vec3 {
        Vec3::new(
            chunk_origin.x + self.local_x,
            chunk_origin.y + self.local_y,
            chunk_origin.z + self.local_z,
        )
    }
}

/// Storage for prop placements within a chunk
/// Typically contains 0-50 props for surface chunks, 0 for underground
pub struct ChunkPropData {
    /// All prop placements in this chunk
    placements: Vec<PropPlacement>,
}

impl ChunkPropData {
    pub fn new() -> Self {
        Self {
            placements: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            placements: Vec::with_capacity(capacity),
        }
    }

    /// Add a prop placement
    #[inline]
    pub fn add(&mut self, placement: PropPlacement) {
        self.placements.push(placement);
    }

    /// Get number of props in this chunk
    #[inline]
    pub fn len(&self) -> usize {
        self.placements.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// Iterate over all placements
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &PropPlacement> {
        self.placements.iter()
    }

    /// Clear all placements
    pub fn clear(&mut self) {
        self.placements.clear();
    }

    /// Remove a prop at the given index, returns the removed prop
    pub fn remove(&mut self, index: usize) -> Option<PropPlacement> {
        if index < self.placements.len() {
            Some(self.placements.swap_remove(index))
        } else {
            None
        }
    }

    /// Find the index of a prop near the given local position (within tolerance)
    pub fn find_near(&self, local_x: f32, local_y: f32, local_z: f32, tolerance: f32) -> Option<usize> {
        let tol_sq = tolerance * tolerance;
        for (i, prop) in self.placements.iter().enumerate() {
            let dx = prop.local_x - local_x;
            let dy = prop.local_y - local_y;
            let dz = prop.local_z - local_z;
            if dx * dx + dy * dy + dz * dz < tol_sq {
                return Some(i);
            }
        }
        None
    }
}

impl Default for ChunkPropData {
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

    /// Get the center position of this chunk in world space
    pub fn center(&self) -> Vec3 {
        let chunk_world_size = CHUNK_SIZE_F32 * BLOCK_SCALE;
        let half_size = chunk_world_size * 0.5;
        Vec3::new(
            self.x as f32 * chunk_world_size + half_size,
            self.y as f32 * chunk_world_size + half_size,
            self.z as f32 * chunk_world_size + half_size,
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
    StoneDirt = 4,      // Dirty stone variant
    Brick = 5,          // Placeable brick block
    Water = 6,
    Wood = 7,
    Leaves = 8,
    StoneRubble2 = 9,   // Stone rubble variant
    IronOre = 10,
    GoldOre = 11,
    DiamondOre = 12,
    RedstoneOre = 13,
    Bedrock = 14,
    Torch = 15,
    Sand = 16,          // Beach/desert sand
    // New ores
    CoalOre = 17,       // Common, upper depths
    CopperOre = 18,     // Common, upper-mid depths
    EmeraldOre = 19,    // Rare, deep
    SapphireOre = 20,   // Rare, very deep
    QuartzOre = 21,     // Uncommon, mid range
    AmethystOre = 22,   // Rare, deep clusters
    SilverOre = 23,     // Uncommon, mid depths
    TinOre = 24,        // Common, upper depths
    NickelOre = 25,     // Uncommon, mid-lower
    LeadOre = 26,       // Uncommon, lower depths
    TitaniumOre = 27,   // Very rare, very deep
    UraniumOre = 28,    // Very rare, deepest, glows!
    // Tree log variants
    OakLog = 29,        // Oak tree log
    BirchLog = 30,      // Birch tree log

    // === NEW BIOME BLOCKS (Phase 3) ===

    // Desert/Badlands biome
    Sandstone = 31,     // Desert subsurface
    RedSandstone = 32,  // Badlands formations

    // Tundra/Snow biome
    SnowGrass = 33,     // Snow-covered grass
    Snow = 34,          // Snow layer
    Ice = 35,           // Frozen water surface
    PackedIce = 36,     // Deep/ancient ice

    // Swamp biome
    MudBlock = 37,      // Swamp surface
    Peat = 38,          // Swamp subsurface

    // Jungle biome
    JungleGrass = 39,   // Dense jungle floor
    JungleLeaves = 40,  // Dense jungle foliage
    Vines = 41,         // Hanging vines (foliage)

    // Mountain biome
    Gravel = 42,        // Mountain scree, riverbeds

    // Structure blocks (ruins, formations)
    MossyStone = 43,    // Decayed ruin walls
    CrackedStone = 44,  // Damaged ruins
    ClayStone = 45,     // Hoodoo/pillar material

    // Cave decoration
    Stalactite = 46,    // Ceiling formations
    Stalagmite = 47,    // Floor formations
    CrystalAmethyst = 48, // Cavern crystals (purple)
    CrystalQuartz = 49,   // Cavern crystals (white)

    // Desert vegetation
    Cactus = 50,        // Desert plant (solid, prickly)
    DeadBush = 51,      // Desert shrub (foliage)

    // Additional tree types
    PineLog = 52,       // Pine tree log
    JungleLog = 53,     // Jungle tree log
    SpruceLeaves = 54,  // Spruce/pine needles

    // Storage/furniture
    Chest = 55,         // Storage container (rendered as vox prop)
}

impl Block {
    pub fn is_solid(&self) -> bool {
        !matches!(
            self,
            Block::Air
                | Block::Water
                | Block::Torch
                | Block::Vines
                | Block::DeadBush
                | Block::Snow
        )
    }

    pub fn is_transparent(&self) -> bool {
        matches!(
            self,
            Block::Air
                | Block::Water
                | Block::Leaves
                | Block::JungleLeaves
                | Block::SpruceLeaves
                | Block::Vines
                | Block::Torch
                | Block::Ice
                | Block::DeadBush
                | Block::Snow
        )
    }

    /// Returns true if this block should be rendered as crossed billboards instead of a cube
    /// Cross-billboard blocks are rendered as two intersecting diagonal planes (X shape)
    /// Currently disabled - all blocks render as cubes
    pub fn is_cross_billboard(&self) -> bool {
        false
    }

    /// Returns true if this block is rendered as a voxel prop (not chunk meshed)
    /// Prop blocks are rendered separately using VoxModel meshes
    pub fn is_prop(&self) -> bool {
        matches!(self, Block::Chest)
    }

    /// Returns true if this block is foliage (renders all 6 faces, no neighbor culling)
    /// Foliage blocks sway in the wind and need all faces visible to avoid gaps
    pub fn is_foliage(&self) -> bool {
        matches!(
            self,
            Block::Leaves
                | Block::JungleLeaves
                | Block::SpruceLeaves
                | Block::Vines
                | Block::DeadBush
        )
    }

    /// Returns true if this block is emissive (light source visible at any distance)
    /// Emissive blocks render at full brightness regardless of lighting calculations
    pub fn is_emissive(&self) -> bool {
        matches!(
            self,
            Block::Torch | Block::UraniumOre | Block::CrystalAmethyst | Block::CrystalQuartz
        )
    }

    /// Get texture indices for each face: [top, bottom, front, back, left, right]
    /// Atlas is 8x8 (64 tiles), indices 0-63
    pub fn texture_indices(&self) -> [u32; 6] {
        match self {
            Block::Air => [0; 6],
            Block::Dirt => [0; 6],
            Block::Grass => [1, 0, 2, 2, 2, 2], // Top=grass, Bottom=dirt, Sides=grass_side
            Block::Stone => [3; 6],
            Block::StoneDirt => [4; 6],         // Dirty stone variant
            Block::Brick => [5; 6],             // Brick building material
            Block::Water => [6; 6],
            Block::Wood => [8, 8, 7, 7, 7, 7], // Top/Bottom=cross-section, Sides=bark
            Block::Leaves => [9; 6],
            Block::StoneRubble2 => [10; 6],    // Stone rubble variant
            Block::IronOre => [11; 6],
            Block::GoldOre => [12; 6],
            Block::DiamondOre => [13; 6],
            Block::RedstoneOre => [14; 6],
            Block::Bedrock => [15; 6],
            Block::Torch => [16; 6],           // Torch procedural orange
            Block::Sand => [17; 6],            // Beach sand
            // Ores (18-29)
            Block::CoalOre => [18; 6],
            Block::CopperOre => [19; 6],
            Block::EmeraldOre => [20; 6],
            Block::SapphireOre => [21; 6],
            Block::QuartzOre => [22; 6],
            Block::AmethystOre => [23; 6],
            Block::SilverOre => [24; 6],
            Block::TinOre => [25; 6],
            Block::NickelOre => [26; 6],
            Block::LeadOre => [27; 6],
            Block::TitaniumOre => [28; 6],
            Block::UraniumOre => [29; 6],
            // Tree logs (30-31, 52-53)
            Block::OakLog => [8, 8, 30, 30, 30, 30],
            Block::BirchLog => [8, 8, 31, 31, 31, 31],

            // === NEW BIOME BLOCKS ===

            // Desert/Badlands (32-33)
            Block::Sandstone => [32; 6],       // Layered sandstone
            Block::RedSandstone => [33; 6],    // Red/orange sandstone

            // Tundra/Snow (34-37)
            Block::SnowGrass => [34, 0, 35, 35, 35, 35], // Snow top, dirt bottom, snowy sides
            Block::Snow => [34; 6],            // Pure snow
            Block::Ice => [36; 6],             // Translucent ice
            Block::PackedIce => [37; 6],       // Dense blue ice

            // Swamp (38-39)
            Block::MudBlock => [38; 6],        // Dark mud
            Block::Peat => [39; 6],            // Brown peat

            // Jungle (40-42)
            Block::JungleGrass => [40, 0, 41, 41, 41, 41], // Lush grass top
            Block::JungleLeaves => [42; 6],    // Dense tropical leaves
            Block::Vines => [43; 6],           // Hanging vines

            // Mountain (44)
            Block::Gravel => [44; 6],          // Gray gravel

            // Structure blocks (45-47)
            Block::MossyStone => [45; 6],      // Mossy cobblestone
            Block::CrackedStone => [46; 6],    // Cracked stone
            Block::ClayStone => [47; 6],       // Tan claystone

            // Cave decoration (48-51)
            Block::Stalactite => [48; 6],      // Pointed stone
            Block::Stalagmite => [48; 6],      // Same texture as stalactite
            Block::CrystalAmethyst => [49; 6], // Purple crystal (emissive)
            Block::CrystalQuartz => [50; 6],   // White crystal (emissive)

            // Desert vegetation (51-52)
            Block::Cactus => [51, 52, 51, 51, 51, 51], // Cactus top different
            Block::DeadBush => [53; 6],        // Dead shrub

            // Additional trees (54-56)
            Block::PineLog => [8, 8, 54, 54, 54, 54],   // Pine bark
            Block::JungleLog => [8, 8, 55, 55, 55, 55], // Jungle bark
            Block::SpruceLeaves => [56; 6],    // Blue-green needles

            // Storage/furniture
            Block::Chest => [7; 6],            // Wood planks (rendered as vox prop)
        }
    }

    /// Get light emission level (0-15) for this block type
    /// Torches emit level 14, providing ~14 block illumination radius
    #[inline]
    pub fn light_emission(&self) -> u8 {
        match self {
            Block::Torch => 14,
            Block::UraniumOre => 8,       // Radioactive glow!
            Block::CrystalAmethyst => 10, // Purple glow
            Block::CrystalQuartz => 12,   // Bright white glow
            _ => 0,
        }
    }

    /// Check if this block allows light to pass through
    #[inline]
    pub fn is_light_transparent(&self) -> bool {
        matches!(
            self,
            Block::Air
                | Block::Water
                | Block::Leaves
                | Block::JungleLeaves
                | Block::SpruceLeaves
                | Block::Vines
                | Block::Torch
                | Block::Ice
                | Block::DeadBush
                | Block::Snow
                | Block::CrystalAmethyst
                | Block::CrystalQuartz
        )
    }

    /// Get texture index for a specific face direction
    /// face_idx: 0=top(+Y), 1=bottom(-Y), 2=front(+Z), 3=back(-Z), 4=left(-X), 5=right(+X)
    #[inline]
    pub fn face_texture(&self, face_idx: usize) -> u32 {
        self.texture_indices()[face_idx]
    }

    /// Get the dominant color for this block type (used for impostor rendering)
    /// Returns [R, G, B] as approximate color matching the block's appearance
    #[inline]
    pub fn dominant_color(&self) -> [u8; 3] {
        match self {
            Block::Air => [0, 0, 0],
            Block::Dirt => [139, 90, 43],           // Brown dirt
            Block::Grass => [86, 125, 70],          // Green grass (weighted toward top)
            Block::Stone => [128, 128, 128],        // Gray stone
            Block::StoneDirt => [115, 100, 85],     // Brownish stone
            Block::Brick => [156, 91, 74],          // Brick red
            Block::Water => [64, 120, 180],         // Blue water
            Block::Wood => [143, 104, 58],          // Wood brown
            Block::Leaves => [56, 118, 29],         // Leaf green
            Block::StoneRubble2 => [120, 120, 115], // Gray rubble
            Block::IronOre => [140, 125, 115],      // Ore gray with iron tint
            Block::GoldOre => [170, 155, 100],      // Ore with gold tint
            Block::DiamondOre => [130, 180, 190],   // Ore with cyan tint
            Block::RedstoneOre => [160, 90, 90],    // Ore with red tint
            Block::Bedrock => [50, 50, 50],         // Dark gray
            Block::Torch => [255, 200, 100],        // Warm light color
            Block::Sand => [220, 200, 140],         // Sandy tan
            Block::CoalOre => [60, 60, 60],         // Dark coal
            Block::CopperOre => [170, 120, 80],     // Copper brown
            Block::EmeraldOre => [80, 180, 100],    // Emerald green
            Block::SapphireOre => [80, 100, 200],   // Sapphire blue
            Block::QuartzOre => [230, 225, 220],    // Near-white quartz
            Block::AmethystOre => [150, 100, 180],  // Purple amethyst
            Block::SilverOre => [190, 195, 200],    // Silver gray
            Block::TinOre => [160, 160, 150],       // Tin gray
            Block::NickelOre => [150, 165, 140],    // Greenish nickel
            Block::LeadOre => [100, 100, 115],      // Dark lead
            Block::TitaniumOre => [180, 180, 195],  // Light titanium
            Block::UraniumOre => [100, 200, 100],   // Glowing green
            Block::OakLog => [130, 95, 50],         // Oak brown
            Block::BirchLog => [210, 200, 180],     // Birch white-tan
            Block::Sandstone => [210, 190, 140],    // Sandy beige
            Block::RedSandstone => [180, 100, 60],  // Terracotta red
            Block::SnowGrass => [200, 230, 200],    // Snow-tinted green
            Block::Snow => [250, 250, 255],         // White snow
            Block::Ice => [180, 220, 255],          // Light blue ice
            Block::PackedIce => [150, 200, 240],    // Deeper blue ice
            Block::MudBlock => [80, 60, 40],        // Dark brown mud
            Block::Peat => [50, 40, 30],            // Very dark brown
            Block::JungleGrass => [50, 90, 35],     // Dark jungle green
            Block::JungleLeaves => [40, 100, 30],   // Dark jungle green
            Block::Vines => [50, 90, 30],           // Vine green
            Block::Gravel => [130, 130, 130],       // Gray gravel
            Block::MossyStone => [90, 115, 85],     // Mossy gray-green
            Block::CrackedStone => [110, 105, 100], // Weathered gray
            Block::ClayStone => [180, 160, 140],    // Tan clay
            Block::Stalactite => [150, 145, 140],   // Limestone color
            Block::Stalagmite => [145, 140, 135],   // Limestone color
            Block::CrystalAmethyst => [180, 120, 200], // Glowing purple
            Block::CrystalQuartz => [255, 250, 240],   // Bright white
            Block::Cactus => [60, 120, 50],         // Cactus green
            Block::DeadBush => [130, 100, 60],      // Brown dead plant
            Block::PineLog => [100, 80, 45],        // Dark pine brown
            Block::JungleLog => [110, 85, 50],      // Tropical wood
            Block::SpruceLeaves => [45, 75, 40],    // Dark spruce green
            Block::Chest => [140, 100, 55],         // Chest wood color
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
            4 => Block::StoneDirt,
            5 => Block::Brick,
            6 => Block::Water,
            7 => Block::Wood,
            8 => Block::Leaves,
            9 => Block::StoneRubble2,
            10 => Block::IronOre,
            11 => Block::GoldOre,
            12 => Block::DiamondOre,
            13 => Block::RedstoneOre,
            14 => Block::Bedrock,
            15 => Block::Torch,
            16 => Block::Sand,
            17 => Block::CoalOre,
            18 => Block::CopperOre,
            19 => Block::EmeraldOre,
            20 => Block::SapphireOre,
            21 => Block::QuartzOre,
            22 => Block::AmethystOre,
            23 => Block::SilverOre,
            24 => Block::TinOre,
            25 => Block::NickelOre,
            26 => Block::LeadOre,
            27 => Block::TitaniumOre,
            28 => Block::UraniumOre,
            29 => Block::OakLog,
            30 => Block::BirchLog,
            // New biome blocks
            31 => Block::Sandstone,
            32 => Block::RedSandstone,
            33 => Block::SnowGrass,
            34 => Block::Snow,
            35 => Block::Ice,
            36 => Block::PackedIce,
            37 => Block::MudBlock,
            38 => Block::Peat,
            39 => Block::JungleGrass,
            40 => Block::JungleLeaves,
            41 => Block::Vines,
            42 => Block::Gravel,
            43 => Block::MossyStone,
            44 => Block::CrackedStone,
            45 => Block::ClayStone,
            46 => Block::Stalactite,
            47 => Block::Stalagmite,
            48 => Block::CrystalAmethyst,
            49 => Block::CrystalQuartz,
            50 => Block::Cactus,
            51 => Block::DeadBush,
            52 => Block::PineLog,
            53 => Block::JungleLog,
            54 => Block::SpruceLeaves,
            55 => Block::Chest,
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
    /// Optional prop placement data - only allocated if chunk has surface vegetation
    props: Option<Box<ChunkPropData>>,
    /// Count of solid blocks for fast empty/full detection
    /// Used to skip meshing for air-only chunks (solid_count == 0)
    solid_count: u16,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            blocks: Box::new([Block::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
            light: Box::new([0u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]),
            water: None,
            props: None,
            solid_count: 0,
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
        let idx = Self::index(x, y, z);
        let old_block = self.blocks[idx];

        // Update solid count when solidity changes
        let was_solid = old_block.is_solid();
        let is_solid = block.is_solid();
        if was_solid && !is_solid {
            self.solid_count = self.solid_count.saturating_sub(1);
        } else if !was_solid && is_solid {
            self.solid_count = self.solid_count.saturating_add(1);
        }

        self.blocks[idx] = block;
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

    /// Get raw blocks data as a byte slice for fast GPU upload
    /// Since Block is #[repr(u8)], this is a zero-cost cast
    /// Layout: x + y * 16 + z * 256 (same as index())
    #[inline]
    pub fn blocks_as_bytes(&self) -> &[u8] {
        // SAFETY: Block is #[repr(u8)] so this cast is valid
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr() as *const u8,
                CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE,
            )
        }
    }

    // ============ SOLID COUNT METHODS (for mesh optimization) ============

    /// Get count of solid blocks in this chunk
    /// Used for fast empty chunk detection to skip meshing
    #[inline]
    pub fn solid_count(&self) -> u16 {
        self.solid_count
    }

    /// Check if chunk is all air (no solid blocks)
    /// When true, meshing can be skipped entirely
    #[inline]
    pub fn is_all_air(&self) -> bool {
        self.solid_count == 0
    }

    /// Check if chunk is completely solid (all 4096 blocks solid)
    /// Useful for detecting interior underground chunks
    #[inline]
    pub fn is_all_solid(&self) -> bool {
        self.solid_count == (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as u16
    }

    /// Recalculate solid count by scanning all blocks
    /// Call after bulk operations that bypass set_block tracking
    pub fn recalculate_solid_count(&mut self) {
        self.solid_count = self.blocks.iter().filter(|b| b.is_solid()).count() as u16;
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

    /// Set water surface with flow data at a column (allocates water data if needed)
    pub fn set_water_surface_with_flow(
        &mut self,
        x: usize,
        z: usize,
        surface_y: f32,
        foam: f32,
        flow_velocity_x: f32,
        flow_velocity_z: f32,
        is_source: bool,
    ) {
        self.ensure_water_data().set_water_with_flow(
            x, z, surface_y, foam, flow_velocity_x, flow_velocity_z, is_source
        );
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

    // ============ PROP ACCESS METHODS ============

    /// Check if this chunk has prop data allocated
    #[inline]
    pub fn has_prop_data(&self) -> bool {
        self.props.is_some()
    }

    /// Check if this chunk has any props
    #[inline]
    pub fn has_props(&self) -> bool {
        self.props.as_ref().map_or(false, |p| !p.is_empty())
    }

    /// Get number of props in this chunk
    #[inline]
    pub fn prop_count(&self) -> usize {
        self.props.as_ref().map_or(0, |p| p.len())
    }

    /// Get reference to prop data if present
    #[inline]
    pub fn prop_data(&self) -> Option<&ChunkPropData> {
        self.props.as_deref()
    }

    /// Get mutable reference to prop data if present
    #[inline]
    pub fn prop_data_mut(&mut self) -> Option<&mut ChunkPropData> {
        self.props.as_deref_mut()
    }

    /// Ensure prop data is allocated and return mutable reference
    pub fn ensure_prop_data(&mut self) -> &mut ChunkPropData {
        if self.props.is_none() {
            self.props = Some(Box::new(ChunkPropData::new()));
        }
        self.props.as_deref_mut().unwrap()
    }

    /// Ensure prop data with capacity hint
    pub fn ensure_prop_data_with_capacity(&mut self, capacity: usize) -> &mut ChunkPropData {
        if self.props.is_none() {
            self.props = Some(Box::new(ChunkPropData::with_capacity(capacity)));
        }
        self.props.as_deref_mut().unwrap()
    }

    /// Add a prop placement (allocates prop data if needed)
    pub fn add_prop(&mut self, placement: PropPlacement) {
        self.ensure_prop_data().add(placement);
    }

    /// Clear all prop data from this chunk
    pub fn clear_props(&mut self) {
        self.props = None;
    }

    /// Iterate over all prop placements (empty iterator if no props)
    pub fn iter_props(&self) -> impl Iterator<Item = &PropPlacement> {
        self.props.iter().flat_map(|p| p.iter())
    }

    /// Remove a prop near the given local position, returns the removed prop
    pub fn remove_prop_near(&mut self, local_x: f32, local_y: f32, local_z: f32, tolerance: f32) -> Option<PropPlacement> {
        if let Some(prop_data) = self.props.as_deref_mut() {
            if let Some(idx) = prop_data.find_near(local_x, local_y, local_z, tolerance) {
                return prop_data.remove(idx);
            }
        }
        None
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
        assert_eq!(Block::from(16), Block::Sand);
        assert_eq!(Block::from(30), Block::BirchLog);

        // New biome blocks
        assert_eq!(Block::from(31), Block::Sandstone);
        assert_eq!(Block::from(42), Block::Gravel);
        assert_eq!(Block::from(54), Block::SpruceLeaves);

        // Invalid values default to Air
        assert_eq!(Block::from(55), Block::Air);
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
