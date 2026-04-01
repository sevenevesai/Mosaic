//! Chunk data structures

use glam::Vec3;

use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_F32, CHUNK_SIZE_I32, BLOCK_SCALE};

/// Water data for a single column in a chunk
/// Stores the water surface height for heightfield rendering
///
/// Uses a cellular automata (CA) level system (0-7) for flow simulation:
/// - Level 0 = no water (empty)
/// - Level 1-6 = flowing water (decays with distance from source)
/// - Level 7 = full water (source blocks or filled pools)
#[derive(Debug, Clone, Copy)]
pub struct WaterColumn {
    /// Water surface height in world Y coordinates
    /// This is the actual Y position of the water surface
    pub surface_y: f32,
    /// Whether this column has water at all
    pub has_water: bool,
    /// Foam factor (0.0-1.0) - higher near block edges and flow points
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
    /// Water level for CA simulation (0-7)
    /// 0 = empty, 7 = full, 1-6 = flowing with decay
    pub level: u8,
    /// Primary flow direction X (-1, 0, +1) for visual animation
    pub flow_dir_x: i8,
    /// Primary flow direction Z (-1, 0, +1) for visual animation
    pub flow_dir_z: i8,
    /// Distance from nearest source (for flow decay and effects)
    /// 255 = unknown/far, 0 = is source
    pub source_distance: u8,
    /// Whether this cell is actively flowing (needs simulation)
    pub is_flowing: bool,
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
            level: 0,
            flow_dir_x: 0,
            flow_dir_z: 0,
            source_distance: 255,
            is_flowing: false,
        }
    }
}

/// Maximum water level for CA simulation
pub const WATER_LEVEL_MAX: u8 = 7;

/// Minimum water level that counts as having water
pub const WATER_LEVEL_MIN: u8 = 1;

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

    /// Update flow angle from flow direction integers
    #[inline]
    pub fn update_flow_angle_from_dir(&mut self) {
        if self.flow_dir_x != 0 || self.flow_dir_z != 0 {
            self.flow_angle = (self.flow_dir_z as f32).atan2(self.flow_dir_x as f32);
            // Also update velocity for compatibility
            let speed = if self.is_flowing { 1.0 } else { 0.2 };
            self.flow_velocity_x = self.flow_dir_x as f32 * speed;
            self.flow_velocity_z = self.flow_dir_z as f32 * speed;
        }
    }

    /// Get the water level as a normalized value (0.0-1.0)
    #[inline]
    pub fn level_normalized(&self) -> f32 {
        self.level as f32 / WATER_LEVEL_MAX as f32
    }

    /// Check if this column has significant water (level > 0)
    #[inline]
    pub fn has_significant_water(&self) -> bool {
        self.has_water && self.level >= WATER_LEVEL_MIN
    }

    /// Check if this is a full water column (level 7 or source)
    #[inline]
    pub fn is_full(&self) -> bool {
        self.level >= WATER_LEVEL_MAX || self.is_source
    }

    /// Set as a water source (infinite, full level)
    pub fn set_as_source(&mut self, surface_y: f32) {
        self.has_water = true;
        self.is_source = true;
        self.level = WATER_LEVEL_MAX;
        self.surface_y = surface_y;
        self.source_distance = 0;
        self.is_flowing = false;
        self.foam = 0.1; // Slight foam at sources
    }

    /// Set as flowing water with a specific level
    pub fn set_flowing(&mut self, level: u8, surface_y: f32, source_dist: u8) {
        self.has_water = level >= WATER_LEVEL_MIN;
        self.is_source = false;
        self.level = level.min(WATER_LEVEL_MAX);
        self.surface_y = surface_y;
        self.source_distance = source_dist;
        self.is_flowing = level >= WATER_LEVEL_MIN;
        // Foam increases with flow and decreases with level
        self.foam = if self.is_flowing {
            0.2 + (1.0 - self.level_normalized()) * 0.3
        } else {
            0.0
        };
    }

    /// Clear water from this column
    pub fn clear(&mut self) {
        self.has_water = false;
        self.is_source = false;
        self.level = 0;
        self.surface_y = 0.0;
        self.source_distance = 255;
        self.is_flowing = false;
        self.foam = 0.0;
        self.flow_dir_x = 0;
        self.flow_dir_z = 0;
        self.flow_velocity_x = 0.0;
        self.flow_velocity_z = 0.0;
        self.flow_angle = 0.0;
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

        // Set default CA fields for static water
        col.level = WATER_LEVEL_MAX;
        col.is_source = true;
        col.source_distance = 0;
        col.is_flowing = false;
        col.flow_dir_x = 0;
        col.flow_dir_z = 0;

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

        // Set CA system fields
        col.level = WATER_LEVEL_MAX; // Sea level water is always full
        col.source_distance = if is_source { 0 } else { 1 };
        col.is_flowing = !is_source && (flow_velocity_x.abs() > 0.01 || flow_velocity_z.abs() > 0.01);

        // Calculate flow direction from velocity
        if flow_velocity_x.abs() > 0.01 || flow_velocity_z.abs() > 0.01 {
            col.flow_dir_x = if flow_velocity_x > 0.05 { 1 } else if flow_velocity_x < -0.05 { -1 } else { 0 };
            col.flow_dir_z = if flow_velocity_z > 0.05 { 1 } else if flow_velocity_z < -0.05 { -1 } else { 0 };
        } else {
            col.flow_dir_x = 0;
            col.flow_dir_z = 0;
        }

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

// ============ GRASS TILE DATA ============

/// A grass tile for procedural grass rendering (16 bytes)
/// Re-exported from renderer::grass for storage in chunks
pub use crate::renderer::grass::GrassTile;

/// Storage for grass tiles within a chunk
/// Typically 0-16 tiles per surface chunk (4x4 tile grid)
pub struct ChunkGrassData {
    /// Grass tiles in this chunk
    tiles: Vec<GrassTile>,
}

impl ChunkGrassData {
    pub fn new() -> Self {
        Self {
            tiles: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tiles: Vec::with_capacity(capacity),
        }
    }

    /// Add a grass tile
    #[inline]
    pub fn add(&mut self, tile: GrassTile) {
        self.tiles.push(tile);
    }

    /// Get number of tiles
    #[inline]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Iterate over tiles
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &GrassTile> {
        self.tiles.iter()
    }

    /// Get tiles as slice
    #[inline]
    pub fn tiles(&self) -> &[GrassTile] {
        &self.tiles
    }

    /// Clear all tiles
    pub fn clear(&mut self) {
        self.tiles.clear();
    }
}

impl Default for ChunkGrassData {
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

    // === TEXTURE CATALOG BLOCKS ===
    // Additional blocks for every texture in assets/blocks/
    Bark03 = 56, Bark05 = 57, Bark06 = 58, Bark08 = 59, Bark09 = 60,
    Bark11 = 61, Bark12 = 62, Bark13 = 63, Bark14 = 64,
    Leaf01 = 65, Leaf02 = 66, Leaf03 = 67, Leaf04 = 68,
    Leaf06 = 69, Leaf07 = 70, Leaf08 = 71, Leaf09 = 72, Leaf10 = 73,
    Leaf11 = 74, Leaf12 = 75, Leaf13 = 76, Leaf14 = 77, Leaf15 = 78, Leaf16 = 79,
    Leaf18 = 80, Leaf19 = 81, Leaf20 = 82, Leaf21 = 83, Leaf22 = 84, Leaf23 = 85,
    Leaf25 = 86, Leaf27 = 87, Leaf28 = 88, Leaf29 = 89, Leaf31 = 90, Leaf32 = 91,
    Magma01 = 92,
    Misc05 = 93, Misc06 = 94, Misc07 = 95, Misc08 = 96, Misc09 = 97,
    Misc10 = 98, Misc11 = 99, Misc12 = 100, Misc13 = 101, Misc18 = 102, Misc19 = 103,
    Sand03 = 104,
    Snow01 = 105, Snow03 = 106, Snow05 = 107, Snow08 = 108,
    Stone02 = 109, Stone03 = 110, Stone06 = 111, Stone08 = 112,
    Stone10 = 113, Stone11 = 114, Stone12 = 115, Stone13 = 116,
    Stone14 = 117, Stone15 = 118, Stone16 = 119, Stone17 = 120,
    Stone18 = 121, Stone19 = 122, Stone20 = 123, Stone21 = 124,
    Stone22 = 125, Stone23 = 126, Stone24 = 127, Stone25 = 128,
    Stone26 = 129, Stone27 = 130, Stone28 = 131, Stone29 = 132,
    Stone30 = 133, Stone31 = 134, Stone32 = 135, Stone33 = 136,
    Stone34 = 137, Stone35 = 138, Stone36 = 139, Stone37 = 140,
    Stone38 = 141, Stone39 = 142, Stone40 = 143, Stone41 = 144,
    Stone42 = 145, Stone43 = 146, Stone44 = 147, Stone45 = 148,
    Stone46 = 149, Stone47 = 150, Stone48 = 151, Stone51 = 152, Stone52 = 153,
    Tile01 = 154, Tile02 = 155, Tile03 = 156, Tile04 = 157, Tile05 = 158,
    Wood01 = 159, Wood02 = 160, Wood03 = 161, Wood04 = 162, Wood05 = 163,
    Wood06 = 164, Wood07 = 165, Wood08 = 166, Wood09 = 167, Wood10 = 168,
    Wood11 = 169, Wood12 = 170, Wood14 = 171, Wood15 = 172, Wood16 = 173,
    Wood17 = 174, Wood18 = 175, Wood19 = 176, Wood20 = 177, Wood21 = 178,
    Wood23 = 179,
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

    /// Hardness in seconds (time to mine by hand). 0.0 = instant. -1.0 = unbreakable.
    pub fn hardness(&self) -> f32 {
        match self {
            Block::Air | Block::Water => 0.0,
            Block::Bedrock => -1.0,
            // Instant breaks
            Block::Torch | Block::DeadBush | Block::Snow | Block::Vines => 0.0,
            // Soft blocks (shovel preferred)
            Block::Dirt | Block::Grass | Block::SnowGrass | Block::JungleGrass
            | Block::MudBlock | Block::Peat | Block::Sand | Block::Gravel => 0.75,
            // Leaves
            Block::Leaves | Block::JungleLeaves | Block::SpruceLeaves => 0.3,
            // Wood (axe preferred)
            Block::Wood | Block::OakLog | Block::BirchLog | Block::PineLog
            | Block::JungleLog | Block::Chest => 2.0,
            Block::Cactus => 0.5,
            // Stone variants (pickaxe preferred)
            Block::Stone | Block::StoneDirt | Block::StoneRubble2 | Block::MossyStone
            | Block::CrackedStone | Block::ClayStone | Block::Sandstone
            | Block::RedSandstone | Block::Brick | Block::Stalactite | Block::Stalagmite => 1.5,
            // Ice
            Block::Ice | Block::PackedIce => 0.5,
            // Common ores
            Block::CoalOre | Block::CopperOre | Block::TinOre => 3.0,
            // Mid-tier ores
            Block::IronOre | Block::NickelOre | Block::LeadOre
            | Block::QuartzOre | Block::RedstoneOre | Block::SilverOre
            | Block::GoldOre | Block::AmethystOre => 3.0,
            // Precious ores
            Block::EmeraldOre | Block::SapphireOre => 4.0,
            Block::DiamondOre | Block::TitaniumOre | Block::UraniumOre => 5.0,
            // Crystals
            Block::CrystalAmethyst | Block::CrystalQuartz => 1.5,
            // Texture catalog blocks — default to stone hardness
            _ => 1.5,
        }
    }

    /// Preferred tool type for mining speed bonus. None = no preference.
    pub fn preferred_tool(&self) -> Option<crate::inventory::item::ToolKind> {
        use crate::inventory::item::ToolKind;
        match self {
            // Shovel blocks
            Block::Dirt | Block::Grass | Block::SnowGrass | Block::JungleGrass
            | Block::MudBlock | Block::Peat | Block::Sand | Block::Gravel
            | Block::Snow => Some(ToolKind::Shovel),
            // Axe blocks
            Block::Wood | Block::OakLog | Block::BirchLog | Block::PineLog
            | Block::JungleLog | Block::Leaves | Block::JungleLeaves
            | Block::SpruceLeaves | Block::Chest | Block::Cactus => Some(ToolKind::Axe),
            // Pickaxe blocks
            Block::Stone | Block::StoneDirt | Block::StoneRubble2 | Block::MossyStone
            | Block::CrackedStone | Block::ClayStone | Block::Sandstone
            | Block::RedSandstone | Block::Brick | Block::Stalactite | Block::Stalagmite
            | Block::Ice | Block::PackedIce
            | Block::CoalOre | Block::CopperOre | Block::TinOre | Block::IronOre
            | Block::NickelOre | Block::LeadOre | Block::QuartzOre | Block::RedstoneOre
            | Block::SilverOre | Block::GoldOre | Block::AmethystOre | Block::EmeraldOre
            | Block::SapphireOre | Block::DiamondOre | Block::TitaniumOre | Block::UraniumOre
            | Block::CrystalAmethyst | Block::CrystalQuartz => Some(ToolKind::Pickaxe),
            _ => None,
        }
    }

    /// Minimum harvest level required to drop items from this block.
    /// 0 = any tool/hand, 1 = stone+, 2 = iron+, 3 = diamond
    pub fn required_harvest_level(&self) -> u8 {
        match self {
            Block::DiamondOre | Block::TitaniumOre | Block::UraniumOre
            | Block::EmeraldOre | Block::SapphireOre | Block::GoldOre => 2,
            Block::IronOre | Block::RedstoneOre => 1,
            _ => 0,
        }
    }

    /// Get texture indices for each face: [top, bottom, front, back, left, right]
    /// Indices resolved from block registry (assets/blocks/ textures)
    pub fn texture_indices(&self) -> [u32; 6] {
        crate::world::block_registry::registry().block_texture_indices(*self as u8)
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
            Block::SpruceLeaves => [140, 180, 200],   // Frost blue
            Block::Chest => [140, 100, 55],         // Chest wood color
            Block::Bark03 | Block::Bark05 | Block::Bark06 | Block::Bark08 | Block::Bark09 |
            Block::Bark11 | Block::Bark12 | Block::Bark13 | Block::Bark14 => [90, 70, 40],
            Block::Leaf01 | Block::Leaf02 | Block::Leaf03 | Block::Leaf04 | Block::Leaf06 |
            Block::Leaf07 | Block::Leaf08 | Block::Leaf09 | Block::Leaf10 | Block::Leaf11 |
            Block::Leaf12 | Block::Leaf13 | Block::Leaf14 | Block::Leaf15 | Block::Leaf16 |
            Block::Leaf18 | Block::Leaf19 | Block::Leaf20 | Block::Leaf21 | Block::Leaf22 |
            Block::Leaf23 | Block::Leaf25 | Block::Leaf27 | Block::Leaf28 | Block::Leaf29 |
            Block::Leaf31 | Block::Leaf32 => [60, 130, 50],
            Block::Magma01 => [200, 80, 20],
            Block::Misc05 | Block::Misc06 | Block::Misc07 | Block::Misc08 | Block::Misc09 |
            Block::Misc10 | Block::Misc11 | Block::Misc12 | Block::Misc13 |
            Block::Misc18 | Block::Misc19 => [128, 128, 128],
            Block::Sand03 => [210, 190, 140],
            Block::Snow01 | Block::Snow03 | Block::Snow05 | Block::Snow08 => [230, 235, 240],
            Block::Stone02 | Block::Stone03 | Block::Stone06 | Block::Stone08 |
            Block::Stone10 | Block::Stone11 | Block::Stone12 | Block::Stone13 |
            Block::Stone14 | Block::Stone15 | Block::Stone16 | Block::Stone17 |
            Block::Stone18 | Block::Stone19 | Block::Stone20 | Block::Stone21 |
            Block::Stone22 | Block::Stone23 | Block::Stone24 | Block::Stone25 |
            Block::Stone26 | Block::Stone27 | Block::Stone28 | Block::Stone29 |
            Block::Stone30 | Block::Stone31 | Block::Stone32 | Block::Stone33 |
            Block::Stone34 | Block::Stone35 | Block::Stone36 | Block::Stone37 |
            Block::Stone38 | Block::Stone39 | Block::Stone40 | Block::Stone41 |
            Block::Stone42 | Block::Stone43 | Block::Stone44 | Block::Stone45 |
            Block::Stone46 | Block::Stone47 | Block::Stone48 | Block::Stone51 | Block::Stone52 => [130, 130, 130],
            Block::Tile01 | Block::Tile02 | Block::Tile03 | Block::Tile04 | Block::Tile05 => [150, 140, 120],
            Block::Wood01 | Block::Wood02 | Block::Wood03 | Block::Wood04 | Block::Wood05 |
            Block::Wood06 | Block::Wood07 | Block::Wood08 | Block::Wood09 | Block::Wood10 |
            Block::Wood11 | Block::Wood12 | Block::Wood14 | Block::Wood15 | Block::Wood16 |
            Block::Wood17 | Block::Wood18 | Block::Wood19 | Block::Wood20 | Block::Wood21 |
            Block::Wood23 => [160, 120, 70],
        }
    }

    /// Human-readable name for UI display (tooltips, catalog)
    pub fn display_name(&self) -> &'static str {
        match self {
            Block::Air => "Air",
            Block::Dirt => "Dirt",
            Block::Grass => "Grass",
            Block::Stone => "Stone",
            Block::StoneDirt => "Stone Dirt",
            Block::Brick => "Brick",
            Block::Water => "Water",
            Block::Wood => "Wood",
            Block::Leaves => "Leaves",
            Block::StoneRubble2 => "Stone Rubble",
            Block::IronOre => "Iron Ore",
            Block::GoldOre => "Gold Ore",
            Block::DiamondOre => "Diamond Ore",
            Block::RedstoneOre => "Redstone Ore",
            Block::Bedrock => "Bedrock",
            Block::Torch => "Torch",
            Block::Sand => "Sand",
            Block::CoalOre => "Coal Ore",
            Block::CopperOre => "Copper Ore",
            Block::EmeraldOre => "Emerald Ore",
            Block::SapphireOre => "Sapphire Ore",
            Block::QuartzOre => "Quartz Ore",
            Block::AmethystOre => "Amethyst Ore",
            Block::SilverOre => "Silver Ore",
            Block::TinOre => "Tin Ore",
            Block::NickelOre => "Nickel Ore",
            Block::LeadOre => "Lead Ore",
            Block::TitaniumOre => "Titanium Ore",
            Block::UraniumOre => "Uranium Ore",
            Block::OakLog => "Oak Log",
            Block::BirchLog => "Birch Log",
            Block::Sandstone => "Sandstone",
            Block::RedSandstone => "Red Sandstone",
            Block::SnowGrass => "Snow Grass",
            Block::Snow => "Snow",
            Block::Ice => "Ice",
            Block::PackedIce => "Packed Ice",
            Block::MudBlock => "Mud",
            Block::Peat => "Peat",
            Block::JungleGrass => "Jungle Grass",
            Block::JungleLeaves => "Jungle Leaves",
            Block::Vines => "Vines",
            Block::Gravel => "Gravel",
            Block::MossyStone => "Mossy Stone",
            Block::CrackedStone => "Cracked Stone",
            Block::ClayStone => "Clay Stone",
            Block::Stalactite => "Stalactite",
            Block::Stalagmite => "Stalagmite",
            Block::CrystalAmethyst => "Amethyst Crystal",
            Block::CrystalQuartz => "Quartz Crystal",
            Block::Cactus => "Cactus",
            Block::DeadBush => "Dead Bush",
            Block::PineLog => "Pine Log",
            Block::JungleLog => "Jungle Log",
            Block::SpruceLeaves => "Spruce Leaves",
            Block::Chest => "Chest",
            Block::Bark03 => "Bark 03", Block::Bark05 => "Bark 05", Block::Bark06 => "Bark 06",
            Block::Bark08 => "Bark 08", Block::Bark09 => "Bark 09", Block::Bark11 => "Bark 11",
            Block::Bark12 => "Bark 12", Block::Bark13 => "Bark 13", Block::Bark14 => "Bark 14",
            Block::Leaf01 => "Leaf 01", Block::Leaf02 => "Leaf 02", Block::Leaf03 => "Leaf 03",
            Block::Leaf04 => "Leaf 04", Block::Leaf06 => "Leaf 06", Block::Leaf07 => "Leaf 07",
            Block::Leaf08 => "Leaf 08", Block::Leaf09 => "Leaf 09", Block::Leaf10 => "Leaf 10",
            Block::Leaf11 => "Leaf 11", Block::Leaf12 => "Leaf 12", Block::Leaf13 => "Leaf 13",
            Block::Leaf14 => "Leaf 14", Block::Leaf15 => "Leaf 15", Block::Leaf16 => "Leaf 16",
            Block::Leaf18 => "Leaf 18", Block::Leaf19 => "Leaf 19", Block::Leaf20 => "Leaf 20",
            Block::Leaf21 => "Leaf 21", Block::Leaf22 => "Leaf 22", Block::Leaf23 => "Leaf 23",
            Block::Leaf25 => "Leaf 25", Block::Leaf27 => "Leaf 27", Block::Leaf28 => "Leaf 28",
            Block::Leaf29 => "Leaf 29", Block::Leaf31 => "Leaf 31", Block::Leaf32 => "Leaf 32",
            Block::Magma01 => "Magma",
            Block::Misc05 => "Misc 05", Block::Misc06 => "Misc 06", Block::Misc07 => "Misc 07",
            Block::Misc08 => "Misc 08", Block::Misc09 => "Misc 09", Block::Misc10 => "Misc 10",
            Block::Misc11 => "Misc 11", Block::Misc12 => "Misc 12", Block::Misc13 => "Misc 13",
            Block::Misc18 => "Misc 18", Block::Misc19 => "Misc 19",
            Block::Sand03 => "Sand 03",
            Block::Snow01 => "Snow 01", Block::Snow03 => "Snow 03",
            Block::Snow05 => "Snow 05", Block::Snow08 => "Snow 08",
            Block::Stone02 => "Stone 02", Block::Stone03 => "Stone 03",
            Block::Stone06 => "Stone 06", Block::Stone08 => "Stone 08",
            Block::Stone10 => "Stone 10", Block::Stone11 => "Stone 11",
            Block::Stone12 => "Stone 12", Block::Stone13 => "Stone 13",
            Block::Stone14 => "Stone 14", Block::Stone15 => "Stone 15",
            Block::Stone16 => "Stone 16", Block::Stone17 => "Stone 17",
            Block::Stone18 => "Stone 18", Block::Stone19 => "Stone 19",
            Block::Stone20 => "Stone 20", Block::Stone21 => "Stone 21",
            Block::Stone22 => "Stone 22", Block::Stone23 => "Stone 23",
            Block::Stone24 => "Stone 24", Block::Stone25 => "Stone 25",
            Block::Stone26 => "Stone 26", Block::Stone27 => "Stone 27",
            Block::Stone28 => "Stone 28", Block::Stone29 => "Stone 29",
            Block::Stone30 => "Stone 30", Block::Stone31 => "Stone 31",
            Block::Stone32 => "Stone 32", Block::Stone33 => "Stone 33",
            Block::Stone34 => "Stone 34", Block::Stone35 => "Stone 35",
            Block::Stone36 => "Stone 36", Block::Stone37 => "Stone 37",
            Block::Stone38 => "Stone 38", Block::Stone39 => "Stone 39",
            Block::Stone40 => "Stone 40", Block::Stone41 => "Stone 41",
            Block::Stone42 => "Stone 42", Block::Stone43 => "Stone 43",
            Block::Stone44 => "Stone 44", Block::Stone45 => "Stone 45",
            Block::Stone46 => "Stone 46", Block::Stone47 => "Stone 47",
            Block::Stone48 => "Stone 48", Block::Stone51 => "Stone 51", Block::Stone52 => "Stone 52",
            Block::Tile01 => "Tile 01", Block::Tile02 => "Tile 02", Block::Tile03 => "Tile 03",
            Block::Tile04 => "Tile 04", Block::Tile05 => "Tile 05",
            Block::Wood01 => "Wood 01", Block::Wood02 => "Wood 02", Block::Wood03 => "Wood 03",
            Block::Wood04 => "Wood 04", Block::Wood05 => "Wood 05", Block::Wood06 => "Wood 06",
            Block::Wood07 => "Wood 07", Block::Wood08 => "Wood 08", Block::Wood09 => "Wood 09",
            Block::Wood10 => "Wood 10", Block::Wood11 => "Wood 11", Block::Wood12 => "Wood 12",
            Block::Wood14 => "Wood 14", Block::Wood15 => "Wood 15", Block::Wood16 => "Wood 16",
            Block::Wood17 => "Wood 17", Block::Wood18 => "Wood 18", Block::Wood19 => "Wood 19",
            Block::Wood20 => "Wood 20", Block::Wood21 => "Wood 21", Block::Wood23 => "Wood 23",
        }
    }

    /// Returns all placeable block types (everything except Air and Water)
    pub fn all_placeable() -> Vec<Block> {
        (1..=179u8)
            .map(Block::from)
            .filter(|b| !matches!(b, Block::Air | Block::Water))
            .collect()
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
            56 => Block::Bark03, 57 => Block::Bark05, 58 => Block::Bark06,
            59 => Block::Bark08, 60 => Block::Bark09, 61 => Block::Bark11,
            62 => Block::Bark12, 63 => Block::Bark13, 64 => Block::Bark14,
            65 => Block::Leaf01, 66 => Block::Leaf02, 67 => Block::Leaf03,
            68 => Block::Leaf04, 69 => Block::Leaf06, 70 => Block::Leaf07,
            71 => Block::Leaf08, 72 => Block::Leaf09, 73 => Block::Leaf10,
            74 => Block::Leaf11, 75 => Block::Leaf12, 76 => Block::Leaf13,
            77 => Block::Leaf14, 78 => Block::Leaf15, 79 => Block::Leaf16,
            80 => Block::Leaf18, 81 => Block::Leaf19, 82 => Block::Leaf20,
            83 => Block::Leaf21, 84 => Block::Leaf22, 85 => Block::Leaf23,
            86 => Block::Leaf25, 87 => Block::Leaf27, 88 => Block::Leaf28,
            89 => Block::Leaf29, 90 => Block::Leaf31, 91 => Block::Leaf32,
            92 => Block::Magma01,
            93 => Block::Misc05, 94 => Block::Misc06, 95 => Block::Misc07,
            96 => Block::Misc08, 97 => Block::Misc09, 98 => Block::Misc10,
            99 => Block::Misc11, 100 => Block::Misc12, 101 => Block::Misc13,
            102 => Block::Misc18, 103 => Block::Misc19,
            104 => Block::Sand03,
            105 => Block::Snow01, 106 => Block::Snow03, 107 => Block::Snow05, 108 => Block::Snow08,
            109 => Block::Stone02, 110 => Block::Stone03, 111 => Block::Stone06,
            112 => Block::Stone08, 113 => Block::Stone10, 114 => Block::Stone11,
            115 => Block::Stone12, 116 => Block::Stone13, 117 => Block::Stone14,
            118 => Block::Stone15, 119 => Block::Stone16, 120 => Block::Stone17,
            121 => Block::Stone18, 122 => Block::Stone19, 123 => Block::Stone20,
            124 => Block::Stone21, 125 => Block::Stone22, 126 => Block::Stone23,
            127 => Block::Stone24, 128 => Block::Stone25, 129 => Block::Stone26,
            130 => Block::Stone27, 131 => Block::Stone28, 132 => Block::Stone29,
            133 => Block::Stone30, 134 => Block::Stone31, 135 => Block::Stone32,
            136 => Block::Stone33, 137 => Block::Stone34, 138 => Block::Stone35,
            139 => Block::Stone36, 140 => Block::Stone37, 141 => Block::Stone38,
            142 => Block::Stone39, 143 => Block::Stone40, 144 => Block::Stone41,
            145 => Block::Stone42, 146 => Block::Stone43, 147 => Block::Stone44,
            148 => Block::Stone45, 149 => Block::Stone46, 150 => Block::Stone47,
            151 => Block::Stone48, 152 => Block::Stone51, 153 => Block::Stone52,
            154 => Block::Tile01, 155 => Block::Tile02, 156 => Block::Tile03,
            157 => Block::Tile04, 158 => Block::Tile05,
            159 => Block::Wood01, 160 => Block::Wood02, 161 => Block::Wood03,
            162 => Block::Wood04, 163 => Block::Wood05, 164 => Block::Wood06,
            165 => Block::Wood07, 166 => Block::Wood08, 167 => Block::Wood09,
            168 => Block::Wood10, 169 => Block::Wood11, 170 => Block::Wood12,
            171 => Block::Wood14, 172 => Block::Wood15, 173 => Block::Wood16,
            174 => Block::Wood17, 175 => Block::Wood18, 176 => Block::Wood19,
            177 => Block::Wood20, 178 => Block::Wood21, 179 => Block::Wood23,
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
    /// Optional grass tile data - only allocated if chunk has grass-capable surface
    grass: Option<Box<ChunkGrassData>>,
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
            grass: None,
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

    // ============ GRASS TILE METHODS ============

    /// Get reference to grass data if present
    #[inline]
    pub fn grass_data(&self) -> Option<&ChunkGrassData> {
        self.grass.as_deref()
    }

    /// Get mutable reference to grass data if present
    #[inline]
    pub fn grass_data_mut(&mut self) -> Option<&mut ChunkGrassData> {
        self.grass.as_deref_mut()
    }

    /// Ensure grass data is allocated and return mutable reference
    pub fn ensure_grass_data(&mut self) -> &mut ChunkGrassData {
        if self.grass.is_none() {
            self.grass = Some(Box::new(ChunkGrassData::new()));
        }
        self.grass.as_deref_mut().unwrap()
    }

    /// Ensure grass data with capacity hint
    pub fn ensure_grass_data_with_capacity(&mut self, capacity: usize) -> &mut ChunkGrassData {
        if self.grass.is_none() {
            self.grass = Some(Box::new(ChunkGrassData::with_capacity(capacity)));
        }
        self.grass.as_deref_mut().unwrap()
    }

    /// Add a grass tile (allocates grass data if needed)
    pub fn add_grass_tile(&mut self, tile: GrassTile) {
        self.ensure_grass_data().add(tile);
    }

    /// Clear all grass data from this chunk
    pub fn clear_grass(&mut self) {
        self.grass = None;
    }

    /// Iterate over all grass tiles (empty iterator if no grass)
    pub fn iter_grass_tiles(&self) -> impl Iterator<Item = &GrassTile> {
        self.grass.iter().flat_map(|g| g.iter())
    }

    /// Get grass tiles as slice (empty slice if no grass)
    pub fn grass_tiles(&self) -> &[GrassTile] {
        self.grass.as_ref().map(|g| g.tiles()).unwrap_or(&[])
    }

    /// Check if chunk has any grass tiles
    pub fn has_grass(&self) -> bool {
        self.grass.as_ref().map(|g| !g.is_empty()).unwrap_or(false)
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
        use crate::world::block_registry::registry;
        let reg = registry();

        // Grass: top=misc_03, bottom=misc_16, sides=misc_04
        let grass = Block::Grass.texture_indices();
        assert_eq!(grass[0], reg.texture_index("misc_03"), "Grass top");
        assert_eq!(grass[1], reg.texture_index("misc_16"), "Grass bottom");
        assert_eq!(grass[2], reg.texture_index("misc_04"), "Grass side");

        // Stone: all faces = stone_05
        let stone = Block::Stone.texture_indices();
        let stone_idx = reg.texture_index("stone_05");
        assert_eq!(stone, [stone_idx; 6], "Stone texture indices incorrect");

        // All 6 faces returned
        assert_eq!(grass.len(), 6);
        assert_eq!(stone.len(), 6);
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

        // Chest
        assert_eq!(Block::from(55), Block::Chest);

        // Texture catalog blocks
        assert_eq!(Block::from(56), Block::Bark03);
        assert_eq!(Block::from(179), Block::Wood23);

        // Invalid values default to Air
        assert_eq!(Block::from(180), Block::Air);
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

    // === Block hardness & mining tests ===

    #[test]
    fn test_block_hardness_air_water_zero() {
        assert_eq!(Block::Air.hardness(), 0.0);
        assert_eq!(Block::Water.hardness(), 0.0);
    }

    #[test]
    fn test_block_hardness_bedrock_unbreakable() {
        assert!(Block::Bedrock.hardness() < 0.0); // Negative = unbreakable
    }

    #[test]
    fn test_block_hardness_ordering() {
        // Dirt should be softer than stone
        assert!(Block::Dirt.hardness() < Block::Stone.hardness());
        // Stone should be softer than ores
        assert!(Block::Stone.hardness() <= Block::IronOre.hardness());
        // Diamond ore should be among the hardest
        assert!(Block::DiamondOre.hardness() >= Block::Stone.hardness());
    }

    #[test]
    fn test_block_hardness_all_positive_or_special() {
        // Solid breakable blocks should have positive hardness
        let test_blocks = [
            Block::Dirt, Block::Grass, Block::Stone, Block::Sand,
            Block::Wood, Block::Leaves, Block::CoalOre,
            Block::IronOre, Block::GoldOre, Block::DiamondOre,
        ];
        for block in test_blocks {
            assert!(block.hardness() > 0.0, "{:?} should have positive hardness", block);
        }
        // Instant-break blocks have zero hardness
        assert_eq!(Block::Torch.hardness(), 0.0);
        assert_eq!(Block::DeadBush.hardness(), 0.0);
    }

    #[test]
    fn test_block_preferred_tool() {
        use crate::inventory::item::ToolKind;
        // Dirt/sand prefer shovel
        assert_eq!(Block::Dirt.preferred_tool(), Some(ToolKind::Shovel));
        assert_eq!(Block::Sand.preferred_tool(), Some(ToolKind::Shovel));
        // Stone/ores prefer pickaxe
        assert_eq!(Block::Stone.preferred_tool(), Some(ToolKind::Pickaxe));
        assert_eq!(Block::IronOre.preferred_tool(), Some(ToolKind::Pickaxe));
        // Wood prefers axe
        assert_eq!(Block::Wood.preferred_tool(), Some(ToolKind::Axe));
        // Air has no preferred tool
        assert_eq!(Block::Air.preferred_tool(), None);
    }
}
