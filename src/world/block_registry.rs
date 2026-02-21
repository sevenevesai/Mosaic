//! Block and texture registry — single source of truth for block properties and texture mappings.
//!
//! Call `init_registry()` early in App::new() before any rendering or world generation.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::path::{Path, PathBuf};

/// Atlas grid size (16x16 = 256 slots)
pub const TILES_PER_ROW: u32 = 16;

/// Directory containing block textures
const BLOCK_TEXTURE_DIR: &str = "assets/blocks";

static REGISTRY: OnceLock<BlockRegistry> = OnceLock::new();

pub fn registry() -> &'static BlockRegistry {
    REGISTRY.get_or_init(BlockRegistry::new)
}

pub fn init_registry() {
    let reg = BlockRegistry::new();
    let _ = REGISTRY.set(reg);
}

/// Maps texture names to atlas indices and block IDs to texture index arrays.
pub struct BlockRegistry {
    /// texture name → atlas tile index
    name_to_index: HashMap<String, u32>,
    /// Ordered list of texture names (position = atlas slot)
    pub texture_names: Vec<String>,
    /// block_id → [atlas_index; 6] (top, bottom, front, back, left, right)
    block_textures: Vec<[u32; 6]>,
}

impl BlockRegistry {
    fn new() -> Self {
        // Scan assets/blocks/ for texture files
        let texture_dir = Self::find_texture_dir();
        let mut texture_names: Vec<String> = Vec::new();

        if let Some(dir) = &texture_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut files: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.ends_with(".png") {
                            Some(name.trim_end_matches(".png").to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                files.sort();
                texture_names = files;
            }
        }

        log::info!("Block registry: found {} textures in assets/blocks/", texture_names.len());

        // Build name → index map
        let mut name_to_index = HashMap::new();
        for (i, name) in texture_names.iter().enumerate() {
            name_to_index.insert(name.clone(), i as u32);
        }

        // Add procedural texture slots at end
        let torch_idx = texture_names.len() as u32;
        name_to_index.insert("_torch".to_string(), torch_idx);
        texture_names.push("_torch".to_string());

        let water_idx = texture_names.len() as u32;
        name_to_index.insert("_water".to_string(), water_idx);
        texture_names.push("_water".to_string());

        // Build per-block texture index arrays
        let block_textures = Self::build_block_textures(&name_to_index);

        Self {
            name_to_index,
            texture_names,
            block_textures,
        }
    }

    fn find_texture_dir() -> Option<PathBuf> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        let candidates = [
            Path::new(BLOCK_TEXTURE_DIR).to_path_buf(),
            exe_dir.join(BLOCK_TEXTURE_DIR),
            exe_dir.join("..").join("..").join("..").join(BLOCK_TEXTURE_DIR),
        ];

        candidates.into_iter().find(|p| p.exists())
    }

    /// Resolve a texture name to atlas index. Returns 0 if not found.
    pub fn texture_index(&self, name: &str) -> u32 {
        *self.name_to_index.get(name).unwrap_or_else(|| {
            log::warn!("Unknown texture name '{}', using index 0", name);
            &0
        })
    }

    /// Get precomputed [top, bottom, front, back, left, right] atlas indices for a block.
    #[inline]
    pub fn block_texture_indices(&self, block_id: u8) -> [u32; 6] {
        self.block_textures.get(block_id as usize).copied().unwrap_or([0; 6])
    }

    /// Number of textures (including procedural slots).
    pub fn texture_count(&self) -> usize {
        self.texture_names.len()
    }

    /// Get the texture directory path (for atlas loader).
    pub fn texture_dir() -> Option<PathBuf> {
        Self::find_texture_dir()
    }

    /// Whether a texture name is procedural (starts with '_').
    pub fn is_procedural(name: &str) -> bool {
        name.starts_with('_')
    }

    /// Build the block_id → texture indices table.
    /// Uses texture names so we're decoupled from file ordering.
    fn build_block_textures(names: &HashMap<String, u32>) -> Vec<[u32; 6]> {
        let r = |name: &str| -> u32 {
            *names.get(name).unwrap_or_else(|| {
                log::warn!("Block texture '{}' not found in registry", name);
                &0
            })
        };

        // Helper: all 6 faces same texture
        let uniform = |name: &str| -> [u32; 6] { [r(name); 6] };

        // Helper: top/bottom/sides pattern (like grass)
        let top_bottom_sides = |top: &str, bottom: &str, side: &str| -> [u32; 6] {
            [r(top), r(bottom), r(side), r(side), r(side), r(side)]
        };

        // Block IDs 0-55, indexed by block discriminant
        let mut table: Vec<[u32; 6]> = vec![[0; 6]; 256];

        // Air (0) — not rendered, placeholder
        table[0] = uniform("stone_05");

        // Dirt (1)
        table[1] = uniform("misc_16");

        // Grass (2) — top=grass, bottom=dirt, sides=grass+mud
        table[2] = top_bottom_sides("misc_03", "misc_16", "misc_04");

        // Stone (3)
        table[3] = uniform("stone_05");

        // StoneDirt (4)
        table[4] = uniform("stone_04");

        // Brick (5)
        table[5] = uniform("stone_07");

        // Water (6)
        table[6] = uniform("_water");

        // Wood (7) — generic bark
        table[7] = uniform("bark_02");

        // Leaves (8)
        table[8] = uniform("leaf_17");

        // StoneRubble2 (9)
        table[9] = uniform("stone_50");

        // IronOre (10)
        table[10] = uniform("ore_07");

        // GoldOre (11)
        table[11] = uniform("ore_06");

        // DiamondOre (12)
        table[12] = uniform("ore_04");

        // RedstoneOre (13)
        table[13] = uniform("ore_11");

        // Bedrock (14) — dark stone
        table[14] = uniform("stone_09");

        // Torch (15) — procedural
        table[15] = uniform("_torch");

        // Sand (16)
        table[16] = uniform("sand_01");

        // CoalOre (17)
        table[17] = uniform("ore_02");

        // CopperOre (18)
        table[18] = uniform("ore_03");

        // EmeraldOre (19)
        table[19] = uniform("ore_05");

        // SapphireOre (20)
        table[20] = uniform("ore_12");

        // QuartzOre (21)
        table[21] = uniform("ore_10");

        // AmethystOre (22)
        table[22] = uniform("ore_01");

        // SilverOre (23)
        table[23] = uniform("ore_13");

        // TinOre (24)
        table[24] = uniform("ore_14");

        // NickelOre (25)
        table[25] = uniform("ore_09");

        // LeadOre (26)
        table[26] = uniform("ore_08");

        // TitaniumOre (27)
        table[27] = uniform("ore_15");

        // UraniumOre (28)
        table[28] = uniform("ore_16");

        // OakLog (29) — redwood-ish bark
        table[29] = uniform("bark_07");

        // BirchLog (30) — white birch
        table[30] = uniform("bark_01");

        // Sandstone (31)
        table[31] = uniform("sand_02");

        // RedSandstone (32)
        table[32] = uniform("misc_01");

        // SnowGrass (33) — snow top, dirt bottom, icy snow sides
        table[33] = top_bottom_sides("snow_04", "misc_16", "snow_06");

        // Snow (34)
        table[34] = uniform("snow_04");

        // Ice (35)
        table[35] = uniform("snow_07");

        // PackedIce (36)
        table[36] = uniform("snow_02");

        // MudBlock (37)
        table[37] = uniform("misc_16");

        // Peat (38)
        table[38] = uniform("misc_17");

        // JungleGrass (39) — same as grass for now
        table[39] = top_bottom_sides("misc_03", "misc_16", "misc_04");

        // JungleLeaves (40) — tropical
        table[40] = uniform("leaf_30");

        // Vines (41)
        table[41] = uniform("leaf_24");

        // Gravel (42)
        table[42] = uniform("stone_50");

        // MossyStone (43)
        table[43] = uniform("stone_53");

        // CrackedStone (44)
        table[44] = uniform("stone_49");

        // ClayStone (45)
        table[45] = uniform("stone_01");

        // Stalactite (46)
        table[46] = uniform("stone_05");

        // Stalagmite (47)
        table[47] = uniform("stone_05");

        // CrystalAmethyst (48)
        table[48] = uniform("misc_02");

        // CrystalQuartz (49)
        table[49] = uniform("ore_10");

        // Cactus (50) — top different
        table[50] = top_bottom_sides("misc_15", "misc_15", "misc_14");

        // DeadBush (51)
        table[51] = uniform("leaf_05");

        // PineLog (52)
        table[52] = uniform("bark_10");

        // JungleLog (53)
        table[53] = uniform("bark_04");

        // SpruceLeaves (54) — pine needles
        table[54] = uniform("leaf_26");

        // Chest (55) — wood planks (rendered as vox prop)
        table[55] = uniform("wood_13");

        table
    }
}
