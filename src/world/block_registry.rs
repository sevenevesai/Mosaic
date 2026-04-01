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

        // === Texture catalog blocks (56-179) ===
        table[56] = uniform("bark_03"); table[57] = uniform("bark_05");
        table[58] = uniform("bark_06"); table[59] = uniform("bark_08");
        table[60] = uniform("bark_09"); table[61] = uniform("bark_11");
        table[62] = uniform("bark_12"); table[63] = uniform("bark_13");
        table[64] = uniform("bark_14");
        table[65] = uniform("leaf_01"); table[66] = uniform("leaf_02");
        table[67] = uniform("leaf_03"); table[68] = uniform("leaf_04");
        table[69] = uniform("leaf_06"); table[70] = uniform("leaf_07");
        table[71] = uniform("leaf_08"); table[72] = uniform("leaf_09");
        table[73] = uniform("leaf_10"); table[74] = uniform("leaf_11");
        table[75] = uniform("leaf_12"); table[76] = uniform("leaf_13");
        table[77] = uniform("leaf_14"); table[78] = uniform("leaf_15");
        table[79] = uniform("leaf_16"); table[80] = uniform("leaf_18");
        table[81] = uniform("leaf_19"); table[82] = uniform("leaf_20");
        table[83] = uniform("leaf_21"); table[84] = uniform("leaf_22");
        table[85] = uniform("leaf_23"); table[86] = uniform("leaf_25");
        table[87] = uniform("leaf_27"); table[88] = uniform("leaf_28");
        table[89] = uniform("leaf_29"); table[90] = uniform("leaf_31");
        table[91] = uniform("leaf_32");
        table[92] = uniform("magma_01");
        table[93] = uniform("misc_05"); table[94] = uniform("misc_06");
        table[95] = uniform("misc_07"); table[96] = uniform("misc_08");
        table[97] = uniform("misc_09"); table[98] = uniform("misc_10");
        table[99] = uniform("misc_11"); table[100] = uniform("misc_12");
        table[101] = uniform("misc_13"); table[102] = uniform("misc_18");
        table[103] = uniform("misc_19");
        table[104] = uniform("sand_03");
        table[105] = uniform("snow_01"); table[106] = uniform("snow_03");
        table[107] = uniform("snow_05"); table[108] = uniform("snow_08");
        table[109] = uniform("stone_02"); table[110] = uniform("stone_03");
        table[111] = uniform("stone_06"); table[112] = uniform("stone_08");
        table[113] = uniform("stone_10"); table[114] = uniform("stone_11");
        table[115] = uniform("stone_12"); table[116] = uniform("stone_13");
        table[117] = uniform("stone_14"); table[118] = uniform("stone_15");
        table[119] = uniform("stone_16"); table[120] = uniform("stone_17");
        table[121] = uniform("stone_18"); table[122] = uniform("stone_19");
        table[123] = uniform("stone_20"); table[124] = uniform("stone_21");
        table[125] = uniform("stone_22"); table[126] = uniform("stone_23");
        table[127] = uniform("stone_24"); table[128] = uniform("stone_25");
        table[129] = uniform("stone_26"); table[130] = uniform("stone_27");
        table[131] = uniform("stone_28"); table[132] = uniform("stone_29");
        table[133] = uniform("stone_30"); table[134] = uniform("stone_31");
        table[135] = uniform("stone_32"); table[136] = uniform("stone_33");
        table[137] = uniform("stone_34"); table[138] = uniform("stone_35");
        table[139] = uniform("stone_36"); table[140] = uniform("stone_37");
        table[141] = uniform("stone_38"); table[142] = uniform("stone_39");
        table[143] = uniform("stone_40"); table[144] = uniform("stone_41");
        table[145] = uniform("stone_42"); table[146] = uniform("stone_43");
        table[147] = uniform("stone_44"); table[148] = uniform("stone_45");
        table[149] = uniform("stone_46"); table[150] = uniform("stone_47");
        table[151] = uniform("stone_48"); table[152] = uniform("stone_51");
        table[153] = uniform("stone_52");
        table[154] = uniform("tile_01"); table[155] = uniform("tile_02");
        table[156] = uniform("tile_03"); table[157] = uniform("tile_04");
        table[158] = uniform("tile_05");
        table[159] = uniform("wood_01"); table[160] = uniform("wood_02");
        table[161] = uniform("wood_03"); table[162] = uniform("wood_04");
        table[163] = uniform("wood_05"); table[164] = uniform("wood_06");
        table[165] = uniform("wood_07"); table[166] = uniform("wood_08");
        table[167] = uniform("wood_09"); table[168] = uniform("wood_10");
        table[169] = uniform("wood_11"); table[170] = uniform("wood_12");
        table[171] = uniform("wood_14"); table[172] = uniform("wood_15");
        table[173] = uniform("wood_16"); table[174] = uniform("wood_17");
        table[175] = uniform("wood_18"); table[176] = uniform("wood_19");
        table[177] = uniform("wood_20"); table[178] = uniform("wood_21");
        table[179] = uniform("wood_23");

        table
    }
}
