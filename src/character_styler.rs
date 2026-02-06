//! Character Styler GUI
//!
//! Allows the player to customize their character's appearance by selecting
//! different body part meshes, textures, hair styles/colors, accessories, etc.
//! Accessed via the "Character" button in the ESC main menu.

use std::path::Path;

/// Color variant suffixes for textures (e.g., Hair001.png, Hair001b.png, ...).
const COLOR_SUFFIXES: &[&str] = &["", "b", "c", "d", "e", "f", "g", "h"];

// ============================================================================
// Asset Catalog - discovers available models and textures at startup
// ============================================================================

/// Sorted list of available asset numbers for a body part category.
#[derive(Clone, Debug)]
pub struct PartCatalog {
    /// Available model numbers (e.g., [1, 2, 3, 5, 8] for files that exist).
    pub models: Vec<u16>,
    /// Available texture numbers (may differ from model numbers).
    pub textures: Vec<u16>,
    /// For hair: number of color variants per style (indexed by position in models vec).
    pub color_variants: Vec<u8>,
}

/// Catalog of all discovered character assets.
#[derive(Clone, Debug)]
pub struct AssetCatalog {
    pub head: PartCatalog,
    pub hair: PartCatalog,
    /// Body/skin textures from Characters/ folder.
    pub skin: PartCatalog,
    pub chest: PartCatalog,
    pub root: PartCatalog,
    pub spine: PartCatalog,
    pub hip: PartCatalog,
    pub knee: PartCatalog,
    pub shoulder: PartCatalog,
    pub elbow: PartCatalog,
    pub accessories: PartCatalog,
}

impl AssetCatalog {
    /// Scan the assets/char/ directory tree and build the catalog.
    pub fn scan() -> Self {
        let models_dir = Path::new("assets/char/models");
        let textures_dir = Path::new("assets/char/textures");

        AssetCatalog {
            head: Self::scan_part(models_dir, "Head", textures_dir, "Head", false),
            hair: Self::scan_part(models_dir, "Hair", textures_dir, "Hair", true),
            skin: Self::scan_skin_textures(textures_dir),
            chest: Self::scan_part(models_dir, "Chest", textures_dir, "Characters", false),
            root: Self::scan_part(models_dir, "Root", textures_dir, "Characters", false),
            spine: Self::scan_part(models_dir, "Spine", textures_dir, "Characters", false),
            hip: Self::scan_part(models_dir, "Hip", textures_dir, "Characters", false),
            knee: Self::scan_part(models_dir, "Knee", textures_dir, "Characters", false),
            shoulder: Self::scan_part(models_dir, "Shoulder", textures_dir, "Characters", false),
            elbow: Self::scan_part(models_dir, "Elbow", textures_dir, "Characters", false),
            accessories: Self::scan_part(models_dir, "Accessories", textures_dir, "Accessories", false),
        }
    }

    fn scan_part(
        models_dir: &Path,
        model_folder: &str,
        textures_dir: &Path,
        texture_folder: &str,
        detect_colors: bool,
    ) -> PartCatalog {
        let models = Self::scan_models(models_dir, model_folder);
        let textures = Self::scan_textures(textures_dir, texture_folder);

        let color_variants = if detect_colors {
            // For each model, count how many color variants of its texture exist.
            models.iter().map(|&num| {
                let mut count: u8 = 0;
                for suffix in COLOR_SUFFIXES {
                    let filename = if suffix.is_empty() {
                        format!("{}{:03}.png", texture_folder, num)
                    } else {
                        format!("{}{:03}{}.png", texture_folder, num, suffix)
                    };
                    if textures_dir.join(texture_folder).join(&filename).exists() {
                        count += 1;
                    }
                }
                count.max(1) // At least 1 (base texture)
            }).collect()
        } else {
            vec![1; models.len()]
        };

        PartCatalog { models, textures, color_variants }
    }

    fn scan_models(models_dir: &Path, folder: &str) -> Vec<u16> {
        let dir = models_dir.join(folder);
        let mut nums = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".fbx") {
                    if let Some(num) = Self::parse_asset_number(&name, folder) {
                        nums.push(num);
                    }
                }
            }
        }
        nums.sort();
        nums.dedup();
        nums
    }

    fn scan_textures(textures_dir: &Path, folder: &str) -> Vec<u16> {
        let dir = textures_dir.join(folder);
        let mut nums = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".png") {
                    if let Some(num) = Self::parse_asset_number(&name, folder) {
                        nums.push(num);
                    }
                }
            }
        }
        nums.sort();
        nums.dedup();
        nums
    }

    /// Skin textures are the base Characters/NNN.png files (no suffix variants).
    fn scan_skin_textures(textures_dir: &Path) -> PartCatalog {
        let textures = Self::scan_textures(textures_dir, "Characters");
        let count = textures.len();
        PartCatalog {
            models: Vec::new(), // No models for skin-only category
            textures,
            color_variants: vec![1; count],
        }
    }

    /// Parse "Head003.fbx" -> 3, "Hair001b.png" -> 1 (ignoring suffix letters).
    fn parse_asset_number(filename: &str, prefix: &str) -> Option<u16> {
        let stem = filename.strip_suffix(".fbx")
            .or_else(|| filename.strip_suffix(".png"))?;
        let after_prefix = stem.strip_prefix(prefix)?;
        // Take only leading digits (ignore suffix letters like 'b', 'c')
        let digits: String = after_prefix.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }
}

// ============================================================================
// Character Style - the player's current appearance selections
// ============================================================================

/// Represents a complete character appearance configuration.
#[derive(Clone, Debug)]
pub struct CharacterStyle {
    pub head_model: u16,
    pub head_texture: u16,
    pub hair_model: u16,
    pub hair_color: u8,    // 0=base, 1=b, 2=c, etc.
    pub skin_texture: u16, // Characters/NNN.png number
    pub chest_model: u16,
    pub root_model: u16,   // Root + Spine share this number
    pub hip_model: u16,    // Hip + Knee share this number
    pub shoulder_model: u16, // Shoulder + Elbow share this number
    pub accessory_model: u16, // 0 = none
}

impl Default for CharacterStyle {
    fn default() -> Self {
        Self {
            head_model: 1,
            head_texture: 1,
            hair_model: 1,
            hair_color: 0,
            skin_texture: 20,
            chest_model: 20,
            root_model: 20,
            hip_model: 20,
            shoulder_model: 20,
            accessory_model: 0,
        }
    }
}

impl CharacterStyle {
    /// Build a Vec<OwnedBodyPart> from the current style selections.
    pub fn build_body_parts(&self) -> Vec<OwnedBodyPart> {
        let skin_tex = format!("Characters/{:03}.png", self.skin_texture);

        let hair_color_suffix = if self.hair_color == 0 {
            String::new()
        } else {
            String::from(COLOR_SUFFIXES.get(self.hair_color as usize).copied().unwrap_or(""))
        };
        let hair_tex = format!("Hair/Hair{:03}{}.png", self.hair_model, hair_color_suffix);
        let head_tex = format!("Head/Head{:03}.png", self.head_texture);

        let mut parts = vec![
            // Center/spine parts (indices 0-4)
            OwnedBodyPart {
                folder: "Root".into(),
                file: format!("Root{:03}.fbx", self.root_model),
                bone_names: vec!["Root_M".into(), "Root".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Spine".into(),
                file: format!("Spine{:03}.fbx", self.root_model),
                bone_names: vec!["Spine1_M".into(), "Spine".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Chest".into(),
                file: format!("Chest{:03}.fbx", self.chest_model),
                bone_names: vec!["Chest_M".into(), "Chest".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Head".into(),
                file: format!("Head{:03}.fbx", self.head_model),
                bone_names: vec!["Head_M".into(), "Head".into()],
                texture: Some(head_tex),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Hair".into(),
                file: format!("Hair{:03}.fbx", self.hair_model),
                bone_names: vec!["Head_M".into(), "Head".into()],
                texture: Some(hair_tex),
                mirror: false,
            },
            // Left side limbs (indices 5-8)
            OwnedBodyPart {
                folder: "Hip".into(),
                file: format!("Hip{:03}.fbx", self.hip_model),
                bone_names: vec!["Hip_L".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Knee".into(),
                file: format!("Knee{:03}.fbx", self.hip_model),
                bone_names: vec!["Knee_L".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Shoulder".into(),
                file: format!("Shoulder{:03}.fbx", self.shoulder_model),
                bone_names: vec!["Shoulder_L".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            OwnedBodyPart {
                folder: "Elbow".into(),
                file: format!("Elbow{:03}.fbx", self.shoulder_model),
                bone_names: vec!["Elbow_L".into()],
                texture: Some(skin_tex.clone()),
                mirror: false,
            },
            // Right side limbs (indices 9-12, mirrored)
            OwnedBodyPart {
                folder: "Hip".into(),
                file: format!("Hip{:03}.fbx", self.hip_model),
                bone_names: vec!["Hip_R".into()],
                texture: Some(skin_tex.clone()),
                mirror: true,
            },
            OwnedBodyPart {
                folder: "Knee".into(),
                file: format!("Knee{:03}.fbx", self.hip_model),
                bone_names: vec!["Knee_R".into()],
                texture: Some(skin_tex.clone()),
                mirror: true,
            },
            OwnedBodyPart {
                folder: "Shoulder".into(),
                file: format!("Shoulder{:03}.fbx", self.shoulder_model),
                bone_names: vec!["Shoulder_R".into()],
                texture: Some(skin_tex.clone()),
                mirror: true,
            },
            OwnedBodyPart {
                folder: "Elbow".into(),
                file: format!("Elbow{:03}.fbx", self.shoulder_model),
                bone_names: vec!["Elbow_R".into()],
                texture: Some(skin_tex),
                mirror: true,
            },
        ];

        // Add accessory if selected (index 13+)
        if self.accessory_model > 0 {
            let acc_tex = format!("Accessories/Accessories{:03}.png", self.accessory_model);
            parts.push(OwnedBodyPart {
                folder: "Accessories".into(),
                file: format!("Accessories{:03}.fbx", self.accessory_model),
                bone_names: vec!["Head_M".into(), "Head".into()],
                texture: Some(acc_tex),
                mirror: false,
            });
        }

        parts
    }

    /// Save style to a config file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let content = format!(
            "head_model={}\nhead_texture={}\nhair_model={}\nhair_color={}\n\
             skin_texture={}\nchest_model={}\nroot_model={}\nhip_model={}\n\
             shoulder_model={}\naccessory_model={}\n",
            self.head_model, self.head_texture, self.hair_model, self.hair_color,
            self.skin_texture, self.chest_model, self.root_model, self.hip_model,
            self.shoulder_model, self.accessory_model,
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }

    /// Load style from a config file.
    pub fn load(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let mut style = Self::default();
        for line in content.lines() {
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 { continue; }
            let key = parts[0].trim();
            let val = parts[1].trim();
            match key {
                "head_model" => { style.head_model = val.parse().unwrap_or(style.head_model); }
                "head_texture" => { style.head_texture = val.parse().unwrap_or(style.head_texture); }
                "hair_model" => { style.hair_model = val.parse().unwrap_or(style.hair_model); }
                "hair_color" => { style.hair_color = val.parse().unwrap_or(style.hair_color); }
                "skin_texture" => { style.skin_texture = val.parse().unwrap_or(style.skin_texture); }
                "chest_model" => { style.chest_model = val.parse().unwrap_or(style.chest_model); }
                "root_model" => { style.root_model = val.parse().unwrap_or(style.root_model); }
                "hip_model" => { style.hip_model = val.parse().unwrap_or(style.hip_model); }
                "shoulder_model" => { style.shoulder_model = val.parse().unwrap_or(style.shoulder_model); }
                "accessory_model" => { style.accessory_model = val.parse().unwrap_or(style.accessory_model); }
                _ => {}
            }
        }
        Some(style)
    }
}

// ============================================================================
// Owned body part (for dynamic body part generation)
// ============================================================================

/// Like loader::BodyPart but with owned strings for dynamic construction.
pub struct OwnedBodyPart {
    pub folder: String,
    pub file: String,
    pub bone_names: Vec<String>,
    pub texture: Option<String>,
    pub mirror: bool,
}

// ============================================================================
// Styler UI categories
// ============================================================================

/// Categories shown as tabs in the styler UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StylerTab {
    Head = 0,
    Hair = 1,
    Skin = 2,
    Chest = 3,
    Waist = 4,
    Legs = 5,
    Arms = 6,
    Extras = 7,
}

impl StylerTab {
    pub const ALL: &[StylerTab] = &[
        StylerTab::Head, StylerTab::Hair, StylerTab::Skin, StylerTab::Chest,
        StylerTab::Waist, StylerTab::Legs, StylerTab::Arms, StylerTab::Extras,
    ];

    pub fn label(self) -> &'static str {
        match self {
            StylerTab::Head => "Head",
            StylerTab::Hair => "Hair",
            StylerTab::Skin => "Skin",
            StylerTab::Chest => "Chest",
            StylerTab::Waist => "Waist",
            StylerTab::Legs => "Legs",
            StylerTab::Arms => "Arms",
            StylerTab::Extras => "Extras",
        }
    }

    /// Whether this category has a secondary row (color/texture variant).
    pub fn has_secondary(self) -> bool {
        matches!(self, StylerTab::Head | StylerTab::Hair)
    }
}

// ============================================================================
// Styler UI state
// ============================================================================

/// Full state for the character styler GUI.
pub struct StylerUI {
    pub open: bool,
    pub selected_tab: usize,
    pub orbit_yaw: f32,
    pub dragging: bool,
    pub drag_start_x: f32,
    pub last_mouse_x: f32,
    pub last_mouse_y: f32,
    /// Working copy of the style (applied to character in real-time).
    pub style: CharacterStyle,
    /// Saved copy to revert on cancel.
    pub saved_style: CharacterStyle,
    /// Asset catalog reference.
    pub catalog: AssetCatalog,
    /// Whether style changed and needs remesh.
    pub dirty: bool,
    /// Screen dimensions for hit testing.
    pub screen_width: f32,
    pub screen_height: f32,
    /// Hovered element for visual feedback.
    pub hovered: HoverTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HoverTarget {
    #[default]
    None,
    Tab(usize),
    PrevArrow,
    NextArrow,
    PrevSecondary,
    NextSecondary,
    DoneButton,
}

impl StylerUI {
    pub fn new(catalog: AssetCatalog) -> Self {
        let style = CharacterStyle::default();
        Self {
            open: false,
            selected_tab: 0,
            orbit_yaw: 0.0, // Camera at +Z, character faces +Z toward camera
            dragging: false,
            drag_start_x: 0.0,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            saved_style: style.clone(),
            style,
            catalog,
            dirty: false,
            screen_width: 1280.0,
            screen_height: 720.0,
            hovered: HoverTarget::None,
        }
    }

    pub fn open(&mut self) {
        self.open = true;
        self.saved_style = self.style.clone();
        self.dirty = false;
        self.dragging = false;
        self.orbit_yaw = 0.0;
    }

    /// Close and save changes.
    pub fn close_save(&mut self) {
        self.open = false;
        self.saved_style = self.style.clone();
    }

    /// Close and revert to saved style.
    pub fn close_revert(&mut self) -> bool {
        let changed = self.styles_differ(&self.style, &self.saved_style);
        self.style = self.saved_style.clone();
        self.open = false;
        if changed {
            self.dirty = true;
        }
        changed
    }

    fn styles_differ(&self, a: &CharacterStyle, b: &CharacterStyle) -> bool {
        a.head_model != b.head_model || a.head_texture != b.head_texture
            || a.hair_model != b.hair_model || a.hair_color != b.hair_color
            || a.skin_texture != b.skin_texture || a.chest_model != b.chest_model
            || a.root_model != b.root_model || a.hip_model != b.hip_model
            || a.shoulder_model != b.shoulder_model || a.accessory_model != b.accessory_model
    }

    pub fn update_screen_size(&mut self, w: f32, h: f32) {
        self.screen_width = w;
        self.screen_height = h;
    }

    // ========================================================================
    // Input handling
    // ========================================================================

    /// Handle mouse movement. Returns true if styler consumed the event.
    pub fn handle_mouse_move(&mut self, x: f32, y: f32) -> bool {
        if !self.open { return false; }

        self.last_mouse_x = x;
        self.last_mouse_y = y;

        if self.dragging {
            let dx = x - self.drag_start_x;
            self.orbit_yaw += dx * 0.01;
            self.drag_start_x = x;
            return true;
        }

        // Update hover state
        self.hovered = self.hit_test(x, y);
        true
    }

    /// Handle mouse button press. Returns true if styler consumed the event.
    pub fn handle_mouse_press(&mut self, x: f32, y: f32) -> bool {
        if !self.open { return false; }

        let target = self.hit_test(x, y);
        match target {
            HoverTarget::Tab(idx) => {
                self.selected_tab = idx;
            }
            HoverTarget::PrevArrow => {
                self.cycle_primary(-1);
            }
            HoverTarget::NextArrow => {
                self.cycle_primary(1);
            }
            HoverTarget::PrevSecondary => {
                self.cycle_secondary(-1);
            }
            HoverTarget::NextSecondary => {
                self.cycle_secondary(1);
            }
            HoverTarget::DoneButton => {
                // Handled by app.rs checking for done action
            }
            HoverTarget::None => {
                // Start drag for orbit rotation
                self.dragging = true;
                self.drag_start_x = x;
            }
        }
        true
    }

    /// Handle mouse button release.
    pub fn handle_mouse_release(&mut self) -> bool {
        if !self.open { return false; }
        self.dragging = false;
        true
    }

    /// Returns true if the Done button was clicked on the last press.
    pub fn was_done_clicked(&self, x: f32, y: f32) -> bool {
        self.hit_test(x, y) == HoverTarget::DoneButton
    }

    /// Handle keyboard input. Returns true if consumed.
    pub fn handle_key(&mut self, key: winit::keyboard::KeyCode, pressed: bool) -> bool {
        use winit::keyboard::KeyCode;
        if !self.open || !pressed { return self.open; }

        match key {
            KeyCode::ArrowUp => {
                if self.selected_tab > 0 {
                    self.selected_tab -= 1;
                }
            }
            KeyCode::ArrowDown => {
                if self.selected_tab < StylerTab::ALL.len() - 1 {
                    self.selected_tab += 1;
                }
            }
            KeyCode::ArrowLeft => {
                self.cycle_primary(-1);
            }
            KeyCode::ArrowRight => {
                self.cycle_primary(1);
            }
            KeyCode::BracketLeft => {
                self.cycle_secondary(-1);
            }
            KeyCode::BracketRight => {
                self.cycle_secondary(1);
            }
            _ => {}
        }
        true
    }

    // ========================================================================
    // Selection cycling
    // ========================================================================

    fn current_tab(&self) -> StylerTab {
        StylerTab::ALL[self.selected_tab.min(StylerTab::ALL.len() - 1)]
    }

    fn cycle_primary(&mut self, dir: i32) {
        let tab = self.current_tab();
        match tab {
            StylerTab::Head => {
                self.style.head_model = Self::cycle_in_list(&self.catalog.head.models, self.style.head_model, dir);
                // Auto-match head texture to model number if it exists
                if self.catalog.head.textures.contains(&self.style.head_model) {
                    self.style.head_texture = self.style.head_model;
                }
            }
            StylerTab::Hair => {
                self.style.hair_model = Self::cycle_in_list(&self.catalog.hair.models, self.style.hair_model, dir);
                // Reset color variant to 0 when changing style
                self.style.hair_color = 0;
            }
            StylerTab::Skin => {
                self.style.skin_texture = Self::cycle_in_list(&self.catalog.skin.textures, self.style.skin_texture, dir);
            }
            StylerTab::Chest => {
                self.style.chest_model = Self::cycle_in_list(&self.catalog.chest.models, self.style.chest_model, dir);
            }
            StylerTab::Waist => {
                self.style.root_model = Self::cycle_in_list(&self.catalog.root.models, self.style.root_model, dir);
            }
            StylerTab::Legs => {
                self.style.hip_model = Self::cycle_in_list(&self.catalog.hip.models, self.style.hip_model, dir);
            }
            StylerTab::Arms => {
                self.style.shoulder_model = Self::cycle_in_list(&self.catalog.shoulder.models, self.style.shoulder_model, dir);
            }
            StylerTab::Extras => {
                let mut acc_list = vec![0u16]; // 0 = None
                acc_list.extend_from_slice(&self.catalog.accessories.models);
                self.style.accessory_model = Self::cycle_in_list(&acc_list, self.style.accessory_model, dir);
            }
        }
        self.dirty = true;
    }

    fn cycle_secondary(&mut self, dir: i32) {
        let tab = self.current_tab();
        match tab {
            StylerTab::Head => {
                self.style.head_texture = Self::cycle_in_list(&self.catalog.head.textures, self.style.head_texture, dir);
            }
            StylerTab::Hair => {
                // Cycle through color variants for current hair model
                let max_colors = self.hair_color_count();
                if max_colors > 0 {
                    let next = (self.style.hair_color as i32 + dir).rem_euclid(max_colors as i32);
                    self.style.hair_color = next as u8;
                }
            }
            _ => {}
        }
        self.dirty = true;
    }

    fn hair_color_count(&self) -> u8 {
        // Find the index of the current hair model in the catalog
        if let Some(idx) = self.catalog.hair.models.iter().position(|&m| m == self.style.hair_model) {
            self.catalog.hair.color_variants.get(idx).copied().unwrap_or(1)
        } else {
            1
        }
    }

    /// Cycle a value through a sorted list, wrapping around.
    fn cycle_in_list(list: &[u16], current: u16, dir: i32) -> u16 {
        if list.is_empty() { return current; }
        let pos = list.iter().position(|&v| v == current).unwrap_or(0);
        let next = (pos as i32 + dir).rem_euclid(list.len() as i32) as usize;
        list[next]
    }

    // ========================================================================
    // Hit testing for UI elements
    // ========================================================================

    fn hit_test(&self, mx: f32, my: f32) -> HoverTarget {
        let sw = self.screen_width;
        let sh = self.screen_height;

        // Layout constants (pixels from top-left)
        let tab_x = sw * 0.04;
        let tab_w = sw * 0.12;
        let tab_start_y = sh * 0.18;
        let tab_h = sh * 0.055;
        let tab_gap = sh * 0.01;

        // Check tab buttons
        for (i, _tab) in StylerTab::ALL.iter().enumerate() {
            let ty = tab_start_y + i as f32 * (tab_h + tab_gap);
            if mx >= tab_x && mx <= tab_x + tab_w && my >= ty && my <= ty + tab_h {
                return HoverTarget::Tab(i);
            }
        }

        // Selection row (centered, lower-mid area)
        // NDC -0.40 → screen_y = (1+0.40)/2 = 0.70
        let sel_y = sh * 0.68;
        let sel_h = sh * 0.05;
        let center_x = sw * 0.5;
        let arrow_w = sw * 0.04;

        // Prev arrow (left of center)
        let prev_x = center_x - sw * 0.18;
        if mx >= prev_x && mx <= prev_x + arrow_w && my >= sel_y && my <= sel_y + sel_h {
            return HoverTarget::PrevArrow;
        }

        // Next arrow (right of center)
        let next_x = center_x + sw * 0.14;
        if mx >= next_x && mx <= next_x + arrow_w && my >= sel_y && my <= sel_y + sel_h {
            return HoverTarget::NextArrow;
        }

        // Secondary row (if applicable)
        // NDC -0.50 → screen_y = (1+0.50)/2 = 0.75
        if self.current_tab().has_secondary() {
            let sec_y = sh * 0.73;
            let sec_h = sel_h;

            let prev_sec_x = center_x - sw * 0.14;
            if mx >= prev_sec_x && mx <= prev_sec_x + arrow_w && my >= sec_y && my <= sec_y + sec_h {
                return HoverTarget::PrevSecondary;
            }

            let next_sec_x = center_x + sw * 0.10;
            if mx >= next_sec_x && mx <= next_sec_x + arrow_w && my >= sec_y && my <= sec_y + sec_h {
                return HoverTarget::NextSecondary;
            }
        }

        // Done button (below selection rows)
        let done_w = sw * 0.10;
        let done_h = sh * 0.05;
        let done_x = center_x - done_w * 0.5;
        let done_y = sh * 0.80; // NDC -0.65 → screen 0.825, with some margin
        if mx >= done_x && mx <= done_x + done_w && my >= done_y && my <= done_y + done_h {
            return HoverTarget::DoneButton;
        }

        HoverTarget::None
    }

    // ========================================================================
    // UI rendering data (for debug_overlay to render)
    // ========================================================================

    /// Get the display string and position for the current primary selection.
    pub fn primary_display(&self) -> (String, usize, usize) {
        let tab = self.current_tab();
        match tab {
            StylerTab::Head => {
                let list = &self.catalog.head.models;
                let pos = list.iter().position(|&v| v == self.style.head_model).map(|p| p + 1).unwrap_or(0);
                (format!("Head{:03}", self.style.head_model), pos, list.len())
            }
            StylerTab::Hair => {
                let list = &self.catalog.hair.models;
                let pos = list.iter().position(|&v| v == self.style.hair_model).map(|p| p + 1).unwrap_or(0);
                (format!("Hair{:03}", self.style.hair_model), pos, list.len())
            }
            StylerTab::Skin => {
                let list = &self.catalog.skin.textures;
                let pos = list.iter().position(|&v| v == self.style.skin_texture).map(|p| p + 1).unwrap_or(0);
                (format!("Skin {:03}", self.style.skin_texture), pos, list.len())
            }
            StylerTab::Chest => {
                let list = &self.catalog.chest.models;
                let pos = list.iter().position(|&v| v == self.style.chest_model).map(|p| p + 1).unwrap_or(0);
                (format!("Chest{:03}", self.style.chest_model), pos, list.len())
            }
            StylerTab::Waist => {
                let list = &self.catalog.root.models;
                let pos = list.iter().position(|&v| v == self.style.root_model).map(|p| p + 1).unwrap_or(0);
                (format!("Waist{:03}", self.style.root_model), pos, list.len())
            }
            StylerTab::Legs => {
                let list = &self.catalog.hip.models;
                let pos = list.iter().position(|&v| v == self.style.hip_model).map(|p| p + 1).unwrap_or(0);
                (format!("Legs {:03}", self.style.hip_model), pos, list.len())
            }
            StylerTab::Arms => {
                let list = &self.catalog.shoulder.models;
                let pos = list.iter().position(|&v| v == self.style.shoulder_model).map(|p| p + 1).unwrap_or(0);
                (format!("Arms {:03}", self.style.shoulder_model), pos, list.len())
            }
            StylerTab::Extras => {
                if self.style.accessory_model == 0 {
                    let total = self.catalog.accessories.models.len() + 1; // +1 for None
                    ("None".to_string(), 1, total)
                } else {
                    let list = &self.catalog.accessories.models;
                    let pos = list.iter().position(|&v| v == self.style.accessory_model).map(|p| p + 2).unwrap_or(0);
                    let total = list.len() + 1;
                    (format!("Acc{:03}", self.style.accessory_model), pos, total)
                }
            }
        }
    }

    /// Get the display string for the secondary selection (if any).
    pub fn secondary_display(&self) -> Option<(String, usize, usize)> {
        let tab = self.current_tab();
        match tab {
            StylerTab::Head => {
                let list = &self.catalog.head.textures;
                let pos = list.iter().position(|&v| v == self.style.head_texture).map(|p| p + 1).unwrap_or(0);
                Some((format!("Face {:03}", self.style.head_texture), pos, list.len()))
            }
            StylerTab::Hair => {
                let max = self.hair_color_count();
                if max <= 1 { return None; }
                let color_name = match self.style.hair_color {
                    0 => "Base",
                    1 => "Alt 1",
                    2 => "Alt 2",
                    3 => "Alt 3",
                    4 => "Alt 4",
                    5 => "Alt 5",
                    6 => "Alt 6",
                    _ => "Alt 7",
                };
                Some((color_name.to_string(), self.style.hair_color as usize + 1, max as usize))
            }
            _ => None,
        }
    }

    /// Build overlay lines for rendering the styler UI text.
    pub fn overlay_data(&self) -> StylerOverlay {
        let (primary_name, primary_pos, primary_total) = self.primary_display();
        let secondary = self.secondary_display();

        StylerOverlay {
            tabs: StylerTab::ALL.iter().map(|t| {
                (t.label().to_string(), *t as usize == self.selected_tab)
            }).collect(),
            primary_label: format!("<  {}  ({}/{})  >", primary_name, primary_pos, primary_total),
            secondary_label: secondary.map(|(name, pos, total)| {
                format!("<  {}  ({}/{})  >", name, pos, total)
            }),
            hovered: self.hovered,
        }
    }
}

/// Data needed to render the styler UI overlay.
#[derive(Clone)]
pub struct StylerOverlay {
    /// Tab labels with selected flag.
    pub tabs: Vec<(String, bool)>,
    /// Primary selection row text.
    pub primary_label: String,
    /// Secondary selection row text (color/texture variant), if applicable.
    pub secondary_label: Option<String>,
    /// Currently hovered element.
    pub hovered: HoverTarget,
}
