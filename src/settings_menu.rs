//! Settings menu for in-game settings toggles
//!
//! Press ESC to open/close the menu. Click on settings to toggle them.

use crate::renderer::Renderer;

/// A menu item that can be toggled
pub struct MenuItem {
    pub label: &'static str,
    pub get_value: fn(&Renderer) -> String,
    pub toggle: fn(&mut Renderer),
}

/// Settings menu state
pub struct SettingsMenu {
    pub is_open: bool,
    pub hovered_index: Option<usize>,
    pub screen_width: f32,
    pub screen_height: f32,
    pub items: Vec<MenuItem>,
    // Mouse position tracking
    mouse_x: f32,
    mouse_y: f32,
}

impl SettingsMenu {
    // Character dimensions matching debug_overlay.rs (6x10 at 2x scale)
    const CHAR_WIDTH: f32 = 12.0;  // 6 * 2
    const CHAR_HEIGHT: f32 = 20.0; // 10 * 2
    const LINE_HEIGHT: f32 = 28.0; // Slightly more spacing for menu items

    // Menu layout
    const TITLE: &'static str = "SETTINGS";
    const FOOTER: &'static str = "ESC TO CLOSE";
    const TITLE_SPACING: f32 = 40.0;  // Extra space after title
    const FOOTER_SPACING: f32 = 30.0; // Extra space before footer

    pub fn new() -> Self {
        let items = vec![
            // === DISPLAY SETTINGS ===
            MenuItem {
                label: "DISPLAY MODE",
                get_value: |r| r.display_mode().as_str().to_string(),
                toggle: |r| { r.toggle_display_mode(); },
            },
            // === GRAPHICS SETTINGS ===
            MenuItem {
                label: "RENDER DISTANCE",
                get_value: |r| format!("{}", r.render_distance()),
                toggle: |r| { r.cycle_render_distance(); },
            },
            MenuItem {
                label: "SHADOW MODE",
                get_value: |r| r.shadow_mode_str().to_string(),
                toggle: |r| { r.toggle_shadow_mode(); },
            },
            MenuItem {
                label: "BLOOM",
                get_value: |r| {
                    if r.is_bloom_enabled() { "ON".to_string() } else { "OFF".to_string() }
                },
                toggle: |r| { r.toggle_bloom(); },
            },
            MenuItem {
                label: "VOLUMETRICS",
                get_value: |r| {
                    if r.volumetrics_enabled() { "ON".to_string() } else { "OFF".to_string() }
                },
                toggle: |r| { r.toggle_volumetrics(); },
            },
            // === GPU SETTINGS ===
            MenuItem {
                label: "GPU GI",
                get_value: |r| {
                    if r.is_gpu_voxel_fill_enabled() { "ON".to_string() } else { "OFF".to_string() }
                },
                toggle: |r| { r.toggle_gpu_voxel_fill(); },
            },
            MenuItem {
                label: "GPU CULLING",
                get_value: |r| {
                    if !r.is_mdi_available() {
                        "N/A".to_string()
                    } else if r.is_gpu_culling_enabled() {
                        "ON".to_string()
                    } else {
                        "OFF".to_string()
                    }
                },
                toggle: |r| { r.toggle_gpu_culling(); },
            },
            // === DEBUG SETTINGS ===
            MenuItem {
                label: "CYCLE SPEED",
                get_value: |r| {
                    let speed = r.cycle_speed();
                    if speed == 0.0 { "PAUSED".to_string() } else { format!("{}X", speed as i32) }
                },
                toggle: |r| { r.toggle_cycle_speed(); },
            },
            MenuItem {
                label: "DEBUG OVERLAY",
                get_value: |r| {
                    if r.is_debug_overlay_enabled() { "ON".to_string() } else { "OFF".to_string() }
                },
                toggle: |r| { r.toggle_debug_overlay(); },
            },
        ];

        Self {
            is_open: false,
            hovered_index: None,
            screen_width: 1280.0,
            screen_height: 720.0,
            items,
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    /// Update screen dimensions
    pub fn update_screen_size(&mut self, width: f32, height: f32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Update mouse position and recalculate hover
    pub fn update_mouse(&mut self, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
        self.update_hover();
    }

    /// Calculate which item (if any) the mouse is hovering over
    fn update_hover(&mut self) {
        if !self.is_open {
            self.hovered_index = None;
            return;
        }

        // Calculate menu layout
        let max_label_len = self.items.iter()
            .map(|item| item.label.len())
            .max()
            .unwrap_or(0);

        // Format: "  LABEL: [VALUE]  " with padding
        let item_width = (max_label_len + 15) as f32 * Self::CHAR_WIDTH;

        // Total menu height: title + spacing + items + spacing + footer
        let menu_height = Self::CHAR_HEIGHT + Self::TITLE_SPACING
            + (self.items.len() as f32 * Self::LINE_HEIGHT)
            + Self::FOOTER_SPACING + Self::CHAR_HEIGHT;

        // Center the menu
        let menu_left = (self.screen_width - item_width) / 2.0;
        let menu_top = (self.screen_height - menu_height) / 2.0;

        // Items start after title + spacing
        let items_top = menu_top + Self::CHAR_HEIGHT + Self::TITLE_SPACING;

        self.hovered_index = None;

        // Check each item
        for (i, _item) in self.items.iter().enumerate() {
            let item_top = items_top + i as f32 * Self::LINE_HEIGHT;
            let item_bottom = item_top + Self::LINE_HEIGHT;

            if self.mouse_y >= item_top && self.mouse_y < item_bottom
               && self.mouse_x >= menu_left && self.mouse_x < menu_left + item_width
            {
                self.hovered_index = Some(i);
                break;
            }
        }
    }

    /// Execute the toggle for the currently hovered item
    pub fn click(&mut self, renderer: &mut Renderer) {
        if let Some(idx) = self.hovered_index {
            if idx < self.items.len() {
                (self.items[idx].toggle)(renderer);
            }
        }
    }

    /// Get the maximum item width in characters (for rendering)
    pub fn max_item_width(&self) -> usize {
        self.items.iter()
            .map(|item| item.label.len() + 15) // "  LABEL: [VALUE]  "
            .max()
            .unwrap_or(20)
    }

    /// Generate display lines for the menu
    pub fn generate_lines(&self, renderer: &Renderer) -> Vec<(String, bool)> {
        let mut lines = Vec::new();

        // Title (not hoverable)
        lines.push((Self::TITLE.to_string(), false));
        lines.push((String::new(), false)); // Spacing line

        // Menu items
        for (i, item) in self.items.iter().enumerate() {
            let value = (item.get_value)(renderer);
            let is_hovered = self.hovered_index == Some(i);
            let prefix = if is_hovered { "> " } else { "  " };
            let line = format!("{}{}: [{}]", prefix, item.label, value);
            lines.push((line, is_hovered));
        }

        // Footer
        lines.push((String::new(), false)); // Spacing line
        lines.push((Self::FOOTER.to_string(), false));

        lines
    }

    /// Get menu center position in screen coordinates
    pub fn get_menu_center(&self) -> (f32, f32) {
        (self.screen_width / 2.0, self.screen_height / 2.0)
    }
}

impl Default for SettingsMenu {
    fn default() -> Self {
        Self::new()
    }
}
