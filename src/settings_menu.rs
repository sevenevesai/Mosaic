//! Settings menu with tabbed categories and appropriate controls
//!
//! Press ESC to open main menu. Click Settings to access tabbed settings.
//! Controls: Toggle, Slider, Cycle arrows.

use crate::renderer::Renderer;

/// Menu mode - main menu vs settings
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuMode {
    /// Main menu with buttons (Settings, Quit)
    Main,
    /// Settings with tabs
    Settings,
}

/// Main menu button types
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MainMenuButton {
    Settings,
    Character,
    SaveAndQuit,
}

impl MainMenuButton {
    pub fn label(&self) -> &'static str {
        match self {
            MainMenuButton::Settings => "SETTINGS",
            MainMenuButton::Character => "CHARACTER",
            MainMenuButton::SaveAndQuit => "SAVE & QUIT",
        }
    }

    pub fn all() -> &'static [MainMenuButton] {
        &[MainMenuButton::Settings, MainMenuButton::Character, MainMenuButton::SaveAndQuit]
    }
}

/// Result of a menu click action.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuAction {
    None,
    Quit,
    OpenCharacterStyler,
}

/// Data returned by generate_lines for rendering
pub struct MenuRenderData {
    /// Lines to render (text, is_hovered)
    pub lines: Vec<(String, bool)>,
    /// True if this is main menu (different layout)
    pub is_main_menu: bool,
    /// Tab line for settings menu (rendered separately at top)
    pub tabs_line: Option<String>,
    /// Whether up scroll arrow should show
    pub scroll_up: bool,
    /// Whether down scroll arrow should show
    pub scroll_down: bool,
}

/// Menu categories (tabs)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuTab {
    Display,
    Graphics,
    Audio,
    Debug,
}

impl MenuTab {
    pub fn as_str(&self) -> &'static str {
        match self {
            MenuTab::Display => "DISPLAY",
            MenuTab::Graphics => "GRAPHICS",
            MenuTab::Audio => "AUDIO",
            MenuTab::Debug => "DEBUG",
        }
    }

    pub fn all() -> &'static [MenuTab] {
        &[MenuTab::Display, MenuTab::Graphics, MenuTab::Audio, MenuTab::Debug]
    }
}

/// Control type determines how the setting renders and interacts
#[derive(Clone)]
pub enum ControlType {
    /// Simple ON/OFF toggle - click anywhere to flip
    Toggle,
    /// Discrete options with arrows: [< VALUE >]
    EnumCycle,
    /// Percentage slider with visual bar: [============]
    PercentSlider,
}

/// Interaction state (simplified - no dropdowns)
#[derive(Clone, Debug, PartialEq)]
pub enum ControlState {
    Idle,
}

/// A menu item with its control type and callbacks
pub struct MenuItem {
    pub label: &'static str,
    pub tab: MenuTab,
    pub control_type: ControlType,
    /// Get display value as string
    pub get_value: fn(&Renderer) -> String,
    /// Get current index for presets/enums (0-based)
    pub get_index: fn(&Renderer) -> usize,
    /// Get float value 0.0-1.0 for sliders
    pub get_float: fn(&Renderer) -> f32,
    /// Cycle to next value
    pub cycle_next: fn(&mut Renderer),
    /// Cycle to previous value
    pub cycle_prev: fn(&mut Renderer),
    /// Set by index (for dropdowns)
    pub set_index: fn(&mut Renderer, usize),
    /// Set float value (for sliders)
    pub set_float: fn(&mut Renderer, f32),
}

/// What part of a control was clicked
#[derive(Debug, Clone, PartialEq)]
pub enum ClickRegion {
    None,
    Button(usize),      // main menu button index
    Tab(usize),
    LeftArrow(usize),   // item index
    Center(usize),      // item index - for toggle or simple click
    RightArrow(usize),  // item index
    SliderBar(usize, f32), // item index, normalized position 0.0-1.0
    Outside,
}

/// Settings menu state
pub struct SettingsMenu {
    pub is_open: bool,
    pub menu_mode: MenuMode,
    pub active_tab: MenuTab,
    pub hovered_tab: Option<usize>,
    pub hovered_index: Option<usize>,
    pub hovered_button: Option<usize>,
    pub control_state: ControlState,
    pub items: Vec<MenuItem>,
    // Screen and mouse state
    pub screen_width: f32,
    pub screen_height: f32,
    mouse_x: f32,
    mouse_y: f32,
    // Cached layout values
    menu_left: f32,
    menu_top: f32,
    menu_width: f32,
    tab_positions: Vec<(f32, f32, f32)>, // (x, width, tab_idx as f32)
    items_start_y: f32,
    // Scrolling
    scroll_offset: usize,
    max_visible_items: usize,
    // Button positions for main menu
    button_positions: Vec<(f32, f32, f32, f32)>, // (x, y, width, height)
}

impl SettingsMenu {
    // Character dimensions - must match debug_overlay.rs exactly
    // debug_overlay uses: scale=2.0, CHAR_WIDTH=6, CHAR_HEIGHT=10
    // In NDC: char_w = (6 * 2) / screen_width * 2, char_h = (10 * 2) / screen_height * 2
    // In screen pixels: 12 x 20 per character
    pub const CHAR_WIDTH_PX: f32 = 12.0;
    pub const CHAR_HEIGHT_PX: f32 = 20.0;
    // debug_overlay uses line_height = char_h * 1.4, so in pixels:
    pub const LINE_HEIGHT_PX: f32 = 28.0; // 20 * 1.4 = 28

    // Layout constants (FOOTER now dynamic per mode)

    // Two-column layout (in characters)
    const LABEL_WIDTH: usize = 18;     // left column for labels
    const CONTROL_WIDTH: usize = 16;   // right column for controls
    const COLUMN_GAP: usize = 2;       // gap between columns
    const TOTAL_WIDTH: usize = Self::LABEL_WIDTH + Self::COLUMN_GAP + Self::CONTROL_WIDTH; // 36 chars
    const SLIDER_BAR_WIDTH: usize = 12; // chars for slider bar

    pub fn new() -> Self {
        let items = Self::create_menu_items();

        Self {
            is_open: false,
            menu_mode: MenuMode::Main,
            active_tab: MenuTab::Graphics, // Start on Graphics since it has the most items
            hovered_tab: None,
            hovered_index: None,
            hovered_button: None,
            control_state: ControlState::Idle,
            items,
            screen_width: 1280.0,
            screen_height: 720.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            menu_left: 0.0,
            menu_top: 0.0,
            menu_width: 0.0,
            tab_positions: Vec::new(),
            items_start_y: 0.0,
            scroll_offset: 0,
            max_visible_items: 10,
            button_positions: Vec::new(),
        }
    }

    fn create_menu_items() -> Vec<MenuItem> {
        vec![
            // === DISPLAY TAB ===
            MenuItem {
                label: "DISPLAY MODE",
                tab: MenuTab::Display,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.display_mode().as_str().to_string(),
                get_index: |r| r.display_mode() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_display_mode(); },
                cycle_prev: |r| { r.cycle_display_mode_prev(); },
                set_index: |r, i| { r.set_display_mode_index(i); },
                set_float: |_, _| {},
            },

            // === GRAPHICS TAB ===
            MenuItem {
                label: "RENDER DISTANCE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| format!("{}", r.render_distance()),
                get_index: |r| {
                    let d = r.render_distance();
                    match d {
                        8 => 0, 12 => 1, 16 => 2, 20 => 3, 24 => 4, 28 => 5, _ => 6,
                    }
                },
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_render_distance(); },
                cycle_prev: |r| { r.cycle_render_distance_prev(); },
                set_index: |r, i| {
                    let distances = [8, 12, 16, 20, 24, 28, 32];
                    if i < distances.len() {
                        r.set_render_distance(distances[i]);
                    }
                },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "LOD DISTANCE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.lod_distance_str(),
                get_index: |r| r.lod_distance_index(),
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_lod_distance(); },
                cycle_prev: |r| { r.cycle_lod_distance_prev(); },
                set_index: |r, i| { r.set_lod_distance_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "SHADOW QUALITY",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.shadow_quality_str().to_string(),
                get_index: |r| r.shadow_quality_index(),
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_shadow_quality_next(); },
                cycle_prev: |r| { r.cycle_shadow_quality_prev(); },
                set_index: |r, i| { r.set_shadow_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "SHADOW MODE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.shadow_mode_str().to_string(),
                get_index: |r| if r.shadow_mode_str() == "HARD" { 0 } else { 1 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_shadow_mode(); },
                cycle_prev: |r| { r.toggle_shadow_mode(); },
                set_index: |r, i| { r.set_shadow_mode(if i == 0 { 0.0 } else { 1.0 }); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "SHADOW UPDATE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| format!("{}", r.get_shadow_update_step() as i32),
                get_index: |r| {
                    let s = r.get_shadow_update_step() as i32;
                    match s { 2 => 0, 4 => 1, 8 => 2, 16 => 3, 32 => 4, _ => 5 }
                },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_shadow_update_step(); },
                cycle_prev: |r| { r.cycle_shadow_update_step_prev(); },
                set_index: |r, i| {
                    let steps = [2.0, 4.0, 8.0, 16.0, 32.0, 48.0];
                    if i < steps.len() { r.set_shadow_update_step(steps[i]); }
                },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "BLOOM",
                tab: MenuTab::Graphics,
                control_type: ControlType::Toggle,
                get_value: |r| if r.is_bloom_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.is_bloom_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_bloom(); },
                cycle_prev: |r| { r.toggle_bloom(); },
                set_index: |r, i| { r.set_bloom_enabled(i == 1); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "VOLUMETRICS",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.volumetrics_quality_str().to_string(),
                get_index: |r| r.volumetrics_quality_index(),
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_volumetrics_quality_next(); },
                cycle_prev: |r| { r.cycle_volumetrics_quality_prev(); },
                set_index: |r, i| { r.set_volumetrics_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "TORCH SHADOWS",
                tab: MenuTab::Graphics,
                control_type: ControlType::Toggle,
                get_value: |r| if r.is_torch_shadows_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.is_torch_shadows_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_torch_shadows(); },
                cycle_prev: |r| { r.toggle_torch_shadows(); },
                set_index: |r, i| { r.set_torch_shadows_enabled(i == 1); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "AMBIENT OCCLUSION",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.ao_quality_str().to_string(),
                get_index: |r| r.ao_quality_index(),
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_ao_quality_next(); },
                cycle_prev: |r| { r.cycle_ao_quality_prev(); },
                set_index: |r, i| { r.set_ao_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "ANTI-ALIASING",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.aa_quality().as_str().to_string(),
                get_index: |r| r.aa_quality() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_aa_quality(); },
                cycle_prev: |r| { r.cycle_aa_quality_prev(); },
                set_index: |r, i| { r.set_aa_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "DOF",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.dof_quality().as_str().to_string(),
                get_index: |r| r.dof_quality() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_dof_quality(); },
                cycle_prev: |r| { r.cycle_dof_quality_prev(); },
                set_index: |r, i| { r.set_dof_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "DOF FOCUS",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| format!("{}", r.dof_focus_distance() as i32),
                get_index: |r| {
                    let d = r.dof_focus_distance() as i32;
                    match d { 5 => 0, 10 => 1, 15 => 2, 20 => 3, 30 => 4, 50 => 5, 75 => 6, _ => 7 }
                },
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_dof_focus_distance(); },
                cycle_prev: |r| { r.cycle_dof_focus_distance_prev(); },
                set_index: |r, i| {
                    let distances = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
                    if i < distances.len() { r.set_dof_focus_distance(distances[i]); }
                },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "DOF BLUR",
                tab: MenuTab::Graphics,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{:.0}%", r.dof_blur_strength() * 100.0),
                get_index: |_| 0,
                get_float: |r| r.dof_blur_strength(),
                cycle_next: |r| { r.cycle_dof_blur_strength(); },
                cycle_prev: |r| { r.cycle_dof_blur_strength_prev(); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_dof_blur_strength(v); },
            },
            MenuItem {
                label: "DOF RANGE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| format!("{}", r.dof_focus_range() as i32),
                get_index: |r| {
                    let d = r.dof_focus_range() as i32;
                    match d { 5 => 0, 10 => 1, 15 => 2, 20 => 3, 30 => 4, 50 => 5, 75 => 6, _ => 7 }
                },
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_dof_focus_range(); },
                cycle_prev: |r| { r.cycle_dof_focus_range_prev(); },
                set_index: |r, i| {
                    let ranges = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
                    if i < ranges.len() { r.set_dof_focus_range(ranges[i]); }
                },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "WATER REFRACTION",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.refraction_quality().as_str().to_string(),
                get_index: |r| r.refraction_quality() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_refraction_quality(); },
                cycle_prev: |r| { r.cycle_refraction_quality_prev(); },
                set_index: |r, i| { r.set_refraction_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "CLOUDS",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.cloud_quality().as_str().to_string(),
                get_index: |r| r.cloud_quality() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_cloud_quality(); },
                cycle_prev: |r| { r.cycle_cloud_quality_prev(); },
                set_index: |r, i| { r.set_cloud_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "CLOUD COVERAGE",
                tab: MenuTab::Graphics,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{:.0}%", r.cloud_coverage() * 100.0),
                get_index: |_| 0,
                get_float: |r| r.cloud_coverage(),
                cycle_next: |r| { let c = (r.cloud_coverage() + 0.1).min(1.0); r.set_cloud_coverage(c); },
                cycle_prev: |r| { let c = (r.cloud_coverage() - 0.1).max(0.0); r.set_cloud_coverage(c); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_cloud_coverage(v); },
            },
            MenuItem {
                label: "CLOUD SPEED",
                tab: MenuTab::Graphics,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{:.0}%", r.cloud_speed() * 100.0),
                get_index: |_| 0,
                get_float: |r| r.cloud_speed(),
                cycle_next: |r| { let s = (r.cloud_speed() + 0.1).min(1.0); r.set_cloud_speed(s); },
                cycle_prev: |r| { let s = (r.cloud_speed() - 0.1).max(0.0); r.set_cloud_speed(s); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_cloud_speed(v); },
            },
            MenuItem {
                label: "CLOUD SHADOWS",
                tab: MenuTab::Graphics,
                control_type: ControlType::Toggle,
                get_value: |r| if r.cloud_shadows_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.cloud_shadows_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_cloud_shadows(); },
                cycle_prev: |r| { r.toggle_cloud_shadows(); },
                set_index: |r, i| { r.set_cloud_shadows_enabled(i == 1); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "SCENERY",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.scenery_quality().as_str().to_string(),
                get_index: |r| r.scenery_quality() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_scenery_quality(); },
                cycle_prev: |r| { r.cycle_scenery_quality_prev(); },
                set_index: |r, i| { r.set_scenery_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "GRASS",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.grass_quality().as_str().to_string(),
                get_index: |r| r.grass_quality() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_grass_quality(); },
                cycle_prev: |r| { r.cycle_grass_quality_prev(); },
                set_index: |r, i| { r.set_grass_quality_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "GRASS DISTANCE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.grass_distance().as_str().to_string(),
                get_index: |r| r.grass_distance() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_grass_distance(); },
                cycle_prev: |r| { r.cycle_grass_distance_prev(); },
                set_index: |r, i| { r.set_grass_distance_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "GRASS SHADOW",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.grass_shadow().as_str().to_string(),
                get_index: |r| r.grass_shadow() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_grass_shadow(); },
                cycle_prev: |r| { r.cycle_grass_shadow_prev(); },
                set_index: |r, i| { r.set_grass_shadow_index(i); },
                set_float: |_, _| {},
            },
            // === WATER FLOW SETTINGS ===
            MenuItem {
                label: "WATER FLOW",
                tab: MenuTab::Graphics,
                control_type: ControlType::Toggle,
                get_value: |r| if r.water_flow_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.water_flow_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_water_flow(); },
                cycle_prev: |r| { r.toggle_water_flow(); },
                set_index: |r, i| { r.set_water_flow_enabled(i == 1); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "FLOW DISTANCE",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.water_flow_distance().to_string(),
                get_index: |r| r.water_flow_distance_index(),
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_water_flow_distance(); },
                cycle_prev: |r| { r.cycle_water_flow_distance_prev(); },
                set_index: |r, i| { r.set_water_flow_distance_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "EXTENDED VIEW",
                tab: MenuTab::Graphics,
                control_type: ControlType::EnumCycle,
                get_value: |r| r.impostor_range().as_str().to_string(),
                get_index: |r| r.impostor_range() as usize,
                get_float: |_| 0.0,
                cycle_next: |r| { r.cycle_impostor_range(); },
                cycle_prev: |r| { r.cycle_impostor_range_prev(); },
                set_index: |r, i| { r.set_impostor_range_index(i); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "GPU GI",
                tab: MenuTab::Graphics,
                control_type: ControlType::Toggle,
                get_value: |r| if r.is_gpu_voxel_fill_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.is_gpu_voxel_fill_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_gpu_voxel_fill(); },
                cycle_prev: |r| { r.toggle_gpu_voxel_fill(); },
                set_index: |r, i| { r.set_gpu_voxel_fill_enabled(i == 1); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "GPU CULLING",
                tab: MenuTab::Graphics,
                control_type: ControlType::Toggle,
                get_value: |r| {
                    if !r.is_mdi_available() { "N/A".to_string() }
                    else if r.is_gpu_culling_enabled() { "ON".to_string() }
                    else { "OFF".to_string() }
                },
                get_index: |r| if r.is_gpu_culling_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_gpu_culling(); },
                cycle_prev: |r| { r.toggle_gpu_culling(); },
                set_index: |r, i| { if r.is_mdi_available() { r.set_gpu_culling_enabled(i == 1); } },
                set_float: |_, _| {},
            },

            // === AUDIO TAB ===
            MenuItem {
                label: "MASTER VOLUME",
                tab: MenuTab::Audio,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{}%", (r.audio_master() * 100.0).round() as i32),
                get_index: |_| 0,
                get_float: |r| r.audio_master(),
                cycle_next: |r| { r.cycle_audio_master(); },
                cycle_prev: |r| { r.cycle_audio_master_prev(); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_audio_master(v); },
            },
            MenuItem {
                label: "SFX VOLUME",
                tab: MenuTab::Audio,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{}%", (r.audio_sfx() * 100.0).round() as i32),
                get_index: |_| 0,
                get_float: |r| r.audio_sfx(),
                cycle_next: |r| { r.cycle_audio_sfx(); },
                cycle_prev: |r| { r.cycle_audio_sfx_prev(); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_audio_sfx(v); },
            },
            MenuItem {
                label: "AMBIENT VOLUME",
                tab: MenuTab::Audio,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{}%", (r.audio_ambient() * 100.0).round() as i32),
                get_index: |_| 0,
                get_float: |r| r.audio_ambient(),
                cycle_next: |r| { r.cycle_audio_ambient(); },
                cycle_prev: |r| { r.cycle_audio_ambient_prev(); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_audio_ambient(v); },
            },
            MenuItem {
                label: "MUSIC",
                tab: MenuTab::Audio,
                control_type: ControlType::Toggle,
                get_value: |r| if r.is_music_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.is_music_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_music(); },
                cycle_prev: |r| { r.toggle_music(); },
                set_index: |r, i| { r.set_music_enabled(i == 1); },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "MUSIC VOLUME",
                tab: MenuTab::Audio,
                control_type: ControlType::PercentSlider,
                get_value: |r| format!("{}%", (r.audio_music() * 100.0).round() as i32),
                get_index: |_| 0,
                get_float: |r| r.audio_music(),
                cycle_next: |r| { r.cycle_audio_music(); },
                cycle_prev: |r| { r.cycle_audio_music_prev(); },
                set_index: |_, _| {},
                set_float: |r, v| { r.set_audio_music(v); },
            },

            // === DEBUG TAB ===
            MenuItem {
                label: "CYCLE SPEED",
                tab: MenuTab::Debug,
                control_type: ControlType::EnumCycle,
                get_value: |r| {
                    let speed = r.cycle_speed();
                    if speed == 0.0 { "PAUSED".to_string() } else { format!("{}X", speed as i32) }
                },
                get_index: |r| {
                    let speed = r.cycle_speed() as i32;
                    match speed { 0 => 0, 1 => 1, 5 => 2, 20 => 3, 50 => 4, _ => 5 }
                },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_cycle_speed(); },
                cycle_prev: |r| { r.cycle_speed_prev(); },
                set_index: |r, i| {
                    let speeds = [0.0, 1.0, 5.0, 20.0, 50.0, 100.0];
                    if i < speeds.len() { r.set_cycle_speed(speeds[i]); }
                },
                set_float: |_, _| {},
            },
            MenuItem {
                label: "DEBUG OVERLAY",
                tab: MenuTab::Debug,
                control_type: ControlType::Toggle,
                get_value: |r| if r.is_debug_overlay_enabled() { "ON".to_string() } else { "OFF".to_string() },
                get_index: |r| if r.is_debug_overlay_enabled() { 1 } else { 0 },
                get_float: |_| 0.0,
                cycle_next: |r| { r.toggle_debug_overlay(); },
                cycle_prev: |r| { r.toggle_debug_overlay(); },
                set_index: |r, i| { r.set_debug_overlay_enabled(i == 1); },
                set_float: |_, _| {},
            },
        ]
    }

    /// Update screen dimensions
    pub fn update_screen_size(&mut self, width: f32, height: f32) {
        self.screen_width = width;
        self.screen_height = height;
        self.recalculate_layout();
    }

    /// Update mouse position and recalculate hover
    pub fn update_mouse(&mut self, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
        self.update_hover();
    }

    /// Recalculate menu layout based on screen size
    /// This must match exactly how debug_overlay.rs renders the menu
    fn recalculate_layout(&mut self) {
        // Menu width in pixels (all lines padded to TOTAL_WIDTH)
        self.menu_width = Self::TOTAL_WIDTH as f32 * Self::CHAR_WIDTH_PX;
        self.menu_left = (self.screen_width - self.menu_width) / 2.0;

        match self.menu_mode {
            MenuMode::Main => {
                self.recalculate_main_menu_layout();
            }
            MenuMode::Settings => {
                self.recalculate_settings_layout();
            }
        }
    }

    /// Convert NDC Y coordinate to screen pixel Y
    /// NDC: Y=1.0 is top, Y=-1.0 is bottom
    /// Screen: Y=0 is top, Y=screen_height is bottom
    fn ndc_y_to_pixel(&self, ndc_y: f32) -> f32 {
        self.screen_height * (1.0 - ndc_y) / 2.0
    }

    /// Layout for main menu (buttons centered on screen)
    fn recalculate_main_menu_layout(&mut self) {
        self.button_positions.clear();

        let buttons = MainMenuButton::all();
        let button_width = 18.0 * Self::CHAR_WIDTH_PX; // Match render width
        let button_height = Self::LINE_HEIGHT_PX * 1.4; // Match line_height in render
        let button_gap = Self::LINE_HEIGHT_PX * 1.4;

        // Main menu renders centered - title + space + buttons
        // In debug_overlay: total_height = lines.len() * line_height, start y = total_height/2
        // We have: title, space, button1, button2 = 4 lines
        let line_height_ndc = (Self::CHAR_HEIGHT_PX * 2.0 * 1.4) / self.screen_height * 2.0;
        let num_lines = 4.0; // title, space, 2 buttons
        let total_height_ndc = num_lines * line_height_ndc;

        // Start Y in NDC is total_height/2 - char_h/2, then decreases
        // Button 1 is at line 2 (after title, space), button 2 at line 3
        let char_h_ndc = (Self::CHAR_HEIGHT_PX * 2.0) / self.screen_height * 2.0;
        let start_y_ndc = total_height_ndc / 2.0 - char_h_ndc / 2.0;

        let start_x = (self.screen_width - button_width) / 2.0;

        for i in 0..buttons.len() {
            // Button i is at line (2 + i) from top
            let line_idx = 2 + i;
            let button_y_ndc = start_y_ndc - line_idx as f32 * line_height_ndc;
            let button_y_pixel = self.ndc_y_to_pixel(button_y_ndc + char_h_ndc); // top of button
            self.button_positions.push((start_x, button_y_pixel, button_width, button_height));
        }
    }

    /// Layout for settings menu (fixed tabs at top, scrollable content)
    fn recalculate_settings_layout(&mut self) {
        // Tabs rendered at NDC Y = 0.85, content at NDC Y = 0.70
        // Convert to pixel coordinates
        let tabs_y_ndc = 0.85;
        let content_start_y_ndc = 0.70;

        let char_h_ndc = (Self::CHAR_HEIGHT_PX * 2.0) / self.screen_height * 2.0;
        let line_height_ndc = char_h_ndc * 1.4;

        // Tabs position in pixels (top of tab line)
        let tabs_y_pixel = self.ndc_y_to_pixel(tabs_y_ndc);
        self.menu_top = tabs_y_pixel;

        // Calculate tab positions
        self.tab_positions.clear();
        let tabs = MenuTab::all();
        let tab_width = 10.0 * Self::CHAR_WIDTH_PX;
        let gap_width = 1.0 * Self::CHAR_WIDTH_PX;
        let total_tabs_width = tabs.len() as f32 * tab_width + (tabs.len() - 1) as f32 * gap_width;
        let tabs_start_x = (self.screen_width - total_tabs_width) / 2.0;

        for i in 0..tabs.len() {
            let x = tabs_start_x + i as f32 * (tab_width + gap_width);
            self.tab_positions.push((x, tab_width, i as f32));
        }

        // Content area starts at NDC 0.70
        self.items_start_y = self.ndc_y_to_pixel(content_start_y_ndc);

        // Calculate how many items can fit in the content area
        // Content goes from items_start_y down to near bottom
        let content_bottom = self.screen_height - Self::LINE_HEIGHT_PX * 2.0;
        let content_height = content_bottom - self.items_start_y;
        self.max_visible_items = (content_height / Self::LINE_HEIGHT_PX).floor() as usize;
        self.max_visible_items = self.max_visible_items.max(1); // At least 1

        // Clamp scroll offset
        let visible_count = self.visible_items().count();
        if visible_count <= self.max_visible_items {
            self.scroll_offset = 0;
        } else if self.scroll_offset > visible_count - self.max_visible_items {
            self.scroll_offset = visible_count - self.max_visible_items;
        }
    }

    /// Get items visible in the current tab
    pub fn visible_items(&self) -> impl Iterator<Item = (usize, &MenuItem)> {
        self.items.iter().enumerate().filter(|(_, item)| item.tab == self.active_tab)
    }

    /// Calculate which item (if any) the mouse is hovering over
    fn update_hover(&mut self) {
        if !self.is_open {
            self.hovered_index = None;
            self.hovered_tab = None;
            self.hovered_button = None;
            return;
        }

        match self.menu_mode {
            MenuMode::Main => {
                self.update_hover_main_menu();
            }
            MenuMode::Settings => {
                self.update_hover_settings();
            }
        }
    }

    /// Update hover state for main menu
    fn update_hover_main_menu(&mut self) {
        self.hovered_button = None;
        self.hovered_tab = None;
        self.hovered_index = None;

        for (i, &(x, y, w, h)) in self.button_positions.iter().enumerate() {
            if self.mouse_x >= x && self.mouse_x < x + w
                && self.mouse_y >= y && self.mouse_y < y + h
            {
                self.hovered_button = Some(i);
                break;
            }
        }
    }

    /// Update hover state for settings menu
    fn update_hover_settings(&mut self) {
        self.hovered_button = None;

        // Check tab hover - tabs are at fixed position (menu_top is pixel Y of tabs)
        self.hovered_tab = None;
        let tab_y_top = self.menu_top;
        let tab_y_bottom = tab_y_top + Self::LINE_HEIGHT_PX;
        if self.mouse_y >= tab_y_top && self.mouse_y < tab_y_bottom {
            for (i, &(x, width, _)) in self.tab_positions.iter().enumerate() {
                if self.mouse_x >= x && self.mouse_x < x + width {
                    self.hovered_tab = Some(i);
                    break;
                }
            }
        }

        // Check item hover (only visible items after scroll)
        self.hovered_index = None;
        let visible: Vec<_> = self.visible_items().collect();
        let visible_count = visible.len();

        // Account for scroll arrow if present
        let scroll_up_offset = if self.can_scroll_up() { 1 } else { 0 };

        for vis_idx in 0..self.max_visible_items.min(visible_count.saturating_sub(self.scroll_offset)) {
            let actual_idx = vis_idx + self.scroll_offset;
            if actual_idx >= visible_count {
                break;
            }

            let (global_idx, _) = visible[actual_idx];
            // Items start after potential scroll-up arrow
            let item_top = self.items_start_y + (vis_idx + scroll_up_offset) as f32 * Self::LINE_HEIGHT_PX;
            let item_bottom = item_top + Self::LINE_HEIGHT_PX;

            if self.mouse_y >= item_top && self.mouse_y < item_bottom
                && self.mouse_x >= self.menu_left && self.mouse_x < self.menu_left + self.menu_width
            {
                self.hovered_index = Some(global_idx);
                break;
            }
        }
    }

    /// Determine what region was clicked
    fn hit_test(&self, x: f32, y: f32) -> ClickRegion {
        match self.menu_mode {
            MenuMode::Main => self.hit_test_main_menu(x, y),
            MenuMode::Settings => self.hit_test_settings(x, y),
        }
    }

    /// Hit test for main menu buttons
    fn hit_test_main_menu(&self, x: f32, y: f32) -> ClickRegion {
        for (i, &(bx, by, bw, bh)) in self.button_positions.iter().enumerate() {
            if x >= bx && x < bx + bw && y >= by && y < by + bh {
                return ClickRegion::Button(i);
            }
        }
        ClickRegion::None
    }

    /// Hit test for settings menu
    fn hit_test_settings(&self, x: f32, y: f32) -> ClickRegion {
        // Check tabs first - tabs are at fixed position
        let tab_y_top = self.menu_top;
        let tab_y_bottom = tab_y_top + Self::LINE_HEIGHT_PX;
        if y >= tab_y_top && y < tab_y_bottom {
            for (i, &(tx, width, _)) in self.tab_positions.iter().enumerate() {
                if x >= tx && x < tx + width {
                    return ClickRegion::Tab(i);
                }
            }
        }

        // Check items - only visible items after scroll
        let visible: Vec<_> = self.visible_items().collect();
        let visible_count = visible.len();
        let control_col_start = self.menu_left + (Self::LABEL_WIDTH + Self::COLUMN_GAP) as f32 * Self::CHAR_WIDTH_PX;

        // Account for scroll arrow if present
        let scroll_up_offset = if self.can_scroll_up() { 1 } else { 0 };

        for vis_idx in 0..self.max_visible_items.min(visible_count.saturating_sub(self.scroll_offset)) {
            let actual_idx = vis_idx + self.scroll_offset;
            if actual_idx >= visible_count {
                break;
            }

            let (global_idx, item) = &visible[actual_idx];
            // Items start after potential scroll-up arrow
            let item_top = self.items_start_y + (vis_idx + scroll_up_offset) as f32 * Self::LINE_HEIGHT_PX;
            let item_bottom = item_top + Self::LINE_HEIGHT_PX;

            if y >= item_top && y < item_bottom {
                match &item.control_type {
                    ControlType::Toggle => {
                        // Toggle is ~7 chars wide: "[  ON ]"
                        let control_width = 7.0 * Self::CHAR_WIDTH_PX;
                        if x >= control_col_start && x < control_col_start + control_width {
                            return ClickRegion::Center(*global_idx);
                        }
                    }
                    ControlType::EnumCycle => {
                        // Control is 14 chars: "[< MEDIUM ^ >]" or "[<   24    >]"
                        let control_width = 14.0 * Self::CHAR_WIDTH_PX;
                        let control_end = control_col_start + control_width;

                        // Left arrow "[<" = first 2 chars, Right arrow ">]" = last 2 chars
                        let arrow_width = 2.0 * Self::CHAR_WIDTH_PX;
                        let left_arrow_end = control_col_start + arrow_width;
                        let right_arrow_start = control_end - arrow_width;

                        if x >= control_col_start && x < left_arrow_end {
                            return ClickRegion::LeftArrow(*global_idx);
                        } else if x >= right_arrow_start && x < control_end {
                            return ClickRegion::RightArrow(*global_idx);
                        } else if x >= left_arrow_end && x < right_arrow_start {
                            return ClickRegion::Center(*global_idx);
                        }
                    }
                    ControlType::PercentSlider => {
                        // Slider "[============]" = 14 chars (12 bar + 2 brackets)
                        let slider_width = (Self::SLIDER_BAR_WIDTH + 2) as f32 * Self::CHAR_WIDTH_PX;
                        let bar_start = control_col_start + Self::CHAR_WIDTH_PX; // After '['
                        let bar_end = bar_start + Self::SLIDER_BAR_WIDTH as f32 * Self::CHAR_WIDTH_PX;

                        if x >= bar_start && x < bar_end {
                            // Continuous value - no snapping
                            let normalized = ((x - bar_start) / (bar_end - bar_start)).clamp(0.0, 1.0);
                            return ClickRegion::SliderBar(*global_idx, normalized);
                        } else if x >= control_col_start && x < control_col_start + slider_width {
                            return ClickRegion::Center(*global_idx);
                        }
                    }
                }
            }
        }

        ClickRegion::None
    }

    /// Handle a click event. Returns true if quit was requested.
    pub fn click(&mut self, renderer: &mut Renderer) -> MenuAction {
        let region = self.hit_test(self.mouse_x, self.mouse_y);

        match region {
            ClickRegion::Button(idx) => {
                let buttons = MainMenuButton::all();
                if idx < buttons.len() {
                    match buttons[idx] {
                        MainMenuButton::Settings => {
                            self.menu_mode = MenuMode::Settings;
                            self.recalculate_layout();
                        }
                        MainMenuButton::Character => {
                            return MenuAction::OpenCharacterStyler;
                        }
                        MainMenuButton::SaveAndQuit => {
                            return MenuAction::Quit;
                        }
                    }
                }
            }
            ClickRegion::Tab(idx) => {
                let tabs = MenuTab::all();
                if idx < tabs.len() {
                    self.active_tab = tabs[idx];
                    self.scroll_offset = 0; // Reset scroll when changing tabs
                    self.control_state = ControlState::Idle;
                    self.recalculate_layout();
                }
            }
            ClickRegion::LeftArrow(item_idx) => {
                if let Some(item) = self.items.get(item_idx) {
                    (item.cycle_prev)(renderer);
                }
            }
            ClickRegion::RightArrow(item_idx) => {
                if let Some(item) = self.items.get(item_idx) {
                    (item.cycle_next)(renderer);
                }
            }
            ClickRegion::Center(item_idx) => {
                if let Some(item) = self.items.get(item_idx) {
                    match &item.control_type {
                        ControlType::Toggle => {
                            (item.cycle_next)(renderer);
                        }
                        ControlType::EnumCycle | ControlType::PercentSlider => {
                            (item.cycle_next)(renderer);
                        }
                    }
                }
            }
            ClickRegion::SliderBar(item_idx, normalized) => {
                if let Some(item) = self.items.get(item_idx) {
                    // Continuous value - no snapping for accurate clicking
                    (item.set_float)(renderer, normalized);
                }
            }
            ClickRegion::Outside | ClickRegion::None => {}
        }
        MenuAction::None
    }

    /// Handle scroll wheel input. Returns true if scroll was consumed.
    pub fn scroll(&mut self, delta: f32) -> bool {
        if self.menu_mode != MenuMode::Settings {
            return false;
        }

        let visible_count = self.visible_items().count();
        if visible_count <= self.max_visible_items {
            return false; // No scrolling needed
        }

        let max_scroll = visible_count - self.max_visible_items;

        if delta > 0.0 && self.scroll_offset > 0 {
            self.scroll_offset = self.scroll_offset.saturating_sub(1);
            return true;
        } else if delta < 0.0 && self.scroll_offset < max_scroll {
            self.scroll_offset = (self.scroll_offset + 1).min(max_scroll);
            return true;
        }
        false
    }

    /// Check if we can scroll up
    pub fn can_scroll_up(&self) -> bool {
        self.menu_mode == MenuMode::Settings && self.scroll_offset > 0
    }

    /// Check if we can scroll down
    pub fn can_scroll_down(&self) -> bool {
        if self.menu_mode != MenuMode::Settings {
            return false;
        }
        let visible_count = self.visible_items().count();
        visible_count > self.max_visible_items && self.scroll_offset < visible_count - self.max_visible_items
    }

    /// Get the maximum item width in characters (for rendering)
    pub fn max_item_width(&self) -> usize {
        Self::TOTAL_WIDTH
    }

    /// Pad a string to fixed width, centered
    fn center_pad(s: &str, width: usize) -> String {
        if s.len() >= width {
            s[..width].to_string()
        } else {
            let pad = width - s.len();
            let left = pad / 2;
            let right = pad - left;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
        }
    }

    /// Generate display lines for the menu (for rendering)
    /// Returns (lines, is_main_menu) - main menu has different rendering
    pub fn generate_lines(&self, renderer: &Renderer) -> MenuRenderData {
        match self.menu_mode {
            MenuMode::Main => self.generate_main_menu_lines(),
            MenuMode::Settings => self.generate_settings_lines(renderer),
        }
    }

    /// Generate lines for main menu
    fn generate_main_menu_lines(&self) -> MenuRenderData {
        let mut lines = Vec::new();

        // Title
        lines.push((Self::center_pad("MOSAIC", 20), false));
        lines.push((Self::center_pad("", 20), false));

        // Buttons
        let buttons = MainMenuButton::all();
        for (i, button) in buttons.iter().enumerate() {
            let is_hovered = self.hovered_button == Some(i);
            let label = button.label();
            // Wider buttons with brackets when hovered
            let text = if is_hovered {
                format!("[ {:^14} ]", label)
            } else {
                format!("  {:^14}  ", label)
            };
            lines.push((text, is_hovered));
        }

        MenuRenderData {
            lines,
            is_main_menu: true,
            tabs_line: None,
            scroll_up: false,
            scroll_down: false,
        }
    }

    /// Generate lines for settings menu
    fn generate_settings_lines(&self, renderer: &Renderer) -> MenuRenderData {
        let mut lines = Vec::new();

        // Tab bar - fixed 10-char width per tab
        let mut tab_line = String::new();
        for (i, tab) in MenuTab::all().iter().enumerate() {
            if i > 0 { tab_line.push(' '); }
            let name = tab.as_str();
            let is_active = *tab == self.active_tab;
            let is_hovered = self.hovered_tab == Some(i);

            if is_active {
                tab_line.push_str(&format!("[{:^8}]", name));
            } else if is_hovered {
                tab_line.push_str(&format!("[{:^8}]", name));
            } else {
                tab_line.push_str(&format!(" {:^8} ", name));
            }
        }

        // Scroll indicator (up arrow)
        if self.can_scroll_up() {
            lines.push((Self::center_pad("^", Self::TOTAL_WIDTH), false));
        }

        // Menu items for active tab - only visible after scroll
        let visible: Vec<_> = self.visible_items().collect();
        let visible_count = visible.len();

        for vis_idx in 0..self.max_visible_items.min(visible_count.saturating_sub(self.scroll_offset)) {
            let actual_idx = vis_idx + self.scroll_offset;
            if actual_idx >= visible_count {
                break;
            }

            let (global_idx, item) = &visible[actual_idx];
            let value = (item.get_value)(renderer);
            let is_hovered = self.hovered_index == Some(*global_idx);

            // Left column: label (left-aligned with hover indicator)
            let prefix = if is_hovered { "> " } else { "  " };
            let label_col = format!("{}{}", prefix, item.label);
            let label_padded = format!("{:width$}", label_col, width = Self::LABEL_WIDTH);

            // Right column: control
            let control_str = match &item.control_type {
                ControlType::Toggle => {
                    format!("[{:^5}]", value)
                }
                ControlType::EnumCycle => {
                    format!("[<{:>10} >]", value)
                }
                ControlType::PercentSlider => {
                    let pct = (item.get_float)(renderer);
                    let filled = (pct * Self::SLIDER_BAR_WIDTH as f32).round() as usize;
                    let empty = Self::SLIDER_BAR_WIDTH - filled;
                    let bar: String = "=".repeat(filled) + &".".repeat(empty);
                    format!("[{}]", bar)
                }
            };

            let gap = " ".repeat(Self::COLUMN_GAP);
            let line = format!("{}{}{}", label_padded, gap, control_str);
            let line_padded = format!("{:width$}", line, width = Self::TOTAL_WIDTH);
            lines.push((line_padded, is_hovered));
        }

        // Scroll indicator (down arrow)
        if self.can_scroll_down() {
            lines.push((Self::center_pad("v", Self::TOTAL_WIDTH), false));
        }

        // Footer
        lines.push((Self::center_pad("", Self::TOTAL_WIDTH), false));
        lines.push((Self::center_pad("ESC TO GO BACK", Self::TOTAL_WIDTH), false));

        MenuRenderData {
            lines,
            is_main_menu: false,
            tabs_line: Some(tab_line),
            scroll_up: self.can_scroll_up(),
            scroll_down: self.can_scroll_down(),
        }
    }

    /// Get menu center position in screen coordinates
    pub fn get_menu_center(&self) -> (f32, f32) {
        (self.screen_width / 2.0, self.screen_height / 2.0)
    }

    /// Called when menu opens to ensure layout is calculated
    pub fn on_open(&mut self) {
        self.menu_mode = MenuMode::Main; // Always start at main menu
        self.scroll_offset = 0;
        self.recalculate_layout();
        self.control_state = ControlState::Idle;
    }

    /// Go back from settings to main menu
    pub fn go_back(&mut self) {
        if self.menu_mode == MenuMode::Settings {
            self.menu_mode = MenuMode::Main;
            self.recalculate_layout();
        }
    }

    /// Check if we're in main menu mode
    pub fn is_main_menu(&self) -> bool {
        self.menu_mode == MenuMode::Main
    }
}

impl Default for SettingsMenu {
    fn default() -> Self {
        Self::new()
    }
}
