//! Application state and main loop

use std::sync::Arc;
use std::time::Instant;

use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, Window, WindowBuilder},
};

use crate::audio::{AudioManager, AudioEnvironment, AudioSettings};
use crate::inventory::Item;
use crate::player::{Player, PlayerData, BlockInteraction};
use crate::renderer::{DisplayMode, Renderer, DebugMetrics};
use crate::settings::Settings;
use crate::settings_menu::{SettingsMenu, MenuAction};
use crate::character_styler::{StylerUI, AssetCatalog, CharacterStyle};
use crate::utils::profiler::FrameProfiler;
use crate::entity::{EntityManager, EntityType, Transform, AnimationComponent};
use crate::entity::loader::{BodyPartAdjustment, BODY_PART_NAMES, default_adjustments};
use crate::world::{WorldManager, ChunkPos, ItemEntityManager, ChestManager, Block, MESH_STATS, save_items, load_items};

/// In-game body part position/scale editor.
/// Toggle with F4. Adjusts per-part offsets and scale, re-merges mesh live.
struct BodyPartEditor {
    active: bool,
    selected: usize, // 0..8 index into BODY_PART_NAMES
    adjustments: Vec<BodyPartAdjustment>, // 13 entries (9 unique + 4 right-side)
    dirty: bool, // needs re-merge
    shift_held: bool, // for fine adjustment
}

impl BodyPartEditor {
    fn new() -> Self {
        Self {
            active: false,
            selected: 0,
            adjustments: default_adjustments(),
            dirty: false,
            shift_held: false,
        }
    }

    fn toggle(&mut self) {
        self.active = !self.active;
        tracing::info!("Body part editor: {}", if self.active { "ON" } else { "OFF" });
    }

    fn select_next(&mut self) {
        self.selected = (self.selected + 1) % BODY_PART_NAMES.len();
    }

    fn select_prev(&mut self) {
        self.selected = if self.selected == 0 { BODY_PART_NAMES.len() - 1 } else { self.selected - 1 };
    }

    fn adjust_offset(&mut self, dx: f32, dy: f32, dz: f32) {
        let adj = &mut self.adjustments[self.selected];
        adj.offset.x += dx;
        adj.offset.y += dy;
        adj.offset.z += dz;
        self.dirty = true;
    }

    fn adjust_scale(&mut self, ds: f32) {
        let adj = &mut self.adjustments[self.selected];
        adj.scale = (adj.scale + ds).max(0.1);
        self.dirty = true;
    }

    fn overlay_lines(&self) -> Vec<String> {
        if !self.active {
            return Vec::new();
        }
        let mut lines = Vec::new();
        lines.push("== BODY PART EDITOR (F4) ==".to_string());
        lines.push("[/] part  Arrows X/Z".to_string());
        lines.push("PgUp/Dn Y  +/- scale".to_string());
        lines.push("Shift=fine  Enter=log".to_string());
        lines.push("Tab=cycle animation".to_string());
        lines.push(String::new());

        for (i, name) in BODY_PART_NAMES.iter().enumerate() {
            let adj = &self.adjustments[i];
            let marker = if i == self.selected { ">" } else { " " };
            lines.push(format!(
                "{} {:<11} X:{:+.1} Y:{:+.1} Z:{:+.1} S:{:.2}",
                marker, name,
                adj.offset.x, adj.offset.y, adj.offset.z, adj.scale
            ));
        }
        lines
    }

    fn log_values(&self) {
        tracing::info!("=== Body Part Editor Values ===");
        for (i, name) in BODY_PART_NAMES.iter().enumerate() {
            let adj = &self.adjustments[i];
            tracing::info!(
                "  {}: offset=({:.2}, {:.2}, {:.2}), scale={:.3}",
                name, adj.offset.x, adj.offset.y, adj.offset.z, adj.scale
            );
        }
        tracing::info!("=== End Editor Values ===");
    }
}

/// Main application state
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    world: WorldManager,
    player: Player,
    item_entities: ItemEntityManager,
    entity_manager: EntityManager,
    chests: ChestManager,
    profiler: FrameProfiler,
    last_frame: Instant,
    cursor_grabbed: bool,
    audio: Option<AudioManager>,
    settings_menu: SettingsMenu,
    /// Remembered window size for returning from fullscreen
    windowed_size: PhysicalSize<u32>,
    /// Flag to indicate quit was requested from menu
    quit_requested: bool,
    /// Counter for periodic LOD stats logging
    lod_log_counter: u32,
    /// Body part position/scale editor
    body_part_editor: BodyPartEditor,
    /// Character appearance styler GUI
    character_styler: StylerUI,
}

impl App {
    pub async fn run() {
        let event_loop = EventLoop::new().expect("Failed to create event loop");

        // Load settings from file (or use defaults)
        let settings = Settings::load();
        let initial_size = PhysicalSize::new(settings.windowed_width, settings.windowed_height);

        let window = Arc::new(
            WindowBuilder::new()
                .with_title("Mosaic")
                .with_inner_size(initial_size)
                .build(&event_loop)
                .expect("Failed to create window"),
        );

        // Initialize renderer and apply loaded settings
        let mut renderer = Renderer::new(window.clone()).await;
        renderer.apply_settings(&settings);

        // Get LOD thresholds from renderer (will be applied to world after it's created)
        let lod_thresholds = renderer.lod_thresholds();

        // Initialize world manager with save directory
        let save_dir = std::path::PathBuf::from("saves/world1");
        let mut world = match WorldManager::new_with_save(&save_dir, None) {
            Ok(w) => {
                tracing::info!("World loaded/created with seed {}", w.seed());
                w
            }
            Err(e) => {
                tracing::error!("Failed to initialize world storage: {}, using in-memory only", e);
                WorldManager::new()
            }
        };

        // Apply LOD thresholds from renderer settings
        world.set_lod_thresholds(lod_thresholds.0, lod_thresholds.1);

        // Try to load player data, fall back to spawn position
        let player_path = save_dir.join("player.dat");
        let player = if PlayerData::exists(&player_path) {
            match PlayerData::load(&player_path) {
                Ok(data) => {
                    tracing::info!("Loaded player save data");
                    Player::from_save_data(&data)
                }
                Err(e) => {
                    tracing::warn!("Failed to load player data: {}, using spawn position", e);
                    let spawn_pos = world.spawn_position()
                        .map(|s| glam::Vec3::new(s[0], s[1], s[2]))
                        .unwrap_or(glam::Vec3::new(0.0, 80.0, 0.0));
                    Player::new(spawn_pos)
                }
            }
        } else {
            // New world - use spawn position from world metadata
            let spawn_pos = world.spawn_position()
                .map(|s| glam::Vec3::new(s[0], s[1], s[2]))
                .unwrap_or(glam::Vec3::new(0.0, 80.0, 0.0));
            tracing::info!("New player spawning at ({:.1}, {:.1}, {:.1})", spawn_pos.x, spawn_pos.y, spawn_pos.z);
            Player::new(spawn_pos)
        };

        // Trigger initial chunk loading - distance-prioritized, non-blocking
        // Chunks will load progressively outward from player position
        let player_pos = player.position();
        world.update(&player_pos);

        // Initialize audio system
        let audio = AudioManager::new();

        // Initialize settings menu with screen size
        let window_size = window.inner_size();
        let mut settings_menu = SettingsMenu::new();
        settings_menu.update_screen_size(window_size.width as f32, window_size.height as f32);

        // Load item entities from disk
        let mut item_entities = ItemEntityManager::new();
        let items_path = save_dir.join("items.dat");
        if let Err(e) = load_items(&items_path, &mut item_entities) {
            tracing::warn!("Failed to load item entities: {}", e);
        }

        // Load character style (or use default)
        let style_path = save_dir.join("character.cfg");
        let character_style = CharacterStyle::load(&style_path).unwrap_or_default();

        // Initialize entity manager and load player character
        let mut entity_manager = EntityManager::new();
        Self::load_player_character(&mut entity_manager, &mut renderer, &character_style);

        // Initialize chest manager
        // TODO: Load chests from disk
        let chests = ChestManager::new();

        let mut app = App {
            window,
            renderer,
            world,
            player,
            item_entities,
            entity_manager,
            chests,
            profiler: FrameProfiler::new(),
            last_frame: Instant::now(),
            cursor_grabbed: false,
            audio,
            settings_menu,
            windowed_size: window_size,
            quit_requested: false,
            lod_log_counter: 0,
            body_part_editor: BodyPartEditor::new(),
            character_styler: {
                let mut styler = StylerUI::new(AssetCatalog::scan());
                styler.style = character_style;
                styler.saved_style = styler.style.clone();
                styler
            },
        };

        tracing::info!("Engine initialized successfully");

        let _ = event_loop.run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => {
                    app.handle_window_event(event, elwt);
                }
                Event::DeviceEvent { event, .. } => {
                    app.handle_device_event(event);
                }
                Event::AboutToWait => {
                    app.window.request_redraw();
                }
                _ => {}
            }
        });
    }

    fn toggle_cursor_grab(&mut self) {
        self.cursor_grabbed = !self.cursor_grabbed;
        if self.cursor_grabbed {
            self.window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked))
                .unwrap_or_else(|e| tracing::warn!("Failed to grab cursor: {}", e));
            self.window.set_cursor_visible(false);
        } else {
            self.window.set_cursor_grab(CursorGrabMode::None).ok();
            self.window.set_cursor_visible(true);
        }
    }

    /// Apply pending display mode change from settings
    fn apply_pending_display_mode(&mut self) {
        if !self.renderer.is_display_mode_pending() {
            return;
        }

        // Save current window size if we're in windowed mode before switching away
        if self.window.fullscreen().is_none() {
            self.windowed_size = self.window.inner_size();
        }

        let mode = self.renderer.display_mode();
        match mode {
            DisplayMode::Windowed => {
                self.window.set_fullscreen(None);
                // Restore the remembered window size
                let _ = self.window.request_inner_size(self.windowed_size);
                tracing::info!("Display mode: WINDOWED ({}x{})", self.windowed_size.width, self.windowed_size.height);
            }
            DisplayMode::Borderless => {
                self.window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                tracing::info!("Display mode: BORDERLESS FULLSCREEN");
            }
            DisplayMode::Fullscreen => {
                // Use the primary monitor's native video mode
                if let Some(monitor) = self.window.current_monitor() {
                    if let Some(video_mode) = monitor.video_modes().next() {
                        self.window.set_fullscreen(Some(Fullscreen::Exclusive(video_mode)));
                        tracing::info!("Display mode: EXCLUSIVE FULLSCREEN");
                    } else {
                        // Fallback to borderless if no video mode found
                        self.window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                        tracing::warn!("No video mode found, using borderless fullscreen");
                    }
                } else {
                    // Fallback to borderless if no monitor found
                    self.window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                    tracing::warn!("No monitor found, using borderless fullscreen");
                }
            }
        }

        self.renderer.clear_display_mode_pending();
    }

    fn handle_window_event(&mut self, event: WindowEvent, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        match event {
            WindowEvent::CloseRequested => {
                self.shutdown(elwt);
            }

            WindowEvent::Resized(new_size) => {
                self.renderer.resize(new_size);
                self.player
                    .camera
                    .update_aspect(new_size.width as f32 / new_size.height as f32);
                self.settings_menu.update_screen_size(new_size.width as f32, new_size.height as f32);
                self.character_styler.update_screen_size(new_size.width as f32, new_size.height as f32);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    // Character styler gets first priority for input
                    if self.character_styler.open {
                        if key_code == KeyCode::Escape && pressed {
                            // ESC closes styler, reverts changes
                            let needs_remesh = self.character_styler.close_revert();
                            if needs_remesh {
                                self.remesh_character();
                            }
                            // Re-open the main menu
                            self.settings_menu.is_open = true;
                            self.settings_menu.on_open();
                            return;
                        }
                        self.character_styler.handle_key(key_code, pressed);
                        return;
                    }

                    match key_code {
                        KeyCode::Escape if pressed => {
                            if self.settings_menu.is_open {
                                if self.settings_menu.is_main_menu() {
                                    // In main menu: close menu and resume gameplay
                                    self.settings_menu.is_open = false;
                                    // Save settings when closing menu
                                    let settings = self.renderer.collect_settings((self.windowed_size.width, self.windowed_size.height));
                                    settings.save();
                                    if !self.cursor_grabbed {
                                        self.toggle_cursor_grab();
                                    }
                                } else {
                                    // In settings: go back to main menu
                                    self.settings_menu.go_back();
                                }
                            } else {
                                // Open menu and release cursor
                                self.settings_menu.is_open = true;
                                self.settings_menu.on_open();
                                if self.cursor_grabbed {
                                    self.toggle_cursor_grab();
                                }
                            }
                        }
                        KeyCode::F3 if pressed => {
                            self.renderer.toggle_debug_overlay();
                        }
                        KeyCode::KeyT if pressed => {
                            let speed = self.renderer.toggle_cycle_speed();
                            if speed == 0.0 {
                                tracing::info!("Day/night cycle: PAUSED");
                            } else {
                                tracing::info!("Day/night cycle speed: {}x", speed);
                            }
                        }
                        KeyCode::KeyY if pressed => {
                            let mode = self.renderer.toggle_shadow_mode();
                            if mode > 0.5 {
                                tracing::info!("Shadow mode: SOFT (old style - gradient)");
                            } else {
                                tracing::info!("Shadow mode: HARD (new style - solid)");
                            }
                        }
                        KeyCode::KeyU if pressed => {
                            let debug = self.renderer.toggle_shadow_debug();
                            if debug > 0.5 {
                                tracing::info!("Shadow DEBUG: ON (R=ndotl, G=shadow, B=traced)");
                            } else {
                                tracing::info!("Shadow DEBUG: OFF");
                            }
                        }
                        KeyCode::KeyV if pressed => {
                            let enabled = self.renderer.toggle_volumetrics();
                            if enabled {
                                tracing::info!("Volumetric light shafts: ON");
                            } else {
                                tracing::info!("Volumetric light shafts: OFF");
                            }
                        }
                        KeyCode::KeyG if pressed => {
                            let enabled = self.renderer.toggle_gpu_culling();
                            if enabled {
                                tracing::info!("GPU frustum culling: ON (compute shader)");
                            } else {
                                tracing::info!("GPU frustum culling: OFF (CPU fallback)");
                            }
                        }
                        KeyCode::KeyB if pressed => {
                            let enabled = self.renderer.toggle_gpu_voxel_fill();
                            if enabled {
                                tracing::info!("GPU voxel fill: ON (compute shader)");
                            } else {
                                tracing::info!("GPU voxel fill: OFF (CPU fallback)");
                            }
                        }
                        KeyCode::KeyP if pressed => {
                            self.profiler.toggle_verbose();
                        }
                        KeyCode::KeyM if pressed => {
                            // Skip to next music track
                            if let Some(ref mut audio) = self.audio {
                                if self.renderer.is_music_enabled() {
                                    audio.skip_track();
                                    tracing::info!("Skipping to next music track...");
                                }
                            }
                        }
                        KeyCode::KeyE if pressed && !self.settings_menu.is_open => {
                            // Toggle inventory with cursor grab handling
                            let was_open = self.player.is_inventory_open();
                            self.player.handle_key(key_code, pressed);
                            let is_open = self.player.is_inventory_open();

                            if !was_open && is_open && self.cursor_grabbed {
                                // Inventory opened: release cursor
                                self.toggle_cursor_grab();
                            } else if was_open && !is_open && !self.cursor_grabbed {
                                // Inventory closed: re-grab cursor
                                self.toggle_cursor_grab();
                            }
                        }
                        KeyCode::F5 if pressed => {
                            self.player.toggle_camera_mode();
                        }
                        // Body part editor controls
                        KeyCode::F4 if pressed => {
                            self.body_part_editor.toggle();
                        }
                        KeyCode::ShiftLeft | KeyCode::ShiftRight if self.body_part_editor.active => {
                            self.body_part_editor.shift_held = pressed;
                            // Also pass to player for normal gameplay
                            self.player.handle_key(key_code, pressed);
                        }
                        KeyCode::BracketRight if pressed && self.body_part_editor.active => {
                            self.body_part_editor.select_next();
                        }
                        KeyCode::BracketLeft if pressed && self.body_part_editor.active => {
                            self.body_part_editor.select_prev();
                        }
                        KeyCode::ArrowRight if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.1 } else { 0.5 };
                            self.body_part_editor.adjust_offset(step, 0.0, 0.0);
                        }
                        KeyCode::ArrowLeft if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.1 } else { 0.5 };
                            self.body_part_editor.adjust_offset(-step, 0.0, 0.0);
                        }
                        KeyCode::ArrowUp if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.1 } else { 0.5 };
                            self.body_part_editor.adjust_offset(0.0, 0.0, step);
                        }
                        KeyCode::ArrowDown if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.1 } else { 0.5 };
                            self.body_part_editor.adjust_offset(0.0, 0.0, -step);
                        }
                        KeyCode::PageUp if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.1 } else { 0.5 };
                            self.body_part_editor.adjust_offset(0.0, step, 0.0);
                        }
                        KeyCode::PageDown if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.1 } else { 0.5 };
                            self.body_part_editor.adjust_offset(0.0, -step, 0.0);
                        }
                        KeyCode::Equal if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.01 } else { 0.05 };
                            self.body_part_editor.adjust_scale(step);
                        }
                        KeyCode::Minus if pressed && self.body_part_editor.active => {
                            let step = if self.body_part_editor.shift_held { 0.01 } else { 0.05 };
                            self.body_part_editor.adjust_scale(-step);
                        }
                        KeyCode::Enter if pressed && self.body_part_editor.active => {
                            self.body_part_editor.log_values();
                        }
                        KeyCode::Tab if pressed && self.body_part_editor.active => {
                            // Cycle animation clip on all animated entities
                            let clip_count = self.entity_manager.animation_clips.len();
                            if clip_count > 0 {
                                for anim in self.entity_manager.animations.values_mut() {
                                    let next = (anim.current_clip + 1) % clip_count;
                                    anim.transition_to(next, 0.2);
                                }
                                let idx = self.entity_manager.animations.values().next()
                                    .map(|a| a.current_clip).unwrap_or(0);
                                let name = self.entity_manager.animation_clips.get(idx)
                                    .map(|c| c.name.as_str()).unwrap_or("?");
                                tracing::info!("Animation: [{}] {}", idx, name);
                            }
                        }
                        _ => {
                            self.player.handle_key(key_code, pressed);
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::MouseButton;
                if self.character_styler.open {
                    if button == MouseButton::Left {
                        if state == ElementState::Pressed {
                            // Check for Done button click
                            let mx = self.character_styler.last_mouse_x;
                            let my = self.character_styler.last_mouse_y;
                            if self.character_styler.was_done_clicked(mx, my) {
                                // Save style and close
                                self.character_styler.close_save();
                                let style_path = std::path::PathBuf::from("saves/world1/character.cfg");
                                if let Err(e) = self.character_styler.style.save(&style_path) {
                                    tracing::warn!("Failed to save character style: {}", e);
                                }
                                // Re-open main menu
                                self.settings_menu.is_open = true;
                                self.settings_menu.on_open();
                            } else {
                                self.character_styler.handle_mouse_press(mx, my);
                            }
                        } else {
                            self.character_styler.handle_mouse_release();
                        }
                    }
                    return;
                }
                if self.settings_menu.is_open {
                    // Handle menu clicks
                    if state == ElementState::Pressed && button == MouseButton::Left {
                        let action = self.settings_menu.click(&mut self.renderer);
                        match action {
                            MenuAction::Quit => {
                                self.quit_requested = true;
                            }
                            MenuAction::OpenCharacterStyler => {
                                self.settings_menu.is_open = false;
                                self.character_styler.open();
                            }
                            MenuAction::None => {}
                        }
                        // Save settings immediately after each toggle
                        let settings = self.renderer.collect_settings((self.windowed_size.width, self.windowed_size.height));
                        settings.save();
                        // Note: display mode changes are applied at start of next frame
                        // to avoid race conditions with the render pipeline
                    }
                } else if self.player.is_inventory_open() {
                    // Handle inventory clicks
                    let size = self.window.inner_size();
                    let (mx, my) = self.player.inventory.mouse_pos();
                    if state == ElementState::Pressed {
                        match button {
                            MouseButton::Left => {
                                self.player.inventory.handle_click(mx, my, size.width as f32, size.height as f32);
                            }
                            MouseButton::Right => {
                                self.player.inventory.handle_right_click(mx, my, size.width as f32, size.height as f32);
                            }
                            _ => {}
                        }
                    }
                } else if !self.cursor_grabbed {
                    if state == ElementState::Pressed {
                        self.toggle_cursor_grab();
                    }
                } else {
                    self.player
                        .handle_mouse_button(button, state == ElementState::Pressed);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 50.0) as f32,
                };

                if self.settings_menu.is_open {
                    // Handle menu scrolling
                    self.settings_menu.scroll(scroll);
                } else if !self.player.is_inventory_open() {
                    if self.player.is_third_person() && self.player.is_sprinting() {
                        // Ctrl+scroll in third person: adjust camera distance
                        self.player.adjust_camera_zoom(scroll);
                    } else {
                        // Normal scroll: cycle hotbar slots
                        self.player.handle_scroll(scroll);
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if self.character_styler.open {
                    self.character_styler.handle_mouse_move(x, y);
                }
                if self.settings_menu.is_open {
                    self.settings_menu.update_mouse(x, y);
                }
                if self.player.is_inventory_open() {
                    let size = self.window.inner_size();
                    self.player.inventory.update_mouse(x, y);
                    self.player.inventory.update_hover(x, y, size.width as f32, size.height as f32);
                }
            }

            WindowEvent::RedrawRequested => {
                self.update_and_render(elwt);
            }

            _ => {}
        }

        // Check if quit was requested from menu
        if self.quit_requested {
            self.quit_requested = false; // Reset in case exit fails
            self.shutdown(elwt);
        }
    }

    /// Perform shutdown: save data and exit
    fn shutdown(&mut self, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        tracing::info!("Shutting down...");

        // Save player data (position, rotation, inventory)
        let player_data = self.player.to_save_data();
        let player_path = std::path::PathBuf::from("saves/world1/player.dat");
        if let Err(e) = player_data.save(&player_path) {
            tracing::error!("Failed to save player data: {}", e);
        }

        // Save item entities
        let items_path = std::path::PathBuf::from("saves/world1/items.dat");
        if let Err(e) = save_items(&items_path, &self.item_entities) {
            tracing::error!("Failed to save item entities: {}", e);
        }

        // Update spawn in world metadata
        let pos = self.player.position();
        self.world.set_spawn(pos.x, pos.y, pos.z);

        // Save settings
        let settings = self.renderer.collect_settings((self.windowed_size.width, self.windowed_size.height));
        settings.save();

        elwt.exit();
    }

    fn handle_device_event(&mut self, event: DeviceEvent) {
        if self.cursor_grabbed {
            if let DeviceEvent::MouseMotion { delta } = event {
                self.player
                    .handle_mouse_motion(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn update_and_render(&mut self, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        // Apply pending display mode changes at start of frame (before rendering)
        // to avoid race conditions with surface/swapchain changes
        if self.renderer.is_display_mode_pending() {
            self.apply_pending_display_mode();
            // Skip this frame to let the window/surface stabilize after mode change
            self.last_frame = Instant::now();
            return;
        }

        // Calculate delta time
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame);
        self.last_frame = now;

        self.profiler.begin_frame();

        // Update day/night cycle
        self.renderer.update_time(dt.as_secs_f32());

        // Update player (physics, collision, block interactions)
        self.profiler.begin_section("player_update");
        if self.cursor_grabbed {
            self.player.update(dt, Some(&self.world));
            // Process block break/place interactions
            let interaction = self.player.process_block_interactions(&mut self.world, dt.as_secs_f32());
            match interaction {
                BlockInteraction::Break { x, y, z, block } => {
                    // If breaking a chest, drop its contents first
                    if block == Block::Chest {
                        let chest_pos = glam::IVec3::new(x, y, z);
                        if let Some(chest_data) = self.chests.remove_chest(chest_pos) {
                            // Drop all items from the chest
                            for (_, stack) in chest_data.container.iter() {
                                if let Some(stack) = stack {
                                    self.item_entities.spawn_from_block([x, y, z], stack.item, stack.count);
                                }
                            }
                        }
                    }
                    // Spawn item entity for broken block
                    if let Some(item) = Item::from_block(block) {
                        self.item_entities.spawn_from_block([x, y, z], item, 1);
                    }
                    // Mark only the affected block for voxel GI update (incremental)
                    self.renderer.mark_voxel_dirty_at(x, y, z);
                    // Play block break sound (material-based, 3D spatial)
                    if let Some(ref mut audio) = self.audio {
                        let block_pos = glam::Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                        let player_pos = self.player.position();
                        let player_dir = self.player.forward();
                        audio.play_block_break(
                            Some(block),
                            Some(block_pos),
                            player_pos,
                            player_dir,
                            Some(&self.world),
                        );
                    }
                }
                BlockInteraction::Place { x, y, z, block } => {
                    // If placing a chest, register it in ChestManager
                    if block == Block::Chest {
                        let chest_pos = glam::IVec3::new(x, y, z);
                        // Chest faces toward player (opposite of player's forward direction)
                        // Use a fixed rotation for now - facing north (0 radians)
                        // TODO: Calculate based on player facing when we have access to yaw
                        let rotation = 0.0;
                        self.chests.add_chest(chest_pos, rotation);
                    }
                    // Mark only the affected block for voxel GI update (incremental)
                    self.renderer.mark_voxel_dirty_at(x, y, z);
                    // Play block place sound (material-based, 3D spatial)
                    if let Some(ref mut audio) = self.audio {
                        let block_pos = glam::Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                        let player_pos = self.player.position();
                        let player_dir = self.player.forward();
                        audio.play_block_place(
                            Some(block),
                            Some(block_pos),
                            player_pos,
                            player_dir,
                            Some(&self.world),
                        );
                    }
                }
                BlockInteraction::HarvestProp { x, y, z, prop_id } => {
                    tracing::debug!("Harvested prop {} at ({:.2}, {:.2}, {:.2})", prop_id, x, y, z);
                    // Play break sound for foliage/grass
                    if let Some(ref mut audio) = self.audio {
                        let prop_pos = glam::Vec3::new(x, y, z);
                        let player_pos = self.player.position();
                        let player_dir = self.player.forward();
                        // Use foliage sound for grass/flowers, stone for rocks
                        let block = if prop_id < 15 {
                            Block::Leaves // Foliage sound for grass/flowers
                        } else {
                            Block::Stone // Stone sound for rocks
                        };
                        audio.play_block_break(
                            Some(block),
                            Some(prop_pos),
                            player_pos,
                            player_dir,
                            Some(&self.world),
                        );
                    }
                    // TODO: Spawn item drops based on prop type
                }
                BlockInteraction::None => {}
            }

            // Process item drop request (Q key)
            if let Some((item, count)) = self.player.process_drop_request() {
                self.item_entities.spawn_from_player(
                    self.player.position(),
                    self.player.forward(),
                    item,
                    count,
                );
            }

            // Update footstep audio (only when not flying)
            if let Some(ref mut audio) = self.audio {
                if !self.player.is_flying() {
                    // Get the block the player is standing on for material-based sounds
                    let feet_pos = self.player.position();
                    let surface_block = self.world.get_block(
                        feet_pos.x.floor() as i32,
                        (feet_pos.y - 0.1).floor() as i32,
                        feet_pos.z.floor() as i32,
                    );
                    audio.update_footsteps(
                        dt,
                        self.player.horizontal_speed(),
                        self.player.is_on_ground(),
                        self.player.is_sprinting(),
                        surface_block,
                    );
                }
            }
        }
        self.profiler.end_section("player_update");

        // Update audio system (ambient sounds, music, volume settings sync)
        if let Some(ref mut audio) = self.audio {
            // Sync volume settings from renderer to audio manager
            let new_settings = AudioSettings {
                master: self.renderer.audio_master(),
                sfx: self.renderer.audio_sfx(),
                ambient: self.renderer.audio_ambient(),
                music: self.renderer.audio_music(),
                music_enabled: self.renderer.is_music_enabled(),
            };
            if audio.get_settings().master != new_settings.master
                || audio.get_settings().sfx != new_settings.sfx
                || audio.get_settings().ambient != new_settings.ambient
                || audio.get_settings().music != new_settings.music
                || audio.get_settings().music_enabled != new_settings.music_enabled
            {
                audio.set_settings(new_settings.clone());
                // If music enabled state changed, notify the music system
                if audio.get_settings().music_enabled != new_settings.music_enabled {
                    audio.set_music_enabled(new_settings.music_enabled);
                }
            }

            // Update ambient sounds based on environment
            let player_pos = self.player.position();
            let (_, sky_light, _) = self.world.get_light_at(
                player_pos.x.floor() as i32,
                player_pos.y.floor() as i32,
                player_pos.z.floor() as i32,
            );

            // Determine if player is in cave (low sky light)
            let in_cave = sky_light < 4;

            // Determine if underwater (head below water level)
            // For now just check if at sea level or below with water nearby
            let underwater = player_pos.y < crate::constants::WATER_SEA_LEVEL as f32
                && self.world.get_block(
                    player_pos.x.floor() as i32,
                    player_pos.y.floor() as i32,
                    player_pos.z.floor() as i32,
                ) == Some(Block::Water);

            // Get biome ID (default to grasslands)
            // TODO: Get actual biome from world generation
            let biome_id = 0u8;

            let env = AudioEnvironment {
                in_cave,
                underwater,
                biome_id,
                height: player_pos.y,
                sky_light,
            };

            audio.update_ambient(dt.as_secs_f32(), env);

            // Update background music system
            audio.update_music(dt.as_secs_f32());
        }

        // Update world (chunk loading/generation, meshing, lighting)
        self.profiler.begin_section("world_update");
        // Sync render distance, LOD thresholds, and total load distance to world manager
        self.world.set_render_distance(self.renderer.render_distance());
        let (lod_full, lod_half) = self.renderer.lod_thresholds();
        self.world.set_lod_thresholds(lod_full, lod_half);
        self.world.set_load_distance(self.renderer.total_view_distance());
        self.world.update(&self.player.position());
        // Update player position for water flow simulation culling
        let player_chunk = crate::world::ChunkPos::from_world_pos(self.player.position());
        self.world.update_water_player_position(player_chunk);
        // Run water flow simulation
        self.world.tick_water_simulation(dt.as_secs_f32());
        self.profiler.end_section("world_update");

        // Log LOD distribution every 600 frames (~10 seconds at 60fps)
        self.lod_log_counter += 1;
        if self.lod_log_counter >= 600 {
            self.lod_log_counter = 0;
            let (full, half, quarter) = self.world.lod_distribution();
            let total = full + half + quarter;
            if total > 0 {
                tracing::info!(
                    "LOD distribution: Full={} ({:.0}%), Half={} ({:.0}%), Quarter={} ({:.0}%) | Total={} meshes",
                    full, full as f32 / total as f32 * 100.0,
                    half, half as f32 / total as f32 * 100.0,
                    quarter, quarter as f32 / total as f32 * 100.0,
                    total
                );
            }
        }

        // Update item entities (physics, pickup, merging)
        self.profiler.begin_section("item_entities");
        // Use a position between feet and eyes for pickup detection
        let player_pos = self.player.position();
        let pickup_pos = glam::Vec3::new(player_pos.x, player_pos.y - 0.8, player_pos.z);
        let world_ref = &self.world;
        let inventory = &self.player.inventory;
        let collected = self.item_entities.update(
            dt.as_secs_f32(),
            pickup_pos,
            |x, y, z| world_ref.get_block(x, y, z),
            |item, count| inventory.has_space_for(*item, count),
        );
        // Add collected items to inventory
        for (item, count) in collected {
            self.player.inventory.add_item(item, count);
        }
        self.profiler.end_section("item_entities");

        // Upload meshes to GPU (per-chunk buffers - for fallback)
        self.profiler.begin_section("mesh_upload");
        self.world.upload_meshes(self.renderer.device());
        self.world.upload_water_meshes(self.renderer.device());
        self.profiler.end_section("mesh_upload");

        // Get target block for outline rendering
        let target_block = if self.cursor_grabbed {
            self.player.get_target_block(&self.world)
        } else {
            None
        };

        // Update item entity renderer
        use crate::renderer::ItemInstanceData;
        self.renderer.update_item_entities(
            self.item_entities.entities().map(ItemInstanceData::from_entity)
        );

        // Update prop renderer with chest positions from ChestManager
        use crate::renderer::PropInstance;
        let chest_instances: Vec<PropInstance> = self.chests.positions()
            .map(|(pos, data)| {
                // Chest is 2 blocks wide, center between this block and +X neighbor
                // Rotate 90 degrees counterclockwise from stored rotation
                PropInstance::new(
                    glam::Vec3::new(pos.x as f32 + 1.0, pos.y as f32, pos.z as f32 + 0.5),
                    data.rotation + std::f32::consts::FRAC_PI_2,
                    1.0,
                )
            })
            .collect();
        self.renderer.update_props("chest", &chest_instances);

        // Update character entity rendering (advance animation + upload)
        self.update_character_entities(dt.as_secs_f64());

        // Build debug metrics
        let pos = self.player.position();
        let chunk_pos = ChunkPos::from_world_pos(pos);

        // Update profiler settings snapshot (for context in verbose logging)
        self.profiler.update_settings(self.renderer.settings_snapshot());

        // Clone render stats to avoid borrow conflict
        let render_stats = self.profiler.render_stats().clone();

        // Get frame time percentiles for stutter detection
        let (fps_1pct_low, fps_01pct_low) = self.profiler.fps_percentiles();

        // Get light level at player's feet position
        let light_pos = (pos.x.floor() as i32, (pos.y - 0.1).floor() as i32, pos.z.floor() as i32);
        let (light_level, sky_light, block_light) = self.world.get_light_at(light_pos.0, light_pos.1, light_pos.2);

        let debug_metrics = DebugMetrics {
            fps: self.profiler.fps() as f32,
            frame_time_ms: self.profiler.frame_time_ms() as f32,
            world_update_ms: self.profiler.section_time_ms("world_update") as f32,
            render_ms: self.profiler.section_time_ms("render") as f32,
            fps_1pct_low: fps_1pct_low as f32,
            fps_01pct_low: fps_01pct_low as f32,
            chunks_pending_gen: self.world.pending_generation_count() as u32,
            chunks_pending_mesh: self.world.pending_meshing_count() as u32,
            triangles: render_stats.triangles,
            draw_calls: render_stats.chunks_rendered,
            chunks_rendered: render_stats.chunks_rendered,
            chunks_culled: render_stats.chunks_culled,
            impostors_rendered: render_stats.impostors_rendered,
            gpu_time_ms: render_stats.gpu_time_ms,
            shadow_chunks_rendered: render_stats.shadow_chunks_rendered,
            shadow_chunks_culled: render_stats.shadow_chunks_culled,
            shadow_pass_ms: render_stats.shadow_pass_ms,
            main_pass_ms: render_stats.main_pass_ms,
            loaded_chunks: self.world.loaded_chunks_count() as u32,
            total_vertices: self.world.total_vertices(),
            pos_x: pos.x,
            pos_y: pos.y,
            pos_z: pos.z,
            chunk_x: chunk_pos.x,
            chunk_y: chunk_pos.y,
            chunk_z: chunk_pos.z,
            yaw: self.player.camera.yaw,
            pitch: self.player.camera.pitch,
            light_level,
            sky_light,
            block_light,
            // Meshing stats
            mesh_chunks_total: MESH_STATS.chunks_meshed.load(std::sync::atomic::Ordering::Relaxed),
            mesh_avg_time_us: MESH_STATS.avg_mesh_time_us(),
            mesh_total_time_ms: MESH_STATS.total_mesh_time_ms(),
            mesh_total_quads: MESH_STATS.total_quads.load(std::sync::atomic::Ordering::Relaxed),
            // Settings context for log files
            settings_context: self.profiler.settings().as_compact_string(),
            // Per-pass timing breakdown
            pass_times: render_stats.pass_times.clone(),
            // Body part editor overlay
            body_part_editor_lines: self.body_part_editor.overlay_lines(),
            // Character styler overlay
            styler_overlay: if self.character_styler.open {
                Some(self.character_styler.overlay_data())
            } else {
                None
            },
        };

        // Re-merge character mesh if body part editor or styler changed values
        if self.body_part_editor.dirty {
            self.body_part_editor.dirty = false;
            self.remesh_character();
        }
        if self.character_styler.dirty {
            self.character_styler.dirty = false;
            self.remesh_character();
        }

        // Render
        self.profiler.begin_section("render");
        let held_light = self.player.selected_block().light_emission();
        let mining_target = self.player.mining_target();
        let mining_progress = self.player.mining_progress();
        let player_feet_pos = self.player.feet_position();
        match self.renderer.render(&self.player.camera, &mut self.world, &self.profiler, target_block, mining_target, mining_progress, player_feet_pos, &debug_metrics, held_light, &self.settings_menu, &self.player.inventory) {
            Ok(_) => {
                // Update profiler with render stats
                self.profiler.update_render_stats(self.renderer.stats.clone());
            }
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.renderer.resize(self.window.inner_size());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                tracing::error!("Out of GPU memory!");
                elwt.exit();
            }
            Err(e) => {
                tracing::warn!("Render error: {:?}", e);
            }
        }
        self.profiler.end_section("render");

        self.profiler.end_frame();
    }

    /// Load the player character model from assets.
    fn load_player_character(entity_manager: &mut EntityManager, renderer: &mut Renderer, style: &CharacterStyle) {
        use crate::entity::loader;
        use crate::entity::RenderComponent;
        use std::sync::Arc;

        // Phase 2 approach: separate mesh, skeleton, and animation files
        // 1. Mesh from body part FBX (no skeleton, no skin weights)
        // 2. Skeleton from Avatar_for_Animation.fbx or EmptyBody.fbx
        // 3. Animations from Anim@xxx.fbx files
        // Body part mesh vertices default to bone 0 (root bone) - whole mesh follows root motion

        // Step 1: Load skeleton first (needed for animation binding)
        let skeleton_path = std::path::Path::new("assets/char/animations/v1/Avatar_for_Animation.fbx");
        let skeleton = match loader::load_skeleton_from_fbx(skeleton_path) {
            Ok(skel) => {
                tracing::info!("Loaded skeleton: {} bones", skel.bone_names.len());
                Arc::new(skel)
            }
            Err(e) => {
                tracing::warn!("Failed to load skeleton from Avatar: {}", e);
                // Fallback to EmptyBody
                let empty_path = std::path::Path::new("assets/char/models/EmptyBody.fbx");
                match loader::load_skeleton_from_fbx(empty_path) {
                    Ok(skel) => {
                        tracing::info!("Loaded skeleton from EmptyBody.fbx: {} bones", skel.bone_names.len());
                        Arc::new(skel)
                    }
                    Err(e2) => {
                        tracing::warn!("Failed to load any skeleton: {}", e2);
                        return;
                    }
                }
            }
        };

        // Step 2: Merge body part meshes into a single skinned mesh
        let models_dir = std::path::Path::new("assets/char/models");
        let defaults = loader::default_adjustments();
        let body_parts = style.build_body_parts();
        let merged = match loader::merge_body_parts(models_dir, &skeleton, &body_parts, Some(&defaults)) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to merge body parts: {}", e);
                return;
            }
        };

        // Log mesh bounds
        if !merged.vertices.is_empty() {
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for v in &merged.vertices {
                for i in 0..3 {
                    min[i] = min[i].min(v.position[i]);
                    max[i] = max[i].max(v.position[i]);
                }
            }
            tracing::info!(
                "Merged mesh: {} verts, {} indices, bounds ({:.1},{:.1},{:.1})->({:.1},{:.1},{:.1})",
                merged.vertices.len(), merged.indices.len(),
                min[0], min[1], min[2], max[0], max[1], max[2],
            );
        }

        // Upload mesh to GPU
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let char_renderer = renderer.character_renderer_mut();
        let mesh_slot = char_renderer.upload_mesh(&device, &merged.vertices, &merged.indices);

        let bone_count = skeleton.bone_names.len();

        // Upload T-pose bone matrices as initial state
        let tpose_matrices = skeleton.compute_tpose_matrices();
        char_renderer.update_bones(&queue, &tpose_matrices);

        // Store skeleton
        entity_manager.skeletons.push(Arc::clone(&skeleton));

        // Step 3: Load animation clips
        let anim_files = [
            "assets/char/animations/v1/Anim@Idle.fbx",
            "assets/char/animations/v1/Anim@Move.fbx",
            "assets/char/animations/v1/Anim@ATK1.fbx",
            "assets/char/animations/v1/Anim@Jump.fbx",
        ];

        for anim_path_str in &anim_files {
            let anim_path = std::path::Path::new(anim_path_str);
            if !anim_path.exists() {
                tracing::warn!("Animation not found: {}", anim_path_str);
                continue;
            }
            match loader::load_animations(anim_path, &skeleton) {
                Ok(clips) => {
                    for clip in clips {
                        let tracked = clip.bone_tracks.iter().filter(|t| t.is_some()).count();
                        tracing::info!(
                            "Loaded animation '{}': {:.2}s, {}/{} bones",
                            clip.name, clip.duration, tracked, bone_count,
                        );
                        entity_manager.animation_clips.push(clip);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load animation '{}': {}", anim_path_str, e);
                }
            }
        }

        // Load character textures from merged mesh into texture array
        if !merged.texture_paths.is_empty() {
            // Filter to only existing paths
            let valid_paths: Vec<_> = merged.texture_paths.iter()
                .filter(|p| p.exists())
                .cloned()
                .collect();

            if !valid_paths.is_empty() {
                if let Err(e) = char_renderer.load_texture_array(&device, &queue, &valid_paths) {
                    tracing::warn!("Failed to load character textures: {}", e);
                    char_renderer.create_placeholder_texture(&device, &queue);
                }
            } else {
                tracing::warn!("No character textures found, using placeholder");
                char_renderer.create_placeholder_texture(&device, &queue);
            }
        } else {
            tracing::warn!("No texture paths from merged mesh, using placeholder");
            char_renderer.create_placeholder_texture(&device, &queue);
        }

        // Spawn player character entity (position will be synced from Player each frame)
        let key = entity_manager.spawn(
            EntityType::Player,
            Transform {
                position: glam::Vec3::ZERO, // Synced from Player::feet_pos each frame
                rotation: glam::Quat::IDENTITY,
                scale: 0.09, // Mesh ~20 units tall (scaled to skeleton space), 0.09 * 20 ≈ 1.8 blocks
            },
        );
        entity_manager.player_entity = Some(key);

        entity_manager.render.insert(key, RenderComponent {
            mesh_slot,
            index_count: merged.indices.len() as u32,
            bone_offset: 0,
            bone_count: bone_count as u32,
        });

        // Add animation component if we have clips
        if !entity_manager.animation_clips.is_empty() {
            entity_manager.animations.insert(
                key,
                AnimationComponent::new(Arc::clone(&skeleton), 0),
            );
            tracing::info!(
                "Character spawned with {} animation clips (bone 0 = root motion)",
                entity_manager.animation_clips.len()
            );
        } else {
            tracing::info!("Character spawned (no animations loaded, T-pose)");
        }
    }

    /// Re-merge character body parts with current adjustments and re-upload to GPU.
    /// Uses the character styler's current style for body part selection.
    fn remesh_character(&mut self) {
        use crate::entity::loader;

        if self.entity_manager.skeletons.is_empty() {
            return;
        }
        let skeleton = &self.entity_manager.skeletons[0];
        let models_dir = std::path::Path::new("assets/char/models");

        let body_parts = self.character_styler.style.build_body_parts();
        let merged = match loader::merge_body_parts(
            models_dir,
            skeleton,
            &body_parts,
            Some(&self.body_part_editor.adjustments),
        ) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Remesh failed: {}", e);
                return;
            }
        };

        // Re-upload mesh to GPU (replace slot 0)
        let device = self.renderer.device().clone();
        let queue = self.renderer.queue().clone();
        let char_renderer = self.renderer.character_renderer_mut();
        char_renderer.replace_mesh(&device, 0, &merged.vertices, &merged.indices);

        // Update render component index count
        if let Some(player_key) = self.entity_manager.player_entity {
            if let Some(render) = self.entity_manager.render.get_mut(player_key) {
                render.index_count = merged.indices.len() as u32;
            }
        }

        // Reload textures if they changed
        if !merged.texture_paths.is_empty() {
            let valid_paths: Vec<_> = merged.texture_paths.iter()
                .filter(|p| p.exists())
                .cloned()
                .collect();
            if !valid_paths.is_empty() {
                if let Err(e) = char_renderer.load_texture_array(&device, &queue, &valid_paths) {
                    tracing::warn!("Failed to reload character textures: {}", e);
                }
            }
        }
    }

    /// Update character entity rendering data each frame.
    fn update_character_entities(&mut self, dt: f64) {
        use crate::renderer::character_renderer::CharacterInstance;
        use crate::player::MovementState;

        let player_feet = self.player.feet_position();
        let player_facing = self.player.facing_yaw();
        let movement_state = self.player.movement_state();
        let is_third_person = self.player.is_third_person();

        let styler_open = self.character_styler.open;

        // Show/hide player entity based on camera mode (always visible in styler)
        if let Some(player_key) = self.entity_manager.player_entity {
            if let Some(base) = self.entity_manager.entities.get_mut(player_key) {
                base.active = is_third_person || styler_open;
            }
        }

        // When styler is open, override camera to orbit around player
        if styler_open {
            let orbit_yaw = self.character_styler.orbit_yaw;
            let orbit_distance = 5.0_f32;
            let orbit_height = 1.0_f32;
            let center = player_feet + glam::Vec3::new(0.0, 0.9, 0.0); // Chest height

            let cam_x = center.x + orbit_distance * orbit_yaw.sin();
            let cam_z = center.z + orbit_distance * orbit_yaw.cos();
            let cam_y = center.y + orbit_height;

            self.player.camera.position = glam::Vec3::new(cam_x, cam_y, cam_z);
            // Look at the character center
            let look_dir = (center - self.player.camera.position).normalize();
            self.player.camera.yaw = look_dir.x.atan2(look_dir.z);
            self.player.camera.pitch = (-look_dir.y).asin();
        }

        // Select animation clip and speed based on player movement state
        let clip_count = self.entity_manager.animation_clips.len();
        if clip_count > 0 {
            if let Some(player_key) = self.entity_manager.player_entity {
                if let Some(anim) = self.entity_manager.animations.get_mut(player_key) {
                    if styler_open {
                        // In styler: idle animation
                        if anim.current_clip != 0 {
                            anim.transition_to(0, 0.3);
                        }
                        anim.speed = 1.0;
                    } else {
                        let (target_clip, target_speed, blend_time, looping) = match movement_state {
                            MovementState::Idle => (0, 1.0, 0.2, true),
                            MovementState::Walking => (1.min(clip_count - 1), 1.0, 0.2, true),
                            MovementState::Sprinting => (1.min(clip_count - 1), 1.5, 0.15, true),
                            MovementState::Jumping => (3.min(clip_count - 1), 1.0, 0.1, false),  // Play once, clamp at peak
                            MovementState::Falling => (3.min(clip_count - 1), 0.3, 0.15, false),  // Slow to near-freeze, clamp
                            MovementState::Swimming => (1.min(clip_count - 1), 0.8, 0.3, true),
                        };

                        // Crossfade to new clip if changed
                        if anim.current_clip != target_clip {
                            anim.transition_to(target_clip, blend_time);
                        }
                        anim.speed = target_speed;
                        anim.looping = looping;
                    }
                }

                // Update player entity transform from Player position
                if let Some(transform) = self.entity_manager.transforms.get_mut(player_key) {
                    transform.position = player_feet;
                    if styler_open {
                        // In styler: face toward camera
                        transform.rotation = glam::Quat::from_rotation_y(self.character_styler.orbit_yaw);
                    } else {
                        transform.rotation = glam::Quat::from_rotation_y(player_facing);
                    }
                }
            }
        }

        // Update all animations and collect bone matrices
        let mut all_bone_matrices = Vec::new();
        let clips = &self.entity_manager.animation_clips;

        let anim_keys: Vec<_> = self.entity_manager.animations.keys().collect();
        for key in &anim_keys {
            if let Some(anim) = self.entity_manager.animations.get_mut(*key) {
                // Update animation (ignore root motion - player position is physics-driven)
                let _root_motion = anim.update(dt, clips);

                // Record where this entity's bones start in the shared buffer
                if let Some(render) = self.entity_manager.render.get_mut(*key) {
                    render.bone_offset = all_bone_matrices.len() as u32;
                    render.bone_count = anim.bone_matrices.len() as u32;
                }
                all_bone_matrices.extend_from_slice(&anim.bone_matrices);
            }
        }

        let queue = self.renderer.queue().clone();

        // Upload bone matrices to GPU
        if !all_bone_matrices.is_empty() {
            self.renderer.character_renderer_mut().update_bones(&queue, &all_bone_matrices);
        }

        // Build instance data for GPU
        let mut instances = Vec::new();
        for (_key, transform, render) in self.entity_manager.renderable_entities() {
            instances.push(CharacterInstance::new(
                transform.position,
                transform.rotation,
                transform.scale,
                render.bone_offset,
            ));
        }

        self.renderer.character_renderer_mut().update_instances(&queue, &instances);
    }
}
