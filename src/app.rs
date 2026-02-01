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

use crate::audio::AudioManager;
use crate::inventory::Item;
use crate::player::{Player, PlayerData, BlockInteraction};
use crate::renderer::{DisplayMode, Renderer, DebugMetrics};
use crate::settings::Settings;
use crate::settings_menu::SettingsMenu;
use crate::utils::profiler::FrameProfiler;
use crate::world::{WorldManager, ChunkPos, ItemEntityManager, ChestManager, Block, MESH_STATS, save_items, load_items};

/// Main application state
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    world: WorldManager,
    player: Player,
    item_entities: ItemEntityManager,
    chests: ChestManager,
    profiler: FrameProfiler,
    last_frame: Instant,
    cursor_grabbed: bool,
    audio: Option<AudioManager>,
    settings_menu: SettingsMenu,
    /// Remembered window size for returning from fullscreen
    windowed_size: PhysicalSize<u32>,
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

        // Initialize chest manager
        // TODO: Load chests from disk
        let chests = ChestManager::new();

        let mut app = App {
            window,
            renderer,
            world,
            player,
            item_entities,
            chests,
            profiler: FrameProfiler::new(),
            last_frame: Instant::now(),
            cursor_grabbed: false,
            audio,
            settings_menu,
            windowed_size: window_size,
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

                // Also update spawn in world metadata
                let pos = self.player.position();
                self.world.set_spawn(pos.x, pos.y, pos.z);

                // Save settings before exit
                let settings = self.renderer.collect_settings((self.windowed_size.width, self.windowed_size.height));
                settings.save();

                elwt.exit();
            }

            WindowEvent::Resized(new_size) => {
                self.renderer.resize(new_size);
                self.player
                    .camera
                    .update_aspect(new_size.width as f32 / new_size.height as f32);
                self.settings_menu.update_screen_size(new_size.width as f32, new_size.height as f32);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    match key_code {
                        KeyCode::Escape if pressed => {
                            if self.settings_menu.is_open {
                                // Close menu and resume gameplay
                                self.settings_menu.is_open = false;
                                if !self.cursor_grabbed {
                                    self.toggle_cursor_grab();
                                }
                            } else {
                                // Open menu and release cursor
                                self.settings_menu.is_open = true;
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
                        _ => {
                            self.player.handle_key(key_code, pressed);
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::MouseButton;
                if self.settings_menu.is_open {
                    // Handle menu clicks
                    if state == ElementState::Pressed && button == MouseButton::Left {
                        self.settings_menu.click(&mut self.renderer);
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
                if !self.settings_menu.is_open && !self.player.is_inventory_open() {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y / 50.0) as f32,
                    };
                    self.player.handle_scroll(scroll);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
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
            let interaction = self.player.process_block_interactions(&mut self.world);
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
                    if let Some(ref mut audio) = self.audio {
                        audio.play_block_break();
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
                    if let Some(ref mut audio) = self.audio {
                        audio.play_block_place();
                    }
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
                    audio.update_footsteps(
                        dt,
                        self.player.horizontal_speed(),
                        self.player.is_on_ground(),
                        self.player.is_sprinting(),
                    );
                }
            }
        }
        self.profiler.end_section("player_update");

        // Update world (chunk loading/generation, meshing, lighting)
        self.profiler.begin_section("world_update");
        // Sync render distance and total load distance (including impostor range) to world manager
        self.world.set_render_distance(self.renderer.render_distance());
        self.world.set_load_distance(self.renderer.total_view_distance());
        self.world.update(&self.player.position());
        // Run water flow simulation
        self.world.tick_water_simulation(dt.as_secs_f32());
        self.profiler.end_section("world_update");

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

        // Build debug metrics
        let pos = self.player.position();
        let chunk_pos = ChunkPos::from_world_pos(pos);
        let render_stats = self.profiler.render_stats();

        // Get light level at player's feet position
        let light_pos = (pos.x.floor() as i32, (pos.y - 0.1).floor() as i32, pos.z.floor() as i32);
        let (light_level, sky_light, block_light) = self.world.get_light_at(light_pos.0, light_pos.1, light_pos.2);

        let debug_metrics = DebugMetrics {
            fps: self.profiler.fps() as f32,
            frame_time_ms: self.profiler.frame_time_ms() as f32,
            world_update_ms: self.profiler.section_time_ms("world_update") as f32,
            render_ms: self.profiler.section_time_ms("render") as f32,
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
        };

        // Render
        self.profiler.begin_section("render");
        let held_light = self.player.selected_block().light_emission();
        match self.renderer.render(&self.player.camera, &mut self.world, &self.profiler, target_block, &debug_metrics, held_light, &self.settings_menu, &self.player.inventory) {
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
}
