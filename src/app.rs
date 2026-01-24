//! Application state and main loop

use std::sync::Arc;
use std::time::Instant;

use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowBuilder},
};

use crate::audio::AudioManager;
use crate::player::Player;
use crate::renderer::{Renderer, DebugMetrics};
use crate::utils::profiler::FrameProfiler;
use crate::world::{WorldManager, ChunkPos, MESH_STATS};

/// Main application state
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    world: WorldManager,
    player: Player,
    profiler: FrameProfiler,
    last_frame: Instant,
    cursor_grabbed: bool,
    audio: Option<AudioManager>,
}

impl App {
    pub async fn run() {
        let event_loop = EventLoop::new().expect("Failed to create event loop");

        let window = Arc::new(
            WindowBuilder::new()
                .with_title("Neolith")
                .with_inner_size(PhysicalSize::new(1280, 720))
                .build(&event_loop)
                .expect("Failed to create window"),
        );

        // Initialize renderer
        let mut renderer = Renderer::new(window.clone()).await;

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

        // Load spawn position from storage
        let spawn_pos = if let Some(spawn) = world.spawn_position() {
            glam::Vec3::new(spawn[0], spawn[1], spawn[2])
        } else {
            glam::Vec3::new(0.0, 80.0, 0.0)
        };
        tracing::info!("Player spawning at ({:.1}, {:.1}, {:.1})", spawn_pos.x, spawn_pos.y, spawn_pos.z);

        // Trigger initial chunk loading - distance-prioritized, non-blocking
        // Chunks will load progressively outward from spawn position
        world.update(&spawn_pos);

        // Initialize audio system
        let audio = AudioManager::new();

        let mut app = App {
            window,
            renderer,
            world,
            player: Player::new(spawn_pos),
            profiler: FrameProfiler::new(),
            last_frame: Instant::now(),
            cursor_grabbed: false,
            audio,
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

    fn handle_window_event(&mut self, event: WindowEvent, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("Shutting down...");
                // Save player position before exit
                let pos = self.player.position();
                self.world.set_spawn(pos.x, pos.y, pos.z);
                tracing::info!("Saved player position: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z);
                elwt.exit();
            }

            WindowEvent::Resized(new_size) => {
                self.renderer.resize(new_size);
                self.player
                    .camera
                    .update_aspect(new_size.width as f32 / new_size.height as f32);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;

                    match key_code {
                        KeyCode::Escape if pressed => {
                            self.toggle_cursor_grab();
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
                        _ => {
                            self.player.handle_key(key_code, pressed);
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if !self.cursor_grabbed {
                    if state == ElementState::Pressed {
                        self.toggle_cursor_grab();
                    }
                } else {
                    self.player
                        .handle_mouse_button(button, state == ElementState::Pressed);
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
        // Calculate delta time
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame);
        self.last_frame = now;

        self.profiler.begin_frame();

        // Update day/night cycle
        self.renderer.update_time(dt.as_secs_f32());

        // Update player
        if self.cursor_grabbed {
            self.player.update(dt, Some(&self.world));
            // Process block break/place interactions
            self.player.process_block_interactions(&mut self.world);

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

        // Update world (chunk loading, meshing)
        self.profiler.begin_section("world_update");
        self.world.update(&self.player.position());
        self.profiler.end_section("world_update");

        // Upload meshes to GPU (per-chunk buffers - for fallback)
        // Note: Pool upload now happens during render after frustum culling
        self.world.upload_meshes(self.renderer.device());

        // Get target block for outline rendering
        let target_block = if self.cursor_grabbed {
            self.player.get_target_block(&self.world)
        } else {
            None
        };

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
        match self.renderer.render(&self.player.camera, &self.world, &self.profiler, target_block, &debug_metrics) {
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
