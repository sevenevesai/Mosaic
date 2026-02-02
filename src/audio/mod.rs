//! Audio system
//!
//! Handles sound effects and music playback.
//!
//! ## Sound Assets
//!
//! Place WAV files in `assets/sfx/`:
//! - `footstep-grass-1.wav`, `footstep-grass-2.wav` - Walking sounds
//! - `block-break-1.wav`, `block-break-2.wav` - Block breaking sounds
//! - `block-place-1.wav`, `block-place-2.wav` - Block placing sounds
//!
//! Missing sounds are gracefully handled (no crash, just silent).

use std::time::Duration;

use glam::Vec3;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source, SpatialSink};

use crate::world::WorldManager;

/// Volume level for footsteps (0.0 - 1.0)
const FOOTSTEP_VOLUME: f32 = 0.15;

/// Volume level for block interactions (slightly louder for feedback)
const BLOCK_SOUND_VOLUME: f32 = 0.25;

/// Time between footstep sounds when walking (seconds)
const FOOTSTEP_INTERVAL_SECS: f32 = 0.40;

/// Time between footstep sounds when sprinting (seconds)
const FOOTSTEP_INTERVAL_SPRINT_SECS: f32 = 0.28;

/// Audio manager handles all sound playback
pub struct AudioManager {
    /// Audio output stream (must be kept alive)
    _stream: OutputStream,
    /// Stream handle for creating sinks
    stream_handle: OutputStreamHandle,
    /// Footstep sound buffers (pre-loaded)
    footstep_buffers: Vec<Vec<u8>>,
    /// Block break sound buffers
    break_buffers: Vec<Vec<u8>>,
    /// Block place sound buffers
    place_buffers: Vec<Vec<u8>>,
    /// Which footstep sound to play next (alternates 0, 1, 0, 1...)
    footstep_index: usize,
    /// Which break sound to play next
    break_index: usize,
    /// Which place sound to play next
    place_index: usize,
    /// Accumulated time since last footstep (for even timing)
    footstep_timer: f32,
    /// Active spatial sinks (kept alive while playing)
    spatial_sinks: Vec<SpatialSink>,
}

impl AudioManager {
    /// Create a new audio manager and load sound assets
    pub fn new() -> Option<Self> {
        // Try to create audio output
        let (stream, stream_handle) = match OutputStream::try_default() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to initialize audio: {}", e);
                return None;
            }
        };

        // Helper to load sounds from paths
        fn load_sounds(paths: &[&str], name: &str) -> Vec<Vec<u8>> {
            let mut buffers = Vec::new();
            for path in paths {
                match std::fs::read(path) {
                    Ok(data) => {
                        buffers.push(data);
                        tracing::debug!("Loaded {} sound: {}", name, path);
                    }
                    Err(e) => {
                        tracing::debug!("Optional {} sound not found: {} ({})", name, path, e);
                    }
                }
            }
            buffers
        }

        // Load footstep sounds
        let footstep_buffers = load_sounds(
            &[
                "assets/sfx/footstep-grass-1.wav",
                "assets/sfx/footstep-grass-2.wav",
            ],
            "footstep",
        );

        // Load block break sounds
        let break_buffers = load_sounds(
            &[
                "assets/sfx/block-break-1.wav",
                "assets/sfx/block-break-2.wav",
            ],
            "block-break",
        );

        // Load block place sounds
        let place_buffers = load_sounds(
            &[
                "assets/sfx/block-place-1.wav",
                "assets/sfx/block-place-2.wav",
            ],
            "block-place",
        );

        // Count total sounds loaded
        let total = footstep_buffers.len() + break_buffers.len() + place_buffers.len();
        if total == 0 {
            tracing::warn!("No audio files loaded - audio disabled");
            return None;
        }

        tracing::info!(
            "Audio initialized: {} footstep, {} break, {} place sounds",
            footstep_buffers.len(),
            break_buffers.len(),
            place_buffers.len()
        );

        Some(Self {
            _stream: stream,
            stream_handle,
            footstep_buffers,
            break_buffers,
            place_buffers,
            footstep_index: 0,
            break_index: 0,
            place_index: 0,
            footstep_timer: 0.0,
            spatial_sinks: Vec::new(),
        })
    }

    /// Update footsteps based on player movement
    ///
    /// Call this every frame with:
    /// - `dt`: Delta time since last frame
    /// - `horizontal_speed`: Player's horizontal movement speed (magnitude of XZ velocity)
    /// - `on_ground`: Whether the player is on the ground
    /// - `sprinting`: Whether the player is sprinting
    /// - `position`: Player's current position (listener)
    /// - `forward`: Player's look direction
    /// - `world`: World manager for occlusion checks
    pub fn update_footsteps(
        &mut self,
        dt: Duration,
        horizontal_speed: f32,
        on_ground: bool,
        sprinting: bool,
        position: Vec3,
        forward: Vec3,
        world: &WorldManager
    ) {
        // Clean up finished spatial sinks
        self.spatial_sinks.retain(|sink| !sink.empty());

        // Only play footsteps when moving on ground
        if !on_ground || horizontal_speed < 0.5 {
            // Reset timer when not walking (so first step plays after short delay)
            self.footstep_timer = 0.0;
            return;
        }

        let interval = if sprinting {
            FOOTSTEP_INTERVAL_SPRINT_SECS
        } else {
            FOOTSTEP_INTERVAL_SECS
        };

        // Accumulate time
        self.footstep_timer += dt.as_secs_f32();

        // Play footstep when interval reached
        if self.footstep_timer >= interval {
            // Footstep comes from slightly below the player
            let foot_pos = position - Vec3::new(0.0, 1.5, 0.0);
            self.play_footstep(foot_pos, position, forward, world);
            // Subtract interval (not reset to 0) for smoother timing if frames are uneven
            self.footstep_timer -= interval;
        }
    }

    /// Play the next footstep sound
    fn play_footstep(&mut self, position: Vec3, listener_pos: Vec3, listener_dir: Vec3, world: &WorldManager) {
        if self.footstep_buffers.is_empty() {
            return;
        }

        // Get the current footstep buffer
        let buffer = self.footstep_buffers[self.footstep_index].clone();

        // Alternate to next footstep
        self.footstep_index = (self.footstep_index + 1) % self.footstep_buffers.len();

        self.play_spatial(&buffer, position, listener_pos, listener_dir, FOOTSTEP_VOLUME, world);
    }

    /// Play a spatial sound effect
    ///
    /// - `buffer`: Audio data
    /// - `position`: World position of sound source
    /// - `listener_pos`: World position of player/camera
    /// - `listener_dir`: Look direction of player/camera
    /// - `base_volume`: Base volume (0.0 - 1.0)
    /// - `world`: World manager for occlusion checks
    pub fn play_spatial(
        &mut self,
        buffer: &[u8],
        position: Vec3,
        listener_pos: Vec3,
        listener_dir: Vec3,
        base_volume: f32,
        world: &WorldManager,
    ) {
        // Create spatial sink
        let sink = match SpatialSink::try_new(
            &self.stream_handle,
            [listener_pos.x, listener_pos.y, listener_pos.z],
            [listener_dir.x, listener_dir.y, listener_dir.z], // Left ear (approx)
            [-listener_dir.z, 0.0, listener_dir.x], // Right ear (approx) - actually up/left vectors needed
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to create spatial sink: {}", e);
                return;
            }
        };

        // rodio's SpatialSink constructor takes:
        // try_new(handle, emitter_pos, left_ear_pos, right_ear_pos)
        // Wait, standard rodio docs say:
        // try_new(handle, position, left_ear, right_ear)
        // Where left_ear and right_ear are positions of ears relative to listener? Or world pos?
        // Actually, looking at rodio 0.17+:
        // SpatialSink::try_new(handle, emitter_pos, left_ear_pos, right_ear_pos)
        // This is tricky for a moving player.
        // Simplified approach: Calculate volume attenuation manually and use a standard sink?
        // No, let's use SpatialSink but we need to define ear positions.
        // Listener is at listener_pos. Ears are +/- 0.1 units perpendicular to dir.
        let right_vec = listener_dir.cross(Vec3::Y).normalize_or_zero() * 0.1;
        let left_ear = listener_pos - right_vec;
        let right_ear = listener_pos + right_vec;

        let sink = match SpatialSink::try_new(
            &self.stream_handle,
            [position.x, position.y, position.z],
            [left_ear.x, left_ear.y, left_ear.z],
            [right_ear.x, right_ear.y, right_ear.z],
        ) {
             Ok(s) => s,
             Err(_) => return,
        };

        // Calculate occlusion using DDA
        let occlusion = calculate_occlusion(listener_pos, position, world);
        let occlusion_vol = 1.0 - (occlusion * 0.7); // Max 70% reduction

        // Decode source
        let cursor = std::io::Cursor::new(buffer.to_vec());
        let source = match Decoder::new(cursor) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Apply volume
        sink.set_volume(base_volume * occlusion_vol);
        sink.append(source);
        
        // Store sink to keep it alive
        self.spatial_sinks.push(sink);
    }

    /// Play a block break sound
    pub fn play_block_break(&mut self, position: Option<Vec3>, listener_pos: Vec3, listener_dir: Vec3, world: Option<&WorldManager>) {
        if self.break_buffers.is_empty() {
            return;
        }

        let buffer = self.break_buffers[self.break_index].clone();
        self.break_index = (self.break_index + 1) % self.break_buffers.len();

        if let (Some(pos), Some(world)) = (position, world) {
            self.play_spatial(&buffer, pos, listener_pos, listener_dir, BLOCK_SOUND_VOLUME, world);
        } else {
            self.play_buffer(&buffer, BLOCK_SOUND_VOLUME);
        }
    }

    /// Play a block place sound
    pub fn play_block_place(&mut self, position: Option<Vec3>, listener_pos: Vec3, listener_dir: Vec3, world: Option<&WorldManager>) {
        if self.place_buffers.is_empty() {
            return;
        }

        let buffer = self.place_buffers[self.place_index].clone();
        self.place_index = (self.place_index + 1) % self.place_buffers.len();

        if let (Some(pos), Some(world)) = (position, world) {
            self.play_spatial(&buffer, pos, listener_pos, listener_dir, BLOCK_SOUND_VOLUME, world);
        } else {
            self.play_buffer(&buffer, BLOCK_SOUND_VOLUME);
        }
    }

    /// Play a sound from a buffer at the specified volume
    fn play_buffer(&self, buffer: &[u8], volume: f32) {
        let cursor = std::io::Cursor::new(buffer.to_vec());
        let source = match Decoder::new(cursor) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to decode sound: {}", e);
                return;
            }
        };

        let source = source.amplify(volume);

        if let Ok(sink) = Sink::try_new(&self.stream_handle) {
            sink.append(source);
            sink.detach();
        }
    }
}

/// Calculate audio occlusion factor between two points
/// Returns 0.0 (no occlusion) to 1.0 (fully occluded)
fn calculate_occlusion(start: Vec3, end: Vec3, world: &WorldManager) -> f32 {
    let dir = end - start;
    let dist = dir.length();
    
    if dist < 0.1 {
        return 0.0;
    }

    let dir_norm = dir / dist;
    
    // DDA Raycasting Setup
    let mut x = start.x.floor() as i32;
    let mut y = start.y.floor() as i32;
    let mut z = start.z.floor() as i32;
    
    let end_x = end.x.floor() as i32;
    let end_y = end.y.floor() as i32;
    let end_z = end.z.floor() as i32;

    let step_x = if dir_norm.x >= 0.0 { 1 } else { -1 };
    let step_y = if dir_norm.y >= 0.0 { 1 } else { -1 };
    let step_z = if dir_norm.z >= 0.0 { 1 } else { -1 };

    let mut t_max_x = if dir_norm.x != 0.0 {
        let next_boundary = if step_x > 0 { x as f32 + 1.0 } else { x as f32 };
        (next_boundary - start.x) / dir_norm.x
    } else {
        f32::MAX
    };

    let mut t_max_y = if dir_norm.y != 0.0 {
        let next_boundary = if step_y > 0 { y as f32 + 1.0 } else { y as f32 };
        (next_boundary - start.y) / dir_norm.y
    } else {
        f32::MAX
    };

    let mut t_max_z = if dir_norm.z != 0.0 {
        let next_boundary = if step_z > 0 { z as f32 + 1.0 } else { z as f32 };
        (next_boundary - start.z) / dir_norm.z
    } else {
        f32::MAX
    };

    let t_delta_x = if dir_norm.x != 0.0 { (1.0 / dir_norm.x).abs() } else { f32::MAX };
    let t_delta_y = if dir_norm.y != 0.0 { (1.0 / dir_norm.y).abs() } else { f32::MAX };
    let t_delta_z = if dir_norm.z != 0.0 { (1.0 / dir_norm.z).abs() } else { f32::MAX };

    let mut occlusion: f32 = 0.0;
    let mut t = 0.0;
    
    // Walk grid
    while t < dist {
        // Check block
        if let Some(block) = world.get_block(x, y, z) {
            if block.is_solid() {
                // Accumulate occlusion
                occlusion += 0.4;
                if occlusion >= 1.0 {
                    return 1.0;
                }
            } else if block.is_foliage() || block == crate::world::Block::Water {
                // Slight muffling for foliage/water
                occlusion += 0.1;
            }
        }
        
        // Reached target voxel?
        if x == end_x && y == end_y && z == end_z {
            break;
        }

        // Step
        if t_max_x < t_max_y && t_max_x < t_max_z {
            t = t_max_x;
            t_max_x += t_delta_x;
            x += step_x;
        } else if t_max_y < t_max_z {
            t = t_max_y;
            t_max_y += t_delta_y;
            y += step_y;
        } else {
            t = t_max_z;
            t_max_z += t_delta_z;
            z += step_z;
        }
    }

    occlusion.min(1.0)
}

