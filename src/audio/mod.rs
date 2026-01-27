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

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};

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
        })
    }

    /// Update footsteps based on player movement
    ///
    /// Call this every frame with:
    /// - `dt`: Delta time since last frame
    /// - `horizontal_speed`: Player's horizontal movement speed (magnitude of XZ velocity)
    /// - `on_ground`: Whether the player is on the ground
    /// - `sprinting`: Whether the player is sprinting
    pub fn update_footsteps(&mut self, dt: Duration, horizontal_speed: f32, on_ground: bool, sprinting: bool) {
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
            self.play_footstep();
            // Subtract interval (not reset to 0) for smoother timing if frames are uneven
            self.footstep_timer -= interval;
        }
    }

    /// Play the next footstep sound
    fn play_footstep(&mut self) {
        if self.footstep_buffers.is_empty() {
            return;
        }

        // Get the current footstep buffer
        let buffer = &self.footstep_buffers[self.footstep_index];

        // Alternate to next footstep
        self.footstep_index = (self.footstep_index + 1) % self.footstep_buffers.len();

        // Create a cursor from the buffer and decode
        let cursor = std::io::Cursor::new(buffer.clone());
        let source = match Decoder::new(cursor) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to decode footstep: {}", e);
                return;
            }
        };

        // Play at reduced volume
        let source = source.amplify(FOOTSTEP_VOLUME);

        // Create a sink and play (fire-and-forget)
        if let Ok(sink) = Sink::try_new(&self.stream_handle) {
            sink.append(source);
            sink.detach(); // Let it play and clean up automatically
        }
    }

    /// Play a block break sound
    ///
    /// Call this when a block is successfully broken.
    /// Alternates between available break sounds for variety.
    pub fn play_block_break(&mut self) {
        if self.break_buffers.is_empty() {
            return;
        }

        let buffer = self.break_buffers[self.break_index].clone();
        self.break_index = (self.break_index + 1) % self.break_buffers.len();
        self.play_buffer(&buffer, BLOCK_SOUND_VOLUME);
    }

    /// Play a block place sound
    ///
    /// Call this when a block is successfully placed.
    /// Alternates between available place sounds for variety.
    pub fn play_block_place(&mut self) {
        if self.place_buffers.is_empty() {
            return;
        }

        let buffer = self.place_buffers[self.place_index].clone();
        self.place_index = (self.place_index + 1) % self.place_buffers.len();
        self.play_buffer(&buffer, BLOCK_SOUND_VOLUME);
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
