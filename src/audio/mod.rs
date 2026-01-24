//! Audio system
//!
//! Handles sound effects and music playback.

use std::time::Duration;

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};

/// Volume level for footsteps (0.0 - 1.0)
/// Keep it subtle - just slightly present
const FOOTSTEP_VOLUME: f32 = 0.15;

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
    /// Which footstep sound to play next (alternates 0, 1, 0, 1...)
    footstep_index: usize,
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

        // Load footstep sounds
        let footstep_paths = [
            "assets/sfx/footstep-grass-1.wav",
            "assets/sfx/footstep-grass-2.wav",
        ];

        let mut footstep_buffers = Vec::new();
        for path in &footstep_paths {
            match std::fs::read(path) {
                Ok(data) => {
                    footstep_buffers.push(data);
                    tracing::debug!("Loaded footstep sound: {}", path);
                }
                Err(e) => {
                    tracing::warn!("Failed to load {}: {}", path, e);
                }
            }
        }

        if footstep_buffers.is_empty() {
            tracing::warn!("No footstep sounds loaded");
            return None;
        }

        tracing::info!("Audio initialized with {} footstep sounds", footstep_buffers.len());

        Some(Self {
            _stream: stream,
            stream_handle,
            footstep_buffers,
            footstep_index: 0,
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
}
