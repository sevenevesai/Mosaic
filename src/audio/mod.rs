//! Audio system
//!
//! Comprehensive sound system with material-based sounds, ambient soundscapes,
//! 3D spatial audio with occlusion, and convolution reverb for caves.
//!
//! ## Sound Assets
//!
//! Place WAV files in `assets/sfx/`:
//! ```
//! assets/sfx/
//! ├── footstep/
//! │   ├── grass-1.wav, grass-2.wav
//! │   ├── stone-1.wav, stone-2.wav
//! │   ├── sand-1.wav, sand-2.wav
//! │   ├── wood-1.wav, wood-2.wav
//! │   ├── snow-1.wav, snow-2.wav
//! │   └── gravel-1.wav, gravel-2.wav
//! ├── break/
//! │   ├── stone-1.wav, stone-2.wav
//! │   ├── wood-1.wav, wood-2.wav
//! │   └── dirt-1.wav, dirt-2.wav
//! ├── place/
//! │   └── (same structure as break)
//! └── ambient/
//!     ├── cave-drips.wav
//!     ├── surface-birds.wav
//!     └── (biome-specific loops)
//! ```
//!
//! Missing sounds are gracefully handled (no crash, just silent).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use glam::Vec3;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source, SpatialSink};

use crate::world::{Block, WorldManager};

/// Base volume level for music (0.0 - 1.0), before category scaling
const MUSIC_BASE_VOLUME: f32 = 0.6;

/// Crossfade duration for music transitions (seconds)
const MUSIC_CROSSFADE_SECS: f32 = 4.0;

/// Minimum gap between tracks (seconds)
const MUSIC_GAP_MIN_SECS: f32 = 30.0;

/// Maximum gap between tracks (seconds)
const MUSIC_GAP_MAX_SECS: f32 = 90.0;

/// Initial delay before music starts (seconds)
const MUSIC_STARTUP_DELAY_SECS: f32 = 45.0;

/// Cooldown between skip requests (seconds)
const MUSIC_SKIP_COOLDOWN_SECS: f32 = 2.0;

/// Music playback state machine
#[derive(Debug, Clone, Copy, PartialEq)]
enum MusicState {
    /// Waiting for startup delay or gap timer
    Idle,
    /// Track is playing normally
    Playing,
    /// Fading out current track (for skip or transition)
    FadingOut,
    /// In silence gap between tracks
    Gap,
}

/// Track category for filtering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackCategory {
    /// Calm ambient tracks for exploration
    Ambient,
    /// Combat/intense tracks (excluded from normal rotation)
    Combat,
    /// Short stingers/discovery sounds (excluded from rotation)
    Stinger,
}

/// Sound material categories for audio feedback
/// Maps blocks to distinct sound types for footsteps, breaking, and placing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoundMaterial {
    /// Grass, dirt, mud - soft organic sounds
    Grass,
    /// Stone, brick, ore - hard mineral sounds
    Stone,
    /// Sand, gravel - granular sounds
    Sand,
    /// Wood, logs, planks - hollow woody sounds
    Wood,
    /// Snow, ice - crunchy cold sounds
    Snow,
    /// Metal ores - metallic clink sounds
    Metal,
    /// Gravel, rubble - loose rocky sounds
    Gravel,
    /// Leaves, vines - rustling foliage sounds
    Foliage,
    /// Glass, crystals - crystalline sounds
    Crystal,
    /// Default fallback
    Generic,
}

impl SoundMaterial {
    /// Get the sound material for a block type
    pub fn from_block(block: Block) -> Self {
        match block {
            // Soft organic materials
            Block::Grass | Block::Dirt | Block::JungleGrass | Block::MudBlock | Block::Peat => {
                SoundMaterial::Grass
            }

            // Hard stone materials
            Block::Stone
            | Block::StoneDirt
            | Block::Brick
            | Block::StoneRubble2
            | Block::Bedrock
            | Block::Sandstone
            | Block::RedSandstone
            | Block::MossyStone
            | Block::CrackedStone
            | Block::ClayStone
            | Block::Stalactite
            | Block::Stalagmite => SoundMaterial::Stone,

            // Sandy/granular materials
            Block::Sand => SoundMaterial::Sand,

            // Gravel and loose rocks
            Block::Gravel => SoundMaterial::Gravel,

            // Wooden materials
            Block::Wood
            | Block::OakLog
            | Block::BirchLog
            | Block::PineLog
            | Block::JungleLog
            | Block::Chest => SoundMaterial::Wood,

            // Snowy/icy materials
            Block::Snow | Block::SnowGrass | Block::Ice | Block::PackedIce => SoundMaterial::Snow,

            // Metal ores
            Block::IronOre
            | Block::GoldOre
            | Block::CopperOre
            | Block::SilverOre
            | Block::TinOre
            | Block::NickelOre
            | Block::LeadOre
            | Block::TitaniumOre => SoundMaterial::Metal,

            // Crystalline materials
            Block::DiamondOre
            | Block::EmeraldOre
            | Block::SapphireOre
            | Block::QuartzOre
            | Block::AmethystOre
            | Block::CrystalAmethyst
            | Block::CrystalQuartz => SoundMaterial::Crystal,

            // Foliage
            Block::Leaves
            | Block::JungleLeaves
            | Block::SpruceLeaves
            | Block::Vines
            | Block::DeadBush
            | Block::Cactus => SoundMaterial::Foliage,

            // Coal, redite, uranium are stone-like
            Block::CoalOre | Block::RedstoneOre | Block::UraniumOre => SoundMaterial::Stone,

            // Special cases
            Block::Air | Block::Water | Block::Torch => SoundMaterial::Generic,
        }
    }

    /// Get the subfolder name for this material's sounds
    pub fn folder_name(&self) -> &'static str {
        match self {
            SoundMaterial::Grass => "grass",
            SoundMaterial::Stone => "stone",
            SoundMaterial::Sand => "sand",
            SoundMaterial::Wood => "wood",
            SoundMaterial::Snow => "snow",
            SoundMaterial::Metal => "metal",
            SoundMaterial::Gravel => "gravel",
            SoundMaterial::Foliage => "foliage",
            SoundMaterial::Crystal => "crystal",
            SoundMaterial::Generic => "generic",
        }
    }
}

/// Base volume level for footsteps (0.0 - 1.0), before category scaling
const FOOTSTEP_VOLUME: f32 = 0.30;

/// Base volume level for block interactions (slightly louder for feedback)
const BLOCK_SOUND_VOLUME: f32 = 0.25;

/// Base volume level for ambient sounds
const AMBIENT_VOLUME: f32 = 0.35;

/// Walk cycle: step1 → step2 gap (shorter — second foot lands early in the animation)
const FOOTSTEP_WALK_SHORT: f32 = 0.35;
/// Walk cycle: step2 → step1 gap (longer — completes the cycle back to first foot)
const FOOTSTEP_WALK_LONG: f32 = 0.53;

/// Sprint cycle: same asymmetry scaled down
const FOOTSTEP_SPRINT_SHORT: f32 = 0.24;
const FOOTSTEP_SPRINT_LONG: f32 = 0.37;

/// Crossfade duration for ambient transitions (seconds)
const AMBIENT_CROSSFADE_SECS: f32 = 2.5;

/// Audio volume settings by category
#[derive(Debug, Clone)]
pub struct AudioSettings {
    /// Master volume (0.0 - 1.0)
    pub master: f32,
    /// Sound effects volume (0.0 - 1.0)
    pub sfx: f32,
    /// Ambient sounds volume (0.0 - 1.0)
    pub ambient: f32,
    /// Music volume (0.0 - 1.0)
    pub music: f32,
    /// Whether music is enabled
    pub music_enabled: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master: 0.8,
            sfx: 1.0,
            ambient: 0.6,
            music: 0.5,
            music_enabled: true,
        }
    }
}

impl AudioSettings {
    /// Get effective SFX volume (master * sfx)
    pub fn effective_sfx(&self) -> f32 {
        self.master * self.sfx
    }

    /// Get effective ambient volume (master * ambient)
    pub fn effective_ambient(&self) -> f32 {
        self.master * self.ambient
    }

    /// Get effective music volume (master * music)
    pub fn effective_music(&self) -> f32 {
        self.master * self.music
    }
}

/// Environment context for audio effects
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioEnvironment {
    /// Player is in a cave (sky light < 4)
    pub in_cave: bool,
    /// Player is underwater
    pub underwater: bool,
    /// Current biome ID (0-7)
    pub biome_id: u8,
    /// Player Y position (for height-based effects)
    pub height: f32,
    /// Sky light level at player position
    pub sky_light: u8,
}

/// Audio manager handles all sound playback
pub struct AudioManager {
    /// Audio output stream (must be kept alive)
    _stream: OutputStream,
    /// Stream handle for creating sinks
    stream_handle: OutputStreamHandle,
    /// Volume settings
    pub settings: AudioSettings,

    // === Material-based sound buffers ===
    /// Footstep sound buffers per material
    footstep_buffers: HashMap<SoundMaterial, Vec<Vec<u8>>>,
    /// Block break sound buffers per material
    break_buffers: HashMap<SoundMaterial, Vec<Vec<u8>>>,
    /// Block place sound buffers per material
    place_buffers: HashMap<SoundMaterial, Vec<Vec<u8>>>,

    // === Sound indices (for variation) ===
    /// Which footstep sound to play next per material
    footstep_indices: HashMap<SoundMaterial, usize>,
    /// Which break sound to play next per material
    break_indices: HashMap<SoundMaterial, usize>,
    /// Which place sound to play next per material
    place_indices: HashMap<SoundMaterial, usize>,

    /// Accumulated time since last footstep
    footstep_timer: f32,
    /// Which foot is next (false = first/left, true = second/right)
    footstep_right: bool,
    /// Whether footsteps were active last frame (for detecting walk start)
    footstep_active: bool,
    /// Last material walked on (for smooth transitions)
    last_footstep_material: SoundMaterial,

    // === Active sinks ===
    /// Active spatial sinks (kept alive while playing)
    spatial_sinks: Vec<SpatialSink>,
    /// Active non-spatial sinks (kept alive while playing)
    sinks: Vec<Sink>,

    // === Ambient system ===
    /// Current ambient sink (for crossfading)
    ambient_sink_a: Option<Sink>,
    /// Secondary ambient sink (for crossfading)
    ambient_sink_b: Option<Sink>,
    /// Ambient crossfade progress (0.0 = A only, 1.0 = B only)
    ambient_crossfade: f32,
    /// Whether we're currently crossfading
    ambient_crossfading: bool,
    /// Current ambient environment
    current_environment: Option<AudioEnvironment>,
    /// Ambient sound buffers by environment type
    ambient_buffers: HashMap<String, Vec<u8>>,

    // === Reverb system ===
    /// Impulse response for cave reverb
    cave_impulse: Option<Vec<f32>>,
    /// Current reverb mix (0.0 = dry, 1.0 = full reverb)
    reverb_mix: f32,
    /// Target reverb mix (smoothly interpolated)
    target_reverb_mix: f32,

    // === Music system ===
    /// Available music track paths (ambient only, shuffled)
    music_tracks: Vec<PathBuf>,
    /// Current position in shuffled track list
    music_track_index: usize,
    /// Recently played tracks (to avoid repeats)
    music_recent: Vec<usize>,
    /// Current music state
    music_state: MusicState,
    /// Timer for gap/delay/fade
    music_timer: f32,
    /// Current music sink
    music_sink: Option<Sink>,
    /// Current track volume (for fading)
    music_current_volume: f32,
    /// Target track volume
    music_target_volume: f32,
    /// Skip cooldown timer
    music_skip_cooldown: f32,
    /// Name of currently playing track (for logging)
    music_current_name: String,
}

impl AudioManager {
    /// Create a new audio manager and load sound assets
    pub fn new() -> Option<Self> {
        Self::new_with_settings(AudioSettings::default())
    }

    /// Create a new audio manager with custom settings
    pub fn new_with_settings(settings: AudioSettings) -> Option<Self> {
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
                        tracing::trace!("Optional {} sound not found: {} ({})", name, path, e);
                    }
                }
            }
            buffers
        }

        // Helper to load material sounds
        fn load_material_sounds(
            base_path: &str,
            sound_type: &str,
        ) -> HashMap<SoundMaterial, Vec<Vec<u8>>> {
            let mut map = HashMap::new();

            let materials = [
                SoundMaterial::Grass,
                SoundMaterial::Stone,
                SoundMaterial::Sand,
                SoundMaterial::Wood,
                SoundMaterial::Snow,
                SoundMaterial::Metal,
                SoundMaterial::Gravel,
                SoundMaterial::Foliage,
                SoundMaterial::Crystal,
                SoundMaterial::Generic,
            ];

            for material in materials {
                let folder = material.folder_name();
                let paths: Vec<String> = (1..=4)
                    .map(|i| format!("{}/{}/{}-{}.wav", base_path, sound_type, folder, i))
                    .collect();

                let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
                let buffers = load_sounds(&path_refs, &format!("{}-{}", sound_type, folder));

                if !buffers.is_empty() {
                    map.insert(material, buffers);
                }
            }

            map
        }

        // Load material-based footstep sounds
        let mut footstep_buffers = load_material_sounds("assets/sfx", "footstep");

        // Fallback: load legacy footstep sounds as Grass material if no new sounds found
        if footstep_buffers.is_empty() {
            let legacy = load_sounds(
                &[
                    "assets/sfx/footstep-grass-1.wav",
                    "assets/sfx/footstep-grass-2.wav",
                ],
                "footstep-legacy",
            );
            if !legacy.is_empty() {
                footstep_buffers.insert(SoundMaterial::Grass, legacy.clone());
                // Also use grass sounds as generic fallback
                footstep_buffers.insert(SoundMaterial::Generic, legacy);
            }
        }

        // Load material-based break sounds
        let mut break_buffers = load_material_sounds("assets/sfx", "break");

        // Fallback: load legacy break sounds
        if break_buffers.is_empty() {
            let legacy = load_sounds(
                &[
                    "assets/sfx/block-break-1.wav",
                    "assets/sfx/block-break-2.wav",
                ],
                "break-legacy",
            );
            if !legacy.is_empty() {
                break_buffers.insert(SoundMaterial::Generic, legacy);
            }
        }

        // Load material-based place sounds
        let mut place_buffers = load_material_sounds("assets/sfx", "place");

        // Fallback: load legacy place sounds
        if place_buffers.is_empty() {
            let legacy = load_sounds(
                &[
                    "assets/sfx/block-place-1.wav",
                    "assets/sfx/block-place-2.wav",
                ],
                "place-legacy",
            );
            if !legacy.is_empty() {
                place_buffers.insert(SoundMaterial::Generic, legacy);
            }
        }

        // Load ambient sounds
        let ambient_buffers = Self::load_ambient_sounds();

        // Generate cave impulse response for reverb
        let cave_impulse = Some(Self::generate_cave_impulse());

        // Load and categorize music tracks
        let music_tracks = Self::load_music_tracks();

        // Count total sounds loaded
        let footstep_count: usize = footstep_buffers.values().map(|v| v.len()).sum();
        let break_count: usize = break_buffers.values().map(|v| v.len()).sum();
        let place_count: usize = place_buffers.values().map(|v| v.len()).sum();
        let ambient_count = ambient_buffers.len();
        let music_count = music_tracks.len();

        let total = footstep_count + break_count + place_count;
        if total == 0 {
            tracing::warn!("No audio files loaded - audio will use procedural sounds");
        }

        tracing::info!(
            "Audio initialized: {} footstep ({} materials), {} break, {} place, {} ambient, {} music tracks",
            footstep_count,
            footstep_buffers.len(),
            break_count,
            place_count,
            ambient_count,
            music_count,
        );

        Some(Self {
            _stream: stream,
            stream_handle,
            settings,
            footstep_buffers,
            break_buffers,
            place_buffers,
            footstep_indices: HashMap::new(),
            break_indices: HashMap::new(),
            place_indices: HashMap::new(),
            footstep_timer: 0.0,
            footstep_right: false,
            footstep_active: false,
            last_footstep_material: SoundMaterial::Grass,
            spatial_sinks: Vec::new(),
            sinks: Vec::new(),
            ambient_sink_a: None,
            ambient_sink_b: None,
            ambient_crossfade: 0.0,
            ambient_crossfading: false,
            current_environment: None,
            ambient_buffers,
            cave_impulse,
            reverb_mix: 0.0,
            target_reverb_mix: 0.0,
            // Music system
            music_tracks,
            music_track_index: 0,
            music_recent: Vec::new(),
            music_state: MusicState::Idle,
            music_timer: MUSIC_STARTUP_DELAY_SECS,
            music_sink: None,
            music_current_volume: 0.0,
            music_target_volume: 0.0,
            music_skip_cooldown: 0.0,
            music_current_name: String::new(),
        })
    }

    /// Load ambient sound files
    fn load_ambient_sounds() -> HashMap<String, Vec<u8>> {
        let mut map = HashMap::new();

        let ambient_files = [
            ("cave", "assets/sfx/ambient/cave-drips.wav"),
            ("surface", "assets/sfx/ambient/surface-birds.wav"),
            ("desert", "assets/sfx/ambient/desert-wind.wav"),
            ("jungle", "assets/sfx/ambient/jungle-insects.wav"),
            ("tundra", "assets/sfx/ambient/tundra-wind.wav"),
            ("swamp", "assets/sfx/ambient/swamp-frogs.wav"),
            ("ocean", "assets/sfx/ambient/ocean-waves.wav"),
            ("underwater", "assets/sfx/ambient/underwater.wav"),
        ];

        for (name, path) in ambient_files {
            if let Ok(data) = std::fs::read(path) {
                map.insert(name.to_string(), data);
                tracing::debug!("Loaded ambient sound: {}", name);
            }
        }

        map
    }

    /// Generate a simple cave impulse response for convolution reverb
    /// Creates a synthetic IR with early reflections and exponential decay
    fn generate_cave_impulse() -> Vec<f32> {
        let sample_rate = 44100;
        let duration_secs = 1.5; // 1.5 second reverb tail
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        let mut impulse = vec![0.0f32; num_samples];

        // Initial impulse (direct sound)
        impulse[0] = 1.0;

        // Early reflections (discrete echoes from cave walls)
        // Simulate reflections at varying distances
        let reflections = [
            (0.015, 0.7),  // 15ms, 70% amplitude - close wall
            (0.025, 0.5),  // 25ms, 50% - side wall
            (0.040, 0.4),  // 40ms, 40% - far wall
            (0.055, 0.3),  // 55ms, 30% - ceiling
            (0.075, 0.25), // 75ms, 25% - floor bounce
            (0.100, 0.2),  // 100ms, 20% - distant
            (0.150, 0.15), // 150ms, 15% - very distant
            (0.200, 0.1),  // 200ms, 10% - cave depth
        ];

        for (delay_secs, amplitude) in reflections {
            let sample_idx = (delay_secs * sample_rate as f32) as usize;
            if sample_idx < num_samples {
                impulse[sample_idx] += amplitude;
            }
        }

        // Late reverb (diffuse tail with exponential decay)
        let decay_rate = 4.0; // Higher = faster decay
        let diffuse_start = (0.08 * sample_rate as f32) as usize; // Start diffuse after 80ms

        // Use a simple pseudo-random for diffuse reflections
        let mut rng_state = 12345u32;
        for i in diffuse_start..num_samples {
            // Simple LCG random
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = ((rng_state >> 16) as f32 / 32768.0) - 1.0; // -1 to 1

            let t = i as f32 / sample_rate as f32;
            let decay = (-decay_rate * t).exp();
            let diffuse_amplitude = 0.15 * decay * random;

            impulse[i] += diffuse_amplitude;
        }

        // Normalize to prevent clipping
        let max_val = impulse.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for sample in &mut impulse {
                *sample /= max_val;
            }
        }

        impulse
    }

    /// Update footsteps based on player movement
    ///
    /// Call this every frame with:
    /// - `dt`: Delta time since last frame
    /// - `horizontal_speed`: Player's horizontal movement speed (magnitude of XZ velocity)
    /// - `on_ground`: Whether the player is on the ground
    /// - `sprinting`: Whether the player is sprinting
    /// - `surface_block`: The block type the player is standing on
    pub fn update_footsteps(
        &mut self,
        dt: Duration,
        horizontal_speed: f32,
        on_ground: bool,
        sprinting: bool,
        surface_block: Option<Block>,
    ) {
        // Clean up finished sinks
        self.spatial_sinks.retain(|sink| !sink.empty());
        self.sinks.retain(|sink| !sink.empty());

        // Not moving: full reset so next walk starts clean
        if horizontal_speed < 0.5 {
            self.footstep_active = false;
            return;
        }

        // Moving but airborne (falling off edge, jumping): just pause the timer,
        // don't reset. The cycle resumes naturally on landing.
        if !on_ground {
            return;
        }

        // Fresh start after a complete stop: reset cycle and prime timer
        if !self.footstep_active {
            self.footstep_active = true;
            self.footstep_right = false;
            let first_interval = if sprinting { FOOTSTEP_SPRINT_SHORT } else { FOOTSTEP_WALK_SHORT };
            self.footstep_timer = first_interval - 0.1;
        }

        // Alternating intervals: short gap after first foot, long gap after second foot.
        // This matches the walk animation where the second foot lands early in the cycle.
        let interval = if sprinting {
            if self.footstep_right { FOOTSTEP_SPRINT_LONG } else { FOOTSTEP_SPRINT_SHORT }
        } else {
            if self.footstep_right { FOOTSTEP_WALK_LONG } else { FOOTSTEP_WALK_SHORT }
        };

        // Accumulate time
        self.footstep_timer += dt.as_secs_f32();

        // Play footstep when interval reached
        if self.footstep_timer >= interval {
            self.play_footstep(SoundMaterial::Generic);
            self.footstep_timer -= interval;
            self.footstep_right = !self.footstep_right;
        }
    }

    /// Play the next footstep sound for a specific material
    fn play_footstep(&mut self, material: SoundMaterial) {
        // Try to get buffers for this material, fall back to Generic, then Grass
        let buffers = self
            .footstep_buffers
            .get(&material)
            .or_else(|| self.footstep_buffers.get(&SoundMaterial::Generic))
            .or_else(|| self.footstep_buffers.get(&SoundMaterial::Grass));

        let buffers = match buffers {
            Some(b) if !b.is_empty() => b,
            _ => return, // No sounds available
        };

        // Get and increment index for this material
        let index = self.footstep_indices.entry(material).or_insert(0);
        let buffer = &buffers[*index % buffers.len()];
        *index = (*index + 1) % buffers.len();

        // Store last material for potential transitions
        self.last_footstep_material = material;

        // Create a cursor from the buffer and decode
        let cursor = std::io::Cursor::new(buffer.clone());
        let source = match Decoder::new(cursor) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to decode footstep: {}", e);
                return;
            }
        };

        // Apply volume (base * sfx category * master)
        let volume = FOOTSTEP_VOLUME * self.settings.effective_sfx();
        let source = source.amplify(volume);

        // Create a sink and play
        if let Ok(sink) = Sink::try_new(&self.stream_handle) {
            sink.append(source);
            self.sinks.push(sink);
        }
    }

    /// Update reverb mix based on environment (call each frame)
    pub fn update_reverb(&mut self, dt: f32, in_cave: bool) {
        self.target_reverb_mix = if in_cave { 0.4 } else { 0.0 };

        // Smooth interpolation
        let speed = 2.0; // Transition over ~0.5 seconds
        if self.reverb_mix < self.target_reverb_mix {
            self.reverb_mix = (self.reverb_mix + speed * dt).min(self.target_reverb_mix);
        } else if self.reverb_mix > self.target_reverb_mix {
            self.reverb_mix = (self.reverb_mix - speed * dt).max(self.target_reverb_mix);
        }
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
        // Calculate ear positions for spatial audio
        // Ears are +/- 0.1 units perpendicular to look direction
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
    ///
    /// - `block`: The block type being broken (for material-specific sound)
    /// - `position`: World position of the block (for 3D audio)
    /// - `listener_pos`: Player/camera position
    /// - `listener_dir`: Player look direction
    /// - `world`: World manager for occlusion calculation
    pub fn play_block_break(
        &mut self,
        block: Option<Block>,
        position: Option<Vec3>,
        listener_pos: Vec3,
        listener_dir: Vec3,
        world: Option<&WorldManager>,
    ) {
        let material = block
            .map(SoundMaterial::from_block)
            .unwrap_or(SoundMaterial::Generic);

        // Try to get buffers for this material, fall back to Generic
        let buffers = self
            .break_buffers
            .get(&material)
            .or_else(|| self.break_buffers.get(&SoundMaterial::Generic));

        let buffers = match buffers {
            Some(b) if !b.is_empty() => b,
            _ => return,
        };

        // Get and increment index for this material
        let index = self.break_indices.entry(material).or_insert(0);
        let buffer = buffers[*index % buffers.len()].clone();
        *index = (*index + 1) % buffers.len();

        let volume = BLOCK_SOUND_VOLUME * self.settings.effective_sfx();

        if let (Some(pos), Some(world)) = (position, world) {
            self.play_spatial(&buffer, pos, listener_pos, listener_dir, volume, world);
        } else {
            self.play_buffer(&buffer, volume);
        }
    }

    /// Play a block place sound
    ///
    /// - `block`: The block type being placed (for material-specific sound)
    /// - `position`: World position of the block (for 3D audio)
    /// - `listener_pos`: Player/camera position
    /// - `listener_dir`: Player look direction
    /// - `world`: World manager for occlusion calculation
    pub fn play_block_place(
        &mut self,
        block: Option<Block>,
        position: Option<Vec3>,
        listener_pos: Vec3,
        listener_dir: Vec3,
        world: Option<&WorldManager>,
    ) {
        let material = block
            .map(SoundMaterial::from_block)
            .unwrap_or(SoundMaterial::Generic);

        // Try to get buffers for this material, fall back to Generic
        let buffers = self
            .place_buffers
            .get(&material)
            .or_else(|| self.place_buffers.get(&SoundMaterial::Generic));

        let buffers = match buffers {
            Some(b) if !b.is_empty() => b,
            _ => return,
        };

        // Get and increment index for this material
        let index = self.place_indices.entry(material).or_insert(0);
        let buffer = buffers[*index % buffers.len()].clone();
        *index = (*index + 1) % buffers.len();

        let volume = BLOCK_SOUND_VOLUME * self.settings.effective_sfx();

        if let (Some(pos), Some(world)) = (position, world) {
            self.play_spatial(&buffer, pos, listener_pos, listener_dir, volume, world);
        } else {
            self.play_buffer(&buffer, volume);
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

    // ========== AMBIENT SOUND SYSTEM ==========

    /// Update ambient sounds based on environment
    ///
    /// Call this every frame with the player's environment context.
    /// Handles crossfading between different ambient soundscapes.
    pub fn update_ambient(&mut self, dt: f32, env: AudioEnvironment) {
        // Check if environment changed
        let needs_transition = match &self.current_environment {
            None => true,
            Some(current) => {
                current.in_cave != env.in_cave
                    || current.underwater != env.underwater
                    || current.biome_id != env.biome_id
            }
        };

        if needs_transition {
            self.start_ambient_transition(&env);
            self.current_environment = Some(env);
        }

        // Update crossfade
        if self.ambient_crossfading {
            self.ambient_crossfade += dt / AMBIENT_CROSSFADE_SECS;

            if self.ambient_crossfade >= 1.0 {
                // Crossfade complete - switch to new ambient
                self.ambient_crossfade = 1.0;
                self.ambient_crossfading = false;

                // Stop old ambient
                if let Some(sink) = self.ambient_sink_a.take() {
                    sink.stop();
                }

                // Move B to A
                self.ambient_sink_a = self.ambient_sink_b.take();
            }

            // Update volumes during crossfade
            let vol_a = (1.0 - self.ambient_crossfade) * AMBIENT_VOLUME * self.settings.effective_ambient();
            let vol_b = self.ambient_crossfade * AMBIENT_VOLUME * self.settings.effective_ambient();

            if let Some(ref sink) = self.ambient_sink_a {
                sink.set_volume(vol_a);
            }
            if let Some(ref sink) = self.ambient_sink_b {
                sink.set_volume(vol_b);
            }
        } else {
            // Not crossfading - just update volume in case settings changed
            let vol = AMBIENT_VOLUME * self.settings.effective_ambient();
            if let Some(ref sink) = self.ambient_sink_a {
                sink.set_volume(vol);
            }
        }

        // Update reverb based on cave state
        self.update_reverb(dt, env.in_cave);
    }

    /// Start a transition to a new ambient sound
    fn start_ambient_transition(&mut self, env: &AudioEnvironment) {
        // Determine which ambient to play
        let ambient_key = self.get_ambient_key(env);

        // Check if we have this ambient sound
        let buffer = match self.ambient_buffers.get(&ambient_key) {
            Some(b) => b.clone(),
            None => {
                // No ambient for this environment - fade to silence
                self.ambient_crossfading = true;
                self.ambient_crossfade = 0.0;
                return;
            }
        };

        // Create new ambient sink
        if let Ok(sink) = Sink::try_new(&self.stream_handle) {
            let cursor = std::io::Cursor::new(buffer);
            if let Ok(source) = Decoder::new(cursor) {
                // Loop the ambient sound
                let source = source.repeat_infinite();
                sink.append(source);
                sink.set_volume(0.0); // Start silent, will fade in

                // If no current ambient, skip crossfade
                if self.ambient_sink_a.is_none() {
                    self.ambient_sink_a = Some(sink);
                    self.ambient_crossfade = 1.0;
                    self.ambient_crossfading = false;
                } else {
                    // Start crossfade
                    self.ambient_sink_b = Some(sink);
                    self.ambient_crossfade = 0.0;
                    self.ambient_crossfading = true;
                }
            }
        }
    }

    /// Get the ambient sound key for an environment
    fn get_ambient_key(&self, env: &AudioEnvironment) -> String {
        if env.underwater {
            return "underwater".to_string();
        }

        if env.in_cave {
            return "cave".to_string();
        }

        // Biome-specific surface ambient
        match env.biome_id {
            0 => "surface".to_string(),      // Grasslands
            1 => "surface".to_string(),      // CavernousHighlands
            2 => "desert".to_string(),       // Desert
            3 => "tundra".to_string(),       // Tundra
            4 => "jungle".to_string(),       // Jungle
            5 => "swamp".to_string(),        // Swamp
            6 => "surface".to_string(),      // Mountains
            7 => "ocean".to_string(),        // Ocean
            _ => "surface".to_string(),      // Fallback
        }
    }

    /// Stop all ambient sounds
    pub fn stop_ambient(&mut self) {
        if let Some(sink) = self.ambient_sink_a.take() {
            sink.stop();
        }
        if let Some(sink) = self.ambient_sink_b.take() {
            sink.stop();
        }
        self.current_environment = None;
    }

    /// Update volume settings
    pub fn set_settings(&mut self, settings: AudioSettings) {
        self.settings = settings;
    }

    /// Get current volume settings
    pub fn get_settings(&self) -> &AudioSettings {
        &self.settings
    }

    // ========== VOLUME CONTROL HELPERS ==========

    /// Set master volume (0.0 - 1.0)
    pub fn set_master_volume(&mut self, volume: f32) {
        self.settings.master = volume.clamp(0.0, 1.0);
    }

    /// Set SFX volume (0.0 - 1.0)
    pub fn set_sfx_volume(&mut self, volume: f32) {
        self.settings.sfx = volume.clamp(0.0, 1.0);
    }

    /// Set ambient volume (0.0 - 1.0)
    pub fn set_ambient_volume(&mut self, volume: f32) {
        self.settings.ambient = volume.clamp(0.0, 1.0);
    }

    /// Set music volume (0.0 - 1.0)
    pub fn set_music_volume(&mut self, volume: f32) {
        self.settings.music = volume.clamp(0.0, 1.0);
    }

    /// Set music enabled state
    pub fn set_music_enabled(&mut self, enabled: bool) {
        let was_enabled = self.settings.music_enabled;
        self.settings.music_enabled = enabled;

        if !enabled && was_enabled {
            // Turning off: fade out current music
            if self.music_state == MusicState::Playing {
                self.music_state = MusicState::FadingOut;
                self.music_target_volume = 0.0;
            }
        } else if enabled && !was_enabled {
            // Turning on: start music system (will pick up in update_music)
            if self.music_state != MusicState::Playing && self.music_state != MusicState::FadingOut {
                self.music_state = MusicState::Idle;
                self.music_timer = 5.0; // Short delay before starting
            }
        }
    }

    // ========== MUSIC SYSTEM ==========

    /// Load music tracks from assets/music directory
    fn load_music_tracks() -> Vec<PathBuf> {
        let music_dir = std::path::Path::new("assets/music");
        let mut tracks = Vec::new();

        if !music_dir.exists() {
            tracing::warn!("Music directory not found: assets/music");
            return tracks;
        }

        // Patterns for tracks to EXCLUDE (combat, stingers)
        let exclude_patterns = [
            "enemy",
            "boss",
            "combat",
            "battle",
            "fight",
            "ghosta",
            "demon",
            "cinematic",
            "discovery",  // Short stinger
            "lava castle",
        ];

        if let Ok(entries) = std::fs::read_dir(music_dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                // Check for audio file extensions
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if !["flac", "ogg", "mp3", "wav"].contains(&ext_lower.as_str()) {
                        continue;
                    }
                } else {
                    continue;
                }

                // Check against exclude patterns
                let filename_lower = path.file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default();

                let should_exclude = exclude_patterns.iter()
                    .any(|pattern| filename_lower.contains(pattern));

                if should_exclude {
                    tracing::debug!("Excluding music track (combat/stinger): {:?}", path.file_name());
                    continue;
                }

                tracing::debug!("Found ambient music track: {:?}", path.file_name());
                tracks.push(path);
            }
        }

        // Shuffle the tracks initially
        Self::shuffle_vec(&mut tracks);

        tracing::info!("Loaded {} ambient music tracks", tracks.len());
        tracks
    }

    /// Fisher-Yates shuffle for track randomization
    fn shuffle_vec<T>(vec: &mut Vec<T>) {
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);

        let mut rng_state = seed;

        for i in (1..vec.len()).rev() {
            // Simple LCG random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            vec.swap(i, j);
        }
    }

    /// Get a random gap duration between tracks
    fn random_gap_duration() -> f32 {
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);

        let normalized = ((seed % 10000) as f32) / 10000.0;
        MUSIC_GAP_MIN_SECS + normalized * (MUSIC_GAP_MAX_SECS - MUSIC_GAP_MIN_SECS)
    }

    /// Update music system - call every frame
    pub fn update_music(&mut self, dt: f32) {
        // Update skip cooldown
        if self.music_skip_cooldown > 0.0 {
            self.music_skip_cooldown -= dt;
        }

        // If music disabled, just maintain fade out
        if !self.settings.music_enabled {
            if self.music_state == MusicState::FadingOut {
                self.update_music_fade(dt);
            }
            return;
        }

        // No tracks available
        if self.music_tracks.is_empty() {
            return;
        }

        match self.music_state {
            MusicState::Idle => {
                // Waiting for startup delay
                self.music_timer -= dt;
                if self.music_timer <= 0.0 {
                    self.start_next_track();
                }
            }

            MusicState::Playing => {
                // Update volume to match settings
                self.music_target_volume = MUSIC_BASE_VOLUME * self.settings.effective_music();
                self.update_music_fade(dt);

                // Check if track finished
                if let Some(ref sink) = self.music_sink {
                    if sink.empty() {
                        tracing::info!("Music track finished: {}", self.music_current_name);
                        self.music_sink = None;
                        self.music_state = MusicState::Gap;
                        self.music_timer = Self::random_gap_duration();
                        tracing::debug!("Music gap: {:.1}s", self.music_timer);
                    }
                }
            }

            MusicState::FadingOut => {
                self.update_music_fade(dt);

                // Check if fade complete
                if self.music_current_volume <= 0.01 {
                    // Stop the sink
                    if let Some(sink) = self.music_sink.take() {
                        sink.stop();
                    }
                    self.music_current_volume = 0.0;

                    if self.settings.music_enabled {
                        // Transition to gap, then next track
                        self.music_state = MusicState::Gap;
                        self.music_timer = 2.0; // Short gap after skip
                    } else {
                        self.music_state = MusicState::Idle;
                        self.music_timer = 5.0;
                    }
                }
            }

            MusicState::Gap => {
                // Silence gap between tracks
                self.music_timer -= dt;
                if self.music_timer <= 0.0 {
                    self.start_next_track();
                }
            }
        }
    }

    /// Update volume fading
    fn update_music_fade(&mut self, dt: f32) {
        let fade_speed = 1.0 / MUSIC_CROSSFADE_SECS;

        if self.music_current_volume < self.music_target_volume {
            self.music_current_volume = (self.music_current_volume + fade_speed * dt)
                .min(self.music_target_volume);
        } else if self.music_current_volume > self.music_target_volume {
            self.music_current_volume = (self.music_current_volume - fade_speed * dt)
                .max(self.music_target_volume);
        }

        // Apply volume to sink
        if let Some(ref sink) = self.music_sink {
            sink.set_volume(self.music_current_volume);
        }
    }

    /// Start playing the next track in the shuffle
    fn start_next_track(&mut self) {
        if self.music_tracks.is_empty() {
            self.music_state = MusicState::Idle;
            self.music_timer = 30.0;
            return;
        }

        // Get next track, avoiding recent repeats
        let mut attempts = 0;
        loop {
            self.music_track_index = (self.music_track_index + 1) % self.music_tracks.len();

            // Check if recently played (avoid last 3 tracks)
            if !self.music_recent.contains(&self.music_track_index) || attempts > self.music_tracks.len() {
                break;
            }
            attempts += 1;
        }

        // Update recent list
        self.music_recent.push(self.music_track_index);
        if self.music_recent.len() > 3 {
            self.music_recent.remove(0);
        }

        let track_path = &self.music_tracks[self.music_track_index];
        self.music_current_name = track_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        // Load and play the track
        match std::fs::File::open(track_path) {
            Ok(file) => {
                let buf_reader = std::io::BufReader::new(file);
                match Decoder::new(buf_reader) {
                    Ok(source) => {
                        // Play directly without filtering for now
                        // (Low-pass filtering requires sample conversion - can add later)
                        if let Ok(sink) = Sink::try_new(&self.stream_handle) {
                            sink.set_volume(0.0); // Start silent for fade-in
                            sink.append(source);

                            self.music_sink = Some(sink);
                            self.music_state = MusicState::Playing;
                            self.music_current_volume = 0.0;
                            self.music_target_volume = MUSIC_BASE_VOLUME * self.settings.effective_music();

                            tracing::info!("Playing music: {}", self.music_current_name);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to decode music track {:?}: {}", track_path, e);
                        // Try next track after a short delay
                        self.music_timer = 2.0;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to open music track {:?}: {}", track_path, e);
                self.music_timer = 2.0;
            }
        }
    }

    /// Skip to the next track (user-initiated)
    pub fn skip_track(&mut self) {
        // Check cooldown
        if self.music_skip_cooldown > 0.0 {
            tracing::debug!("Music skip on cooldown ({:.1}s remaining)", self.music_skip_cooldown);
            return;
        }

        if !self.settings.music_enabled {
            return;
        }

        // Set cooldown
        self.music_skip_cooldown = MUSIC_SKIP_COOLDOWN_SECS;

        if self.music_state == MusicState::Playing {
            tracing::info!("Skipping music track: {}", self.music_current_name);
            self.music_state = MusicState::FadingOut;
            self.music_target_volume = 0.0;
        } else if self.music_state == MusicState::Gap || self.music_state == MusicState::Idle {
            // If in gap, start next track immediately
            self.music_timer = 0.0;
        }
    }

    /// Stop all music playback
    pub fn stop_music(&mut self) {
        if let Some(sink) = self.music_sink.take() {
            sink.stop();
        }
        self.music_state = MusicState::Idle;
        self.music_current_volume = 0.0;
        self.music_timer = MUSIC_STARTUP_DELAY_SECS;
    }

    /// Get the name of the currently playing track (or empty if none)
    pub fn current_track_name(&self) -> &str {
        if self.music_state == MusicState::Playing || self.music_state == MusicState::FadingOut {
            &self.music_current_name
        } else {
            ""
        }
    }

    /// Check if music is currently playing
    pub fn is_music_playing(&self) -> bool {
        self.music_state == MusicState::Playing
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

