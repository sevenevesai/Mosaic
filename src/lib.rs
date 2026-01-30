//! Mosaic Engine Library
//!
//! Core modules for the voxel engine.

pub mod app;
pub mod audio;
pub mod renderer;
pub mod settings_menu;
pub mod world;
pub mod player;
pub mod utils;

pub use app::App;

/// Engine-wide constants
pub mod constants {
    /// Chunk dimensions
    pub const CHUNK_SIZE: usize = 16;
    pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
    pub const CHUNK_SIZE_F32: f32 = CHUNK_SIZE as f32;

    /// Vertical chunks per column (16 chunks * 16 = 256 blocks tall)
    pub const CHUNKS_VERTICAL: usize = 16;

    /// World height in blocks
    pub const WORLD_HEIGHT: usize = CHUNK_SIZE * CHUNKS_VERTICAL;

    /// Render distance in chunks
    pub const RENDER_DISTANCE: i32 = 24;

    /// LOD levels - distances at which to switch detail levels
    /// Set all to RENDER_DISTANCE to disable LOD and use full detail everywhere
    pub const LOD_FULL_DETAIL: i32 = 24;     // Full mesh detail (all chunks)
    pub const LOD_HALF_DETAIL: i32 = 24;     // Disabled - same as full
    pub const LOD_QUARTER_DETAIL: i32 = 24;  // Disabled - same as full

    /// Texture atlas tile size (pixel art)
    pub const TEXTURE_TILE_SIZE: u32 = 256;

    /// Block scale - size of each block in world units
    /// 1.0 = standard Minecraft-like blocks
    /// 0.5 = half-size blocks (2x visual resolution)
    /// 0.25 = quarter-size blocks (4x visual resolution)
    pub const BLOCK_SCALE: f32 = 1.0;

    // Player physics constants
    /// Player hitbox width (X and Z)
    pub const PLAYER_WIDTH: f32 = 0.6;
    /// Player hitbox height
    pub const PLAYER_HEIGHT: f32 = 1.8;
    /// Camera height above player feet
    pub const PLAYER_EYE_HEIGHT: f32 = 1.62;

    /// Player reach distance for block interaction (blocks)
    pub const PLAYER_REACH: f32 = 5.0;

    /// Gravity acceleration (blocks/s²)
    pub const GRAVITY: f32 = 28.0;
    /// Jump initial velocity (blocks/s)
    pub const JUMP_VELOCITY: f32 = 9.0;
    /// Terminal falling velocity (blocks/s)
    pub const TERMINAL_VELOCITY: f32 = 78.0;

    // ============ WATER CONSTANTS ============

    /// Sea level Y coordinate (world blocks)
    pub const WATER_SEA_LEVEL: i32 = 32;

    /// Water surface height within a full water block (0.0-1.0)
    /// Slightly below 1.0 to allow for wave animation headroom
    pub const WATER_FULL_HEIGHT: f32 = 0.875;

    /// Maximum wave amplitude (blocks)
    pub const WATER_WAVE_AMPLITUDE: f32 = 0.15;

    /// Base wave speed multiplier
    pub const WATER_WAVE_SPEED: f32 = 0.8;

    /// Water transparency (alpha) - shallow water
    pub const WATER_ALPHA_SHALLOW: f32 = 0.7;

    /// Water transparency (alpha) - deep water
    pub const WATER_ALPHA_DEEP: f32 = 0.92;

    /// Light absorption rate per block depth (exponential decay factor)
    pub const WATER_LIGHT_ABSORPTION: f32 = 0.15;

    /// Underwater visibility distance (blocks) before full fog
    pub const WATER_VISIBILITY: f32 = 24.0;

    /// Refraction distortion strength
    pub const WATER_REFRACTION_STRENGTH: f32 = 0.04;

    /// Fresnel power for reflection blend
    pub const WATER_FRESNEL_POWER: f32 = 4.0;

    // ============ WATER FLOW SIMULATION CONSTANTS ============

    /// Water flow simulation tick rate (ticks per second)
    /// 20 Hz provides smooth flow without excessive CPU usage
    pub const WATER_FLOW_TICK_RATE: f32 = 1.0 / 20.0;

    /// Gravity constant for water pressure calculation (blocks/s^2)
    pub const WATER_FLOW_GRAVITY: f32 = 9.8;

    /// Cross-sectional area of virtual pipes between cells
    pub const WATER_FLOW_PIPE_AREA: f32 = 1.0;

    /// Length between adjacent cell centers (blocks)
    pub const WATER_FLOW_PIPE_LENGTH: f32 = 1.0;

    /// Velocity damping factor per tick (0.0-1.0)
    /// Higher values = slower energy loss = longer flowing water
    pub const WATER_FLOW_DAMPING: f32 = 0.985;

    /// Minimum flow velocity before considered stationary (blocks/s)
    pub const WATER_FLOW_MIN_VELOCITY: f32 = 0.001;

    /// Minimum height difference to trigger flow (blocks)
    pub const WATER_FLOW_MIN_HEIGHT_DIFF: f32 = 0.001;

    /// Maximum number of active water cells to simulate per tick
    pub const WATER_FLOW_MAX_ACTIVE_CELLS: usize = 50000;

    /// Radius around block changes to activate water simulation (blocks)
    pub const WATER_FLOW_ACTIVATION_RADIUS: i32 = 4;

    /// Number of simulation ticks before a stable cell becomes inactive
    pub const WATER_FLOW_STABILITY_TICKS: u32 = 10;

    /// Initial flow scale when generating water from terrain gradient
    pub const WATER_FLOW_INITIAL_SCALE: f32 = 0.5;
}
