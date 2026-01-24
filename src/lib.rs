//! Blocks Engine Library
//!
//! Core modules for the voxel engine.

pub mod app;
pub mod audio;
pub mod renderer;
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
    pub const LOD_FULL_DETAIL: i32 = 8;      // Full mesh detail
    pub const LOD_HALF_DETAIL: i32 = 16;     // Skip every other block
    pub const LOD_QUARTER_DETAIL: i32 = 24;  // Skip 3/4 blocks

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
}
