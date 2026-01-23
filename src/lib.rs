//! Blocks Engine Library
//!
//! Core modules for the voxel engine.

pub mod app;
pub mod renderer;
pub mod world;
pub mod player;
pub mod utils;

pub use app::App;

/// Engine-wide constants
pub mod constants {
    /// Chunk dimensions (16x16x16 for cache-friendly iteration)
    pub const CHUNK_SIZE: usize = 16;
    pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
    pub const CHUNK_SIZE_F32: f32 = CHUNK_SIZE as f32;

    /// Vertical chunks per column (16 chunks * 16 = 256 blocks tall)
    pub const CHUNKS_VERTICAL: usize = 16;

    /// World height in blocks
    pub const WORLD_HEIGHT: usize = CHUNK_SIZE * CHUNKS_VERTICAL;

    /// Render distance in chunks (increased for stress testing)
    pub const RENDER_DISTANCE: i32 = 24;

    /// LOD levels - distances at which to switch detail levels
    pub const LOD_FULL_DETAIL: i32 = 8;      // Full mesh detail
    pub const LOD_HALF_DETAIL: i32 = 16;     // Skip every other block
    pub const LOD_QUARTER_DETAIL: i32 = 24;  // Skip 3/4 blocks

    /// Texture atlas tile size (pixel art)
    pub const TEXTURE_TILE_SIZE: u32 = 16;
}
