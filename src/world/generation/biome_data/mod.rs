//! Biome definitions
//!
//! Each biome defines its terrain characteristics, cave behavior, and decorations.

mod grasslands;
mod highlands;
mod desert;
mod tundra;
mod jungle;
mod swamp;
mod mountains;
mod ocean;

pub use grasslands::GrasslandsBiome;
pub use highlands::CavernousHighlandsBiome;
pub use desert::DesertBiome;
pub use tundra::TundraBiome;
pub use jungle::JungleBiome;
pub use swamp::SwampBiome;
pub use mountains::MountainsBiome;
pub use ocean::OceanBiome;
