//! Ocean biome - deep water with underwater features

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Ocean biome - deep water areas
pub struct OceanBiome;

impl Biome for OceanBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Ocean
    }

    fn name(&self) -> &'static str {
        "Ocean"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 20,         // Below sea level
            height_scale: 15.0,      // Gentle ocean floor
            mountain_influence: 0.1, // Minimal mountains (underwater ridges)
            cliff_threshold: 0.9,    // Almost no cliffs
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 15,  // Deep underwater caves only
            cave_density: 0.4,       // Fewer caves
            cavern_enabled: true,    // Underwater caverns
            worm_threshold: 0.06,    // Narrow tunnels
            chamber_threshold: 0.75, // Rare chambers
            cavern_threshold: 0.80,  // Rare caverns
            variable_width: false,
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 0.0,      // No trees
            formations_enabled: false,
            surface_block: Block::Sand, // Ocean floor
            subsurface_block: Block::Sand,
        }
    }

    fn is_cave(
        &self,
        x: i32,
        y: i32,
        z: i32,
        surface_height: i32,
        cave_noise: &CaveNoise,
    ) -> bool {
        let params = self.cave_params();
        let depth = surface_height - y;

        // Only deep underwater caves
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }
}
