//! Swamp biome - wet, muddy, shallow water throughout

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Swamp biome - murky wetlands with shallow water
pub struct SwampBiome;

impl Biome for SwampBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Swamp
    }

    fn name(&self) -> &'static str {
        "Swamp"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 34,         // Low and flat (near water level)
            height_scale: 8.0,       // Very flat terrain
            mountain_influence: 0.0, // No mountains
            cliff_threshold: 0.9,    // Almost never rocky
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 10,  // Caves very deep (waterlogged surface)
            cave_density: 0.5,       // Fewer caves (waterlogged)
            cavern_enabled: false,   // No large caverns (flood risk)
            worm_threshold: 0.06,    // Tight tunnels
            chamber_threshold: 0.80, // Very rare chambers
            cavern_threshold: 0.90,  // Almost no caverns
            variable_width: false,
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 0.6,      // Moderate trees (mangroves)
            formations_enabled: false,
            surface_block: Block::MudBlock,
            subsurface_block: Block::Peat,
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

        // No shallow caves in swamp (water table)
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }
}
