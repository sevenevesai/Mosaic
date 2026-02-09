//! Tundra biome - cold, snowy, sparse vegetation

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Tundra biome - frozen landscape with snow and ice
pub struct TundraBiome;

impl Biome for TundraBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Tundra
    }

    fn name(&self) -> &'static str {
        "Tundra"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 42,         // Match grasslands baseline
            height_scale: 35.0,      // Dramatic height variation for proper mountains
            mountain_influence: 1.3, // Strong mountains with snow-capped peaks
            cliff_threshold: 0.4,    // More exposed rock faces on steep slopes
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 6,   // Permafrost keeps caves deeper
            cave_density: 0.8,       // Somewhat fewer caves
            cavern_enabled: true,    // Ice caverns
            worm_threshold: 0.07,    // Standard tunnels
            chamber_threshold: 0.65, // Some chambers
            cavern_threshold: 0.70,  // Ice caverns
            variable_width: false,
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 0.15,     // Sparse spruce trees
            formations_enabled: true, // Stone spires and boulders
            surface_block: Block::SnowGrass,
            subsurface_block: Block::Dirt,
        }
    }

    fn get_surface_block(&self, _x: i32, y: i32, _z: i32) -> Block {
        if y > 75 {
            Block::Snow       // High peaks: pure snow
        } else if y > 60 {
            Block::Gravel     // Mid altitude: exposed gravel/rock
        } else {
            Block::SnowGrass  // Low altitude: normal tundra grass
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

        // Permafrost prevents shallow caves
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }
}
