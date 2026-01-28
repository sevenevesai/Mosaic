//! Mountains biome - high peaks, rocky terrain, cliff bands

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Mountains biome - dramatic peaks and rocky terrain
pub struct MountainsBiome;

impl Biome for MountainsBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Mountains
    }

    fn name(&self) -> &'static str {
        "Mountains"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 55,         // Higher base elevation
            height_scale: 60.0,      // Dramatic height variation
            mountain_influence: 1.5, // Enhanced mountain features
            cliff_threshold: 0.3,    // More cliffs and rock faces
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 3,   // Some cliff-face cave entrances
            cave_density: 1.3,       // Many caves in mountains
            cavern_enabled: true,    // Large underground chambers
            worm_threshold: 0.09,    // Wider tunnels
            chamber_threshold: 0.50, // Common chambers
            cavern_threshold: 0.55,  // Common caverns
            variable_width: true,    // Variable width
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 0.3,      // Sparse alpine trees
            formations_enabled: false, // Mountains ARE the formation
            surface_block: Block::Stone, // Rocky surface
            subsurface_block: Block::Stone,
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

        // Allow cliff-face caves
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }

    fn get_surface_block(&self, _x: i32, y: i32, _z: i32) -> Block {
        // High altitude -> snow
        if y > 80 {
            Block::Snow
        } else if y > 70 {
            Block::Gravel
        } else {
            Block::Stone
        }
    }
}
