//! Jungle biome - hot, humid, dense vegetation

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Jungle biome - dense tropical forest with tall trees
pub struct JungleBiome;

impl Biome for JungleBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Jungle
    }

    fn name(&self) -> &'static str {
        "Jungle"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 42,         // Standard height
            height_scale: 30.0,      // More varied terrain
            mountain_influence: 0.6, // Some hills
            cliff_threshold: 0.7,    // Rocky outcrops
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 4,   // Some cave entrances visible
            cave_density: 1.2,       // More caves
            cavern_enabled: true,    // Large underground chambers
            worm_threshold: 0.08,    // Wider tunnels
            chamber_threshold: 0.55, // More chambers
            cavern_threshold: 0.60,  // More caverns
            variable_width: true,    // Variable width tunnels
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 2.5,      // Very dense vegetation
            formations_enabled: false,
            surface_block: Block::JungleGrass,
            subsurface_block: Block::Dirt,
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

        // Allow some surface cave entrances
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }
}
