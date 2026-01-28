//! Desert biome - hot, arid, sandy terrain with rock formations

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Desert biome - sand dunes, sandstone, sparse vegetation
pub struct DesertBiome;

impl Biome for DesertBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Desert
    }

    fn name(&self) -> &'static str {
        "Desert"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 45,        // Slightly higher base
            height_scale: 15.0,     // Flatter terrain (dunes are gentle)
            mountain_influence: 0.2, // Low mountains (mesas instead)
            cliff_threshold: 0.8,    // High threshold for cliff detection
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 8,   // Caves deeper below surface
            cave_density: 0.6,       // Fewer caves in desert
            cavern_enabled: true,    // Underground oases possible
            worm_threshold: 0.06,    // Tighter tunnels
            chamber_threshold: 0.72, // Rarer chambers
            cavern_threshold: 0.75,  // Rare caverns
            variable_width: false,
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 0.02,     // Very sparse (cacti only)
            formations_enabled: true, // Arches and hoodoos
            surface_block: Block::Sand,
            subsurface_block: Block::Sandstone,
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

        // No caves near surface in desert
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }
}
