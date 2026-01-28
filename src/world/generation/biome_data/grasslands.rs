//! Grasslands biome - rolling hills with underground caves

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Default grasslands biome - rolling hills, caves stay underground
pub struct GrasslandsBiome;

impl Biome for GrasslandsBiome {
    fn id(&self) -> BiomeId {
        BiomeId::Grasslands
    }

    fn name(&self) -> &'static str {
        "Grasslands"
    }

    fn terrain_params(&self) -> TerrainParams {
        TerrainParams {
            base_height: 42,
            height_scale: 25.0,
            mountain_influence: 1.0,
            cliff_threshold: 0.6,
        }
    }

    fn cave_params(&self) -> CaveParams {
        CaveParams {
            surface_cave_depth: 5,  // No caves within 5 blocks of surface
            cave_density: 1.0,
            cavern_enabled: true,
            worm_threshold: 0.07,   // Narrower tunnels
            chamber_threshold: 0.68, // Rarer chambers
            cavern_threshold: 0.72,  // Very rare caverns
            variable_width: false,
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 1.0,
            formations_enabled: false,
            surface_block: Block::Grass,
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

        // No caves within 5 blocks of surface (prevents surface openings)
        if depth < params.surface_cave_depth {
            return false;
        }

        // Use specialized grasslands cave logic
        self.is_cave_grasslands(x, y, z, depth, cave_noise)
    }
}

impl GrasslandsBiome {
    /// Grasslands caves: stay underground, tunnel entrances, no surface openings
    #[inline]
    fn is_cave_grasslands(
        &self,
        x: i32,
        y: i32,
        z: i32,
        depth: i32,
        cave_noise: &CaveNoise,
    ) -> bool {
        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Primary worm cave system - narrower tunnels
        let worm_scale = 0.04;
        let worm1 = cave_noise.get_worm1(xf, yf, zf, worm_scale);
        let worm2 = cave_noise.get_worm2(xf, yf, zf, worm_scale);

        // Narrower threshold for Grasslands (0.07 vs 0.09)
        if worm1.abs() < 0.07 && worm2.abs() < 0.07 {
            return true;
        }

        // Small chambers deep underground only
        if y > 8 && y < 35 && depth > 10 {
            let chamber = cave_noise.get_chamber(xf, yf, zf, 0.03);
            // Higher threshold = rarer chambers (0.68)
            if chamber > 0.68 {
                return true;
            }
        }

        // Rare large caverns - only very deep underground
        if y > 5 && y < 25 && depth > 20 {
            let cavern = cave_noise.get_cavern(xf, yf, zf, 0.015);
            // Very high threshold = rare (0.72)
            if cavern > 0.72 {
                return true;
            }
        }

        false
    }
}
