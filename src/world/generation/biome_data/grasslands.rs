//! Grasslands biome - rolling hills with underground caves

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams, VegetationParams,
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

    fn vegetation_params(&self) -> VegetationParams {
        VegetationParams::grassland()
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
    /// Grasslands caves: connected multi-layer system with rooms, corridors, and vertical shafts
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

        // === Layer 1: Primary worm tunnels (shallow-mid depth) ===
        let worm_scale = 0.04;
        let worm1 = cave_noise.get_worm1(xf, yf, zf, worm_scale);
        let worm2 = cave_noise.get_worm2(xf, yf, zf, worm_scale);

        if worm1.abs() < 0.07 && worm2.abs() < 0.07 {
            return true;
        }

        // === Layer 2: Mid-depth longer tunnels (different scale for variety) ===
        // These use a larger scale (0.03) creating longer, more winding passages
        if y > 12 && y < 50 && depth > 8 {
            let worm3 = cave_noise.get_worm1_offset(xf, yf, zf, 0.03, 500.0);
            let worm4 = cave_noise.get_worm2_offset(xf, yf, zf, 0.03, 500.0);
            if worm3.abs() < 0.065 && worm4.abs() < 0.065 {
                return true;
            }
        }

        // === Layer 3: Room nodes (medium chambers at tunnel intersections) ===
        // Lowered threshold and expanded Y range for more frequent, better-connected rooms
        if y > 10 && y < 45 && depth > 8 {
            let chamber = cave_noise.get_chamber(xf, yf, zf, 0.025);
            if chamber > 0.55 {
                return true;
            }
        }

        // === Layer 4: Vertical shafts connecting depth layers ===
        // Thin vertical passages that connect horizontal tunnel layers
        if depth > 12 && y > 8 && y < 40 {
            let shaft1 = cave_noise.get_worm1(xf, yf * 0.3, zf, 0.06);
            let shaft2 = cave_noise.get_worm2(xf, yf * 0.3, zf, 0.06);
            // Very narrow threshold + compressed Y = vertical tendency
            if shaft1.abs() < 0.035 && shaft2.abs() < 0.035 {
                return true;
            }
        }

        // === Layer 5: Deep caverns (larger underground spaces) ===
        if y > 5 && y < 35 && depth > 15 {
            let cavern = cave_noise.get_cavern(xf, yf, zf, 0.015);
            if cavern > 0.65 {
                return true;
            }
        }

        false
    }
}
