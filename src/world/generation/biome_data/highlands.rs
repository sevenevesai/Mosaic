//! Cavernous Highlands biome - hilly terrain with surface caverns

use crate::world::chunk::Block;
use crate::world::generation::biome::{
    Biome, BiomeId, CaveParams, SurfaceParams, TerrainParams,
};
use crate::world::generation::caves::CaveNoise;

/// Cavernous Highlands - hilly green terrain with many surface caverns
pub struct CavernousHighlandsBiome;

impl Biome for CavernousHighlandsBiome {
    fn id(&self) -> BiomeId {
        BiomeId::CavernousHighlands
    }

    fn name(&self) -> &'static str {
        "Cavernous Highlands"
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
            surface_cave_depth: 0,  // Caves CAN open to surface
            cave_density: 1.2,      // More caves
            cavern_enabled: true,
            worm_threshold: 0.08,   // Slightly wider tunnels
            chamber_threshold: 0.55, // More common chambers
            cavern_threshold: 0.48,  // More common caverns
            variable_width: true,    // Variable width tunnels
        }
    }

    fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            tree_density: 0.8, // Slightly fewer trees due to caverns
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
        _surface_height: i32,
        cave_noise: &CaveNoise,
    ) -> bool {
        // Cavernous Highlands allows surface openings (no depth check)
        self.is_cave_cavernous(x, y, z, cave_noise)
    }
}

impl CavernousHighlandsBiome {
    /// Cavernous Highlands caves: many surface openings, large caverns
    #[inline]
    fn is_cave_cavernous(&self, x: i32, y: i32, z: i32, cave_noise: &CaveNoise) -> bool {
        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Variable tunnel width
        let width_mod = cave_noise.get_width_modifier(x, y, z);
        let threshold = 0.08 + width_mod.abs() * 0.06;

        // Primary worm caves with variable width
        let worm_scale = 0.04;
        let worm1 = cave_noise.get_worm1(xf, yf, zf, worm_scale);
        let worm2 = cave_noise.get_worm2(xf, yf, zf, worm_scale);

        if worm1.abs() < threshold && worm2.abs() < threshold {
            return true;
        }

        // Secondary larger tunnel network
        let worm_scale2 = 0.025;
        let worm3 = cave_noise.get_worm1_offset(xf, yf, zf, worm_scale2, 500.0);
        let worm4 = cave_noise.get_worm2_offset(xf, yf, zf, worm_scale2, 600.0);
        if worm3.abs() < 0.07 && worm4.abs() < 0.07 {
            return true;
        }

        // Large surface caverns (the signature feature)
        if y > 8 && y < 55 {
            let cavern = cave_noise.get_cavern(xf, yf, zf, 0.012);
            if cavern > 0.5 {
                let detail = cave_noise.get_chamber(xf, yf, zf, 0.06);
                if cavern + detail * 0.15 > 0.55 {
                    return true;
                }
            }
        }

        // Medium chambers throughout
        if y > 10 && y < 45 {
            let chamber = cave_noise.get_chamber(xf, yf, zf, 0.025);
            if chamber > 0.55 {
                return true;
            }
        }

        // Deep caverns
        if y > 5 && y < 30 {
            let deep = cave_noise.get_cavern_offset(xf, yf, zf, 0.018, 200.0);
            if deep > 0.48 {
                return true;
            }
        }

        false
    }
}
