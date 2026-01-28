//! Cave generation system
//!
//! Multi-layer noise carving for natural cave systems.
//! Implements Minecraft-style cheese/spaghetti/noodle caves plus ridged tunnel networks.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

use super::biome::CaveParams;

/// Bitflags for enabled cave types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaveTypes(u8);

impl CaveTypes {
    pub const NONE: Self = Self(0);
    pub const CHEESE: Self = Self(1 << 0);      // Large open chambers
    pub const SPAGHETTI: Self = Self(1 << 1);   // Long thin tunnels
    pub const NOODLE: Self = Self(1 << 2);      // Very thin passages
    pub const RIDGED: Self = Self(1 << 3);      // Connected tunnel networks
    pub const ALL: Self = Self(0b1111);

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl Default for CaveTypes {
    fn default() -> Self {
        // Default: spaghetti + noodle (traditional worm caves)
        Self::SPAGHETTI.or(Self::NOODLE)
    }
}

/// Cave noise generators for different cave types
pub struct CaveNoise {
    // Cheese caves: large open chambers
    cheese_noise: Fbm<Perlin>,

    // Spaghetti: long thin tunnels (zero-crossing detection)
    spaghetti_noise: Perlin,

    // Noodle: very thin passages
    noodle_noise: Perlin,

    // Ridged tunnel networks (intersection of two ridged fractals)
    tunnel_a: RidgedMulti<Perlin>,
    tunnel_b: RidgedMulti<Perlin>,

    // Legacy worm caves (for compatibility)
    worm_noise: Perlin,
    worm_noise2: Perlin,

    // Large chamber caves
    chamber_noise: Perlin,

    // Large open caverns
    cavern_noise: Perlin,

    // Variable tunnel width
    width_noise: Perlin,
}

impl CaveNoise {
    /// Create a new cave noise generator with the given seed
    pub fn new(seed: u32) -> Self {
        Self {
            // Cheese caves: low frequency Fbm for large chambers
            cheese_noise: Fbm::new(seed.wrapping_add(100))
                .set_octaves(3)
                .set_frequency(0.015)
                .set_lacunarity(2.0)
                .set_persistence(0.5),

            // Spaghetti: single Perlin for zero-crossing detection
            spaghetti_noise: Perlin::new(seed.wrapping_add(101)),

            // Noodle: higher frequency for thin passages
            noodle_noise: Perlin::new(seed.wrapping_add(102)),

            // Ridged tunnel networks
            tunnel_a: RidgedMulti::new(seed.wrapping_add(103))
                .set_octaves(1)
                .set_frequency(0.04),
            tunnel_b: RidgedMulti::new(seed.wrapping_add(104))
                .set_octaves(1)
                .set_frequency(0.04),

            // Legacy worm caves
            worm_noise: Perlin::new(seed.wrapping_add(20)),
            worm_noise2: Perlin::new(seed.wrapping_add(21)),

            // Chamber and cavern noise
            chamber_noise: Perlin::new(seed.wrapping_add(22)),
            cavern_noise: Perlin::new(seed.wrapping_add(23)),

            // Width variation
            width_noise: Perlin::new(seed.wrapping_add(24)),
        }
    }

    /// Calculate depth-based cave density multiplier
    /// Peak density around y=51 (20% of height), fading at surface and bedrock
    #[inline]
    pub fn cave_density_multiplier(y: i32) -> f64 {
        let norm_y = y as f64 / 256.0;

        // Peak at y≈51 (norm_y ≈ 0.2)
        let peak = (-((norm_y - 0.2).powi(2)) / 0.125).exp();

        // Fade out near surface (top 76 blocks)
        let surface_fade = ((256.0 - y as f64) / 76.0).min(1.0);

        // Fade out near bedrock (bottom 8 blocks)
        let bedrock_fade = (y as f64 / 8.0).min(1.0);

        peak * surface_fade * bedrock_fade
    }

    /// Check if a position is a cave using the new multi-layer system
    /// This is the enhanced cave detection with all cave types
    #[inline]
    pub fn is_cave_enhanced(
        &self,
        x: i32,
        y: i32,
        z: i32,
        cave_types: CaveTypes,
        density_mult: f64,
    ) -> bool {
        // Early exit if density is too low
        if density_mult < 0.1 {
            return false;
        }

        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Scale factor for 3D sampling (Y compressed for horizontal caves)
        let scale = 0.02;
        let (nx, ny, nz) = (xf * scale, yf * scale * 0.5, zf * scale);

        // Cheese caves: large open chambers
        if cave_types.contains(CaveTypes::CHEESE) {
            let cheese_val = self.cheese_noise.get([nx * 0.75, ny * 0.75, nz * 0.75]);
            if cheese_val > 0.65 {
                return true;
            }
        }

        // Spaghetti caves: tunnels at noise zero-crossings
        if cave_types.contains(CaveTypes::SPAGHETTI) {
            let spaghetti_val = self.spaghetti_noise.get([nx * 1.5, ny * 1.5, nz * 1.5]);
            if spaghetti_val.abs() < 0.03 {
                return true;
            }
        }

        // Noodle caves: very thin passages
        if cave_types.contains(CaveTypes::NOODLE) {
            let noodle_val = self.noodle_noise.get([nx * 3.0, ny * 3.0, nz * 3.0]);
            if noodle_val.abs() < 0.015 {
                return true;
            }
        }

        // Ridged tunnel intersection: connected passages
        if cave_types.contains(CaveTypes::RIDGED) {
            let t1 = self.tunnel_a.get([nx, ny, nz]);
            let t2 = self.tunnel_b.get([nx, ny, nz]);
            if t1 * t2 > 0.6 {
                return true;
            }
        }

        false
    }

    /// Check if a position is a cave using biome-specific parameters
    /// This maintains compatibility with the existing biome system
    #[inline]
    pub fn is_cave(&self, x: i32, y: i32, z: i32, params: &CaveParams) -> bool {
        let xf = x as f64;
        let yf = y as f64;
        let zf = z as f64;

        // Variable tunnel width if enabled
        let threshold = if params.variable_width {
            let width_mod = self.width_noise.get([xf * 0.02, yf * 0.015, zf * 0.02]);
            params.worm_threshold + width_mod.abs() * 0.06
        } else {
            params.worm_threshold
        };

        // Primary worm cave system
        let worm_scale = 0.04;
        let worm1 = self.worm_noise.get([
            xf * worm_scale,
            yf * worm_scale * 0.7,
            zf * worm_scale,
        ]);
        let worm2 = self.worm_noise2.get([
            xf * worm_scale + 100.0,
            yf * worm_scale * 0.7,
            zf * worm_scale + 100.0,
        ]);

        if worm1.abs() < threshold && worm2.abs() < threshold {
            return true;
        }

        // Secondary larger tunnel network (if variable width enabled)
        if params.variable_width {
            let worm_scale2 = 0.025;
            let worm3 = self.worm_noise.get([
                xf * worm_scale2 + 500.0,
                yf * worm_scale2 * 0.6,
                zf * worm_scale2 + 500.0,
            ]);
            let worm4 = self.worm_noise2.get([
                xf * worm_scale2 + 600.0,
                yf * worm_scale2 * 0.6,
                zf * worm_scale2 + 600.0,
            ]);
            if worm3.abs() < 0.07 && worm4.abs() < 0.07 {
                return true;
            }
        }

        // Large surface caverns (signature feature of CavernousHighlands)
        if params.cavern_enabled && y > 8 && y < 55 && params.surface_cave_depth == 0 {
            let cavern = self.cavern_noise.get([xf * 0.012, yf * 0.008, zf * 0.012]);
            if cavern > 0.5 {
                let detail = self.chamber_noise.get([xf * 0.06, yf * 0.04, zf * 0.06]);
                if cavern + detail * 0.15 > 0.55 {
                    return true;
                }
            }
        }

        // Medium chambers
        if y > 8 && y < 45 {
            let chamber = self.chamber_noise.get([xf * 0.025, yf * 0.02, zf * 0.025]);
            if chamber > params.chamber_threshold - 0.13 {
                return true;
            }
        }

        // Small chambers deep underground (for grasslands-style caves)
        if y > 8 && y < 35 && params.surface_cave_depth >= 5 {
            let chamber = self.chamber_noise.get([xf * 0.03, yf * 0.025, zf * 0.03]);
            if chamber > params.chamber_threshold {
                return true;
            }
        }

        // Deep caverns
        if params.cavern_enabled && y > 5 && y < 30 {
            let deep = self.cavern_noise.get([
                xf * 0.018 + 200.0,
                yf * 0.01,
                zf * 0.018 + 200.0,
            ]);
            if deep > params.cavern_threshold - 0.24 {
                return true;
            }
        }

        // Rare large caverns very deep (grasslands style)
        if params.cavern_enabled && y > 5 && y < 25 && params.surface_cave_depth >= 5 {
            let cavern = self.cavern_noise.get([xf * 0.015, yf * 0.01, zf * 0.015]);
            if cavern > params.cavern_threshold {
                return true;
            }
        }

        false
    }

    /// Get the width noise value at a position (for variable width tunnels)
    #[inline]
    pub fn get_width_modifier(&self, x: i32, y: i32, z: i32) -> f64 {
        self.width_noise.get([x as f64 * 0.02, y as f64 * 0.015, z as f64 * 0.02])
    }

    // ============ Accessor methods for biome-specific cave logic ============

    /// Get primary worm noise value
    #[inline]
    pub fn get_worm1(&self, xf: f64, yf: f64, zf: f64, scale: f64) -> f64 {
        self.worm_noise.get([xf * scale, yf * scale * 0.7, zf * scale])
    }

    /// Get secondary worm noise value
    #[inline]
    pub fn get_worm2(&self, xf: f64, yf: f64, zf: f64, scale: f64) -> f64 {
        self.worm_noise2.get([xf * scale + 100.0, yf * scale * 0.7, zf * scale + 100.0])
    }

    /// Get primary worm noise with offset
    #[inline]
    pub fn get_worm1_offset(&self, xf: f64, yf: f64, zf: f64, scale: f64, offset: f64) -> f64 {
        self.worm_noise.get([xf * scale + offset, yf * scale * 0.6, zf * scale + offset])
    }

    /// Get secondary worm noise with offset
    #[inline]
    pub fn get_worm2_offset(&self, xf: f64, yf: f64, zf: f64, scale: f64, offset: f64) -> f64 {
        self.worm_noise2.get([xf * scale + offset, yf * scale * 0.6, zf * scale + offset])
    }

    /// Get chamber noise value
    #[inline]
    pub fn get_chamber(&self, xf: f64, yf: f64, zf: f64, scale: f64) -> f64 {
        self.chamber_noise.get([xf * scale, yf * scale * 0.8, zf * scale])
    }

    /// Get cavern noise value
    #[inline]
    pub fn get_cavern(&self, xf: f64, yf: f64, zf: f64, scale: f64) -> f64 {
        self.cavern_noise.get([xf * scale, yf * scale * 0.67, zf * scale])
    }

    /// Get cavern noise with offset
    #[inline]
    pub fn get_cavern_offset(&self, xf: f64, yf: f64, zf: f64, scale: f64, offset: f64) -> f64 {
        self.cavern_noise.get([xf * scale + offset, yf * scale * 0.56, zf * scale + offset])
    }

    // ============ New enhanced cave accessors ============

    /// Get cheese cave noise value (large chambers)
    #[inline]
    pub fn get_cheese(&self, xf: f64, yf: f64, zf: f64) -> f64 {
        let scale = 0.015;
        self.cheese_noise.get([xf * scale, yf * scale * 0.5, zf * scale])
    }

    /// Get spaghetti cave noise value (thin tunnels)
    #[inline]
    pub fn get_spaghetti(&self, xf: f64, yf: f64, zf: f64) -> f64 {
        let scale = 0.035;
        self.spaghetti_noise.get([xf * scale, yf * scale * 0.5, zf * scale])
    }

    /// Get noodle cave noise value (very thin passages)
    #[inline]
    pub fn get_noodle(&self, xf: f64, yf: f64, zf: f64) -> f64 {
        let scale = 0.06;
        self.noodle_noise.get([xf * scale, yf * scale * 0.5, zf * scale])
    }

    /// Get ridged tunnel intersection value
    #[inline]
    pub fn get_ridged_tunnel(&self, xf: f64, yf: f64, zf: f64) -> f64 {
        let scale = 0.04;
        let t1 = self.tunnel_a.get([xf * scale, yf * scale * 0.5, zf * scale]);
        let t2 = self.tunnel_b.get([xf * scale, yf * scale * 0.5, zf * scale]);
        t1 * t2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cave_noise_creation() {
        let _noise = CaveNoise::new(12345);
    }

    #[test]
    fn test_cave_detection_deterministic() {
        let noise = CaveNoise::new(12345);
        let params = CaveParams::default();

        let result1 = noise.is_cave(100, 20, 100, &params);
        let result2 = noise.is_cave(100, 20, 100, &params);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_cave_types_bitflags() {
        assert!(CaveTypes::ALL.contains(CaveTypes::CHEESE));
        assert!(CaveTypes::ALL.contains(CaveTypes::SPAGHETTI));
        assert!(CaveTypes::ALL.contains(CaveTypes::NOODLE));
        assert!(CaveTypes::ALL.contains(CaveTypes::RIDGED));
        assert!(!CaveTypes::NONE.contains(CaveTypes::CHEESE));
    }

    #[test]
    fn test_cave_density_multiplier() {
        // Should be low at surface
        let surface = CaveNoise::cave_density_multiplier(200);
        assert!(surface < 0.5, "Surface density should be low");

        // Should be low at bedrock
        let bedrock = CaveNoise::cave_density_multiplier(2);
        assert!(bedrock < 0.5, "Bedrock density should be low");

        // Should be higher in mid-range
        let mid = CaveNoise::cave_density_multiplier(51);
        assert!(mid > bedrock, "Mid-range should have higher density than bedrock");
    }

    #[test]
    fn test_enhanced_cave_detection() {
        let noise = CaveNoise::new(12345);

        // Test with all cave types enabled
        let _result = noise.is_cave_enhanced(100, 50, 100, CaveTypes::ALL, 1.0);

        // Test with no cave types (should always be false)
        let no_caves = noise.is_cave_enhanced(100, 50, 100, CaveTypes::NONE, 1.0);
        assert!(!no_caves, "No cave types should produce no caves");

        // Test with low density (should usually be false)
        let low_density = noise.is_cave_enhanced(100, 50, 100, CaveTypes::ALL, 0.05);
        assert!(!low_density, "Low density should produce no caves");
    }

    #[test]
    fn test_cheese_cave_noise() {
        let noise = CaveNoise::new(12345);
        let val = noise.get_cheese(100.0, 50.0, 100.0);
        assert!(val >= -1.0 && val <= 1.0, "Cheese noise should be normalized");
    }

    #[test]
    fn test_spaghetti_cave_noise() {
        let noise = CaveNoise::new(12345);
        let val = noise.get_spaghetti(100.0, 50.0, 100.0);
        assert!(val >= -1.0 && val <= 1.0, "Spaghetti noise should be normalized");
    }

    #[test]
    fn test_ridged_tunnel_noise() {
        let noise = CaveNoise::new(12345);
        let val = noise.get_ridged_tunnel(100.0, 50.0, 100.0);
        // Ridged noise product can be > 1 but is typically 0-1 range
        assert!(val >= 0.0, "Ridged tunnel should be non-negative");
    }
}
