//! Biome system with trait-based architecture
//!
//! Provides biome definitions, selection via temperature/humidity, and blending.

use noise::{NoiseFn, Perlin};

use super::caves::CaveNoise;
use crate::world::chunk::Block;

/// Biome identification for serialization and lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BiomeId {
    Grasslands = 0,
    CavernousHighlands = 1,
    Desert = 2,
    Tundra = 3,
    Jungle = 4,
    Swamp = 5,
    Mountains = 6,
    Ocean = 7,
}

/// Parameters controlling terrain shape for a biome
#[derive(Debug, Clone)]
pub struct TerrainParams {
    /// Base terrain height (Y coordinate)
    pub base_height: i32,
    /// Amplitude of height noise
    pub height_scale: f64,
    /// 0.0-1.0 how much mountains affect this biome
    pub mountain_influence: f64,
    /// Slope steepness that triggers cliff material
    pub cliff_threshold: f64,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            base_height: 42,
            height_scale: 25.0,
            mountain_influence: 1.0,
            cliff_threshold: 0.6,
        }
    }
}

/// Parameters controlling cave generation
#[derive(Debug, Clone)]
pub struct CaveParams {
    /// Minimum depth below surface for caves (prevents surface openings)
    pub surface_cave_depth: i32,
    /// Multiplier on cave probability (1.0 = normal)
    pub cave_density: f64,
    /// Whether large caverns can spawn in this biome
    pub cavern_enabled: bool,
    /// Worm cave threshold (lower = more caves)
    pub worm_threshold: f64,
    /// Chamber threshold (higher = rarer chambers)
    pub chamber_threshold: f64,
    /// Cavern threshold (higher = rarer caverns)
    pub cavern_threshold: f64,
    /// Whether to use variable width tunnels
    pub variable_width: bool,
}

impl Default for CaveParams {
    fn default() -> Self {
        Self {
            surface_cave_depth: 5,
            cave_density: 1.0,
            cavern_enabled: true,
            worm_threshold: 0.07,
            chamber_threshold: 0.68,
            cavern_threshold: 0.72,
            variable_width: false,
        }
    }
}

/// Parameters controlling surface decoration
#[derive(Debug, Clone)]
pub struct SurfaceParams {
    /// Tree placement probability multiplier (1.0 = normal)
    pub tree_density: f64,
    /// Whether stone formations (arches, etc.) can spawn
    pub formations_enabled: bool,
    /// Surface block type
    pub surface_block: Block,
    /// Subsurface block type (1-3 blocks below surface)
    pub subsurface_block: Block,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        Self {
            tree_density: 1.0,
            formations_enabled: false,
            surface_block: Block::Grass,
            subsurface_block: Block::Dirt,
        }
    }
}

/// Core biome trait - defines terrain generation behavior
pub trait Biome: Send + Sync {
    /// Get the biome identifier
    fn id(&self) -> BiomeId;

    /// Get the biome name for display
    fn name(&self) -> &'static str;

    /// Get terrain parameters for this biome
    fn terrain_params(&self) -> TerrainParams;

    /// Get cave parameters for this biome
    fn cave_params(&self) -> CaveParams;

    /// Get surface decoration parameters
    fn surface_params(&self) -> SurfaceParams;

    /// Check if a position is a cave in this biome
    /// Default implementation uses CaveNoise with biome-specific params
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

        // Check minimum depth
        if depth < params.surface_cave_depth {
            return false;
        }

        cave_noise.is_cave(x, y, z, &params)
    }

    /// Get surface block for this biome at a position
    fn get_surface_block(&self, _x: i32, _y: i32, _z: i32) -> Block {
        self.surface_params().surface_block
    }

    /// Get subsurface block for this biome
    fn get_subsurface_block(&self) -> Block {
        self.surface_params().subsurface_block
    }
}

/// Registry of all biomes with temperature/humidity selection
pub struct BiomeRegistry {
    /// All registered biomes (indexed by BiomeId)
    biomes: Vec<Box<dyn Biome>>,
    /// Temperature noise (hot/cold axis)
    temperature_noise: Perlin,
    /// Humidity/moisture noise (wet/dry axis)
    humidity_noise: Perlin,
    /// Variation noise for edge detail
    variation_noise: Perlin,
    /// Mountain region detection noise
    mountain_noise: Perlin,
}

impl BiomeRegistry {
    /// Create a new biome registry with the given seed
    pub fn new(seed: u32) -> Self {
        // Pre-allocate for all biome IDs (8 biomes)
        let mut biomes: Vec<Box<dyn Biome>> = Vec::with_capacity(8);

        // Register all biomes in BiomeId order
        biomes.push(Box::new(super::biome_data::GrasslandsBiome));           // 0
        biomes.push(Box::new(super::biome_data::CavernousHighlandsBiome));   // 1
        biomes.push(Box::new(super::biome_data::DesertBiome));               // 2
        biomes.push(Box::new(super::biome_data::TundraBiome));               // 3
        biomes.push(Box::new(super::biome_data::JungleBiome));               // 4
        biomes.push(Box::new(super::biome_data::SwampBiome));                // 5
        biomes.push(Box::new(super::biome_data::MountainsBiome));            // 6
        biomes.push(Box::new(super::biome_data::OceanBiome));                // 7

        Self {
            biomes,
            temperature_noise: Perlin::new(seed.wrapping_add(60)),
            humidity_noise: Perlin::new(seed.wrapping_add(61)),
            variation_noise: Perlin::new(seed.wrapping_add(62)),
            mountain_noise: Perlin::new(seed.wrapping_add(63)),
        }
    }

    /// Get temperature value at world position (-1 to 1, cold to hot)
    fn get_temperature(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        // Very low frequency for large climate zones
        let base = self.temperature_noise.get([xf * 0.0008, zf * 0.0008]);

        // Medium frequency detail
        let detail = self.variation_noise.get([xf * 0.004, zf * 0.004]) * 0.2;

        (base + detail).clamp(-1.0, 1.0)
    }

    /// Get humidity value at world position (-1 to 1, dry to wet)
    fn get_humidity(&self, x: i32, z: i32) -> f64 {
        let xf = x as f64;
        let zf = z as f64;

        // Low frequency for large humid/arid regions
        let base = self.humidity_noise.get([xf * 0.001, zf * 0.001]);

        // Detail variation
        let detail = self.variation_noise.get([xf * 0.005 + 100.0, zf * 0.005 + 100.0]) * 0.15;

        (base + detail).clamp(-1.0, 1.0)
    }

    /// Check if position is in a mountain region
    fn is_mountain_region(&self, x: i32, z: i32) -> bool {
        let xf = x as f64;
        let zf = z as f64;

        let mountain_value = self.mountain_noise.get([xf * 0.001, zf * 0.001]);
        mountain_value > 0.4
    }

    /// Select biome based on temperature and humidity (Whittaker-style)
    fn select_biome(&self, temp: f64, humidity: f64, _is_mountain: bool) -> BiomeId {
        // Cold regions -> Tundra (~15%)
        if temp < -0.3 {
            return BiomeId::Tundra;
        }

        // Hot + Humid -> Jungle (dense forest for testing shadows) (~20%)
        if temp > 0.3 && humidity > 0.2 {
            return BiomeId::Jungle;
        }

        // Warm/dry regions -> Cavernous Highlands (~15%)
        if temp > 0.4 {
            return BiomeId::CavernousHighlands;
        }

        // Default -> Grasslands (~50%)
        BiomeId::Grasslands
    }

    /// Get the biome at a world position
    pub fn get_biome(&self, x: i32, z: i32) -> &dyn Biome {
        let id = self.get_biome_id(x, z);
        &*self.biomes[id as usize]
    }

    /// Get biome ID at a world position using temperature/humidity selection
    pub fn get_biome_id(&self, x: i32, z: i32) -> BiomeId {
        let temp = self.get_temperature(x, z);
        let humidity = self.get_humidity(x, z);
        let is_mountain = self.is_mountain_region(x, z);

        self.select_biome(temp, humidity, is_mountain)
    }

    /// Get temperature and humidity for debugging/display
    pub fn get_climate(&self, x: i32, z: i32) -> (f64, f64) {
        (self.get_temperature(x, z), self.get_humidity(x, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biome_registry_creation() {
        let registry = BiomeRegistry::new(12345);
        assert_eq!(registry.biomes.len(), 8);
    }

    #[test]
    fn test_biome_selection_deterministic() {
        let registry = BiomeRegistry::new(12345);

        // Same position should return same biome
        let biome1 = registry.get_biome_id(100, 100);
        let biome2 = registry.get_biome_id(100, 100);
        assert_eq!(biome1, biome2);
    }

    #[test]
    fn test_temperature_range() {
        let registry = BiomeRegistry::new(12345);

        for i in 0..100 {
            let temp = registry.get_temperature(i * 100, i * 100);
            assert!(temp >= -1.0 && temp <= 1.0, "Temperature out of range: {}", temp);
        }
    }

    #[test]
    fn test_humidity_range() {
        let registry = BiomeRegistry::new(12345);

        for i in 0..100 {
            let humidity = registry.get_humidity(i * 100, i * 100);
            assert!(humidity >= -1.0 && humidity <= 1.0, "Humidity out of range: {}", humidity);
        }
    }

    #[test]
    fn test_biome_selection_extremes() {
        let registry = BiomeRegistry::new(12345);

        // Cold -> Tundra
        assert_eq!(registry.select_biome(-0.5, 0.0, false), BiomeId::Tundra);

        // Hot + Dry -> Desert
        assert_eq!(registry.select_biome(0.5, -0.5, false), BiomeId::Desert);

        // Hot + Wet -> Jungle
        assert_eq!(registry.select_biome(0.5, 0.5, false), BiomeId::Jungle);

        // Very wet -> Swamp
        assert_eq!(registry.select_biome(0.0, 0.6, false), BiomeId::Swamp);

        // Mountain region
        assert_eq!(registry.select_biome(0.0, 0.0, true), BiomeId::Mountains);
    }

    #[test]
    fn test_terrain_params_default() {
        let params = TerrainParams::default();
        assert_eq!(params.base_height, 42);
        assert!(params.height_scale > 0.0);
    }

    #[test]
    fn test_cave_params_default() {
        let params = CaveParams::default();
        assert!(params.surface_cave_depth > 0);
        assert!(params.worm_threshold > 0.0);
    }

    #[test]
    fn test_surface_params_default() {
        let params = SurfaceParams::default();
        assert_eq!(params.surface_block, Block::Grass);
        assert_eq!(params.subsurface_block, Block::Dirt);
    }
}
