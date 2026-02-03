//! Vegetation placement system
//!
//! Places grass, flowers, and rocks during chunk generation using noise-based
//! density modulation and natural clustering patterns.

use noise::{NoiseFn, Perlin, Fbm, MultiFractal};

use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32, WATER_SEA_LEVEL};
use crate::world::chunk::{Block, Chunk, ChunkPos, PropPlacement};
use super::biome::{VegetationParams, BiomeId};
use super::util::fast_hash_3d;

/// Noise generators for vegetation placement
pub struct VegetationNoise {
    /// Base density noise - large scale variation
    density_noise: Fbm<Perlin>,
    /// Detail noise - small scale variation
    detail_noise: Perlin,
    /// Cluster noise - for flower grouping
    cluster_noise: Perlin,
    /// Placement hash seed
    seed: u32,
}

impl VegetationNoise {
    pub fn new(seed: u32) -> Self {
        Self {
            density_noise: Fbm::new(seed.wrapping_add(100))
                .set_octaves(3)
                .set_frequency(0.02)
                .set_persistence(0.5),
            detail_noise: Perlin::new(seed.wrapping_add(101)),
            cluster_noise: Perlin::new(seed.wrapping_add(102)),
            seed,
        }
    }

    /// Get density multiplier at world position (0.0 - 1.5)
    #[inline]
    fn get_density(&self, x: i32, z: i32) -> f32 {
        let xf = x as f64;
        let zf = z as f64;

        // Base density from FBM
        let base = self.density_noise.get([xf, zf]);

        // Small-scale detail
        let detail = self.detail_noise.get([xf * 0.1, zf * 0.1]) * 0.3;

        // Map from [-1, 1] to [0.2, 1.3] - always some variation
        ((base + detail + 1.0) * 0.55 + 0.2) as f32
    }

    /// Check if position is in a flower cluster center
    #[inline]
    fn is_cluster_center(&self, x: i32, z: i32) -> bool {
        let xf = x as f64;
        let zf = z as f64;

        // Cluster centers are rare peaks in noise
        let cluster = self.cluster_noise.get([xf * 0.08, zf * 0.08]);
        cluster > 0.6
    }

    /// Get cluster strength for flower grouping (0.0 - 1.0)
    #[inline]
    fn get_cluster_strength(&self, x: i32, z: i32) -> f32 {
        let xf = x as f64;
        let zf = z as f64;

        let cluster = self.cluster_noise.get([xf * 0.15, zf * 0.15]);
        ((cluster + 1.0) * 0.5) as f32
    }

    /// Hash-based probability check
    #[inline]
    fn hash_check(&self, x: i32, y: i32, z: i32, threshold: f32) -> bool {
        let hash = fast_hash_3d(x.wrapping_add(self.seed as i32), y, z);
        (hash % 10000) as f32 / 10000.0 < threshold
    }

    /// Get random float from hash (0.0 - 1.0)
    #[inline]
    fn hash_float(&self, x: i32, y: i32, z: i32, salt: i32) -> f32 {
        let hash = fast_hash_3d(
            x.wrapping_add(self.seed as i32).wrapping_add(salt),
            y,
            z
        );
        (hash % 10000) as f32 / 10000.0
    }

    /// Get random rotation (0 - 2π)
    #[inline]
    fn hash_rotation(&self, x: i32, z: i32) -> f32 {
        let hash = fast_hash_3d(x.wrapping_add(self.seed as i32), 12345, z);
        (hash % 10000) as f32 / 10000.0 * std::f32::consts::TAU
    }

    /// Get random scale within range
    #[inline]
    fn hash_scale(&self, x: i32, z: i32, min: f32, max: f32) -> f32 {
        let hash = fast_hash_3d(x, z.wrapping_add(self.seed as i32), 67890);
        let t = (hash % 10000) as f32 / 10000.0;
        min + t * (max - min)
    }

    /// Get sub-block offset for natural placement
    #[inline]
    fn hash_offset(&self, x: i32, z: i32, salt: i32) -> (f32, f32) {
        let hash_x = fast_hash_3d(x.wrapping_add(salt), self.seed as i32, z);
        let hash_z = fast_hash_3d(x, z.wrapping_add(salt), self.seed as i32);
        (
            (hash_x % 1000) as f32 / 1000.0 - 0.5, // -0.5 to 0.5
            (hash_z % 1000) as f32 / 1000.0 - 0.5,
        )
    }
}

/// Vegetation placement generator
pub struct VegetationGenerator<'a> {
    noise: &'a VegetationNoise,
}

impl<'a> VegetationGenerator<'a> {
    pub fn new(noise: &'a VegetationNoise) -> Self {
        Self { noise }
    }

    /// Add vegetation to a chunk
    pub fn add_vegetation<F, B>(
        &self,
        chunk: &mut Chunk,
        pos: ChunkPos,
        get_height: F,
        get_biome: B,
    ) where
        F: Fn(i32, i32) -> i32,
        B: Fn(i32, i32) -> BiomeId,
    {
        let world_x = pos.x * CHUNK_SIZE_I32;
        let world_y = pos.y * CHUNK_SIZE_I32;
        let world_z = pos.z * CHUNK_SIZE_I32;

        // Quick exit: if chunk is entirely underground, skip vegetation
        let chunk_top = world_y + CHUNK_SIZE_I32;
        let chunk_bottom = world_y;

        // Sample a few heights to check if any surface is in this chunk
        let sample_heights = [
            get_height(world_x, world_z),
            get_height(world_x + 8, world_z),
            get_height(world_x, world_z + 8),
            get_height(world_x + 8, world_z + 8),
            get_height(world_x + 15, world_z + 15),
        ];

        let min_height = *sample_heights.iter().min().unwrap();
        let max_height = *sample_heights.iter().max().unwrap();

        // Skip if chunk is entirely above or below terrain
        if chunk_bottom > max_height + 2 || chunk_top < min_height - 2 {
            return;
        }

        // Get vegetation params from biome (sample center of chunk)
        let center_biome = get_biome(world_x + 8, world_z + 8);
        let veg_params = self.get_vegetation_params(center_biome);

        if veg_params.max_props_per_chunk == 0 {
            return;
        }

        // Pre-allocate prop data
        chunk.ensure_prop_data_with_capacity(veg_params.max_props_per_chunk);

        let mut prop_count = 0;

        // Iterate over chunk columns
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                if prop_count >= veg_params.max_props_per_chunk {
                    break;
                }

                let wx = world_x + lx as i32;
                let wz = world_z + lz as i32;

                // Get terrain height
                let terrain_height = get_height(wx, wz);

                // Skip if surface not in this chunk
                if terrain_height < world_y || terrain_height >= chunk_top {
                    continue;
                }

                // Local Y of surface
                let surface_ly = (terrain_height - world_y) as usize;

                // Check surface block
                let surface_block = chunk.get_block(lx, surface_ly, lz);
                if !self.is_valid_surface(surface_block) {
                    continue;
                }

                // Check slope (skip steep terrain)
                let slope = self.calculate_slope(&get_height, wx, wz);
                if slope > 0.7 {
                    continue;
                }

                // Get local biome for this exact position
                let biome = get_biome(wx, wz);
                let local_params = self.get_vegetation_params(biome);

                // Density modulation
                let density_mult = self.noise.get_density(wx, wz);

                // Try placing different vegetation types
                prop_count += self.try_place_grass(
                    chunk, wx, wz, lx, surface_ly, lz, world_y,
                    &local_params, density_mult, slope,
                );

                prop_count += self.try_place_flowers(
                    chunk, wx, wz, lx, surface_ly, lz, world_y,
                    &local_params, density_mult, terrain_height,
                );

                prop_count += self.try_place_rocks(
                    chunk, wx, wz, lx, surface_ly, lz, world_y,
                    &local_params, density_mult, slope,
                );
            }
        }
    }

    /// Get vegetation params for a biome
    fn get_vegetation_params(&self, biome: BiomeId) -> VegetationParams {
        match biome {
            BiomeId::Grasslands => VegetationParams::grassland(),
            BiomeId::CavernousHighlands => VegetationParams {
                grass_density: 0.8,
                flower_density: 0.25,
                rock_density: 0.2,
                ..VegetationParams::grassland()
            },
            BiomeId::Jungle => VegetationParams::forest(),
            BiomeId::Tundra => VegetationParams::sparse_dead(),
            BiomeId::Swamp => VegetationParams {
                grass_density: 0.6,
                flower_density: 0.1,
                rock_density: 0.05,
                grass_props: &[2, 3], // Forest grass
                flower_props: &[14, 15], // Water flowers
                ..VegetationParams::default()
            },
            BiomeId::Desert => VegetationParams {
                grass_density: 0.0,
                flower_density: 0.0,
                rock_density: 0.15,
                grass_props: &[],
                flower_props: &[],
                rock_props: &[36], // Sand stone
                ..VegetationParams::none()
            },
            BiomeId::Mountains => VegetationParams {
                grass_density: 0.4,
                flower_density: 0.1,
                rock_density: 0.35,
                ..VegetationParams::sparse_dead()
            },
            BiomeId::Ocean => VegetationParams::none(),
        }
    }

    /// Check if a block is valid for vegetation
    #[inline]
    fn is_valid_surface(&self, block: Block) -> bool {
        matches!(
            block,
            Block::Grass
                | Block::Dirt
                | Block::JungleGrass
                | Block::SnowGrass
                | Block::MudBlock
                | Block::Sand
        )
    }

    /// Calculate terrain slope (0.0 = flat, 1.0 = vertical)
    fn calculate_slope<F>(&self, get_height: &F, x: i32, z: i32) -> f32
    where
        F: Fn(i32, i32) -> i32,
    {
        let h_center = get_height(x, z) as f32;
        let h_n = get_height(x, z - 1) as f32;
        let h_s = get_height(x, z + 1) as f32;
        let h_e = get_height(x + 1, z) as f32;
        let h_w = get_height(x - 1, z) as f32;

        let dx = (h_e - h_w).abs() / 2.0;
        let dz = (h_n - h_s).abs() / 2.0;

        // Slope magnitude (0-1 approximately)
        (dx * dx + dz * dz).sqrt().min(1.0)
    }

    /// Try to place grass at a position
    fn try_place_grass(
        &self,
        chunk: &mut Chunk,
        wx: i32, wz: i32,
        lx: usize, ly: usize, lz: usize,
        world_y: i32,
        params: &VegetationParams,
        density_mult: f32,
        slope: f32,
    ) -> usize {
        if params.grass_props.is_empty() || params.grass_density <= 0.0 {
            return 0;
        }

        // Grass probability: base density * local density * slope falloff
        let slope_factor = 1.0 - slope * 0.5; // Less grass on slopes
        let probability = params.grass_density * density_mult * slope_factor * 0.15;

        if !self.noise.hash_check(wx, world_y, wz, probability) {
            return 0;
        }

        // Select grass variant
        let variant_idx = (fast_hash_3d(wx, wz, 11111) as usize) % params.grass_props.len();
        let prop_id = params.grass_props[variant_idx];

        // Sub-block offset for natural look
        let (ox, oz) = self.noise.hash_offset(wx, wz, 1);
        let rotation = self.noise.hash_rotation(wx, wz);
        let scale = self.noise.hash_scale(wx, wz, 0.9, 1.1);

        // Position: on top of surface, slightly sunk
        let local_x = lx as f32 + 0.5 + ox * 0.6;
        let local_y = ly as f32 + 1.0 - params.grass_sink;
        let local_z = lz as f32 + 0.5 + oz * 0.6;

        chunk.add_prop(PropPlacement::new(
            prop_id,
            local_x,
            local_y,
            local_z,
            rotation,
            scale,
        ));

        1
    }

    /// Try to place flowers at a position
    fn try_place_flowers(
        &self,
        chunk: &mut Chunk,
        wx: i32, wz: i32,
        lx: usize, ly: usize, lz: usize,
        world_y: i32,
        params: &VegetationParams,
        density_mult: f32,
        terrain_height: i32,
    ) -> usize {
        if params.flower_props.is_empty() || params.flower_density <= 0.0 {
            return 0;
        }

        // Flowers less common near water
        let water_factor = if terrain_height < WATER_SEA_LEVEL + 3 { 0.3 } else { 1.0 };

        // Cluster factor: flowers prefer clustering
        let cluster_factor = if params.flowers_cluster {
            self.noise.get_cluster_strength(wx, wz) * 1.5 + 0.5
        } else {
            1.0
        };

        let probability = params.flower_density * density_mult * water_factor * cluster_factor * 0.08;

        if !self.noise.hash_check(wx, world_y + 1, wz, probability) {
            return 0;
        }

        // Select flower variant
        let variant_idx = (fast_hash_3d(wx, wz, 22222) as usize) % params.flower_props.len();
        let prop_id = params.flower_props[variant_idx];

        let (ox, oz) = self.noise.hash_offset(wx, wz, 2);
        let rotation = self.noise.hash_rotation(wx, wz);
        let scale = self.noise.hash_scale(wx, wz, 0.85, 1.15);

        let local_x = lx as f32 + 0.5 + ox * 0.7;
        let local_y = ly as f32 + 1.0 - 0.02;
        let local_z = lz as f32 + 0.5 + oz * 0.7;

        chunk.add_prop(PropPlacement::new(
            prop_id,
            local_x,
            local_y,
            local_z,
            rotation,
            scale,
        ));

        1
    }

    /// Try to place rocks at a position
    fn try_place_rocks(
        &self,
        chunk: &mut Chunk,
        wx: i32, wz: i32,
        lx: usize, ly: usize, lz: usize,
        world_y: i32,
        params: &VegetationParams,
        density_mult: f32,
        slope: f32,
    ) -> usize {
        if params.rock_props.is_empty() || params.rock_density <= 0.0 {
            return 0;
        }

        // Rocks more common on slopes
        let slope_factor = 1.0 + slope * 0.5;

        // Rocks are rarer than grass/flowers
        let probability = params.rock_density * density_mult * slope_factor * 0.03;

        if !self.noise.hash_check(wx, world_y + 2, wz, probability) {
            return 0;
        }

        // Select rock variant
        let variant_idx = (fast_hash_3d(wx, wz, 33333) as usize) % params.rock_props.len();
        let prop_id = params.rock_props[variant_idx];

        let (ox, oz) = self.noise.hash_offset(wx, wz, 3);
        let rotation = self.noise.hash_rotation(wx, wz);
        // Rocks have more size variation
        let scale = self.noise.hash_scale(wx, wz, 0.6, 1.4);

        let local_x = lx as f32 + 0.5 + ox * 0.4;
        let local_y = ly as f32 + 1.0 - 0.1; // Sink rocks slightly
        let local_z = lz as f32 + 0.5 + oz * 0.4;

        chunk.add_prop(PropPlacement::new(
            prop_id,
            local_x,
            local_y,
            local_z,
            rotation,
            scale,
        ));

        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vegetation_noise_density_range() {
        let noise = VegetationNoise::new(12345);

        for x in 0..100 {
            for z in 0..100 {
                let density = noise.get_density(x, z);
                assert!(density >= 0.0 && density <= 2.0,
                    "Density {} out of range at ({}, {})", density, x, z);
            }
        }
    }

    #[test]
    fn test_hash_rotation_range() {
        let noise = VegetationNoise::new(12345);

        for x in 0..100 {
            for z in 0..100 {
                let rot = noise.hash_rotation(x, z);
                assert!(rot >= 0.0 && rot <= std::f32::consts::TAU,
                    "Rotation {} out of range", rot);
            }
        }
    }

    #[test]
    fn test_hash_scale_range() {
        let noise = VegetationNoise::new(12345);

        for x in 0..50 {
            for z in 0..50 {
                let scale = noise.hash_scale(x, z, 0.5, 1.5);
                assert!(scale >= 0.5 && scale <= 1.5,
                    "Scale {} out of range", scale);
            }
        }
    }
}
