//! Prop Registry - Centralized management of voxel prop definitions
//!
//! Loads all nature props at startup and provides efficient access for rendering.
//! Props are small voxel models (grass, flowers, rocks) placed procedurally in the world.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use glam::Vec3;

use super::vox_loader::{VoxMesh, VoxModel};

/// Unique identifier for a prop type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PropId {
    // Grass variants (0-9)
    Grass01 = 0,
    Grass02 = 1,
    ForestGrass01 = 2,
    ForestGrass02 = 3,
    DeadGrass = 4,

    // Flower variants (10-29)
    Flower01 = 10,
    Flower02 = 11,
    Flower03 = 12,
    DeadFlower = 13,
    WaterFlower01 = 14,
    WaterFlower02 = 15,
    FlowerSmallPink = 16,
    FlowerSmallRed = 17,
    FlowerSmallYellow = 18,
    FlowerSmallSkyBlue = 19,

    // Rock variants (30-49)
    Stone01 = 30,
    Stone02 = 31,
    Stone03 = 32,
    ForestStone01 = 33,
    ForestStone02 = 34,
    ForestStone03 = 35,
    SandStone = 36,
    RockFlat = 37,
    Rock = 38,
}

impl PropId {
    /// Get all grass prop IDs
    pub fn grass_variants() -> &'static [PropId] {
        &[PropId::Grass01, PropId::Grass02]
    }

    /// Get all forest grass prop IDs
    pub fn forest_grass_variants() -> &'static [PropId] {
        &[PropId::ForestGrass01, PropId::ForestGrass02]
    }

    /// Get all flower prop IDs for meadows
    pub fn meadow_flower_variants() -> &'static [PropId] {
        &[
            PropId::Flower01, PropId::Flower02, PropId::Flower03,
            PropId::FlowerSmallPink, PropId::FlowerSmallRed,
            PropId::FlowerSmallYellow, PropId::FlowerSmallSkyBlue,
        ]
    }

    /// Get water flower variants
    pub fn water_flower_variants() -> &'static [PropId] {
        &[PropId::WaterFlower01, PropId::WaterFlower02]
    }

    /// Get dead vegetation variants
    pub fn dead_variants() -> &'static [PropId] {
        &[PropId::DeadGrass, PropId::DeadFlower]
    }

    /// Get stone/rock variants for grasslands
    pub fn grassland_rock_variants() -> &'static [PropId] {
        &[PropId::Stone01, PropId::Stone02, PropId::Stone03]
    }

    /// Get forest rock variants
    pub fn forest_rock_variants() -> &'static [PropId] {
        &[PropId::ForestStone01, PropId::ForestStone02, PropId::ForestStone03]
    }

    /// Get the file path for this prop
    pub fn file_path(&self) -> &'static str {
        match self {
            PropId::Grass01 => "assets/props/nature/grass/grass-01.vox",
            PropId::Grass02 => "assets/props/nature/grass/grass-02.vox",
            PropId::ForestGrass01 => "assets/props/nature/grass/forest_grass-01.vox",
            PropId::ForestGrass02 => "assets/props/nature/grass/forest_grass-02.vox",
            PropId::DeadGrass => "assets/props/nature/grass/dead-grass.vox",

            PropId::Flower01 => "assets/props/nature/flowers/flower-01.vox",
            PropId::Flower02 => "assets/props/nature/flowers/flower-02.vox",
            PropId::Flower03 => "assets/props/nature/flowers/flower-03.vox",
            PropId::DeadFlower => "assets/props/nature/flowers/dead-flower.vox",
            PropId::WaterFlower01 => "assets/props/nature/flowers/water-flower-01.vox",
            PropId::WaterFlower02 => "assets/props/nature/flowers/water-flower-02.vox",
            PropId::FlowerSmallPink => "assets/props/nature/flowers/Small/Flower_Small_Pink.vox",
            PropId::FlowerSmallRed => "assets/props/nature/flowers/Small/Flower_Small_Red.vox",
            PropId::FlowerSmallYellow => "assets/props/nature/flowers/Small/Flower_Small_Yellow.vox",
            PropId::FlowerSmallSkyBlue => "assets/props/nature/flowers/Small/Flower_Small_Sky_Blue.vox",

            PropId::Stone01 => "assets/props/nature/rocks/stone-01.vox",
            PropId::Stone02 => "assets/props/nature/rocks/stone-02.vox",
            PropId::Stone03 => "assets/props/nature/rocks/stone-03.vox",
            PropId::ForestStone01 => "assets/props/nature/rocks/forest_stone-01.vox",
            PropId::ForestStone02 => "assets/props/nature/rocks/forest_stone-02.vox",
            PropId::ForestStone03 => "assets/props/nature/rocks/forest_stone-03.vox",
            PropId::SandStone => "assets/props/nature/rocks/sand-stone.vox",
            PropId::RockFlat => "assets/props/nature/rocks/Rock_Flat.vox",
            PropId::Rock => "assets/props/nature/rocks/Rock.vox",
        }
    }

    /// Get all prop IDs that should be loaded
    pub fn all() -> &'static [PropId] {
        &[
            PropId::Grass01, PropId::Grass02,
            PropId::ForestGrass01, PropId::ForestGrass02,
            PropId::DeadGrass,
            PropId::Flower01, PropId::Flower02, PropId::Flower03,
            PropId::DeadFlower,
            PropId::WaterFlower01, PropId::WaterFlower02,
            PropId::FlowerSmallPink, PropId::FlowerSmallRed,
            PropId::FlowerSmallYellow, PropId::FlowerSmallSkyBlue,
            PropId::Stone01, PropId::Stone02, PropId::Stone03,
            PropId::ForestStone01, PropId::ForestStone02, PropId::ForestStone03,
            PropId::SandStone, PropId::RockFlat, PropId::Rock,
        ]
    }
}

/// How a prop interacts with physics/collision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionType {
    /// No collision - player walks through (grass, flowers)
    None,
    /// Solid collision - blocks player movement (rocks)
    Solid,
}

/// Definition of a prop type with rendering and gameplay metadata
pub struct PropDefinition {
    /// Unique identifier
    pub id: PropId,
    /// Display name
    pub name: &'static str,
    /// Pre-built GPU mesh (shared across all instances)
    pub mesh: Arc<VoxMesh>,
    /// Collision behavior
    pub collision: CollisionType,
    /// Wind sway strength (0.0 = none, 1.0 = full grass sway)
    pub sway_strength: f32,
    /// Scale variation range (min, max) - instances pick random in range
    pub scale_range: (f32, f32),
    /// Whether instances can have random Y rotation
    pub can_rotate: bool,
    /// Model center offset for proper placement
    pub center_offset: Vec3,
    /// Approximate bounding radius for culling
    pub bounds_radius: f32,
    /// Number of voxels in the model (for stats)
    pub voxel_count: u32,
}

impl PropDefinition {
    /// Create a new prop definition by loading and processing a .vox file
    fn new(
        device: &wgpu::Device,
        id: PropId,
        name: &'static str,
        collision: CollisionType,
        sway_strength: f32,
        scale_range: (f32, f32),
        can_rotate: bool,
        base_scale: f32,
    ) -> Result<Self, String> {
        let path = Path::new(id.file_path());
        let model = VoxModel::load(path)?;

        let voxel_count = model.voxels.len() as u32;

        // Calculate bounds from model dimensions
        let max_dim = model.size_x.max(model.size_y).max(model.size_z) as f32;
        let bounds_radius = max_dim * base_scale * 0.5 * scale_range.1;

        // Generate mesh with the base scale
        let mesh = VoxMesh::from_model(device, &model, base_scale);
        let center_offset = mesh.center;

        tracing::debug!(
            "Loaded prop '{}': {}x{}x{}, {} voxels, {} indices",
            name, model.size_x, model.size_y, model.size_z,
            voxel_count, mesh.index_count
        );

        Ok(Self {
            id,
            name,
            mesh: Arc::new(mesh),
            collision,
            sway_strength,
            scale_range,
            can_rotate,
            center_offset,
            bounds_radius,
            voxel_count,
        })
    }
}

/// Registry of all loaded prop definitions
pub struct PropRegistry {
    /// Props indexed by PropId
    props: HashMap<PropId, PropDefinition>,
    /// Stats: total voxels across all props
    total_voxels: u32,
    /// Stats: total indices across all props
    total_indices: u32,
}

impl PropRegistry {
    /// Load all nature props from disk
    pub fn new(device: &wgpu::Device) -> Self {
        let mut props = HashMap::new();
        let mut total_voxels = 0u32;
        let mut total_indices = 0u32;
        let mut loaded = 0;
        let mut failed = 0;

        // Define all props with their metadata
        // Scale is in world units per voxel (0.1 = 10 voxels per block)
        let prop_configs: &[(PropId, &str, CollisionType, f32, (f32, f32), bool, f32)] = &[
            // Grass - high sway, small scale variation, random rotation
            (PropId::Grass01, "Grass", CollisionType::None, 0.8, (0.08, 0.12), true, 0.1),
            (PropId::Grass02, "Grass", CollisionType::None, 0.8, (0.08, 0.12), true, 0.1),
            (PropId::ForestGrass01, "Forest Grass", CollisionType::None, 0.7, (0.07, 0.11), true, 0.1),
            (PropId::ForestGrass02, "Forest Grass", CollisionType::None, 0.7, (0.07, 0.11), true, 0.1),
            (PropId::DeadGrass, "Dead Grass", CollisionType::None, 0.3, (0.06, 0.10), true, 0.1),

            // Flowers - moderate sway, varied sizes
            (PropId::Flower01, "Flower", CollisionType::None, 0.5, (0.08, 0.14), true, 0.1),
            (PropId::Flower02, "Flower", CollisionType::None, 0.5, (0.08, 0.14), true, 0.1),
            (PropId::Flower03, "Flower", CollisionType::None, 0.5, (0.08, 0.14), true, 0.1),
            (PropId::DeadFlower, "Dead Flower", CollisionType::None, 0.2, (0.06, 0.10), true, 0.1),
            (PropId::WaterFlower01, "Water Lily", CollisionType::None, 0.3, (0.10, 0.16), true, 0.1),
            (PropId::WaterFlower02, "Water Lily", CollisionType::None, 0.3, (0.10, 0.16), true, 0.1),
            (PropId::FlowerSmallPink, "Small Pink Flower", CollisionType::None, 0.6, (0.06, 0.10), true, 0.1),
            (PropId::FlowerSmallRed, "Small Red Flower", CollisionType::None, 0.6, (0.06, 0.10), true, 0.1),
            (PropId::FlowerSmallYellow, "Small Yellow Flower", CollisionType::None, 0.6, (0.06, 0.10), true, 0.1),
            (PropId::FlowerSmallSkyBlue, "Small Blue Flower", CollisionType::None, 0.6, (0.06, 0.10), true, 0.1),

            // Rocks - no sway, larger scale variation, can rotate
            (PropId::Stone01, "Stone", CollisionType::Solid, 0.0, (0.15, 0.35), true, 0.1),
            (PropId::Stone02, "Stone", CollisionType::Solid, 0.0, (0.15, 0.35), true, 0.1),
            (PropId::Stone03, "Stone", CollisionType::Solid, 0.0, (0.15, 0.35), true, 0.1),
            (PropId::ForestStone01, "Forest Stone", CollisionType::Solid, 0.0, (0.12, 0.30), true, 0.1),
            (PropId::ForestStone02, "Forest Stone", CollisionType::Solid, 0.0, (0.12, 0.30), true, 0.1),
            (PropId::ForestStone03, "Forest Stone", CollisionType::Solid, 0.0, (0.12, 0.30), true, 0.1),
            (PropId::SandStone, "Sand Stone", CollisionType::Solid, 0.0, (0.12, 0.28), true, 0.1),
            (PropId::RockFlat, "Flat Rock", CollisionType::Solid, 0.0, (0.20, 0.50), true, 0.08),
            (PropId::Rock, "Rock", CollisionType::Solid, 0.0, (0.25, 0.60), true, 0.08),
        ];

        for &(id, name, collision, sway, scale_range, can_rotate, base_scale) in prop_configs {
            match PropDefinition::new(device, id, name, collision, sway, scale_range, can_rotate, base_scale) {
                Ok(def) => {
                    total_voxels += def.voxel_count;
                    total_indices += def.mesh.index_count;
                    props.insert(id, def);
                    loaded += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to load prop {:?}: {}", id, e);
                    failed += 1;
                }
            }
        }

        tracing::info!(
            "PropRegistry: loaded {}/{} props ({} voxels, {} indices total)",
            loaded, loaded + failed, total_voxels, total_indices
        );

        Self {
            props,
            total_voxels,
            total_indices,
        }
    }

    /// Get a prop definition by ID
    pub fn get(&self, id: PropId) -> Option<&PropDefinition> {
        self.props.get(&id)
    }

    /// Get mesh for a prop ID
    pub fn get_mesh(&self, id: PropId) -> Option<Arc<VoxMesh>> {
        self.props.get(&id).map(|p| Arc::clone(&p.mesh))
    }

    /// Check if a prop ID is loaded
    pub fn is_loaded(&self, id: PropId) -> bool {
        self.props.contains_key(&id)
    }

    /// Get all loaded prop definitions
    pub fn all(&self) -> impl Iterator<Item = &PropDefinition> {
        self.props.values()
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, u32, u32) {
        (self.props.len(), self.total_voxels, self.total_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_id_file_paths() {
        // Verify all prop IDs have valid file paths
        for id in PropId::all() {
            let path = id.file_path();
            assert!(!path.is_empty(), "PropId {:?} has empty path", id);
            assert!(path.ends_with(".vox"), "PropId {:?} path doesn't end with .vox", id);
        }
    }

    #[test]
    fn test_prop_variants_non_empty() {
        assert!(!PropId::grass_variants().is_empty());
        assert!(!PropId::forest_grass_variants().is_empty());
        assert!(!PropId::meadow_flower_variants().is_empty());
        assert!(!PropId::grassland_rock_variants().is_empty());
    }
}
