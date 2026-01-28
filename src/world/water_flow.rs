//! Water Flow Simulation Engine
//!
//! Implements a pipe-model water simulation for realistic gravity-driven water flow.
//! The simulation runs on a 2D heightfield grid where water flows based on pressure
//! differences between adjacent cells.
//!
//! Key concepts:
//! - Pressure is proportional to water height (hydrostatic)
//! - Virtual pipes connect adjacent cells, allowing water to flow
//! - Flow rate depends on pressure difference between connected cells
//! - Volume is conserved (water doesn't appear or disappear)

use std::sync::Arc;
use ahash::{AHashMap, AHashSet};
use parking_lot::RwLock;

use crate::constants::{
    CHUNK_SIZE, CHUNK_SIZE_I32,
    WATER_FLOW_GRAVITY, WATER_FLOW_PIPE_AREA, WATER_FLOW_PIPE_LENGTH,
    WATER_FLOW_DAMPING, WATER_FLOW_MIN_VELOCITY, WATER_FLOW_MIN_HEIGHT_DIFF,
    WATER_FLOW_MAX_ACTIVE_CELLS, WATER_FLOW_STABILITY_TICKS,
};

use super::chunk::{Chunk, ChunkPos};

/// 4 cardinal directions for water flow: +X, -X, +Z, -Z
const FLOW_DIRECTIONS: [(i32, i32); 4] = [
    (1, 0),   // +X
    (-1, 0),  // -X
    (0, 1),   // +Z
    (0, -1),  // -Z
];

/// Direction indices for flux array
const DIR_POS_X: usize = 0;
const DIR_NEG_X: usize = 1;
const DIR_POS_Z: usize = 2;
const DIR_NEG_Z: usize = 3;

/// Convert world X,Z coordinates to chunk position and local coordinates
#[inline]
pub fn world_to_chunk_local_2d(x: i32, z: i32) -> (ChunkPos, usize, usize) {
    let chunk_pos = ChunkPos {
        x: x.div_euclid(CHUNK_SIZE_I32),
        y: 0, // Water is 2D heightfield, Y component not used for water chunks
        z: z.div_euclid(CHUNK_SIZE_I32),
    };
    let local_x = x.rem_euclid(CHUNK_SIZE_I32) as usize;
    let local_z = z.rem_euclid(CHUNK_SIZE_I32) as usize;
    (chunk_pos, local_x, local_z)
}

/// State of an active water cell being simulated
#[derive(Clone, Copy, Debug)]
pub struct ActiveWaterCell {
    /// World X coordinate
    pub world_x: i32,
    /// World Z coordinate
    pub world_z: i32,
    /// Number of ticks this cell has been stable (low velocity change)
    pub stability_counter: u32,
}

/// Flux data for a water cell - outgoing flow to each neighbor
#[derive(Clone, Copy, Debug, Default)]
pub struct CellFlux {
    /// Outgoing flux in each direction [+X, -X, +Z, -Z]
    /// Positive values = water flowing out in that direction
    pub flux: [f32; 4],
}

impl CellFlux {
    /// Get total outgoing flux (sum of positive flux values)
    #[inline]
    pub fn total_outflow(&self) -> f32 {
        self.flux.iter().map(|&f| f.max(0.0)).sum()
    }

    /// Scale all flux values by a factor
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        for f in &mut self.flux {
            *f *= factor;
        }
    }
}

/// Water flow simulation parameters
#[derive(Clone, Debug)]
pub struct FlowParams {
    /// Gravity constant for pressure calculation
    pub gravity: f32,
    /// Cross-sectional area of virtual pipes
    pub pipe_area: f32,
    /// Length between cell centers
    pub pipe_length: f32,
    /// Velocity damping per tick
    pub damping: f32,
    /// Minimum velocity threshold
    pub min_velocity: f32,
    /// Minimum height difference to trigger flow
    pub min_height_diff: f32,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            gravity: WATER_FLOW_GRAVITY,
            pipe_area: WATER_FLOW_PIPE_AREA,
            pipe_length: WATER_FLOW_PIPE_LENGTH,
            damping: WATER_FLOW_DAMPING,
            min_velocity: WATER_FLOW_MIN_VELOCITY,
            min_height_diff: WATER_FLOW_MIN_HEIGHT_DIFF,
        }
    }
}

/// Water flow simulation engine
///
/// Manages active water cells and runs pipe-model simulation to update
/// water heights and flow velocities across the world.
pub struct WaterFlowEngine {
    /// Active water cells that need simulation
    /// Key: (world_x, world_z) position
    active_cells: AHashMap<(i32, i32), ActiveWaterCell>,

    /// Flux values for each active cell
    /// Stores outgoing flow in each of 4 directions
    flux: AHashMap<(i32, i32), CellFlux>,

    /// Chunks that have been modified and need water mesh regeneration
    dirty_chunks: AHashSet<ChunkPos>,

    /// Simulation tick counter
    tick: u64,

    /// Simulation parameters
    pub params: FlowParams,

    /// Whether simulation is enabled
    pub enabled: bool,
}

impl WaterFlowEngine {
    pub fn new() -> Self {
        Self {
            active_cells: AHashMap::new(),
            flux: AHashMap::new(),
            dirty_chunks: AHashSet::new(),
            tick: 0,
            params: FlowParams::default(),
            enabled: true,
        }
    }

    /// Get current simulation tick
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get number of active water cells
    pub fn active_cell_count(&self) -> usize {
        self.active_cells.len()
    }

    /// Check if a position has active water
    pub fn is_active(&self, x: i32, z: i32) -> bool {
        self.active_cells.contains_key(&(x, z))
    }

    /// Activate a water cell for simulation
    pub fn activate(&mut self, x: i32, z: i32) {
        if self.active_cells.len() < WATER_FLOW_MAX_ACTIVE_CELLS {
            self.active_cells.entry((x, z)).or_insert(ActiveWaterCell {
                world_x: x,
                world_z: z,
                stability_counter: 0,
            });
            self.flux.entry((x, z)).or_insert(CellFlux::default());
        }
    }

    /// Activate all water cells within a radius of a world position
    pub fn activate_region(&mut self, center_x: i32, center_z: i32, radius: i32) {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                self.activate(center_x + dx, center_z + dz);
            }
        }
    }

    /// Deactivate a water cell
    pub fn deactivate(&mut self, x: i32, z: i32) {
        self.active_cells.remove(&(x, z));
        self.flux.remove(&(x, z));
    }

    /// Mark a chunk as dirty (needs water mesh regeneration)
    fn mark_dirty(&mut self, chunk_pos: ChunkPos) {
        self.dirty_chunks.insert(chunk_pos);
    }

    /// Take all dirty chunks, clearing the internal set
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkPos> {
        self.dirty_chunks.drain().collect()
    }

    /// Get water height at a world position from chunk data
    fn get_water_height(
        &self,
        x: i32,
        z: i32,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> Option<f32> {
        let (chunk_pos, local_x, local_z) = world_to_chunk_local_2d(x, z);

        // For water, we need chunks at any Y level that might have water
        // Check Y=0 chunk first (where sea level usually is), then Y=1, Y=2
        for y in 0..3 {
            let pos = ChunkPos { x: chunk_pos.x, y, z: chunk_pos.z };
            if let Some(chunk) = chunks.get(&pos) {
                if let Some(water_data) = chunk.water_data() {
                    let col = water_data.get_column(local_x, local_z);
                    if col.has_water {
                        return Some(col.surface_y);
                    }
                }
            }
        }
        None
    }

    /// Get terrain height at a world position (top of solid blocks)
    fn get_terrain_height(
        &self,
        x: i32,
        z: i32,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> f32 {
        let (chunk_pos, local_x, local_z) = world_to_chunk_local_2d(x, z);

        // Search from top down to find highest solid block
        for y in (0..8).rev() {
            let pos = ChunkPos { x: chunk_pos.x, y, z: chunk_pos.z };
            if let Some(chunk) = chunks.get(&pos) {
                for local_y in (0..CHUNK_SIZE).rev() {
                    let block = chunk.get_block(local_x, local_y, local_z);
                    if block.is_solid() {
                        // Return top of this block
                        return y as f32 * CHUNK_SIZE as f32 + local_y as f32 + 1.0;
                    }
                }
            }
        }

        // No solid found, return 0 (bottom of world)
        0.0
    }

    /// Check if a cell is a water source (shouldn't drain)
    fn is_source(
        &self,
        x: i32,
        z: i32,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> bool {
        let (chunk_pos, local_x, local_z) = world_to_chunk_local_2d(x, z);

        for y in 0..3 {
            let pos = ChunkPos { x: chunk_pos.x, y, z: chunk_pos.z };
            if let Some(chunk) = chunks.get(&pos) {
                if let Some(water_data) = chunk.water_data() {
                    let col = water_data.get_column(local_x, local_z);
                    if col.has_water && col.is_source {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Run one simulation tick
    ///
    /// This implements the pipe model algorithm:
    /// 1. Calculate pressure at each active cell
    /// 2. Update flux based on pressure differences
    /// 3. Scale outflow to prevent over-draining
    /// 4. Apply flux to update water heights
    /// 5. Calculate velocity from net flux
    /// 6. Update flow angles
    pub fn simulate_tick(
        &mut self,
        dt: f32,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) -> Vec<(ChunkPos, Vec<WaterUpdate>)> {
        if !self.enabled || self.active_cells.is_empty() {
            return Vec::new();
        }

        self.tick += 1;
        let chunks_read = chunks.read();

        // Collect all water heights and terrain heights for active cells
        let mut heights: AHashMap<(i32, i32), (f32, f32)> = AHashMap::new(); // (water_height, terrain_height)

        for (&(x, z), _) in &self.active_cells {
            let water_h = self.get_water_height(x, z, &chunks_read).unwrap_or(0.0);
            let terrain_h = self.get_terrain_height(x, z, &chunks_read);
            heights.insert((x, z), (water_h, terrain_h));
        }

        // Also collect heights for neighbors of active cells
        for (&(x, z), _) in &self.active_cells {
            for (dx, dz) in FLOW_DIRECTIONS {
                let nx = x + dx;
                let nz = z + dz;
                if !heights.contains_key(&(nx, nz)) {
                    let water_h = self.get_water_height(nx, nz, &chunks_read).unwrap_or(0.0);
                    let terrain_h = self.get_terrain_height(nx, nz, &chunks_read);
                    heights.insert((nx, nz), (water_h, terrain_h));
                }
            }
        }

        // Step 1: Update flux based on pressure differences
        let pipe_factor = self.params.gravity * self.params.pipe_area / self.params.pipe_length * dt;

        for (&(x, z), _) in &self.active_cells {
            let (self_water, self_terrain) = heights.get(&(x, z)).copied().unwrap_or((0.0, 0.0));
            let self_surface = self_water.max(self_terrain); // Effective surface height

            if self_water <= self_terrain {
                // No water at this cell
                continue;
            }

            let flux_entry = self.flux.entry((x, z)).or_insert(CellFlux::default());

            for (dir_idx, (dx, dz)) in FLOW_DIRECTIONS.iter().enumerate() {
                let nx = x + dx;
                let nz = z + dz;

                let (neighbor_water, neighbor_terrain) = heights.get(&(nx, nz)).copied().unwrap_or((0.0, f32::MAX));
                let neighbor_surface = neighbor_water.max(neighbor_terrain);

                // Pressure difference drives flow
                let delta_h = self_surface - neighbor_surface;

                // Update flux with acceleration from pressure difference
                flux_entry.flux[dir_idx] *= self.params.damping;
                flux_entry.flux[dir_idx] += delta_h * pipe_factor;

                // Clamp to prevent backward flow through terrain
                // Water can't flow "uphill" against terrain
                if neighbor_terrain > self_water {
                    flux_entry.flux[dir_idx] = flux_entry.flux[dir_idx].min(0.0);
                }
            }
        }

        // Step 2: Scale outflow to prevent draining more than available
        for (&(x, z), _) in &self.active_cells {
            let (self_water, self_terrain) = heights.get(&(x, z)).copied().unwrap_or((0.0, 0.0));
            let available = (self_water - self_terrain).max(0.0);

            if let Some(flux_entry) = self.flux.get_mut(&(x, z)) {
                let total_out = flux_entry.total_outflow();
                if total_out > available && total_out > 0.0 {
                    let scale = available / total_out;
                    flux_entry.scale(scale);
                }
            }
        }

        // Step 3: Calculate new heights and velocities
        // First, collect all the data we need without mutating self
        let mut updates: AHashMap<ChunkPos, Vec<WaterUpdate>> = AHashMap::new();
        let mut cells_to_deactivate: Vec<(i32, i32)> = Vec::new();
        let mut stability_updates: Vec<((i32, i32), u32)> = Vec::new(); // (pos, new_counter)
        let mut dirty_chunks: Vec<ChunkPos> = Vec::new();

        // Pre-collect source status for all active cells
        let source_status: AHashMap<(i32, i32), bool> = self.active_cells.keys()
            .map(|&(x, z)| ((x, z), self.is_source(x, z, &chunks_read)))
            .collect();

        // Collect current stability counters
        let stability_counters: AHashMap<(i32, i32), u32> = self.active_cells.iter()
            .map(|(&pos, cell)| (pos, cell.stability_counter))
            .collect();

        for &(x, z) in self.active_cells.keys() {
            let (old_water, terrain) = heights.get(&(x, z)).copied().unwrap_or((0.0, 0.0));

            // Calculate net flux (inflow - outflow)
            let mut net_flux = 0.0;
            let mut velocity_x = 0.0;
            let mut velocity_z = 0.0;

            // Outflow from this cell
            if let Some(flux_entry) = self.flux.get(&(x, z)) {
                net_flux -= flux_entry.flux[DIR_POS_X].max(0.0);
                net_flux -= flux_entry.flux[DIR_NEG_X].max(0.0);
                net_flux -= flux_entry.flux[DIR_POS_Z].max(0.0);
                net_flux -= flux_entry.flux[DIR_NEG_Z].max(0.0);

                velocity_x += flux_entry.flux[DIR_POS_X] - flux_entry.flux[DIR_NEG_X];
                velocity_z += flux_entry.flux[DIR_POS_Z] - flux_entry.flux[DIR_NEG_Z];
            }

            // Inflow from neighbors
            for (dir_idx, (dx, dz)) in FLOW_DIRECTIONS.iter().enumerate() {
                let nx = x + dx;
                let nz = z + dz;

                if let Some(neighbor_flux) = self.flux.get(&(nx, nz)) {
                    // Opposite direction index
                    let opposite_idx = match dir_idx {
                        DIR_POS_X => DIR_NEG_X,
                        DIR_NEG_X => DIR_POS_X,
                        DIR_POS_Z => DIR_NEG_Z,
                        DIR_NEG_Z => DIR_POS_Z,
                        _ => continue,
                    };

                    // Inflow is neighbor's outflow toward us (positive flux in opposite direction)
                    let inflow = neighbor_flux.flux[opposite_idx].max(0.0);
                    net_flux += inflow;
                }
            }

            // Calculate new water height
            let mut new_water = old_water + net_flux * dt;

            // Check if this is a source - sources don't drain below their level
            let is_source = source_status.get(&(x, z)).copied().unwrap_or(false);
            if is_source {
                new_water = new_water.max(old_water);
            }

            // Clamp to terrain (can't go below ground)
            new_water = new_water.max(terrain);

            // Normalize velocity by water depth for flow angle calculation
            let water_depth = (new_water - terrain).max(0.1);
            velocity_x /= water_depth;
            velocity_z /= water_depth;

            // Check stability
            let height_change = (new_water - old_water).abs();
            let velocity_magnitude = (velocity_x * velocity_x + velocity_z * velocity_z).sqrt();

            let current_counter = stability_counters.get(&(x, z)).copied().unwrap_or(0);
            if height_change < self.params.min_height_diff &&
               velocity_magnitude < self.params.min_velocity {
                let new_counter = current_counter + 1;
                stability_updates.push(((x, z), new_counter));
                if new_counter >= WATER_FLOW_STABILITY_TICKS {
                    cells_to_deactivate.push((x, z));
                }
            } else {
                stability_updates.push(((x, z), 0));
            }

            // Create update if height or velocity changed significantly
            if height_change > 0.0001 || velocity_magnitude > 0.001 {
                let (chunk_pos, local_x, local_z) = world_to_chunk_local_2d(x, z);

                // Find the actual chunk Y that has water
                for y in 0..3 {
                    let pos = ChunkPos { x: chunk_pos.x, y, z: chunk_pos.z };
                    if chunks_read.get(&pos).is_some() {
                        let update = WaterUpdate {
                            local_x,
                            local_z,
                            new_height: new_water,
                            velocity_x,
                            velocity_z,
                        };

                        updates.entry(pos).or_insert_with(Vec::new).push(update);
                        dirty_chunks.push(pos);
                        break;
                    }
                }
            }
        }

        drop(chunks_read);

        // Apply stability updates
        for ((x, z), counter) in stability_updates {
            if let Some(cell) = self.active_cells.get_mut(&(x, z)) {
                cell.stability_counter = counter;
            }
        }

        // Mark dirty chunks
        for pos in dirty_chunks {
            self.mark_dirty(pos);
        }

        // Deactivate stable cells
        for (x, z) in cells_to_deactivate {
            self.deactivate(x, z);
        }

        updates.into_iter().collect()
    }

    /// Activate all water cells in newly generated chunks
    pub fn activate_chunk_water(&mut self, chunk: &Chunk, chunk_pos: ChunkPos) {
        if let Some(water_data) = chunk.water_data() {
            for (local_x, local_z, col) in water_data.iter_water_columns() {
                if col.has_water && !col.is_source {
                    let world_x = chunk_pos.x * CHUNK_SIZE_I32 + local_x as i32;
                    let world_z = chunk_pos.z * CHUNK_SIZE_I32 + local_z as i32;
                    self.activate(world_x, world_z);
                }
            }
        }
    }
}

impl Default for WaterFlowEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Water update to apply to a chunk
#[derive(Clone, Debug)]
pub struct WaterUpdate {
    /// Local X coordinate within chunk
    pub local_x: usize,
    /// Local Z coordinate within chunk
    pub local_z: usize,
    /// New water surface height
    pub new_height: f32,
    /// New X velocity component
    pub velocity_x: f32,
    /// New Z velocity component
    pub velocity_z: f32,
}

/// Apply water updates to chunks
pub fn apply_water_updates(
    updates: Vec<(ChunkPos, Vec<WaterUpdate>)>,
    chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
) {
    if updates.is_empty() {
        return;
    }

    let mut chunks_write = chunks.write();

    for (chunk_pos, water_updates) in updates {
        if let Some(old_chunk) = chunks_write.get(&chunk_pos) {
            // Clone the chunk for modification (copy-on-write)
            let mut new_chunk = Chunk::new();

            // Copy block data
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        new_chunk.set_block(x, y, z, old_chunk.get_block(x, y, z));
                        new_chunk.set_light(x, y, z, old_chunk.get_light(x, y, z));
                    }
                }
            }

            // Copy and update water data
            if let Some(old_water) = old_chunk.water_data() {
                let water_data = new_chunk.ensure_water_data();

                // Copy existing water columns
                for z in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let old_col = old_water.get_column(x, z);
                        if old_col.has_water {
                            let col = water_data.get_column_mut(x, z);
                            col.surface_y = old_col.surface_y;
                            col.has_water = true;
                            col.foam = old_col.foam;
                            col.flow_angle = old_col.flow_angle;
                            col.flow_velocity_x = old_col.flow_velocity_x;
                            col.flow_velocity_z = old_col.flow_velocity_z;
                            col.is_source = old_col.is_source;
                        }
                    }
                }

                // Apply updates
                for update in water_updates {
                    let col = water_data.get_column_mut(update.local_x, update.local_z);
                    col.surface_y = update.new_height;
                    col.flow_velocity_x = update.velocity_x;
                    col.flow_velocity_z = update.velocity_z;
                    col.update_flow_angle();

                    // Increase foam based on flow speed
                    let speed = col.flow_speed();
                    col.foam = (col.foam + speed * 0.1).min(1.0);
                }

                water_data.recalculate_bounds();
            }

            // Replace chunk
            chunks_write.insert(chunk_pos, Arc::new(new_chunk));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_to_chunk_local_2d() {
        // Positive coordinates
        let (pos, lx, lz) = world_to_chunk_local_2d(17, 33);
        assert_eq!(pos.x, 1);
        assert_eq!(pos.z, 2);
        assert_eq!(lx, 1);
        assert_eq!(lz, 1);

        // Origin
        let (pos, lx, lz) = world_to_chunk_local_2d(0, 0);
        assert_eq!(pos.x, 0);
        assert_eq!(pos.z, 0);
        assert_eq!(lx, 0);
        assert_eq!(lz, 0);

        // Negative coordinates
        let (pos, lx, lz) = world_to_chunk_local_2d(-1, -1);
        assert_eq!(pos.x, -1);
        assert_eq!(pos.z, -1);
        assert_eq!(lx, 15);
        assert_eq!(lz, 15);

        // Boundary
        let (pos, lx, lz) = world_to_chunk_local_2d(16, 0);
        assert_eq!(pos.x, 1);
        assert_eq!(lx, 0);
    }

    #[test]
    fn test_cell_flux() {
        let mut flux = CellFlux::default();
        flux.flux[DIR_POS_X] = 1.0;
        flux.flux[DIR_NEG_Z] = 0.5;
        flux.flux[DIR_NEG_X] = -0.3; // Negative (inflow)

        assert!((flux.total_outflow() - 1.5).abs() < 0.001);

        flux.scale(0.5);
        assert!((flux.flux[DIR_POS_X] - 0.5).abs() < 0.001);
        assert!((flux.flux[DIR_NEG_Z] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_water_flow_engine_activation() {
        let mut engine = WaterFlowEngine::new();

        assert_eq!(engine.active_cell_count(), 0);

        engine.activate(10, 20);
        assert!(engine.is_active(10, 20));
        assert_eq!(engine.active_cell_count(), 1);

        engine.activate_region(0, 0, 2);
        // Should activate 5x5 = 25 cells, but (10, 20) already exists
        // Actually radius 2 means -2..=2 = 5 values per axis = 25 cells
        assert!(engine.active_cell_count() >= 25);

        engine.deactivate(10, 20);
        assert!(!engine.is_active(10, 20));
    }

    #[test]
    fn test_flow_params_default() {
        let params = FlowParams::default();
        assert!((params.gravity - WATER_FLOW_GRAVITY).abs() < 0.001);
        assert!((params.damping - WATER_FLOW_DAMPING).abs() < 0.001);
    }
}
