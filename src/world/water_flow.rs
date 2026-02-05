//! Water Flow Simulation Engine (Cellular Automata)
//!
//! Implements a level-based cellular automata water simulation for realistic
//! and performant water flow. Inspired by Minecraft's water mechanics but with
//! visual smoothing for better aesthetics.
//!
//! Key concepts:
//! - Water has levels 0-7 (0 = empty, 7 = full/source)
//! - Horizontal flow: water spreads with level decay (7 → 6 → 5 → ...)
//! - Vertical flow: water always flows down at full level (waterfalls)
//! - Sources maintain level 7 indefinitely
//! - Flowing water decays over distance from source
//!
//! Visual integration:
//! - Level affects surface height (for smooth visual transitions)
//! - Flow direction drives wave animation
//! - Foam increases at flow points and waterfalls

use std::collections::VecDeque;
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use parking_lot::RwLock;

use crate::constants::{
    CHUNK_SIZE, CHUNK_SIZE_I32, BLOCK_SCALE,
    WATER_FLOW_MAX_ACTIVE_CELLS, WATER_FLOW_CELLS_PER_TICK,
    WATER_FLOW_STABILITY_TICKS, WATER_FLOW_SIM_RADIUS,
};

use super::chunk::{Block, Chunk, ChunkPos, WATER_LEVEL_MAX, WATER_LEVEL_MIN};

/// 4 cardinal directions for horizontal water flow: +X, -X, +Z, -Z
const FLOW_DIRECTIONS: [(i32, i32); 4] = [
    (1, 0),   // +X (East)
    (-1, 0),  // -X (West)
    (0, 1),   // +Z (South)
    (0, -1),  // -Z (North)
];

/// Convert world X, Y, Z coordinates to chunk position and local coordinates
#[inline]
pub fn world_to_chunk_local(x: i32, y: i32, z: i32) -> (ChunkPos, usize, usize, usize) {
    let chunk_pos = ChunkPos {
        x: x.div_euclid(CHUNK_SIZE_I32),
        y: y.div_euclid(CHUNK_SIZE_I32),
        z: z.div_euclid(CHUNK_SIZE_I32),
    };
    let local_x = x.rem_euclid(CHUNK_SIZE_I32) as usize;
    let local_y = y.rem_euclid(CHUNK_SIZE_I32) as usize;
    let local_z = z.rem_euclid(CHUNK_SIZE_I32) as usize;
    (chunk_pos, local_x, local_y, local_z)
}

/// A water cell position in world coordinates
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaterPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl WaterPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Get the chunk position for this water cell
    pub fn chunk_pos(&self) -> ChunkPos {
        ChunkPos {
            x: self.x.div_euclid(CHUNK_SIZE_I32),
            y: self.y.div_euclid(CHUNK_SIZE_I32),
            z: self.z.div_euclid(CHUNK_SIZE_I32),
        }
    }

    /// Get local coordinates within the chunk
    pub fn local(&self) -> (usize, usize, usize) {
        (
            self.x.rem_euclid(CHUNK_SIZE_I32) as usize,
            self.y.rem_euclid(CHUNK_SIZE_I32) as usize,
            self.z.rem_euclid(CHUNK_SIZE_I32) as usize,
        )
    }

    /// Get position below this one
    pub fn below(&self) -> WaterPos {
        WaterPos::new(self.x, self.y - 1, self.z)
    }

    /// Get position above this one
    pub fn above(&self) -> WaterPos {
        WaterPos::new(self.x, self.y + 1, self.z)
    }

    /// Get neighbor in a horizontal direction
    pub fn neighbor(&self, dx: i32, dz: i32) -> WaterPos {
        WaterPos::new(self.x + dx, self.y, self.z + dz)
    }
}

/// State of an active water cell being simulated
#[derive(Clone, Copy, Debug)]
pub struct ActiveWaterCell {
    /// World position
    pub pos: WaterPos,
    /// Current water level (0-7)
    pub level: u8,
    /// Whether this is a source (infinite water)
    pub is_source: bool,
    /// Distance from nearest source
    pub source_distance: u8,
    /// Number of ticks this cell has been stable
    pub stability_counter: u8,
    /// Primary flow direction for visuals
    pub flow_dir: (i8, i8),
}

/// Water update to apply to the world
#[derive(Clone, Debug)]
pub struct WaterUpdate {
    pub pos: WaterPos,
    pub level: u8,
    pub is_source: bool,
    pub source_distance: u8,
    pub flow_dir: (i8, i8),
    /// Whether to remove water (level = 0)
    pub remove: bool,
}

/// Water flow simulation engine using cellular automata
pub struct WaterFlowEngine {
    /// Active water cells that need simulation
    /// Using a HashMap for O(1) lookup by position
    active_cells: AHashMap<WaterPos, ActiveWaterCell>,

    /// Queue of cells to process (for BFS-style propagation)
    process_queue: VecDeque<WaterPos>,

    /// Set of positions already queued (to avoid duplicates)
    queued: AHashSet<WaterPos>,

    /// Chunks that have been modified and need remeshing
    dirty_chunks: AHashSet<ChunkPos>,

    /// Simulation tick counter
    tick: u64,

    /// Player position (for simulation radius culling)
    player_chunk: Option<ChunkPos>,

    /// Whether simulation is enabled
    pub enabled: bool,

    /// Simulation quality/distance setting
    pub sim_radius: i32,
}

impl WaterFlowEngine {
    pub fn new() -> Self {
        Self {
            active_cells: AHashMap::with_capacity(10000),
            process_queue: VecDeque::with_capacity(1000),
            queued: AHashSet::with_capacity(1000),
            dirty_chunks: AHashSet::new(),
            tick: 0,
            player_chunk: None,
            enabled: true,
            sim_radius: WATER_FLOW_SIM_RADIUS,
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

    /// Update player position for simulation culling
    pub fn set_player_position(&mut self, chunk_pos: ChunkPos) {
        self.player_chunk = Some(chunk_pos);
    }

    /// Check if a chunk is within simulation radius
    fn is_in_sim_radius(&self, chunk_pos: ChunkPos) -> bool {
        match self.player_chunk {
            Some(player) => {
                let dx = (chunk_pos.x - player.x).abs();
                let dy = (chunk_pos.y - player.y).abs();
                let dz = (chunk_pos.z - player.z).abs();
                dx <= self.sim_radius && dy <= self.sim_radius && dz <= self.sim_radius
            }
            None => true, // No player set, simulate everything
        }
    }

    /// Check if a position has active water
    pub fn is_active(&self, pos: WaterPos) -> bool {
        self.active_cells.contains_key(&pos)
    }

    /// Activate a water cell for simulation
    pub fn activate(&mut self, pos: WaterPos, level: u8, is_source: bool, source_dist: u8) {
        if self.active_cells.len() >= WATER_FLOW_MAX_ACTIVE_CELLS {
            return;
        }

        // Check if in simulation radius
        if !self.is_in_sim_radius(pos.chunk_pos()) {
            return;
        }

        let cell = ActiveWaterCell {
            pos,
            level,
            is_source,
            source_distance: source_dist,
            stability_counter: 0,
            flow_dir: (0, 0),
        };

        self.active_cells.insert(pos, cell);

        // Add to process queue if not already queued
        if !self.queued.contains(&pos) {
            self.process_queue.push_back(pos);
            self.queued.insert(pos);
        }
    }

    /// Activate a water source at a position
    pub fn activate_source(&mut self, x: i32, y: i32, z: i32) {
        let pos = WaterPos::new(x, y, z);
        self.activate(pos, WATER_LEVEL_MAX, true, 0);
    }

    /// Activate flowing water at a position
    pub fn activate_flow(&mut self, x: i32, y: i32, z: i32, level: u8, source_dist: u8) {
        let pos = WaterPos::new(x, y, z);
        self.activate(pos, level, false, source_dist);
    }

    /// Deactivate a water cell
    pub fn deactivate(&mut self, pos: WaterPos) {
        self.active_cells.remove(&pos);
        self.queued.remove(&pos);
    }

    /// Queue a position for processing
    fn queue_for_processing(&mut self, pos: WaterPos) {
        if !self.queued.contains(&pos) && self.is_in_sim_radius(pos.chunk_pos()) {
            self.process_queue.push_back(pos);
            self.queued.insert(pos);
        }
    }

    /// Mark a chunk as dirty (needs remeshing)
    fn mark_dirty(&mut self, chunk_pos: ChunkPos) {
        self.dirty_chunks.insert(chunk_pos);
    }

    /// Take all dirty chunks, clearing the internal set
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkPos> {
        self.dirty_chunks.drain().collect()
    }

    /// Get block at a world position
    fn get_block(
        &self,
        pos: WaterPos,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> Option<Block> {
        let chunk_pos = pos.chunk_pos();
        let (lx, ly, lz) = pos.local();
        chunks.get(&chunk_pos).map(|c| c.get_block(lx, ly, lz))
    }

    /// Check if a position can hold water (not solid, not already water source)
    fn can_hold_water(
        &self,
        pos: WaterPos,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> bool {
        match self.get_block(pos, chunks) {
            Some(block) => !block.is_solid() && block != Block::Water,
            None => false, // Chunk not loaded
        }
    }

    /// Check if a position has solid terrain below
    /// Scans down through water blocks to find the actual floor (solid) or air (cave)
    fn has_floor(
        &self,
        pos: WaterPos,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> bool {
        let mut check_pos = pos.below();
        let mut scanned = 0;

        // Scan down through water to find the actual floor
        // Max scan distance to prevent infinite loops
        for _ in 0..32 {
            scanned += 1;
            match self.get_block(check_pos, chunks) {
                Some(Block::Water) => {
                    // Water block - continue scanning down
                    check_pos = check_pos.below();
                }
                Some(block) if block.is_solid() => {
                    // Found solid floor after scanning
                    return true;
                }
                Some(_block) => {
                    // Found air (or other non-solid, non-water block) - this is a cave opening!
                    return false;
                }
                None => {
                    // Chunk not loaded - assume solid to be safe
                    return true;
                }
            }
        }

        // Exceeded scan distance, assume solid
        true
    }

    /// Get the first non-water position below this one (cave opening)
    fn find_cave_below(
        &self,
        pos: WaterPos,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> Option<WaterPos> {
        let mut check_pos = pos.below();

        for _ in 0..32 {
            match self.get_block(check_pos, chunks) {
                Some(Block::Water) => {
                    check_pos = check_pos.below();
                }
                Some(block) if !block.is_solid() && block != Block::Water => {
                    // Found air (cave) - this is where water should flow
                    return Some(check_pos);
                }
                _ => {
                    // Solid block or chunk not loaded
                    return None;
                }
            }
        }

        None
    }

    /// Get water level at a position (from active cells or world data)
    fn get_water_level(
        &self,
        pos: WaterPos,
        chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
    ) -> u8 {
        // Check active cells first
        if let Some(cell) = self.active_cells.get(&pos) {
            return cell.level;
        }

        // Check chunk water data
        let chunk_pos = pos.chunk_pos();
        let (lx, _ly, lz) = pos.local();
        if let Some(chunk) = chunks.get(&chunk_pos) {
            if let Some(water_data) = chunk.water_data() {
                let col = water_data.get_column(lx, lz);
                if col.has_water {
                    return col.level;
                }
            }
        }

        0 // No water
    }

    /// Calculate the surface Y for a water level at a given terrain height
    fn calculate_surface_y(terrain_y: f32, level: u8) -> f32 {
        // Water surface is terrain + (level/7) of a block
        let level_height = (level as f32 / WATER_LEVEL_MAX as f32) * BLOCK_SCALE;
        terrain_y + level_height
    }

    /// Run one simulation tick
    ///
    /// Processes water cells using CA rules:
    /// 1. Water with no floor flows down (waterfall) at full level
    /// 2. Water with floor spreads horizontally with level decay
    /// 3. Water level naturally settles (higher flows to lower)
    /// 4. Sources maintain level 7 indefinitely
    pub fn simulate_tick(
        &mut self,
        _dt: f32,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) -> Vec<WaterUpdate> {
        if !self.enabled || self.active_cells.is_empty() {
            return Vec::new();
        }

        self.tick += 1;

        // Log stats every 300 ticks (~5 seconds)
        if self.tick % 300 == 0 {
            tracing::debug!("Water flow tick {}: {} active cells, {} in queue",
                self.tick, self.active_cells.len(), self.process_queue.len());
        }
        let chunks_read = chunks.read();

        let mut updates: Vec<WaterUpdate> = Vec::new();
        let mut cells_processed = 0;
        let mut cells_to_deactivate: Vec<WaterPos> = Vec::new();
        let mut new_activations: Vec<(WaterPos, u8, bool, u8, (i8, i8))> = Vec::new();

        // Process cells from the queue
        while let Some(pos) = self.process_queue.pop_front() {
            self.queued.remove(&pos);

            if cells_processed >= WATER_FLOW_CELLS_PER_TICK {
                // Re-queue for next tick
                self.process_queue.push_back(pos);
                self.queued.insert(pos);
                break;
            }

            let cell = match self.active_cells.get(&pos) {
                Some(c) => *c,
                None => continue, // Cell was deactivated
            };

            cells_processed += 1;

            // Skip if outside sim radius
            if !self.is_in_sim_radius(pos.chunk_pos()) {
                continue;
            }

            let mut changed = false;
            let mut new_level = cell.level;
            let mut new_flow_dir = cell.flow_dir;

            // === RULE 1: Waterfall (flow down into caves) ===
            if !self.has_floor(pos, &chunks_read) {
                // Find the actual cave opening below (could be multiple blocks down)
                let target_pos = self.find_cave_below(pos, &chunks_read).unwrap_or_else(|| pos.below());
                if self.can_hold_water(target_pos, &chunks_read) {
                    let target_level = self.get_water_level(target_pos, &chunks_read);
                    if target_level < WATER_LEVEL_MAX {
                        // Flow down at full level (waterfall into cave)
                        new_activations.push((
                            target_pos,
                            WATER_LEVEL_MAX,
                            false,
                            cell.source_distance.saturating_add(1),
                            (0, 0), // Vertical flow
                        ));
                        changed = true;

                        // Non-source water drains when flowing down
                        if !cell.is_source {
                            new_level = new_level.saturating_sub(1);
                        }
                    }
                }
            }

            // === RULE 2: Horizontal spread (only if we have a floor) ===
            if cell.level >= WATER_LEVEL_MIN && self.has_floor(pos, &chunks_read) {
                // Calculate spread level (decays by 1)
                let spread_level = if cell.is_source {
                    WATER_LEVEL_MAX - 1 // Sources spread at level 6
                } else {
                    cell.level.saturating_sub(1)
                };

                if spread_level >= WATER_LEVEL_MIN {
                    let mut lowest_neighbor: Option<(WaterPos, u8, i8, i8)> = None;

                    for (dx, dz) in FLOW_DIRECTIONS {
                        let neighbor_pos = pos.neighbor(dx, dz);

                        // Check if neighbor can hold water
                        if !self.can_hold_water(neighbor_pos, &chunks_read) {
                            continue;
                        }

                        let neighbor_level = self.get_water_level(neighbor_pos, &chunks_read);

                        // Flow to neighbors with lower level
                        if neighbor_level < spread_level {
                            // Track lowest for flow direction
                            if lowest_neighbor.is_none()
                                || neighbor_level < lowest_neighbor.unwrap().1
                            {
                                lowest_neighbor =
                                    Some((neighbor_pos, neighbor_level, dx as i8, dz as i8));
                            }

                            // Activate the neighbor
                            let new_dist = cell.source_distance.saturating_add(1);
                            new_activations.push((
                                neighbor_pos,
                                spread_level,
                                false,
                                new_dist,
                                (dx as i8, dz as i8),
                            ));
                            changed = true;
                        }
                    }

                    // Update flow direction toward lowest neighbor
                    if let Some((_, _, dx, dz)) = lowest_neighbor {
                        new_flow_dir = (dx, dz);
                    }
                }
            }

            // === RULE 3: Level equalization ===
            // Water flows from higher to lower levels (gravity settling)
            if !cell.is_source && cell.level > WATER_LEVEL_MIN {
                for (dx, dz) in FLOW_DIRECTIONS {
                    let neighbor_pos = pos.neighbor(dx, dz);
                    let neighbor_level = self.get_water_level(neighbor_pos, &chunks_read);

                    // If neighbor has significantly less water, we lose some
                    if neighbor_level + 2 < cell.level {
                        new_level = new_level.saturating_sub(1);
                        changed = true;
                        break;
                    }
                }
            }

            // === Update cell state ===
            if changed || new_level != cell.level || new_flow_dir != cell.flow_dir {
                if new_level < WATER_LEVEL_MIN && !cell.is_source {
                    // Water dried up
                    cells_to_deactivate.push(pos);
                    updates.push(WaterUpdate {
                        pos,
                        level: 0,
                        is_source: false,
                        source_distance: 255,
                        flow_dir: (0, 0),
                        remove: true,
                    });
                } else {
                    // Update the cell
                    if let Some(cell_mut) = self.active_cells.get_mut(&pos) {
                        cell_mut.level = new_level;
                        cell_mut.flow_dir = new_flow_dir;
                        cell_mut.stability_counter = 0;
                    }

                    updates.push(WaterUpdate {
                        pos,
                        level: new_level,
                        is_source: cell.is_source,
                        source_distance: cell.source_distance,
                        flow_dir: new_flow_dir,
                        remove: false,
                    });
                }

                self.mark_dirty(pos.chunk_pos());
            } else {
                // Cell is stable
                if let Some(cell_mut) = self.active_cells.get_mut(&pos) {
                    cell_mut.stability_counter += 1;
                    if cell_mut.stability_counter >= WATER_FLOW_STABILITY_TICKS as u8
                        && !cell_mut.is_source
                    {
                        // Cell has been stable long enough, deactivate
                        // (but keep the water in the world)
                        cells_to_deactivate.push(pos);
                    }
                }
            }
        }

        drop(chunks_read);

        // Apply deactivations
        for pos in cells_to_deactivate {
            self.deactivate(pos);
        }

        // Apply new activations
        for (pos, level, is_source, source_dist, flow_dir) in new_activations {
            if !self.active_cells.contains_key(&pos) {
                let mut cell = ActiveWaterCell {
                    pos,
                    level,
                    is_source,
                    source_distance: source_dist,
                    stability_counter: 0,
                    flow_dir,
                };

                // Merge with existing if present
                if let Some(existing) = self.active_cells.get(&pos) {
                    cell.level = cell.level.max(existing.level);
                    cell.is_source = cell.is_source || existing.is_source;
                }

                self.active_cells.insert(pos, cell);
                self.queue_for_processing(pos);
            } else if let Some(existing) = self.active_cells.get_mut(&pos) {
                // Update existing cell if new level is higher
                if level > existing.level {
                    existing.level = level;
                    existing.source_distance = existing.source_distance.min(source_dist);
                    existing.stability_counter = 0;
                    self.queue_for_processing(pos);
                }
            }
        }

        updates
    }

    /// Activate water cells from a newly loaded chunk
    pub fn activate_chunk_water(&mut self, chunk: &Chunk, chunk_pos: ChunkPos) {
        let mut activated_count = 0;
        if let Some(water_data) = chunk.water_data() {
            for (local_x, local_z, col) in water_data.iter_water_columns() {
                if col.has_water && col.level >= WATER_LEVEL_MIN {
                    // Find the Y position for this water column
                    // Water is at the surface, need to find the actual block Y
                    let world_x = chunk_pos.x * CHUNK_SIZE_I32 + local_x as i32;
                    let world_z = chunk_pos.z * CHUNK_SIZE_I32 + local_z as i32;
                    let world_y = (col.surface_y / BLOCK_SCALE).floor() as i32;

                    let pos = WaterPos::new(world_x, world_y, world_z);
                    self.activate(pos, col.level, col.is_source, col.source_distance);
                    activated_count += 1;
                }
            }
        }
    }

    /// Remove a water source and trigger flow recalculation
    pub fn remove_water(&mut self, x: i32, y: i32, z: i32) {
        let pos = WaterPos::new(x, y, z);

        // Deactivate this cell
        self.deactivate(pos);

        // Queue neighbors for recalculation
        for (dx, dz) in FLOW_DIRECTIONS {
            let neighbor = pos.neighbor(dx, dz);
            if self.active_cells.contains_key(&neighbor) {
                self.queue_for_processing(neighbor);
            }
        }

        // Also check above and below
        let above = pos.above();
        let below = pos.below();
        if self.active_cells.contains_key(&above) {
            self.queue_for_processing(above);
        }
        if self.active_cells.contains_key(&below) {
            self.queue_for_processing(below);
        }

        self.mark_dirty(pos.chunk_pos());
    }

    /// Clear all active cells (for world reload)
    pub fn clear(&mut self) {
        self.active_cells.clear();
        self.process_queue.clear();
        self.queued.clear();
        self.dirty_chunks.clear();
    }
}

impl Default for WaterFlowEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply water updates to chunks
pub fn apply_water_updates(
    updates: Vec<WaterUpdate>,
    chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
) {
    if updates.is_empty() {
        return;
    }

    // Group updates by chunk
    let mut updates_by_chunk: AHashMap<ChunkPos, Vec<&WaterUpdate>> = AHashMap::new();
    for update in &updates {
        let chunk_pos = update.pos.chunk_pos();
        updates_by_chunk.entry(chunk_pos).or_default().push(update);
    }

    let mut chunks_write = chunks.write();

    for (chunk_pos, chunk_updates) in updates_by_chunk {
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

            // Copy existing water data
            if let Some(old_water) = old_chunk.water_data() {
                let water_data = new_chunk.ensure_water_data();
                for z in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let old_col = old_water.get_column(x, z);
                        if old_col.has_water {
                            let col = water_data.get_column_mut(x, z);
                            *col = *old_col;
                        }
                    }
                }
            }

            // Apply updates
            let water_data = new_chunk.ensure_water_data();
            for update in chunk_updates {
                let (lx, ly, lz) = update.pos.local();
                let col = water_data.get_column_mut(lx, lz);

                if update.remove {
                    col.clear();
                } else {
                    // Calculate surface Y based on terrain and level
                    let terrain_y = (ly as f32 + chunk_pos.y as f32 * CHUNK_SIZE as f32) * BLOCK_SCALE;
                    let surface_y = WaterFlowEngine::calculate_surface_y(terrain_y, update.level);

                    if update.is_source {
                        col.set_as_source(surface_y);
                    } else {
                        col.set_flowing(update.level, surface_y, update.source_distance);
                    }

                    col.flow_dir_x = update.flow_dir.0;
                    col.flow_dir_z = update.flow_dir.1;
                    col.update_flow_angle_from_dir();
                }
            }

            water_data.recalculate_bounds();

            // Replace chunk
            chunks_write.insert(chunk_pos, Arc::new(new_chunk));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_pos_chunk_local() {
        let pos = WaterPos::new(17, 33, -5);
        assert_eq!(pos.chunk_pos(), ChunkPos { x: 1, y: 2, z: -1 });
        assert_eq!(pos.local(), (1, 1, 11));
    }

    #[test]
    fn test_water_pos_neighbors() {
        let pos = WaterPos::new(10, 20, 30);
        assert_eq!(pos.below(), WaterPos::new(10, 19, 30));
        assert_eq!(pos.above(), WaterPos::new(10, 21, 30));
        assert_eq!(pos.neighbor(1, 0), WaterPos::new(11, 20, 30));
        assert_eq!(pos.neighbor(0, -1), WaterPos::new(10, 20, 29));
    }

    #[test]
    fn test_water_flow_engine_activation() {
        let mut engine = WaterFlowEngine::new();

        let pos = WaterPos::new(10, 20, 30);
        engine.activate(pos, 7, true, 0);

        assert!(engine.is_active(pos));
        assert_eq!(engine.active_cell_count(), 1);

        engine.deactivate(pos);
        assert!(!engine.is_active(pos));
        assert_eq!(engine.active_cell_count(), 0);
    }

    #[test]
    fn test_surface_y_calculation() {
        // At terrain Y=32 with full level, surface should be at 32 + 1.0
        let surface = WaterFlowEngine::calculate_surface_y(32.0, 7);
        assert!((surface - 33.0).abs() < 0.01);

        // At terrain Y=32 with half level, surface should be at 32 + 0.5
        let surface = WaterFlowEngine::calculate_surface_y(32.0, 3);
        assert!((surface - 32.43).abs() < 0.01); // 32 + (3/7)
    }
}
