//! World management module
//!
//! Handles chunks, terrain generation, and mesh building.

pub mod chest;
mod chunk;
pub mod generation;
pub mod item_entity;
pub mod item_persistence;
pub mod lighting;
mod meshing;
pub mod storage;
mod water_meshing;
pub mod water_flow;

pub use chest::{ChestManager, ChestData, Container, SavedChest, SavedChestItem, CHEST_SLOTS};
pub use chunk::{Block, Chunk, ChunkPos, ChunkWaterData, WaterColumn, PropPlacement, ChunkPropData};
pub use generation::{generate_chunk, generate_chunk_with_seed};
pub use item_entity::{ItemEntity, ItemEntityManager, SavedItemEntity};
pub use item_persistence::{save_items, load_items};
pub use lighting::{LightEngine, apply_light_updates, world_to_chunk_local};
pub use meshing::{ChunkMesh, ChunkVertex, GpuMesh, LodLevel, build_chunk_mesh, build_chunk_mesh_lod, MESH_STATS};
pub use storage::{WorldStorage, SharedStorage, StorageError, create_shared_storage};
pub use water_meshing::{WaterMesh, WaterVertex, GpuWaterMesh, build_water_mesh, chunk_has_water};
pub use water_flow::{WaterFlowEngine, apply_water_updates};

use std::sync::Arc;

use ahash::AHashMap;
use crossbeam_channel::{Receiver, Sender};
use glam::Vec3;
use parking_lot::RwLock;

use crate::constants::{CHUNK_SIZE_I32, RENDER_DISTANCE, WATER_FLOW_TICK_RATE};

/// Result of a raycast into the world
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    /// World position of the hit block
    pub block_pos: [i32; 3],
    /// World position of the adjacent block (for placement)
    pub adjacent_pos: [i32; 3],
    /// Distance from ray origin to hit
    pub distance: f32,
    /// Whether the hit block is water (for placement logic)
    pub hit_water: bool,
}

/// Perform a raycast through the voxel world using DDA algorithm
pub fn raycast(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    world: &WorldManager,
) -> Option<RaycastHit> {
    // Normalize direction
    let dir = direction.normalize();

    // Current block position
    let mut block_x = origin.x.floor() as i32;
    let mut block_y = origin.y.floor() as i32;
    let mut block_z = origin.z.floor() as i32;

    // Step direction (+1 or -1)
    let step_x = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z = if dir.z >= 0.0 { 1 } else { -1 };

    // Distance along ray to next voxel boundary
    let t_delta_x = if dir.x != 0.0 { (1.0 / dir.x).abs() } else { f32::MAX };
    let t_delta_y = if dir.y != 0.0 { (1.0 / dir.y).abs() } else { f32::MAX };
    let t_delta_z = if dir.z != 0.0 { (1.0 / dir.z).abs() } else { f32::MAX };

    // Distance to first boundary
    let mut t_max_x = if dir.x != 0.0 {
        if dir.x > 0.0 {
            ((block_x + 1) as f32 - origin.x) / dir.x
        } else {
            (block_x as f32 - origin.x) / dir.x
        }
    } else {
        f32::MAX
    };

    let mut t_max_y = if dir.y != 0.0 {
        if dir.y > 0.0 {
            ((block_y + 1) as f32 - origin.y) / dir.y
        } else {
            (block_y as f32 - origin.y) / dir.y
        }
    } else {
        f32::MAX
    };

    let mut t_max_z = if dir.z != 0.0 {
        if dir.z > 0.0 {
            ((block_z + 1) as f32 - origin.z) / dir.z
        } else {
            (block_z as f32 - origin.z) / dir.z
        }
    } else {
        f32::MAX
    };

    // Track previous block for adjacent position
    let mut prev_x = block_x;
    let mut prev_y = block_y;
    let mut prev_z = block_z;

    let mut distance = 0.0;

    while distance < max_distance {
        // Check current block
        if let Some(block) = world.get_block(block_x, block_y, block_z) {
            if block.is_solid() {
                return Some(RaycastHit {
                    block_pos: [block_x, block_y, block_z],
                    adjacent_pos: [prev_x, prev_y, prev_z],
                    distance,
                    hit_water: false,
                });
            }
        }

        // Save previous position
        prev_x = block_x;
        prev_y = block_y;
        prev_z = block_z;

        // Step to next voxel boundary
        if t_max_x < t_max_y && t_max_x < t_max_z {
            distance = t_max_x;
            t_max_x += t_delta_x;
            block_x += step_x;
        } else if t_max_y < t_max_z {
            distance = t_max_y;
            t_max_y += t_delta_y;
            block_y += step_y;
        } else {
            distance = t_max_z;
            t_max_z += t_delta_z;
            block_z += step_z;
        }
    }

    None
}

/// Result of a prop raycast
#[derive(Debug, Clone, Copy)]
pub struct PropHit {
    /// Chunk containing the prop
    pub chunk_pos: ChunkPos,
    /// World position of the prop
    pub world_pos: [f32; 3],
    /// Local position within chunk
    pub local_pos: [f32; 3],
    /// Prop type ID
    pub prop_id: u8,
    /// Distance from ray origin
    pub distance: f32,
}

/// Raycast against props in the world
/// Returns the closest prop hit within max_distance, if any
pub fn raycast_props(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    chunks: &AHashMap<ChunkPos, Arc<Chunk>>,
) -> Option<PropHit> {
    let dir = direction.normalize();
    let mut closest_hit: Option<PropHit> = None;

    // Check all loaded chunks within reach
    // We need to check chunks the ray might pass through
    let check_radius = (max_distance / 16.0).ceil() as i32 + 1;
    let origin_chunk = ChunkPos::from_world_pos(origin);

    for dx in -check_radius..=check_radius {
        for dy in -check_radius..=check_radius {
            for dz in -check_radius..=check_radius {
                let chunk_pos = ChunkPos {
                    x: origin_chunk.x + dx,
                    y: origin_chunk.y + dy,
                    z: origin_chunk.z + dz,
                };

                if let Some(chunk) = chunks.get(&chunk_pos) {
                    if let Some(props) = chunk.prop_data() {
                        let chunk_world_x = chunk_pos.x * CHUNK_SIZE_I32;
                        let chunk_world_y = chunk_pos.y * CHUNK_SIZE_I32;
                        let chunk_world_z = chunk_pos.z * CHUNK_SIZE_I32;

                        for prop in props.iter() {
                            let prop_world = Vec3::new(
                                chunk_world_x as f32 + prop.local_x,
                                chunk_world_y as f32 + prop.local_y,
                                chunk_world_z as f32 + prop.local_z,
                            );

                            // Point-to-line distance check
                            // Project prop position onto ray
                            let to_prop = prop_world - origin;
                            let t = to_prop.dot(dir);

                            // Skip if behind ray origin or too far
                            if t < 0.0 || t > max_distance {
                                continue;
                            }

                            // Skip if we already have a closer hit
                            if let Some(ref hit) = closest_hit {
                                if t >= hit.distance {
                                    continue;
                                }
                            }

                            // Find closest point on ray to prop
                            let closest_point = origin + dir * t;
                            let distance_to_ray = (prop_world - closest_point).length();

                            // Tolerance based on prop type (grass/flowers smaller, rocks larger)
                            let tolerance = match prop.prop_id {
                                // Grass types (0-4)
                                0..=4 => 0.3,
                                // Flower types (5-14)
                                5..=14 => 0.35,
                                // Rock/stone types (15+)
                                _ => 0.5,
                            };

                            if distance_to_ray <= tolerance {
                                closest_hit = Some(PropHit {
                                    chunk_pos,
                                    world_pos: [prop_world.x, prop_world.y, prop_world.z],
                                    local_pos: [prop.local_x, prop.local_y, prop.local_z],
                                    prop_id: prop.prop_id,
                                    distance: t,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    closest_hit
}

/// Messages for chunk worker threads
enum ChunkTask {
    Generate(ChunkPos),
    Mesh(ChunkPos, LodLevel, u64),  // Added generation counter to detect stale results
    MeshWater(ChunkPos),
    Shutdown,
}

/// Results from chunk workers
enum ChunkResult {
    Generated(ChunkPos, Box<Chunk>),
    Meshed(ChunkPos, ChunkMesh, u64),  // Includes generation counter
    WaterMeshed(ChunkPos, WaterMesh),
}

/// Manages all world chunks and coordinates generation/meshing
pub struct WorldManager {
    /// Loaded chunks (thread-safe for parallel access)
    chunks: Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,

    /// Chunk meshes ready for rendering
    meshes: AHashMap<ChunkPos, ChunkMesh>,

    /// Water meshes ready for rendering
    water_meshes: AHashMap<ChunkPos, WaterMesh>,

    /// Chunks pending generation
    pending_generation: AHashMap<ChunkPos, ()>,

    /// Mesh generation counter per chunk - used to discard stale mesh results
    /// Incremented each time a mesh is requested, only accept results matching current gen
    mesh_generation: AHashMap<ChunkPos, u64>,

    /// Chunks pending meshing
    pending_meshing: AHashMap<ChunkPos, ()>,

    /// Chunks pending water meshing
    pending_water_meshing: AHashMap<ChunkPos, ()>,

    /// Channel to send tasks to workers (regular priority - generation, LOD updates)
    task_sender: Sender<ChunkTask>,

    /// Channel to send HIGH PRIORITY tasks (player block changes)
    /// Workers check this channel first before the regular channel
    priority_task_sender: Sender<ChunkTask>,

    /// Channel to receive results from workers
    result_receiver: Receiver<ChunkResult>,

    /// Worker thread handles
    _workers: Vec<std::thread::JoinHandle<()>>,

    /// Last player chunk position (for loading new chunks)
    last_player_chunk: Option<ChunkPos>,

    /// Light propagation engine
    light_engine: LightEngine,

    /// Water flow simulation engine
    water_flow_engine: water_flow::WaterFlowEngine,

    /// Accumulator for water flow simulation fixed timestep
    water_flow_accumulator: f32,

    /// World storage for persistence (None if not saving)
    storage: Option<storage::SharedStorage>,

    /// World generation seed
    seed: u64,

    /// Meshes that need IMMEDIATE upload to chunk pool (player block changes)
    /// These bypass the normal throttled upload and get uploaded the same frame
    urgent_mesh_uploads: Vec<ChunkPos>,

    /// Meshes that need upload to chunk pool (regular mesh generation)
    /// Populated when mesh results arrive from workers, drained by renderer
    pending_mesh_uploads: Vec<ChunkPos>,

    /// Chunks that were unloaded and need eviction from GPU pool
    /// Populated when chunks are removed, drained by renderer
    unloaded_chunks: Vec<ChunkPos>,

    /// Chunks that should be marked urgent when their mesh result arrives
    /// (neighbors of player-modified chunks)
    pending_urgent: ahash::AHashSet<ChunkPos>,

    /// Mesh render distance - chunks within this get full meshes (in chunks)
    render_distance: i32,

    /// Total load distance - includes render + impostor range (in chunks)
    /// Chunks beyond render_distance but within load_distance only compute average_color
    load_distance: i32,

    /// LOD full detail threshold (0 to this = Full detail)
    /// If i32::MAX, LOD is OFF and everything is full detail
    lod_full_threshold: i32,

    /// LOD half detail threshold (lod_full to this = Half detail, beyond = Quarter)
    lod_half_threshold: i32,
}

impl WorldManager {
    /// Create a new world manager without persistence
    pub fn new() -> Self {
        Self::new_internal(None, 12345)
    }

    /// Create a new world manager with persistence to the given save directory
    pub fn new_with_save(save_path: &std::path::Path, seed: Option<u64>) -> Result<Self, storage::StorageError> {
        let shared_storage = storage::create_shared_storage(save_path, seed)?;
        let actual_seed = shared_storage.lock().seed();
        tracing::info!("World storage initialized at {:?} with seed {}", save_path, actual_seed);
        Ok(Self::new_internal(Some(shared_storage), actual_seed))
    }

    fn new_internal(storage: Option<storage::SharedStorage>, seed: u64) -> Self {
        // Regular priority channel for generation and LOD updates
        let (task_sender, task_receiver) = crossbeam_channel::unbounded::<ChunkTask>();
        // HIGH priority channel for player block changes - workers check this first
        let (priority_task_sender, priority_task_receiver) = crossbeam_channel::unbounded::<ChunkTask>();
        let (result_sender, result_receiver) = crossbeam_channel::unbounded::<ChunkResult>();

        let chunks: Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>> =
            Arc::new(RwLock::new(AHashMap::new()));

        // Clone storage for workers
        let worker_storage = storage.clone();

        // Spawn worker threads
        let num_workers = num_cpus::get().max(2).min(8);
        tracing::info!("Spawning {} chunk worker threads", num_workers);

        let mut workers = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let task_rx = task_receiver.clone();
            let priority_rx = priority_task_receiver.clone();
            let result_tx = result_sender.clone();
            let chunks_ref = chunks.clone();
            let storage_ref = worker_storage.clone();
            let worker_seed = seed;

            let handle = std::thread::Builder::new()
                .name(format!("chunk-worker-{}", i))
                .spawn(move || {
                    Self::worker_loop(priority_rx, task_rx, result_tx, chunks_ref, storage_ref, worker_seed);
                })
                .expect("Failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            chunks,
            meshes: AHashMap::new(),
            water_meshes: AHashMap::new(),
            pending_generation: AHashMap::new(),
            pending_meshing: AHashMap::new(),
            pending_water_meshing: AHashMap::new(),
            mesh_generation: AHashMap::new(),
            task_sender,
            priority_task_sender,
            result_receiver,
            _workers: workers,
            last_player_chunk: None,
            light_engine: LightEngine::new(),
            water_flow_engine: water_flow::WaterFlowEngine::new(),
            water_flow_accumulator: 0.0,
            storage,
            seed,
            urgent_mesh_uploads: Vec::new(),
            pending_mesh_uploads: Vec::new(),
            unloaded_chunks: Vec::new(),
            pending_urgent: ahash::AHashSet::new(),
            render_distance: RENDER_DISTANCE,
            load_distance: RENDER_DISTANCE, // Initially same, updated when impostor range is set
            lod_full_threshold: crate::constants::LOD_FULL_DETAIL,
            lod_half_threshold: crate::constants::LOD_HALF_DETAIL,
        }
    }

    /// Set the render distance (in chunks) - affects full mesh rendering
    pub fn set_render_distance(&mut self, distance: i32) {
        self.render_distance = distance.clamp(8, 64);
        // Keep load_distance at least as large as render_distance
        if self.load_distance < self.render_distance {
            self.load_distance = self.render_distance;
        }
    }

    /// Set the total load distance (render_distance + impostor_range)
    /// Chunks beyond render_distance but within load_distance are loaded for impostor rendering
    pub fn set_load_distance(&mut self, total_distance: i32) {
        self.load_distance = total_distance.clamp(self.render_distance, 96);
    }

    /// Get the current render distance (full mesh)
    pub fn render_distance(&self) -> i32 {
        self.render_distance
    }

    /// Get the current total load distance
    pub fn load_distance(&self) -> i32 {
        self.load_distance
    }

    /// Set LOD thresholds
    /// full_threshold: chunks within this distance use full detail
    /// half_threshold: chunks between full and this use half detail, beyond use quarter
    /// Pass (i32::MAX, i32::MAX) to disable LOD (everything full detail)
    pub fn set_lod_thresholds(&mut self, full_threshold: i32, half_threshold: i32) {
        // Only log if thresholds actually changed
        if self.lod_full_threshold != full_threshold || self.lod_half_threshold != half_threshold {
            tracing::info!(
                "LOD thresholds set: Full={}, Half={} (OFF={})",
                if full_threshold == i32::MAX { "∞".to_string() } else { full_threshold.to_string() },
                if half_threshold == i32::MAX { "∞".to_string() } else { half_threshold.to_string() },
                full_threshold == i32::MAX
            );
        }
        self.lod_full_threshold = full_threshold;
        self.lod_half_threshold = half_threshold;
    }

    /// Check if LOD is enabled (false means all chunks use full detail)
    pub fn is_lod_enabled(&self) -> bool {
        self.lod_full_threshold != i32::MAX
    }

    fn worker_loop(
        priority_rx: Receiver<ChunkTask>,
        task_rx: Receiver<ChunkTask>,
        result_tx: Sender<ChunkResult>,
        chunks: Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        storage: Option<storage::SharedStorage>,
        seed: u64,
    ) {
        use crossbeam_channel::select;

        loop {
            // Use select! to prioritize the priority channel
            // Priority tasks (player block changes) always get processed first
            let task = select! {
                recv(priority_rx) -> msg => {
                    match msg {
                        Ok(task) => task,
                        Err(_) => break, // Channel closed
                    }
                }
                recv(task_rx) -> msg => {
                    match msg {
                        Ok(task) => task,
                        Err(_) => break, // Channel closed
                    }
                }
            };

            // Drain any remaining priority tasks before processing this one
            // This ensures ALL priority tasks are handled before regular tasks
            let mut priority_tasks = vec![task];
            while let Ok(t) = priority_rx.try_recv() {
                priority_tasks.push(t);
            }

            for task in priority_tasks {
                match task {
                    ChunkTask::Generate(pos) => {
                        // Try loading from disk first if storage is available
                        let chunk = if let Some(ref storage) = storage {
                            match storage.lock().load_chunk(pos) {
                                Ok(Some(loaded)) => {
                                    tracing::trace!("Loaded chunk {:?} from disk", pos);
                                    loaded
                                }
                                Ok(None) => {
                                    // Not on disk, generate it
                                    generation::generate_chunk_with_seed(pos, seed)
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to load chunk {:?}: {}, regenerating", pos, e);
                                    generation::generate_chunk_with_seed(pos, seed)
                                }
                            }
                        } else {
                            generation::generate_chunk_with_seed(pos, seed)
                        };
                        let _ = result_tx.send(ChunkResult::Generated(pos, Box::new(chunk)));
                    }
                    ChunkTask::Mesh(pos, lod, generation) => {
                        let chunks_read = chunks.read();
                        if let Some(chunk) = chunks_read.get(&pos).cloned() {
                            // Get neighbor chunks for proper face culling
                            let neighbors = [
                                chunks_read.get(&pos.offset(-1, 0, 0)).cloned(),
                                chunks_read.get(&pos.offset(1, 0, 0)).cloned(),
                                chunks_read.get(&pos.offset(0, -1, 0)).cloned(),
                                chunks_read.get(&pos.offset(0, 1, 0)).cloned(),
                                chunks_read.get(&pos.offset(0, 0, -1)).cloned(),
                                chunks_read.get(&pos.offset(0, 0, 1)).cloned(),
                            ];
                            drop(chunks_read);

                            // Build mesh with specified LOD level
                            let mesh = meshing::build_chunk_mesh_lod(pos, &chunk, &neighbors, lod);
                            // Pass through generation so we can discard stale results
                            let _ = result_tx.send(ChunkResult::Meshed(pos, mesh, generation));
                        }
                    }
                    ChunkTask::MeshWater(pos) => {
                        let chunks_read = chunks.read();
                        if let Some(chunk) = chunks_read.get(&pos).cloned() {
                            // Only build water mesh if chunk has water
                            if chunk.has_water() {
                                let neighbors = [
                                    chunks_read.get(&pos.offset(-1, 0, 0)).cloned(),
                                    chunks_read.get(&pos.offset(1, 0, 0)).cloned(),
                                    chunks_read.get(&pos.offset(0, -1, 0)).cloned(),
                                    chunks_read.get(&pos.offset(0, 1, 0)).cloned(),
                                    chunks_read.get(&pos.offset(0, 0, -1)).cloned(),
                                    chunks_read.get(&pos.offset(0, 0, 1)).cloned(),
                                ];
                                drop(chunks_read);

                                let water_mesh = water_meshing::build_water_mesh(pos, &chunk, &neighbors);
                                let _ = result_tx.send(ChunkResult::WaterMeshed(pos, water_mesh));
                            }
                        }
                    }
                    ChunkTask::Shutdown => return, // Exit the entire function
                }
            }
        }
    }

    /// Upload pending meshes to GPU (must be called from main thread)
    pub fn upload_meshes(&mut self, device: &wgpu::Device) {
        for mesh in self.meshes.values_mut() {
            if mesh.gpu_mesh.is_none() && !mesh.vertices.is_empty() {
                mesh.upload(device);
            }
        }
    }

    /// Update world based on player position
    pub fn update(&mut self, player_pos: &Vec3) {
        // Calculate player's chunk position
        let player_chunk = ChunkPos::from_world_pos(*player_pos);

        // Process results from workers - batch process for faster throughput
        // Increased to 200 for faster mesh updates, especially after block changes
        // At ~3ms per mesh result processing, 200 results = ~6ms worst case
        let mut results_processed = 0;
        const MAX_RESULTS_PER_FRAME: usize = 200;

        while results_processed < MAX_RESULTS_PER_FRAME {
            let result = match self.result_receiver.try_recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            results_processed += 1;

            match result {
                ChunkResult::Generated(pos, chunk) => {
                    self.pending_generation.remove(&pos);
                    // Check if chunk has water before inserting
                    let has_water = chunk.has_water();

                    // Scan for light-emitting blocks (ores, torches) and propagate their light
                    let world_x = pos.x * crate::constants::CHUNK_SIZE_I32;
                    let world_y = pos.y * crate::constants::CHUNK_SIZE_I32;
                    let world_z = pos.z * crate::constants::CHUNK_SIZE_I32;
                    let mut light_sources: Vec<(i32, i32, i32, u8)> = Vec::new();
                    for lz in 0..crate::constants::CHUNK_SIZE {
                        for ly in 0..crate::constants::CHUNK_SIZE {
                            for lx in 0..crate::constants::CHUNK_SIZE {
                                let block = chunk.get_block(lx, ly, lz);
                                let emission = block.light_emission();
                                if emission > 0 {
                                    light_sources.push((
                                        world_x + lx as i32,
                                        world_y + ly as i32,
                                        world_z + lz as i32,
                                        emission,
                                    ));
                                }
                            }
                        }
                    }

                    self.chunks.write().insert(pos, Arc::new(*chunk));

                    // Propagate light from all sources found in this chunk
                    for (x, y, z, emission) in light_sources {
                        let updates = self.light_engine.propagate_light(x, y, z, emission, &self.chunks);
                        apply_light_updates(updates, &self.chunks);
                    }

                    // Queue meshing with appropriate LOD
                    let lod = self.calculate_lod(pos, player_chunk);
                    self.queue_meshing_with_lod(pos, lod);
                    // Queue water meshing if chunk has water
                    if has_water {
                        self.queue_water_meshing(pos);
                    }
                    // ALWAYS force remesh neighbors that exist (including edge-adjacent for diagonal structures)
                    // This fixes AO artifacts at chunk boundaries where neighbors
                    // were meshed before this chunk existed (they had no neighbor data)
                    // Include all 26 neighbors (6 face + 12 edge + 8 corner) for trees spanning diagonally
                    let chunks_read = self.chunks.read();
                    let mut neighbors_to_remesh: Vec<ChunkPos> = Vec::new();
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue; // Skip self
                                }
                                let neighbor = pos.offset(dx, dy, dz);
                                if chunks_read.contains_key(&neighbor) {
                                    neighbors_to_remesh.push(neighbor);
                                }
                            }
                        }
                    }
                    drop(chunks_read);

                    // Remesh neighbors with appropriate LOD (not always Full!)
                    for neighbor in neighbors_to_remesh {
                        let neighbor_lod = self.calculate_lod(neighbor, player_chunk);
                        self.force_remesh_with_lod(neighbor, neighbor_lod);
                    }

                    // ALSO remesh the current chunk itself after queuing neighbor remeshes
                    // This handles the case where THIS chunk's initial mesh didn't see all neighbors
                    // The generation system ensures correct ordering - stale meshes are discarded
                    self.force_remesh_with_lod(pos, lod);
                }
                ChunkResult::Meshed(pos, mesh, generation) => {
                    self.pending_meshing.remove(&pos);
                    // Only accept mesh if its generation matches the current expected generation
                    // This discards stale meshes that were built with outdated neighbor data
                    let current_gen = self.mesh_generation.get(&pos).copied().unwrap_or(0);
                    if generation >= current_gen {
                        self.meshes.insert(pos, mesh);
                        // Check if this was a priority neighbor mesh - mark for urgent upload
                        if self.pending_urgent.remove(&pos) {
                            self.urgent_mesh_uploads.push(pos);
                        } else {
                            // Regular mesh - add to pending uploads for throttled upload
                            self.pending_mesh_uploads.push(pos);
                        }
                    }
                    // else: discard stale mesh, a newer one is coming
                }
                ChunkResult::WaterMeshed(pos, water_mesh) => {
                    self.pending_water_meshing.remove(&pos);
                    if !water_mesh.is_empty() {
                        self.water_meshes.insert(pos, water_mesh);
                    }
                }
            }
        }

        // Queue chunks for generation around player, prioritized by distance
        // Higher limit allows workers to stay busy and maximizes parallelism
        const MAX_PENDING_GEN: usize = 500; // Increased for faster initial load

        // Only queue more if we have room
        if self.pending_generation.len() >= MAX_PENDING_GEN {
            return;
        }

        let budget = MAX_PENDING_GEN - self.pending_generation.len();

        // Vertical range: Y chunks 0-9 covers blocks 0-159, enough for terrain + mountains
        let mut chunks_to_queue: Vec<(ChunkPos, i32)> = Vec::new();

        for dy in 0..=9 {
            for dz in -self.load_distance..=self.load_distance {
                for dx in -self.load_distance..=self.load_distance {
                    let pos = ChunkPos {
                        x: player_chunk.x + dx,
                        y: dy,
                        z: player_chunk.z + dz,
                    };

                    if !self.chunks.read().contains_key(&pos)
                        && !self.pending_generation.contains_key(&pos)
                    {
                        // Distance from player chunk (prioritize horizontal, then vertical)
                        let vert_dist = (dy - player_chunk.y).abs();
                        let dist_sq = dx * dx + dz * dz + vert_dist * vert_dist * 2;
                        chunks_to_queue.push((pos, dist_sq));
                    }
                }
            }
        }

        // Sort by distance (closest first) and queue only up to budget
        chunks_to_queue.sort_by_key(|(_, dist)| *dist);
        for (pos, _) in chunks_to_queue.into_iter().take(budget) {
            self.pending_generation.insert(pos, ());
            let _ = self.task_sender.send(ChunkTask::Generate(pos));
        }

        // Only do LOD upgrades and unloading when player moves to a new chunk
        if self.last_player_chunk == Some(player_chunk) {
            return;
        }
        self.last_player_chunk = Some(player_chunk);

        // Queue LOD upgrades for nearby chunks
        let chunks_to_upgrade: Vec<_> = self.meshes
            .iter()
            .filter_map(|(pos, mesh)| {
                let desired_lod = self.calculate_lod(*pos, player_chunk);
                if (desired_lod as usize) < (mesh.lod as usize) {
                    Some((*pos, desired_lod))
                } else {
                    None
                }
            })
            .collect();

        for (pos, lod) in chunks_to_upgrade {
            self.queue_meshing_with_lod(pos, lod);
        }

        // Unload distant chunks
        let mut to_remove = Vec::new();
        for pos in self.chunks.read().keys() {
            let dx = (pos.x - player_chunk.x).abs();
            let dz = (pos.z - player_chunk.z).abs();
            if dx > self.load_distance + 2 || dz > self.load_distance + 2 {
                to_remove.push(*pos);
            }
        }
        for pos in to_remove {
            self.chunks.write().remove(&pos);
            self.meshes.remove(&pos);
            self.water_meshes.remove(&pos);
            // Track for GPU pool eviction
            self.unloaded_chunks.push(pos);
        }
    }

    #[allow(dead_code)]
    fn queue_meshing(&mut self, pos: ChunkPos) {
        self.queue_meshing_with_lod(pos, LodLevel::Full);
    }

    /// Force a remesh regardless of current LOD or pending status
    /// Always queues a new mesh task - used after block modifications (uses Full LOD)
    /// This handles race conditions where a mesh task is already running with stale neighbor data
    fn force_remesh(&mut self, pos: ChunkPos) {
        self.force_remesh_with_lod(pos, LodLevel::Full);
    }

    /// Force a remesh with specified LOD level
    /// Used for neighbor remeshes during chunk generation where LOD should be respected
    fn force_remesh_with_lod(&mut self, pos: ChunkPos, lod: LodLevel) {
        // Increment generation - this invalidates any in-flight mesh tasks for this chunk
        let gen = self.mesh_generation.entry(pos).or_insert(0);
        *gen += 1;
        let current_gen = *gen;

        // Always queue, even if already pending - the in-flight task may have stale data
        // The generation counter ensures only the latest result is accepted
        self.pending_meshing.insert(pos, ());
        let _ = self.task_sender.send(ChunkTask::Mesh(pos, lod, current_gen));
    }

    fn queue_meshing_with_lod(&mut self, pos: ChunkPos, lod: LodLevel) {
        // Check if we already have a mesh with same or better LOD
        if let Some(existing) = self.meshes.get(&pos) {
            if existing.lod as usize <= lod as usize {
                return; // Already have equal or better detail
            }
        }

        if !self.pending_meshing.contains_key(&pos) {
            // Increment generation for new mesh request
            let gen = self.mesh_generation.entry(pos).or_insert(0);
            *gen += 1;
            let current_gen = *gen;
            self.pending_meshing.insert(pos, ());
            let _ = self.task_sender.send(ChunkTask::Mesh(pos, lod, current_gen));
        }
    }

    /// Queue water meshing for a chunk
    fn queue_water_meshing(&mut self, pos: ChunkPos) {
        if !self.pending_water_meshing.contains_key(&pos) {
            self.pending_water_meshing.insert(pos, ());
            let _ = self.task_sender.send(ChunkTask::MeshWater(pos));
        }
    }

    /// Calculate LOD level based on distance from player
    /// Uses dynamic thresholds set via set_lod_thresholds()
    fn calculate_lod(&self, chunk_pos: ChunkPos, player_chunk: ChunkPos) -> LodLevel {
        // Calculate horizontal distance (ignore vertical for LOD)
        let dx = (chunk_pos.x - player_chunk.x).abs();
        let dz = (chunk_pos.z - player_chunk.z).abs();
        let dist = dx.max(dz); // Chebyshev distance

        // Use dynamic thresholds (i32::MAX means LOD is OFF, everything full detail)
        if dist <= self.lod_full_threshold {
            LodLevel::Full
        } else if dist <= self.lod_half_threshold {
            LodLevel::Half
        } else {
            LodLevel::Quarter
        }
    }

    /// Get all visible chunk meshes for rendering
    pub fn visible_meshes(&self) -> impl Iterator<Item = &ChunkMesh> {
        self.meshes.values()
    }

    /// Get all visible water meshes for rendering
    pub fn visible_water_meshes(&self) -> impl Iterator<Item = &WaterMesh> {
        self.water_meshes.values()
    }

    /// Upload pending water meshes to GPU (must be called from main thread)
    pub fn upload_water_meshes(&mut self, device: &wgpu::Device) {
        for mesh in self.water_meshes.values_mut() {
            if mesh.gpu_mesh.is_none() && !mesh.vertices.is_empty() {
                mesh.upload(device);
            }
        }
    }

    /// Get counts for debug display
    pub fn pending_generation_count(&self) -> usize {
        self.pending_generation.len()
    }

    pub fn pending_meshing_count(&self) -> usize {
        self.pending_meshing.len()
    }

    pub fn loaded_chunks_count(&self) -> usize {
        self.chunks.read().len()
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn water_mesh_count(&self) -> usize {
        self.water_meshes.len()
    }

    pub fn total_vertices(&self) -> u64 {
        self.meshes.values().map(|m| m.vertices.len() as u64).sum()
    }

    /// Get LOD distribution for debugging (Full, Half, Quarter counts)
    pub fn lod_distribution(&self) -> (usize, usize, usize) {
        let mut full = 0;
        let mut half = 0;
        let mut quarter = 0;
        for mesh in self.meshes.values() {
            match mesh.lod {
                LodLevel::Full => full += 1,
                LodLevel::Half => half += 1,
                LodLevel::Quarter => quarter += 1,
            }
        }
        (full, half, quarter)
    }

    pub fn total_water_vertices(&self) -> u64 {
        self.water_meshes.values().map(|m| m.vertices.len() as u64).sum()
    }

    /// Take the list of meshes that need urgent upload to chunk pool
    /// Called by renderer to prioritize player-modified meshes
    pub fn take_urgent_uploads(&mut self) -> Vec<ChunkPos> {
        std::mem::take(&mut self.urgent_mesh_uploads)
    }

    /// Take up to `limit` meshes that need upload to chunk pool
    /// Called by renderer each frame - returns newly generated meshes
    /// Remaining meshes stay in the queue for next frame
    pub fn take_pending_uploads(&mut self, limit: usize) -> Vec<ChunkPos> {
        if self.pending_mesh_uploads.len() <= limit {
            std::mem::take(&mut self.pending_mesh_uploads)
        } else {
            // Take first `limit` items, leave rest for next frame
            let remaining = self.pending_mesh_uploads.split_off(limit);
            let taken = std::mem::replace(&mut self.pending_mesh_uploads, remaining);
            taken
        }
    }

    /// Get count of pending mesh uploads (for debug display)
    pub fn pending_upload_count(&self) -> usize {
        self.pending_mesh_uploads.len()
    }

    /// Take the list of chunks that were unloaded and need GPU eviction
    /// Called by renderer to remove stale chunks from the pool
    pub fn take_unloaded_chunks(&mut self) -> Vec<ChunkPos> {
        std::mem::take(&mut self.unloaded_chunks)
    }

    /// Get access to the chunk storage for voxel GI system
    /// Returns a clone of the Arc to avoid holding the lock
    pub fn chunks(&self) -> Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>> {
        Arc::clone(&self.chunks)
    }

    /// Get a snapshot of all loaded chunks for read-only operations
    /// Returns a copy of the chunks hashmap
    pub fn chunks_read(&self) -> AHashMap<ChunkPos, Arc<Chunk>> {
        self.chunks.read().clone()
    }

    /// Get a block at world coordinates
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> Option<Block> {
        let chunk_pos = ChunkPos {
            x: x.div_euclid(CHUNK_SIZE_I32),
            y: y.div_euclid(CHUNK_SIZE_I32),
            z: z.div_euclid(CHUNK_SIZE_I32),
        };

        let local_x = x.rem_euclid(CHUNK_SIZE_I32) as usize;
        let local_y = y.rem_euclid(CHUNK_SIZE_I32) as usize;
        let local_z = z.rem_euclid(CHUNK_SIZE_I32) as usize;

        self.chunks
            .read()
            .get(&chunk_pos)
            .map(|chunk| chunk.get_block(local_x, local_y, local_z))
    }

    /// Get light level at world coordinates
    /// Returns (combined_light, sky_light, block_light)
    pub fn get_light_at(&self, x: i32, y: i32, z: i32) -> (u8, u8, u8) {
        let chunk_pos = ChunkPos {
            x: x.div_euclid(CHUNK_SIZE_I32),
            y: y.div_euclid(CHUNK_SIZE_I32),
            z: z.div_euclid(CHUNK_SIZE_I32),
        };

        let local_x = x.rem_euclid(CHUNK_SIZE_I32) as usize;
        let local_y = y.rem_euclid(CHUNK_SIZE_I32) as usize;
        let local_z = z.rem_euclid(CHUNK_SIZE_I32) as usize;

        if let Some(chunk) = self.chunks.read().get(&chunk_pos) {
            let sky = chunk.get_sky_light(local_x, local_y, local_z);
            let block = chunk.get_block_light(local_x, local_y, local_z);
            (sky.max(block), sky, block)
        } else {
            (15, 15, 0) // Default to bright (outdoor)
        }
    }

    /// Set a block at world coordinates, returns true if successful
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: Block) -> bool {
        let chunk_pos = ChunkPos {
            x: x.div_euclid(CHUNK_SIZE_I32),
            y: y.div_euclid(CHUNK_SIZE_I32),
            z: z.div_euclid(CHUNK_SIZE_I32),
        };

        let local_x = x.rem_euclid(CHUNK_SIZE_I32) as usize;
        let local_y = y.rem_euclid(CHUNK_SIZE_I32) as usize;
        let local_z = z.rem_euclid(CHUNK_SIZE_I32) as usize;

        // Get the old block for light comparison
        let old_block = {
            let chunks_read = self.chunks.read();
            chunks_read.get(&chunk_pos)
                .map(|c| c.get_block(local_x, local_y, local_z))
        };

        let Some(old_block) = old_block else {
            return false;
        };

        // Get the chunk, clone it, modify, and replace
        {
            let mut chunks = self.chunks.write();
            if let Some(old_chunk) = chunks.get(&chunk_pos) {
                let mut new_chunk = Chunk::new();
                // Copy all blocks AND light data from old chunk
                for cz in 0..16 {
                    for cy in 0..16 {
                        for cx in 0..16 {
                            new_chunk.set_block(cx, cy, cz, old_chunk.get_block(cx, cy, cz));
                            new_chunk.set_light(cx, cy, cz, old_chunk.get_light(cx, cy, cz));
                        }
                    }
                }
                // Copy prop data from old chunk (scenery: grass, flowers, rocks)
                if let Some(prop_data) = old_chunk.prop_data() {
                    let new_prop_data = new_chunk.ensure_prop_data_with_capacity(prop_data.len());
                    for prop in prop_data.iter() {
                        new_prop_data.add(*prop);
                    }
                }
                // Copy water data if present
                if let Some(water_data) = old_chunk.water_data() {
                    // Water data is already handled by other systems, but ensure it's preserved
                    let _ = water_data; // water heightfield is regenerated separately
                }
                // Set the new block
                new_chunk.set_block(local_x, local_y, local_z, block);
                chunks.insert(chunk_pos, Arc::new(new_chunk));
            }
        }

        // === LIGHT PROPAGATION ===
        let old_emission = old_block.light_emission();
        let new_emission = block.light_emission();

        // Track chunks that need remeshing due to light changes
        let mut light_dirty_chunks: Vec<ChunkPos> = Vec::new();

        // Case 1: Placed a light source (torch)
        if new_emission > 0 && old_emission == 0 {
            tracing::debug!("Placed light source at ({}, {}, {}) with emission {}", x, y, z, new_emission);
            let updates = self.light_engine.propagate_light(x, y, z, new_emission, &self.chunks);
            apply_light_updates(updates, &self.chunks);
            light_dirty_chunks.extend(self.light_engine.take_dirty_chunks());
        }
        // Case 2: Removed a light source
        else if old_emission > 0 && new_emission == 0 {
            tracing::debug!("Removed light source at ({}, {}, {})", x, y, z);
            let updates = self.light_engine.remove_light(x, y, z, &self.chunks);
            apply_light_updates(updates, &self.chunks);
            light_dirty_chunks.extend(self.light_engine.take_dirty_chunks());
        }
        // Case 3: Placed solid block where transparent was (may block light)
        else if !block.is_light_transparent() && old_block.is_light_transparent() {
            // For simplicity, recalculate light in a small area
            // A more sophisticated approach would trace affected light paths
            let updates = self.light_engine.recalculate_area(x, y, z, 15, &self.chunks);
            apply_light_updates(updates, &self.chunks);
            light_dirty_chunks.extend(self.light_engine.take_dirty_chunks());
        }
        // Case 4: Removed solid block, light may flow in
        else if block.is_light_transparent() && !old_block.is_light_transparent() {
            let updates = self.light_engine.propagate_into(x, y, z, &self.chunks);
            apply_light_updates(updates, &self.chunks);
            light_dirty_chunks.extend(self.light_engine.take_dirty_chunks());
        }

        // SYNCHRONOUS mesh for the main chunk - instant feedback (~2-3ms)
        // This ensures the player sees their block change immediately
        tracing::info!("set_block: ({}, {}, {}) = {:?} in chunk {:?}, local ({}, {}, {})",
            x, y, z, block, chunk_pos, local_x, local_y, local_z);
        self.mesh_chunk_sync(chunk_pos);

        // Determine which neighbor chunks need remeshing based on block position
        // On edges/corners, we need to remesh diagonal neighbors too (for AO and face culling)
        let on_x_min = local_x == 0;
        let on_x_max = local_x == 15;
        let on_y_min = local_y == 0;
        let on_y_max = local_y == 15;
        let on_z_min = local_z == 0;
        let on_z_max = local_z == 15;

        // Collect neighbors that need remeshing
        let mut neighbors_to_remesh: Vec<ChunkPos> = Vec::new();

        // Check all 26 neighbors, but only include those that matter based on position
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue; // Skip self
                    }

                    // Only remesh neighbor if block is on the boundary facing that neighbor
                    let need_x = (dx == -1 && on_x_min) || (dx == 1 && on_x_max) || dx == 0;
                    let need_y = (dy == -1 && on_y_min) || (dy == 1 && on_y_max) || dy == 0;
                    let need_z = (dz == -1 && on_z_min) || (dz == 1 && on_z_max) || dz == 0;

                    // For face neighbors (one axis non-zero), always remesh if on that boundary
                    // For edge/corner neighbors, only remesh if on ALL relevant boundaries
                    let is_face = (dx != 0) as u8 + (dy != 0) as u8 + (dz != 0) as u8 == 1;

                    if is_face {
                        // Face neighbor: remesh if on that boundary
                        if (dx == -1 && on_x_min) || (dx == 1 && on_x_max) ||
                           (dy == -1 && on_y_min) || (dy == 1 && on_y_max) ||
                           (dz == -1 && on_z_min) || (dz == 1 && on_z_max) {
                            neighbors_to_remesh.push(chunk_pos.offset(dx, dy, dz));
                        }
                    } else {
                        // Edge or corner neighbor: need to be on all relevant boundaries
                        // This fixes invisible blocks at chunk corners/edges
                        if need_x && need_y && need_z {
                            neighbors_to_remesh.push(chunk_pos.offset(dx, dy, dz));
                        }
                    }
                }
            }
        }

        // Use priority queue for neighbor remeshes - skip ahead of generation queue
        for neighbor in neighbors_to_remesh {
            self.remesh_neighbor_priority(neighbor);
        }

        // Remesh any additional chunks affected by light changes (also with priority)
        for dirty_pos in light_dirty_chunks {
            if dirty_pos != chunk_pos {
                self.remesh_neighbor_priority(dirty_pos);
            }
        }

        // Mark chunk as dirty for saving
        if let Some(ref storage) = self.storage {
            storage.lock().mark_dirty(chunk_pos);
        }

        true
    }

    /// Try to remove a prop at the given world position
    /// Returns the prop that was removed, if any
    pub fn try_remove_prop(&mut self, world_x: f32, world_y: f32, world_z: f32, tolerance: f32) -> Option<PropPlacement> {
        let chunk_pos = ChunkPos {
            x: (world_x as i32).div_euclid(CHUNK_SIZE_I32),
            y: (world_y as i32).div_euclid(CHUNK_SIZE_I32),
            z: (world_z as i32).div_euclid(CHUNK_SIZE_I32),
        };

        let local_x = world_x - (chunk_pos.x * CHUNK_SIZE_I32) as f32;
        let local_y = world_y - (chunk_pos.y * CHUNK_SIZE_I32) as f32;
        let local_z = world_z - (chunk_pos.z * CHUNK_SIZE_I32) as f32;

        // Get the chunk, clone it, modify, and replace (same pattern as set_block)
        let removed_prop = {
            let mut chunks = self.chunks.write();
            if let Some(old_chunk) = chunks.get(&chunk_pos) {
                let mut new_chunk = Chunk::new();
                // Copy all blocks and light data
                for cz in 0..16 {
                    for cy in 0..16 {
                        for cx in 0..16 {
                            new_chunk.set_block(cx, cy, cz, old_chunk.get_block(cx, cy, cz));
                            new_chunk.set_light(cx, cy, cz, old_chunk.get_light(cx, cy, cz));
                        }
                    }
                }
                // Copy prop data, but try to remove the target prop
                let mut removed: Option<PropPlacement> = None;
                if let Some(prop_data) = old_chunk.prop_data() {
                    let new_prop_data = new_chunk.ensure_prop_data_with_capacity(prop_data.len());
                    for prop in prop_data.iter() {
                        let dx = prop.local_x - local_x;
                        let dy = prop.local_y - local_y;
                        let dz = prop.local_z - local_z;
                        if removed.is_none() && dx * dx + dy * dy + dz * dz < tolerance * tolerance {
                            removed = Some(*prop);
                            // Skip adding this prop - it's removed
                        } else {
                            new_prop_data.add(*prop);
                        }
                    }
                }
                chunks.insert(chunk_pos, Arc::new(new_chunk));
                removed
            } else {
                None
            }
        };

        if removed_prop.is_some() {
            tracing::debug!("Removed prop at ({:.2}, {:.2}, {:.2})", world_x, world_y, world_z);
            // Mark chunk as dirty for saving
            if let Some(ref storage) = self.storage {
                storage.lock().mark_dirty(chunk_pos);
            }
        }

        removed_prop
    }

    #[allow(dead_code)]
    fn remesh_if_exists(&mut self, pos: ChunkPos) {
        if self.chunks.read().contains_key(&pos) {
            // Keep old mesh until new one is ready to avoid flicker
            self.pending_meshing.remove(&pos);
            self.force_remesh(pos);
        }
    }

    /// Force a remesh using the HIGH PRIORITY channel (for player block changes)
    /// This skips ahead of the regular generation queue for immediate feedback
    fn force_remesh_priority(&mut self, pos: ChunkPos) {
        // Increment generation - this invalidates any in-flight mesh tasks for this chunk
        let gen = self.mesh_generation.entry(pos).or_insert(0);
        *gen += 1;
        let current_gen = *gen;

        self.pending_meshing.insert(pos, ());
        let _ = self.priority_task_sender.send(ChunkTask::Mesh(pos, LodLevel::Full, current_gen));
    }

    /// Perform SYNCHRONOUS mesh generation on the main thread for instant feedback
    /// This is used for the chunk the player directly modified - ~2-3ms for instant visual update
    fn mesh_chunk_sync(&mut self, pos: ChunkPos) {
        let chunks_read = self.chunks.read();
        let Some(chunk) = chunks_read.get(&pos).cloned() else {
            tracing::error!("mesh_chunk_sync: chunk {:?} not found!", pos);
            return;
        };

        // Get neighbor chunks for proper face culling
        let neighbors = [
            chunks_read.get(&pos.offset(-1, 0, 0)).cloned(),
            chunks_read.get(&pos.offset(1, 0, 0)).cloned(),
            chunks_read.get(&pos.offset(0, -1, 0)).cloned(),
            chunks_read.get(&pos.offset(0, 1, 0)).cloned(),
            chunks_read.get(&pos.offset(0, 0, -1)).cloned(),
            chunks_read.get(&pos.offset(0, 0, 1)).cloned(),
        ];
        drop(chunks_read);

        // Build mesh synchronously on main thread
        let start = std::time::Instant::now();
        let mesh = meshing::build_chunk_mesh_lod(pos, &chunk, &neighbors, LodLevel::Full);
        let elapsed = start.elapsed();

        tracing::info!(
            "mesh_chunk_sync: {:?} built {} vertices in {:.2}ms (gpu_mesh: {:?})",
            pos,
            mesh.vertices.len(),
            elapsed.as_secs_f64() * 1000.0,
            mesh.gpu_mesh.is_some()
        );

        // Update generation counter
        let gen = self.mesh_generation.entry(pos).or_insert(0);
        *gen += 1;

        // Insert mesh immediately - no waiting for worker
        self.pending_meshing.remove(&pos);
        self.meshes.insert(pos, mesh);

        // Mark for urgent upload to chunk pool (bypass throttling)
        self.urgent_mesh_uploads.push(pos);
    }

    /// Remesh a neighbor chunk using priority queue if it exists
    fn remesh_neighbor_priority(&mut self, pos: ChunkPos) {
        if self.chunks.read().contains_key(&pos) {
            self.pending_meshing.remove(&pos);
            self.force_remesh_priority(pos);
            // Mark as urgent so when the mesh result arrives, it gets priority upload
            self.pending_urgent.insert(pos);
        }
    }

    // ============ WATER FLOW SIMULATION ============

    /// Run water flow simulation for the given delta time
    /// Uses a fixed timestep accumulator for stable physics
    pub fn tick_water_simulation(&mut self, dt: f32) {
        if !self.water_flow_engine.enabled {
            return;
        }

        self.water_flow_accumulator += dt;

        // Run simulation at fixed timestep
        while self.water_flow_accumulator >= WATER_FLOW_TICK_RATE {
            let updates = self.water_flow_engine.simulate_tick(
                WATER_FLOW_TICK_RATE,
                &self.chunks,
            );

            // Apply updates to chunks
            if !updates.is_empty() {
                water_flow::apply_water_updates(updates, &self.chunks);

                // Queue water remeshing for dirty chunks
                for chunk_pos in self.water_flow_engine.take_dirty_chunks() {
                    self.queue_water_meshing(chunk_pos);
                }
            }

            self.water_flow_accumulator -= WATER_FLOW_TICK_RATE;
        }
    }

    /// Activate water flow simulation in a region around a world position
    pub fn activate_water_flow(&mut self, x: i32, z: i32, radius: i32) {
        self.water_flow_engine.activate_region(x, z, radius);
    }

    /// Get number of active water cells being simulated
    pub fn active_water_cell_count(&self) -> usize {
        self.water_flow_engine.active_cell_count()
    }

    /// Enable or disable water flow simulation
    pub fn set_water_flow_enabled(&mut self, enabled: bool) {
        self.water_flow_engine.enabled = enabled;
    }

    /// Check if water flow simulation is enabled
    pub fn water_flow_enabled(&self) -> bool {
        self.water_flow_engine.enabled
    }

    /// Save all dirty chunks and world metadata (blocking)
    pub fn save_all(&mut self) {
        let Some(ref storage) = self.storage else {
            return;
        };

        let mut storage = storage.lock();

        // Get all dirty chunks
        let dirty_positions = storage.take_dirty_chunks();
        let dirty_count = dirty_positions.len();

        if dirty_count > 0 {
            tracing::info!("Saving {} dirty chunks...", dirty_count);

            // Save each dirty chunk
            let chunks = self.chunks.read();
            let mut saved = 0;
            for pos in dirty_positions {
                if let Some(chunk) = chunks.get(&pos) {
                    match storage.save_chunk(pos, chunk) {
                        Ok(_) => saved += 1,
                        Err(e) => tracing::error!("Failed to save chunk {:?}: {}", pos, e),
                    }
                }
            }

            tracing::info!("Saved {}/{} chunks", saved, dirty_count);
        }

        // Save metadata
        if let Err(e) = storage.save_meta() {
            tracing::error!("Failed to save world metadata: {}", e);
        }

        // Flush all region files
        if let Err(e) = storage.flush() {
            tracing::error!("Failed to flush region files: {}", e);
        }
    }

    /// Get the number of dirty chunks waiting to be saved
    pub fn dirty_chunk_count(&self) -> usize {
        self.storage.as_ref().map(|s| s.lock().dirty_count()).unwrap_or(0)
    }

    /// Check if storage is enabled
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// Get the world seed
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get player spawn position from storage
    pub fn spawn_position(&self) -> Option<[f32; 3]> {
        self.storage.as_ref().map(|s| s.lock().spawn_position())
    }

    /// Set player spawn position in storage
    pub fn set_spawn(&mut self, x: f32, y: f32, z: f32) {
        if let Some(ref storage) = self.storage {
            storage.lock().set_spawn(x, y, z);
        }
    }

    /// Load essential chunks around spawn position and wait for them to be ready
    /// This ensures the player doesn't spawn in unloaded terrain
    /// Returns the number of chunks that were loaded
    pub fn load_spawn_area(&mut self, spawn_pos: Vec3, radius: i32) -> usize {
        let spawn_chunk = ChunkPos::from_world_pos(spawn_pos);
        let spawn_y = spawn_chunk.y;

        // Calculate which chunks we need - prioritize by distance
        let mut chunks_needed: Vec<(ChunkPos, i32)> = Vec::new();

        // Load chunks in a cylinder around spawn (full vertical column)
        for dy in -2..=3 {
            let y = spawn_y + dy;
            if y < 0 || y > 9 {
                continue;
            }
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    // Distance for priority (closer = lower = higher priority)
                    let dist_sq = dx * dx + dz * dz + dy * dy * 4; // Weight Y more
                    let pos = ChunkPos {
                        x: spawn_chunk.x + dx,
                        y,
                        z: spawn_chunk.z + dz,
                    };
                    chunks_needed.push((pos, dist_sq));
                }
            }
        }

        // Sort by distance (closest first)
        chunks_needed.sort_by_key(|(_, dist)| *dist);

        // Queue generation for all needed chunks
        let mut queued = 0;
        for (pos, _) in &chunks_needed {
            if !self.chunks.read().contains_key(pos)
                && !self.pending_generation.contains_key(pos)
            {
                self.pending_generation.insert(*pos, ());
                let _ = self.task_sender.send(ChunkTask::Generate(*pos));
                queued += 1;
            }
        }

        tracing::info!(
            "Spawn area: queued {} chunks around ({}, {}, {}), radius {}",
            queued, spawn_chunk.x, spawn_chunk.y, spawn_chunk.z, radius
        );

        // Wait for essential chunks to be generated AND meshed
        // Use most of the radius as essential to prevent gaps
        let essential_radius = radius.max(2) - 1;
        let mut essential_chunks: Vec<ChunkPos> = Vec::new();
        for dy in -1..=2 {
            let y = spawn_y + dy;
            if y < 0 || y > 9 {
                continue;
            }
            for dz in -essential_radius..=essential_radius {
                for dx in -essential_radius..=essential_radius {
                    essential_chunks.push(ChunkPos {
                        x: spawn_chunk.x + dx,
                        y,
                        z: spawn_chunk.z + dz,
                    });
                }
            }
        }

        // Poll until essential chunks are ready (with timeout)
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);

        while start.elapsed() < timeout {
            // Process worker results
            while let Ok(result) = self.result_receiver.try_recv() {
                match result {
                    ChunkResult::Generated(pos, chunk) => {
                        self.pending_generation.remove(&pos);
                        self.chunks.write().insert(pos, Arc::new(*chunk));
                        // Immediately queue meshing for essential chunks
                        if essential_chunks.contains(&pos) {
                            self.queue_meshing_with_lod(pos, LodLevel::Full);
                        }
                    }
                    ChunkResult::Meshed(pos, mesh, generation) => {
                        self.pending_meshing.remove(&pos);
                        // Only accept mesh if its generation matches the current expected generation
                        let current_gen = self.mesh_generation.get(&pos).copied().unwrap_or(0);
                        if generation >= current_gen {
                            self.meshes.insert(pos, mesh);
                        }
                    }
                    ChunkResult::WaterMeshed(pos, water_mesh) => {
                        self.pending_water_meshing.remove(&pos);
                        if !water_mesh.is_empty() {
                            self.water_meshes.insert(pos, water_mesh);
                        }
                    }
                }
            }

            // Check if all essential chunks have meshes
            let ready_count = essential_chunks.iter()
                .filter(|pos| self.meshes.contains_key(pos))
                .count();

            if ready_count >= essential_chunks.len() {
                tracing::info!(
                    "Spawn area ready: {}/{} essential chunks loaded in {:.1}s",
                    ready_count, essential_chunks.len(), start.elapsed().as_secs_f32()
                );
                return ready_count;
            }

            // Small sleep to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Timeout - report what we got
        let ready_count = essential_chunks.iter()
            .filter(|pos| self.meshes.contains_key(pos))
            .count();
        tracing::warn!(
            "Spawn area timeout: only {}/{} essential chunks ready after {:.1}s",
            ready_count, essential_chunks.len(), timeout.as_secs_f32()
        );
        ready_count
    }

    /// Check if the immediate area around a position has loaded chunks
    pub fn is_area_ready(&self, pos: Vec3, radius: i32) -> bool {
        let chunk_pos = ChunkPos::from_world_pos(pos);
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let check = ChunkPos {
                    x: chunk_pos.x + dx,
                    y: chunk_pos.y,
                    z: chunk_pos.z + dz,
                };
                if !self.meshes.contains_key(&check) {
                    return false;
                }
            }
        }
        true
    }
}

impl Drop for WorldManager {
    fn drop(&mut self) {
        // Save all dirty chunks before shutting down
        self.save_all();

        // Signal workers to shut down
        for _ in &self._workers {
            let _ = self.task_sender.send(ChunkTask::Shutdown);
        }
    }
}
