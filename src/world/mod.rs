//! World management module
//!
//! Handles chunks, terrain generation, and mesh building.

mod chunk;
mod generation;
mod meshing;

pub use chunk::{Block, Chunk, ChunkPos};
pub use meshing::{ChunkMesh, ChunkVertex, GpuMesh, LodLevel, build_chunk_mesh, build_chunk_mesh_lod};

use std::sync::Arc;

use ahash::AHashMap;
use crossbeam_channel::{Receiver, Sender};
use glam::Vec3;
use parking_lot::RwLock;

use crate::constants::{CHUNK_SIZE_I32, RENDER_DISTANCE};

/// Result of a raycast into the world
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    /// World position of the hit block
    pub block_pos: [i32; 3],
    /// World position of the adjacent block (for placement)
    pub adjacent_pos: [i32; 3],
    /// Distance from ray origin to hit
    pub distance: f32,
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

/// Messages for chunk worker threads
enum ChunkTask {
    Generate(ChunkPos),
    Mesh(ChunkPos, LodLevel),
    Shutdown,
}

/// Results from chunk workers
enum ChunkResult {
    Generated(ChunkPos, Box<Chunk>),
    Meshed(ChunkPos, ChunkMesh),
}

/// Manages all world chunks and coordinates generation/meshing
pub struct WorldManager {
    /// Loaded chunks (thread-safe for parallel access)
    chunks: Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,

    /// Chunk meshes ready for rendering
    meshes: AHashMap<ChunkPos, ChunkMesh>,

    /// Chunks pending generation
    pending_generation: AHashMap<ChunkPos, ()>,

    /// Chunks pending meshing
    pending_meshing: AHashMap<ChunkPos, ()>,

    /// Channel to send tasks to workers
    task_sender: Sender<ChunkTask>,

    /// Channel to receive results from workers
    result_receiver: Receiver<ChunkResult>,

    /// Worker thread handles
    _workers: Vec<std::thread::JoinHandle<()>>,

    /// Last player chunk position (for loading new chunks)
    last_player_chunk: Option<ChunkPos>,
}

impl WorldManager {
    pub fn new() -> Self {
        let (task_sender, task_receiver) = crossbeam_channel::unbounded::<ChunkTask>();
        let (result_sender, result_receiver) = crossbeam_channel::unbounded::<ChunkResult>();

        let chunks: Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>> =
            Arc::new(RwLock::new(AHashMap::new()));

        // Spawn worker threads
        let num_workers = num_cpus::get().max(2).min(8);
        tracing::info!("Spawning {} chunk worker threads", num_workers);

        let mut workers = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let task_rx = task_receiver.clone();
            let result_tx = result_sender.clone();
            let chunks_ref = chunks.clone();

            let handle = std::thread::Builder::new()
                .name(format!("chunk-worker-{}", i))
                .spawn(move || {
                    Self::worker_loop(task_rx, result_tx, chunks_ref);
                })
                .expect("Failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            chunks,
            meshes: AHashMap::new(),
            pending_generation: AHashMap::new(),
            pending_meshing: AHashMap::new(),
            task_sender,
            result_receiver,
            _workers: workers,
            last_player_chunk: None,
        }
    }

    fn worker_loop(
        task_rx: Receiver<ChunkTask>,
        result_tx: Sender<ChunkResult>,
        chunks: Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) {
        while let Ok(task) = task_rx.recv() {
            match task {
                ChunkTask::Generate(pos) => {
                    let chunk = generation::generate_chunk(pos);
                    let _ = result_tx.send(ChunkResult::Generated(pos, Box::new(chunk)));
                }
                ChunkTask::Mesh(pos, lod) => {
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
                        let _ = result_tx.send(ChunkResult::Meshed(pos, mesh));
                    }
                }
                ChunkTask::Shutdown => break,
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

        // Process results from workers
        while let Ok(result) = self.result_receiver.try_recv() {
            match result {
                ChunkResult::Generated(pos, chunk) => {
                    self.pending_generation.remove(&pos);
                    self.chunks.write().insert(pos, Arc::new(*chunk));
                    // Queue meshing with appropriate LOD
                    let lod = self.calculate_lod(pos, player_chunk);
                    self.queue_meshing_with_lod(pos, lod);
                    // Also update neighbor meshes
                    for offset in &[(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                        let neighbor = pos.offset(offset.0, offset.1, offset.2);
                        if self.chunks.read().contains_key(&neighbor) {
                            let neighbor_lod = self.calculate_lod(neighbor, player_chunk);
                            self.queue_meshing_with_lod(neighbor, neighbor_lod);
                        }
                    }
                }
                ChunkResult::Meshed(pos, mesh) => {
                    self.pending_meshing.remove(&pos);
                    self.meshes.insert(pos, mesh);
                }
            }
        }

        // Only do full update if player moved to a new chunk
        if self.last_player_chunk == Some(player_chunk) {
            return;
        }
        self.last_player_chunk = Some(player_chunk);

        // Queue chunks for generation around player
        // Vertical range: Y chunks 0-9 covers blocks 0-159, enough for terrain + mountains
        for dy in 0..=9 {
            for dz in -RENDER_DISTANCE..=RENDER_DISTANCE {
                for dx in -RENDER_DISTANCE..=RENDER_DISTANCE {
                    let pos = ChunkPos {
                        x: player_chunk.x + dx,
                        y: dy,
                        z: player_chunk.z + dz,
                    };

                    if !self.chunks.read().contains_key(&pos)
                        && !self.pending_generation.contains_key(&pos)
                    {
                        self.pending_generation.insert(pos, ());
                        let _ = self.task_sender.send(ChunkTask::Generate(pos));
                    }
                }
            }
        }

        // Upgrade LOD for chunks that are now closer
        for (pos, mesh) in &self.meshes {
            let desired_lod = self.calculate_lod(*pos, player_chunk);
            if (desired_lod as usize) < (mesh.lod as usize) {
                // Need higher detail - will be handled by queue_meshing_with_lod
            }
        }

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
            if dx > RENDER_DISTANCE + 2 || dz > RENDER_DISTANCE + 2 {
                to_remove.push(*pos);
            }
        }
        for pos in to_remove {
            self.chunks.write().remove(&pos);
            self.meshes.remove(&pos);
        }
    }

    fn queue_meshing(&mut self, pos: ChunkPos) {
        self.queue_meshing_with_lod(pos, LodLevel::Full);
    }

    /// Force a remesh regardless of current LOD (used after block modifications)
    fn force_remesh(&mut self, pos: ChunkPos) {
        if !self.pending_meshing.contains_key(&pos) {
            self.pending_meshing.insert(pos, ());
            let _ = self.task_sender.send(ChunkTask::Mesh(pos, LodLevel::Full));
        }
    }

    fn queue_meshing_with_lod(&mut self, pos: ChunkPos, lod: LodLevel) {
        // Check if we already have a mesh with same or better LOD
        if let Some(existing) = self.meshes.get(&pos) {
            if existing.lod as usize <= lod as usize {
                return; // Already have equal or better detail
            }
        }

        if !self.pending_meshing.contains_key(&pos) {
            self.pending_meshing.insert(pos, ());
            let _ = self.task_sender.send(ChunkTask::Mesh(pos, lod));
        }
    }

    /// Calculate LOD level based on distance from player
    fn calculate_lod(&self, chunk_pos: ChunkPos, player_chunk: ChunkPos) -> LodLevel {
        use crate::constants::{LOD_FULL_DETAIL, LOD_HALF_DETAIL};

        // Calculate horizontal distance (ignore vertical for LOD)
        let dx = (chunk_pos.x - player_chunk.x).abs();
        let dz = (chunk_pos.z - player_chunk.z).abs();
        let dist = dx.max(dz); // Chebyshev distance

        if dist <= LOD_FULL_DETAIL {
            LodLevel::Full
        } else if dist <= LOD_HALF_DETAIL {
            LodLevel::Half
        } else {
            LodLevel::Quarter
        }
    }

    /// Get all visible chunk meshes for rendering
    pub fn visible_meshes(&self) -> impl Iterator<Item = &ChunkMesh> {
        self.meshes.values()
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

    pub fn total_vertices(&self) -> u64 {
        self.meshes.values().map(|m| m.vertices.len() as u64).sum()
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

        // Get the chunk, clone it, modify, and replace
        let mut chunks = self.chunks.write();
        if let Some(old_chunk) = chunks.get(&chunk_pos) {
            let mut new_chunk = Chunk::new();
            // Copy all blocks from old chunk
            for z in 0..16 {
                for y in 0..16 {
                    for x in 0..16 {
                        new_chunk.set_block(x, y, z, old_chunk.get_block(x, y, z));
                    }
                }
            }
            // Set the new block
            new_chunk.set_block(local_x, local_y, local_z, block);
            chunks.insert(chunk_pos, Arc::new(new_chunk));
            drop(chunks);

            // Force remesh (keep old mesh until new one is ready to avoid flicker)
            self.pending_meshing.remove(&chunk_pos); // Clear pending to allow re-queue
            self.force_remesh(chunk_pos);

            // Also remesh neighbors if block is on chunk boundary
            if local_x == 0 {
                self.remesh_if_exists(chunk_pos.offset(-1, 0, 0));
            }
            if local_x == 15 {
                self.remesh_if_exists(chunk_pos.offset(1, 0, 0));
            }
            if local_y == 0 {
                self.remesh_if_exists(chunk_pos.offset(0, -1, 0));
            }
            if local_y == 15 {
                self.remesh_if_exists(chunk_pos.offset(0, 1, 0));
            }
            if local_z == 0 {
                self.remesh_if_exists(chunk_pos.offset(0, 0, -1));
            }
            if local_z == 15 {
                self.remesh_if_exists(chunk_pos.offset(0, 0, 1));
            }

            true
        } else {
            false
        }
    }

    fn remesh_if_exists(&mut self, pos: ChunkPos) {
        if self.chunks.read().contains_key(&pos) {
            // Keep old mesh until new one is ready to avoid flicker
            self.pending_meshing.remove(&pos);
            self.force_remesh(pos);
        }
    }
}

impl Drop for WorldManager {
    fn drop(&mut self) {
        // Signal workers to shut down
        for _ in &self._workers {
            let _ = self.task_sender.send(ChunkTask::Shutdown);
        }
    }
}
