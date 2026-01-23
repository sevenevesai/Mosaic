//! World management module
//!
//! Handles chunks, terrain generation, and mesh building.

mod chunk;
mod generation;
mod meshing;

pub use chunk::{Block, Chunk, ChunkPos};
pub use meshing::{ChunkMesh, ChunkVertex, GpuMesh, LodLevel, build_chunk_mesh};

use std::collections::HashMap;
use std::sync::Arc;

use ahash::AHashMap;
use crossbeam_channel::{Receiver, Sender};
use glam::Vec3;
use parking_lot::RwLock;

use crate::constants::{CHUNK_SIZE_I32, RENDER_DISTANCE, LOD_FULL_DETAIL, LOD_HALF_DETAIL};

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

                        let mesh = meshing::build_chunk_mesh(pos, &chunk, &neighbors);
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
        for dy in -2..=2 {
            for dz in -RENDER_DISTANCE..=RENDER_DISTANCE {
                for dx in -RENDER_DISTANCE..=RENDER_DISTANCE {
                    let pos = ChunkPos {
                        x: player_chunk.x + dx,
                        y: dy, // Only a few vertical chunks for PoC
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
    fn calculate_lod(&self, _chunk_pos: ChunkPos, _player_chunk: ChunkPos) -> LodLevel {
        // TODO: Re-enable LOD once greedy meshing is verified working
        // For now, always use full detail
        LodLevel::Full
    }

    /// Get all visible chunk meshes for rendering
    pub fn visible_meshes(&self) -> impl Iterator<Item = &ChunkMesh> {
        self.meshes.values()
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
}

impl Drop for WorldManager {
    fn drop(&mut self) {
        // Signal workers to shut down
        for _ in &self._workers {
            let _ = self.task_sender.send(ChunkTask::Shutdown);
        }
    }
}
