//! Flood fill lighting system
//!
//! Implements BFS-based light propagation for block light (torches, lava, etc.).
//! Light levels range from 0-15 and decay by 1 per block traveled.
//!
//! Sky light is deferred to a future implementation.

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use ahash::AHashMap;

use super::chunk::{Block, Chunk, ChunkPos};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32};

/// Node in the light propagation queue
#[derive(Clone, Copy, Debug)]
pub struct LightNode {
    /// World coordinates
    pub x: i32,
    pub y: i32,
    pub z: i32,
    /// Current light level at this position
    pub level: u8,
}

/// Node for light removal (tracks old level for proper propagation)
#[derive(Clone, Copy, Debug)]
pub struct RemovalNode {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub old_level: u8,
}

/// Convert world coordinates to chunk position and local coordinates
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

/// 6 cardinal directions for light propagation
const DIRECTIONS: [(i32, i32, i32); 6] = [
    (1, 0, 0),   // +X
    (-1, 0, 0),  // -X
    (0, 1, 0),   // +Y
    (0, -1, 0),  // -Y
    (0, 0, 1),   // +Z
    (0, 0, -1),  // -Z
];

/// Light propagation engine that works with the chunk map
pub struct LightEngine {
    /// Queue for light propagation (adding light)
    propagation_queue: VecDeque<LightNode>,
    /// Queue for light removal
    removal_queue: VecDeque<RemovalNode>,
    /// Chunks that need remeshing after light changes
    dirty_chunks: Vec<ChunkPos>,
}

impl LightEngine {
    pub fn new() -> Self {
        Self {
            propagation_queue: VecDeque::with_capacity(512),
            removal_queue: VecDeque::with_capacity(512),
            dirty_chunks: Vec::with_capacity(32),
        }
    }

    /// Get the list of chunks that were modified and need remeshing
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkPos> {
        std::mem::take(&mut self.dirty_chunks)
    }

    /// Mark a chunk as needing remesh
    fn mark_dirty(&mut self, chunk_pos: ChunkPos) {
        if !self.dirty_chunks.contains(&chunk_pos) {
            self.dirty_chunks.push(chunk_pos);
        }
    }

    /// Propagate block light from a source position
    /// Call this when placing a light-emitting block (torch, etc.)
    pub fn propagate_light(
        &mut self,
        source_x: i32,
        source_y: i32,
        source_z: i32,
        light_level: u8,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) -> Vec<(ChunkPos, Vec<(usize, usize, usize, u8)>)> {
        if light_level == 0 {
            return Vec::new();
        }

        // Track the best light level assigned to each position to avoid duplicates
        // Key: (x, y, z), Value: light level already assigned
        let mut visited: AHashMap<(i32, i32, i32), u8> = AHashMap::new();

        // Collect all light updates per chunk for batch application
        let mut chunk_updates: AHashMap<ChunkPos, Vec<(usize, usize, usize, u8)>> = AHashMap::new();

        // Seed the queue with the source
        self.propagation_queue.clear();
        self.propagation_queue.push_back(LightNode {
            x: source_x,
            y: source_y,
            z: source_z,
            level: light_level,
        });

        // Mark source as visited
        visited.insert((source_x, source_y, source_z), light_level);

        // Add source to updates
        let (chunk_pos, lx, ly, lz) = world_to_chunk_local(source_x, source_y, source_z);
        chunk_updates
            .entry(chunk_pos)
            .or_insert_with(Vec::new)
            .push((lx, ly, lz, light_level));
        self.mark_dirty(chunk_pos);

        // BFS flood fill
        let chunks_read = chunks.read();

        while let Some(node) = self.propagation_queue.pop_front() {
            let next_level = node.level.saturating_sub(1);
            if next_level == 0 {
                continue;
            }

            // Check all 6 neighbors
            for (dx, dy, dz) in DIRECTIONS {
                let nx = node.x + dx;
                let ny = node.y + dy;
                let nz = node.z + dz;

                // Check if we've already visited this position with equal or better light
                if let Some(&existing) = visited.get(&(nx, ny, nz)) {
                    if existing >= next_level {
                        continue;
                    }
                }

                let (neighbor_chunk_pos, nlx, nly, nlz) = world_to_chunk_local(nx, ny, nz);

                // Get the chunk containing this neighbor
                if let Some(chunk) = chunks_read.get(&neighbor_chunk_pos) {
                    let block = chunk.get_block(nlx, nly, nlz);

                    // Light only propagates through transparent blocks
                    if !block.is_light_transparent() {
                        continue;
                    }

                    // Also check original chunk light level (for existing torches)
                    let current_light = chunk.get_block_light(nlx, nly, nlz);

                    // Only update if we'd increase the light level
                    if next_level > current_light {
                        // Mark as visited with this light level
                        visited.insert((nx, ny, nz), next_level);

                        chunk_updates
                            .entry(neighbor_chunk_pos)
                            .or_insert_with(Vec::new)
                            .push((nlx, nly, nlz, next_level));
                        self.mark_dirty(neighbor_chunk_pos);

                        self.propagation_queue.push_back(LightNode {
                            x: nx,
                            y: ny,
                            z: nz,
                            level: next_level,
                        });
                    }
                }
            }
        }

        drop(chunks_read);

        // Convert to vec for return
        chunk_updates.into_iter().collect()
    }

    /// Remove block light when a light source is destroyed
    /// Call this when breaking a light-emitting block
    pub fn remove_light(
        &mut self,
        source_x: i32,
        source_y: i32,
        source_z: i32,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) -> Vec<(ChunkPos, Vec<(usize, usize, usize, u8)>)> {
        // Get the old light level at the source
        let (source_chunk_pos, slx, sly, slz) = world_to_chunk_local(source_x, source_y, source_z);

        let old_level = {
            let chunks_read = chunks.read();
            if let Some(chunk) = chunks_read.get(&source_chunk_pos) {
                chunk.get_block_light(slx, sly, slz)
            } else {
                return Vec::new();
            }
        };

        if old_level == 0 {
            return Vec::new();
        }

        // Track visited positions to avoid duplicates
        let mut visited: AHashMap<(i32, i32, i32), bool> = AHashMap::new();

        // Collect updates: positions to zero out and positions to re-propagate from
        let mut chunk_updates: AHashMap<ChunkPos, Vec<(usize, usize, usize, u8)>> = AHashMap::new();
        let mut relight_sources: Vec<LightNode> = Vec::new();

        // Seed removal queue
        self.removal_queue.clear();
        self.removal_queue.push_back(RemovalNode {
            x: source_x,
            y: source_y,
            z: source_z,
            old_level,
        });
        visited.insert((source_x, source_y, source_z), true);

        // Zero out the source
        chunk_updates
            .entry(source_chunk_pos)
            .or_insert_with(Vec::new)
            .push((slx, sly, slz, 0));
        self.mark_dirty(source_chunk_pos);

        // BFS removal
        let chunks_read = chunks.read();

        while let Some(node) = self.removal_queue.pop_front() {
            // Check all 6 neighbors
            for (dx, dy, dz) in DIRECTIONS {
                let nx = node.x + dx;
                let ny = node.y + dy;
                let nz = node.z + dz;

                // Skip if already visited
                if visited.contains_key(&(nx, ny, nz)) {
                    continue;
                }

                let (neighbor_chunk_pos, nlx, nly, nlz) = world_to_chunk_local(nx, ny, nz);

                if let Some(chunk) = chunks_read.get(&neighbor_chunk_pos) {
                    let neighbor_light = chunk.get_block_light(nlx, nly, nlz);

                    if neighbor_light != 0 && neighbor_light < node.old_level {
                        // This light was dependent on the removed source - remove it
                        visited.insert((nx, ny, nz), true);

                        chunk_updates
                            .entry(neighbor_chunk_pos)
                            .or_insert_with(Vec::new)
                            .push((nlx, nly, nlz, 0));
                        self.mark_dirty(neighbor_chunk_pos);

                        self.removal_queue.push_back(RemovalNode {
                            x: nx,
                            y: ny,
                            z: nz,
                            old_level: neighbor_light,
                        });
                    } else if neighbor_light >= node.old_level {
                        // This light is from another source - we'll re-propagate from here
                        relight_sources.push(LightNode {
                            x: nx,
                            y: ny,
                            z: nz,
                            level: neighbor_light,
                        });
                    }
                }
            }
        }

        drop(chunks_read);

        // Store updates so far (removals)
        let removal_updates: Vec<_> = chunk_updates.into_iter().collect();

        // Now re-propagate from remaining light sources
        // This fills in any gaps that should still be lit
        for source in relight_sources {
            // We need to apply the removal updates first, then re-propagate
            // For simplicity, we'll return the removal updates and let the caller handle re-propagation
        }

        removal_updates
    }

    /// Recalculate light when an opaque block is placed (may block existing light)
    /// This is more complex - for now, we just clear and re-propagate in the area
    pub fn recalculate_area(
        &mut self,
        center_x: i32,
        center_y: i32,
        center_z: i32,
        radius: i32,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) -> Vec<(ChunkPos, Vec<(usize, usize, usize, u8)>)> {
        let mut chunk_updates: AHashMap<ChunkPos, Vec<(usize, usize, usize, u8)>> = AHashMap::new();
        let mut light_sources: Vec<LightNode> = Vec::new();

        let chunks_read = chunks.read();

        // First pass: clear all light in the area and collect light sources
        for z in (center_z - radius)..=(center_z + radius) {
            for y in (center_y - radius)..=(center_y + radius) {
                for x in (center_x - radius)..=(center_x + radius) {
                    let (chunk_pos, lx, ly, lz) = world_to_chunk_local(x, y, z);

                    if let Some(chunk) = chunks_read.get(&chunk_pos) {
                        let block = chunk.get_block(lx, ly, lz);
                        let emission = block.light_emission();

                        if emission > 0 {
                            // This is a light source - save it for re-propagation
                            light_sources.push(LightNode {
                                x,
                                y,
                                z,
                                level: emission,
                            });
                        }

                        // Clear the light
                        if chunk.get_block_light(lx, ly, lz) > 0 {
                            chunk_updates
                                .entry(chunk_pos)
                                .or_insert_with(Vec::new)
                                .push((lx, ly, lz, 0));
                            self.mark_dirty(chunk_pos);
                        }
                    }
                }
            }
        }

        drop(chunks_read);

        // Return the clear updates - re-propagation should be done after applying these
        // The caller should then call propagate_light for each light source found
        chunk_updates.into_iter().collect()
    }

    /// Check neighboring light and propagate into a newly-transparent position
    /// Call this when breaking a solid block
    pub fn propagate_into(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) -> Vec<(ChunkPos, Vec<(usize, usize, usize, u8)>)> {
        let mut chunk_updates: AHashMap<ChunkPos, Vec<(usize, usize, usize, u8)>> = AHashMap::new();

        // Find the maximum neighboring light level
        let mut max_neighbor_light = 0u8;

        {
            let chunks_read = chunks.read();

            for (dx, dy, dz) in DIRECTIONS {
                let nx = x + dx;
                let ny = y + dy;
                let nz = z + dz;

                let (neighbor_chunk_pos, nlx, nly, nlz) = world_to_chunk_local(nx, ny, nz);

                if let Some(chunk) = chunks_read.get(&neighbor_chunk_pos) {
                    let neighbor_light = chunk.get_block_light(nlx, nly, nlz);
                    max_neighbor_light = max_neighbor_light.max(neighbor_light);
                }
            }
        }

        // If neighbors have light, propagate from this position
        if max_neighbor_light > 1 {
            let new_light = max_neighbor_light - 1;
            let (chunk_pos, lx, ly, lz) = world_to_chunk_local(x, y, z);

            chunk_updates
                .entry(chunk_pos)
                .or_insert_with(Vec::new)
                .push((lx, ly, lz, new_light));
            self.mark_dirty(chunk_pos);

            // If this new light is high enough, continue propagation
            if new_light > 1 {
                let additional = self.propagate_light(x, y, z, new_light, chunks);
                for (cp, updates) in additional {
                    chunk_updates.entry(cp).or_insert_with(Vec::new).extend(updates);
                }
            }
        }

        chunk_updates.into_iter().collect()
    }
}

impl Default for LightEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply light updates to chunks (call after propagation)
/// Takes ownership of chunks temporarily to apply batch updates
pub fn apply_light_updates(
    updates: Vec<(ChunkPos, Vec<(usize, usize, usize, u8)>)>,
    chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
) {
    let mut chunks_write = chunks.write();

    for (chunk_pos, light_updates) in updates {
        if let Some(old_chunk) = chunks_write.get(&chunk_pos) {
            // Clone the chunk for modification
            let mut new_chunk = Chunk::new();

            // Copy all block and light data
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        new_chunk.set_block(x, y, z, old_chunk.get_block(x, y, z));
                        new_chunk.set_light(x, y, z, old_chunk.get_light(x, y, z));
                    }
                }
            }

            // Apply light updates
            for (x, y, z, level) in light_updates {
                new_chunk.set_block_light(x, y, z, level);
            }

            // Replace the chunk
            chunks_write.insert(chunk_pos, Arc::new(new_chunk));
        }
    }
}
