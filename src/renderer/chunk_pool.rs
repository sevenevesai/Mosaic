//! Chunk buffer pool for multi-draw indirect rendering
//!
//! Manages a single large vertex/index buffer that holds all chunk meshes,
//! enabling efficient batched rendering with multi_draw_indexed_indirect.

use ahash::AHashMap;

use crate::world::{ChunkMesh, ChunkPos, ChunkVertex};

/// Size of a single chunk slot in vertices (max vertices per chunk)
/// With greedy meshing, typical is ~1500 but complex terrain can hit ~4000
const MAX_VERTICES_PER_CHUNK: u32 = 4096;

/// Size of a single chunk slot in indices (max indices per chunk)
/// Typically 6 indices per 4 vertices (quad), max ~6144
const MAX_INDICES_PER_CHUNK: u32 = 8192;

/// Maximum number of chunk slots in the pool
/// 2200 slots * 4096 verts * 28 bytes = 252MB vertex (under 256MB limit)
/// 2200 slots * 8192 indices * 4 bytes = 72MB index
/// (28-byte vertex now includes emergence_data field)
const MAX_CHUNK_SLOTS: u32 = 2200;

/// Information about a chunk's location in the mega-buffer
#[derive(Clone, Copy, Debug)]
pub struct ChunkSlot {
    /// Slot index in the pool
    pub slot_index: u32,
    /// Actual vertex count (may be less than MAX_VERTICES_PER_CHUNK)
    pub vertex_count: u32,
    /// Actual index count (may be less than MAX_INDICES_PER_CHUNK)
    pub index_count: u32,
}

/// Draw command for indirect rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

impl DrawIndexedIndirect {
    pub fn new(slot: &ChunkSlot) -> Self {
        Self {
            index_count: slot.index_count,
            instance_count: 1,
            first_index: slot.slot_index * MAX_INDICES_PER_CHUNK,
            base_vertex: (slot.slot_index * MAX_VERTICES_PER_CHUNK) as i32,
            first_instance: 0,
        }
    }
}

impl ChunkSlot {
    pub fn first_index(&self) -> u32 {
        self.slot_index * MAX_INDICES_PER_CHUNK
    }

    pub fn base_vertex(&self) -> i32 {
        (self.slot_index * MAX_VERTICES_PER_CHUNK) as i32
    }
}

/// Manages the chunk buffer pool
pub struct ChunkPool {
    /// Mega vertex buffer holding all chunk vertices
    pub vertex_buffer: wgpu::Buffer,
    /// Mega index buffer holding all chunk indices
    pub index_buffer: wgpu::Buffer,
    /// Indirect draw command buffer
    pub indirect_buffer: wgpu::Buffer,

    /// Map of chunk position to slot info
    chunk_slots: AHashMap<ChunkPos, ChunkSlot>,

    /// Free slot indices (for reuse)
    free_slots: Vec<u32>,
    /// Next slot index to allocate if no free slots
    next_slot: u32,

    /// Whether multi-draw indirect is available
    pub mdi_available: bool,

    /// Number of draw commands in the indirect buffer (updated each frame)
    pub draw_count: u32,
}

impl ChunkPool {
    pub fn new(device: &wgpu::Device, mdi_available: bool) -> Self {
        let vertex_buffer_size = MAX_CHUNK_SLOTS as u64
            * MAX_VERTICES_PER_CHUNK as u64
            * std::mem::size_of::<ChunkVertex>() as u64;

        let index_buffer_size = MAX_CHUNK_SLOTS as u64
            * MAX_INDICES_PER_CHUNK as u64
            * std::mem::size_of::<u32>() as u64;

        // Indirect buffer holds one DrawIndexedIndirect per potential visible chunk
        let indirect_buffer_size = MAX_CHUNK_SLOTS as u64
            * std::mem::size_of::<DrawIndexedIndirect>() as u64;

        tracing::info!(
            "Creating chunk pool: {:.1}MB vertices, {:.1}MB indices, {:.1}MB indirect",
            vertex_buffer_size as f64 / 1024.0 / 1024.0,
            index_buffer_size as f64 / 1024.0 / 1024.0,
            indirect_buffer_size as f64 / 1024.0 / 1024.0,
        );

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Pool Vertex Buffer"),
            size: vertex_buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Pool Index Buffer"),
            size: index_buffer_size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Pool Indirect Buffer"),
            size: indirect_buffer_size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            indirect_buffer,
            chunk_slots: AHashMap::new(),
            free_slots: Vec::new(),
            next_slot: 0,
            mdi_available,
            draw_count: 0,
        }
    }

    /// Allocate a slot for a chunk, returns None if pool is full
    fn allocate_slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free_slots.pop() {
            Some(slot)
        } else if self.next_slot < MAX_CHUNK_SLOTS {
            let slot = self.next_slot;
            self.next_slot += 1;
            Some(slot)
        } else {
            // Pool full - logged once per session, not spam
            None
        }
    }

    /// Free a slot for reuse
    fn free_slot(&mut self, slot: u32) {
        self.free_slots.push(slot);
    }

    /// Upload a chunk mesh to the pool
    pub fn upload_chunk(&mut self, queue: &wgpu::Queue, mesh: &ChunkMesh) {
        // Skip empty meshes
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            // Remove from pool if it existed
            if let Some(slot) = self.chunk_slots.remove(&mesh.pos) {
                self.free_slot(slot.slot_index);
            }
            return;
        }

        // Check if mesh fits in a slot - log only once per size category
        if mesh.vertices.len() > MAX_VERTICES_PER_CHUNK as usize {
            // Skip oversized chunks silently (rare edge case)
            return;
        }
        if mesh.indices.len() > MAX_INDICES_PER_CHUNK as usize {
            return;
        }

        // Get or allocate slot
        let slot_index = if let Some(existing) = self.chunk_slots.get(&mesh.pos) {
            existing.slot_index
        } else if let Some(new_slot) = self.allocate_slot() {
            new_slot
        } else {
            return; // Pool full
        };

        // Calculate buffer offsets
        let vertex_offset = slot_index as u64
            * MAX_VERTICES_PER_CHUNK as u64
            * std::mem::size_of::<ChunkVertex>() as u64;
        let index_offset = slot_index as u64
            * MAX_INDICES_PER_CHUNK as u64
            * std::mem::size_of::<u32>() as u64;

        // Upload data
        queue.write_buffer(&self.vertex_buffer, vertex_offset, bytemuck::cast_slice(&mesh.vertices));
        queue.write_buffer(&self.index_buffer, index_offset, bytemuck::cast_slice(&mesh.indices));

        // Update slot info
        self.chunk_slots.insert(mesh.pos, ChunkSlot {
            slot_index,
            vertex_count: mesh.vertices.len() as u32,
            index_count: mesh.indices.len() as u32,
        });
    }

    /// Remove a chunk from the pool
    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        if let Some(slot) = self.chunk_slots.remove(&pos) {
            self.free_slot(slot.slot_index);
        }
    }

    /// Check if a chunk is in the pool
    pub fn has_chunk(&self, pos: &ChunkPos) -> bool {
        self.chunk_slots.contains_key(pos)
    }

    /// Get slot info for a chunk
    pub fn get_slot(&self, pos: &ChunkPos) -> Option<&ChunkSlot> {
        self.chunk_slots.get(pos)
    }

    /// Prepare indirect draw commands for visible chunks
    /// Returns the number of draw commands written
    pub fn prepare_indirect_draws(
        &mut self,
        queue: &wgpu::Queue,
        visible_chunks: &[ChunkPos],
    ) -> u32 {
        let mut commands: Vec<DrawIndexedIndirect> = Vec::with_capacity(visible_chunks.len());

        for pos in visible_chunks {
            if let Some(slot) = self.chunk_slots.get(pos) {
                if slot.index_count > 0 {
                    commands.push(DrawIndexedIndirect::new(slot));
                }
            }
        }

        if !commands.is_empty() {
            queue.write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&commands));
        }

        self.draw_count = commands.len() as u32;
        self.draw_count
    }

    /// Get total number of chunks in pool
    pub fn chunk_count(&self) -> usize {
        self.chunk_slots.len()
    }

    /// Get memory usage stats
    pub fn memory_stats(&self) -> (u64, u64) {
        let vertex_used = self.chunk_slots.values()
            .map(|s| s.vertex_count as u64 * std::mem::size_of::<ChunkVertex>() as u64)
            .sum();
        let index_used = self.chunk_slots.values()
            .map(|s| s.index_count as u64 * std::mem::size_of::<u32>() as u64)
            .sum();
        (vertex_used, index_used)
    }

    /// Evict chunks that are not in the visible set to make room for new ones
    pub fn evict_non_visible(&mut self, visible: &[ChunkPos]) {
        use std::collections::HashSet;
        let visible_set: HashSet<_> = visible.iter().collect();

        // Find chunks to evict (not in visible set)
        let to_evict: Vec<ChunkPos> = self.chunk_slots.keys()
            .filter(|pos| !visible_set.contains(pos))
            .copied()
            .collect();

        // Evict up to half the non-visible chunks to make room
        let evict_count = to_evict.len().min(MAX_CHUNK_SLOTS as usize / 4);
        for pos in to_evict.into_iter().take(evict_count) {
            self.remove_chunk(pos);
        }
    }

    /// Check if pool is nearly full
    pub fn is_nearly_full(&self) -> bool {
        self.chunk_slots.len() >= (MAX_CHUNK_SLOTS as usize * 9 / 10)
    }
}
