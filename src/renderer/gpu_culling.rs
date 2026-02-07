//! GPU-based frustum culling using compute shaders
//!
//! Replaces CPU frustum culling loop with a compute shader that:
//! 1. Tests chunk AABBs against frustum planes in parallel
//! 2. Writes indirect draw commands for visible chunks
//! 3. Outputs draw count via atomic counter
//!
//! This eliminates ~0.7ms of CPU work per frame.

use std::num::NonZeroU64;
use ahash::AHashMap;
use glam::Vec3;

use crate::world::ChunkPos;
use super::chunk_pool::ChunkSlot;

// ============================================================================
// VISIBILITY CACHE
// ============================================================================

/// Visibility cache for frame-to-frame culling reuse
#[derive(Debug)]
pub struct VisibilityCache {
    last_camera_pos: Vec3,
    last_camera_forward: Vec3,
    last_draw_count: u32,
    valid: bool,
    cache_hits: u64,
    cache_misses: u64,
}

impl Default for VisibilityCache {
    fn default() -> Self { Self::new() }
}

impl VisibilityCache {
    const POSITION_THRESHOLD: f32 = 2.0;
    const ROTATION_THRESHOLD: f32 = 0.087;

    pub fn new() -> Self {
        Self {
            last_camera_pos: Vec3::ZERO,
            last_camera_forward: Vec3::NEG_Z,
            last_draw_count: 0,
            valid: false,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    fn camera_moved(&self, camera_pos: Vec3, camera_forward: Vec3) -> bool {
        let pos_delta = (camera_pos - self.last_camera_pos).length();
        let forward_delta = (camera_forward.normalize() - self.last_camera_forward.normalize()).length();
        pos_delta > Self::POSITION_THRESHOLD || forward_delta > Self::ROTATION_THRESHOLD
    }

    pub fn try_use_cache(&mut self, camera_pos: Vec3, camera_forward: Vec3) -> Option<u32> {
        if !self.valid { self.cache_misses += 1; return None; }
        if self.camera_moved(camera_pos, camera_forward) { 
            self.valid = false; 
            self.cache_misses += 1; 
            tracing::trace!("Visibility cache MISS (camera moved)");
            return None; 
        }
        self.cache_hits += 1;
        tracing::trace!("Visibility cache HIT (skipping compute dispatch)");
        Some(self.last_draw_count)
    }

    pub fn update(&mut self, camera_pos: Vec3, camera_forward: Vec3, draw_count: u32) {
        self.last_camera_pos = camera_pos;
        self.last_camera_forward = camera_forward.normalize();
        self.last_draw_count = draw_count;
        self.valid = true;
    }

    pub fn invalidate(&mut self) { self.valid = false; }
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f32 / total as f32 }
    }
    pub fn stats(&self) -> (u64, u64) { (self.cache_hits, self.cache_misses) }
    pub fn is_valid(&self) -> bool { self.valid }
}

// ============================================================================
// GPU CULLING STRUCTS
// ============================================================================

/// Maximum chunks that can be culled (must match chunk pool)
const MAX_CHUNKS: u32 = 2200;

/// Chunk bounds data uploaded to GPU
/// Contains position and draw command info for each chunk slot
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkBoundsGpu {
    /// World position of chunk minimum corner (x, y, z) + padding
    pub min: [f32; 4],
    /// Slot index in chunk pool
    pub slot_index: u32,
    /// Index count for draw command
    pub index_count: u32,
    /// Padding for alignment
    pub _padding: [u32; 2],
}

/// Uniforms for culling compute shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullUniforms {
    /// Frustum planes (6 planes, each is vec4: xyz = normal, w = distance)
    pub frustum_planes: [[f32; 4]; 6],
    /// Chunk size in world units
    pub chunk_size: f32,
    /// Number of chunks to process
    pub chunk_count: u32,
    /// Padding
    pub _padding: [f32; 2],
}

/// Atomic counter for visible chunk count (read back to CPU)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
pub struct DrawCount {
    pub count: u32,
}

/// GPU frustum culler
pub struct GpuCuller {
    /// Compute pipeline for culling
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
    /// Current bind group (recreated when chunk data changes)
    bind_group: Option<wgpu::BindGroup>,

    /// Buffer containing chunk bounds data
    bounds_buffer: wgpu::Buffer,
    /// Uniform buffer for frustum planes
    uniform_buffer: wgpu::Buffer,
    /// Atomic counter buffer (GPU-side)
    count_buffer: wgpu::Buffer,
    /// Staging buffer for reading count back to CPU
    count_staging: wgpu::Buffer,

    /// Output indirect draw buffer (shared with chunk pool)
    /// We write directly to the chunk pool's indirect buffer

    /// Current number of chunks in bounds buffer
    chunk_count: u32,
    /// Mapping from slot index to bounds buffer index
    slot_to_bounds_idx: AHashMap<u32, u32>,
    /// Whether bounds buffer needs rebuild
    dirty: bool,

    /// Visibility cache for skipping redundant culls
    visibility_cache: VisibilityCache,
}

impl GpuCuller {
    pub fn new(device: &wgpu::Device, _indirect_buffer: &wgpu::Buffer) -> Self {
        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(CULL_SHADER.into()),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Cull Bind Group Layout"),
            entries: &[
                // Uniforms (frustum planes)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<CullUniforms>() as u64),
                    },
                    count: None,
                },
                // Chunk bounds (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indirect draw commands (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Atomic counter (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU Cull Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU Cull Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cull_main",
        });

        // Create buffers
        let bounds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Bounds Buffer"),
            size: (MAX_CHUNKS as usize * std::mem::size_of::<ChunkBoundsGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Uniforms Buffer"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Count Buffer"),
            size: 4,
            // INDIRECT added for use with multi_draw_indexed_indirect_count
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Count Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group: None,
            bounds_buffer,
            uniform_buffer,
            count_buffer,
            count_staging,
            chunk_count: 0,
            slot_to_bounds_idx: AHashMap::new(),
            dirty: true,
            visibility_cache: VisibilityCache::new(),
        }
    }

    /// Rebuild the bind group (call when indirect buffer changes)
    pub fn rebuild_bind_group(&mut self, device: &wgpu::Device, indirect_buffer: &wgpu::Buffer) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Cull Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.count_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Update chunk bounds data from chunk pool slots
    pub fn update_bounds(
        &mut self,
        queue: &wgpu::Queue,
        slots: &AHashMap<ChunkPos, ChunkSlot>,
    ) {
        // Build bounds data
        let mut bounds: Vec<ChunkBoundsGpu> = Vec::with_capacity(slots.len());
        self.slot_to_bounds_idx.clear();

        for (pos, slot) in slots {
            if slot.index_count == 0 {
                continue;
            }

            let world_origin = pos.world_origin();
            let idx = bounds.len() as u32;
            self.slot_to_bounds_idx.insert(slot.slot_index, idx);

            bounds.push(ChunkBoundsGpu {
                min: [world_origin.x, world_origin.y, world_origin.z, 0.0],
                slot_index: slot.slot_index,
                index_count: slot.index_count,
                _padding: [0; 2],
            });
        }

        self.chunk_count = bounds.len() as u32;

        if !bounds.is_empty() {
            queue.write_buffer(&self.bounds_buffer, 0, bytemuck::cast_slice(&bounds));
            tracing::debug!("GPU Culler: Updated bounds buffer with {} chunks", self.chunk_count);
        }

        self.dirty = false;

        // Invalidate visibility cache since chunk bounds changed
        self.visibility_cache.invalidate();
    }

    /// Run GPU culling with visibility caching
    pub fn cull_with_cache(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frustum_planes: &[[f32; 4]; 6],
        chunk_size: f32,
        camera_pos: Vec3,
        camera_forward: Vec3,
    ) -> Option<u32> {
        if self.chunk_count == 0 { return Some(0); }

        // Check visibility cache first
        if let Some(cached_count) = self.visibility_cache.try_use_cache(camera_pos, camera_forward) {
            return Some(cached_count);
        }

        // Cache miss - run full culling
        let bind_group = self.bind_group.as_ref()?;
        let uniforms = CullUniforms {
            frustum_planes: *frustum_planes,
            chunk_size,
            chunk_count: self.chunk_count,
            _padding: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(&self.count_buffer, 0, bytemuck::bytes_of(&0u32));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU Cull Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let workgroups = (self.chunk_count + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.count_buffer, 0, &self.count_staging, 0, 4);
        self.visibility_cache.update(camera_pos, camera_forward, self.chunk_count);
        
        // Log cache stats every 300 frames
        let (hits, misses) = self.visibility_cache.stats();
        let total = hits + misses;
        if total > 0 && total % 300 == 0 {
            let hit_rate = hits as f32 / total as f32 * 100.0;
            tracing::debug!("Visibility cache: {:.1}% hit rate ({} hits / {} total)", hit_rate, hits, total);
        }
        
        Some(self.chunk_count)
    }

    /// Run GPU culling and return draw count
    /// Returns None if GPU culling isn't ready
    pub fn cull(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frustum_planes: &[[f32; 4]; 6],
        chunk_size: f32,
    ) -> Option<u32> {
        if self.chunk_count == 0 {
            return Some(0);
        }

        let bind_group = self.bind_group.as_ref()?;

        // Update uniforms
        let uniforms = CullUniforms {
            frustum_planes: *frustum_planes,
            chunk_size,
            chunk_count: self.chunk_count,
            _padding: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Reset counter to 0
        queue.write_buffer(&self.count_buffer, 0, bytemuck::bytes_of(&0u32));

        // Dispatch compute shader
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU Cull Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // 64 threads per workgroup, dispatch enough to cover all chunks
            let workgroups = (self.chunk_count + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy count to staging buffer for CPU read
        encoder.copy_buffer_to_buffer(&self.count_buffer, 0, &self.count_staging, 0, 4);

        // Note: Actual count reading happens asynchronously
        // For now, return chunk_count as upper bound (actual culling happens on GPU)
        Some(self.chunk_count)
    }

    /// Get the draw count (call after submit, with map_async)
    #[allow(dead_code)]
    pub fn get_draw_count_async(&self) -> &wgpu::Buffer {
        &self.count_staging
    }

    /// Mark bounds as needing rebuild
    #[allow(dead_code)]
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.visibility_cache.invalidate();
    }

    /// Check if bounds need rebuild
    #[allow(dead_code)]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the count buffer for use with multi_draw_indexed_indirect_count
    pub fn count_buffer(&self) -> &wgpu::Buffer {
        &self.count_buffer
    }

    /// Get current chunk count (for stats)
    pub fn chunk_count(&self) -> u32 {
        self.chunk_count
    }

    /// Get visibility cache hit rate (0.0 - 1.0)
    pub fn cache_hit_rate(&self) -> f32 {
        self.visibility_cache.hit_rate()
    }

    /// Get visibility cache statistics (hits, misses)
    pub fn cache_stats(&self) -> (u64, u64) {
        self.visibility_cache.stats()
    }

    /// Invalidate visibility cache
    pub fn invalidate_cache(&mut self) {
        self.visibility_cache.invalidate();
    }
}

/// WGSL compute shader for frustum culling
const CULL_SHADER: &str = r#"
struct CullUniforms {
    frustum_planes: array<vec4<f32>, 6>,
    chunk_size: f32,
    chunk_count: u32,
    _padding: vec2<f32>,
}

struct ChunkBounds {
    min: vec4<f32>,
    slot_index: u32,
    index_count: u32,
    _padding: vec2<u32>,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> chunks: array<ChunkBounds>;
@group(0) @binding(2) var<storage, read_write> draw_commands: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read_write> draw_count: atomic<u32>;

// Test if AABB is visible against frustum planes
fn test_aabb_frustum(min: vec3<f32>, max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = uniforms.frustum_planes[i];

        // Find the positive vertex (furthest in plane normal direction)
        var p: vec3<f32>;
        if (plane.x > 0.0) { p.x = max.x; } else { p.x = min.x; }
        if (plane.y > 0.0) { p.y = max.y; } else { p.y = min.y; }
        if (plane.z > 0.0) { p.z = max.z; } else { p.z = min.z; }

        // If positive vertex is behind plane, AABB is outside frustum
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

// Constants matching chunk_pool.rs
const MAX_VERTICES_PER_CHUNK: u32 = 4096u;
const MAX_INDICES_PER_CHUNK: u32 = 8192u;

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.chunk_count) {
        return;
    }

    let chunk = chunks[idx];
    let min = chunk.min.xyz;
    let max = min + vec3<f32>(uniforms.chunk_size);

    if (test_aabb_frustum(min, max)) {
        // Chunk is visible - add draw command
        let draw_idx = atomicAdd(&draw_count, 1u);

        draw_commands[draw_idx].index_count = chunk.index_count;
        draw_commands[draw_idx].instance_count = 1u;
        draw_commands[draw_idx].first_index = chunk.slot_index * MAX_INDICES_PER_CHUNK;
        draw_commands[draw_idx].base_vertex = i32(chunk.slot_index * MAX_VERTICES_PER_CHUNK);
        draw_commands[draw_idx].first_instance = 0u;
    }
}
"#;
