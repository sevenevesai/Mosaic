//! Voxel-based Global Illumination System
//!
//! Implements:
//! - 3D voxel volume texture for world representation
//! - DDA ray marching for shadows (replaces shadow maps)
//! - Voxel Cone Tracing for global illumination
//! - GPU-based light propagation via compute shaders
//!
//! Based on modern voxel GI techniques optimized for our engine.

use glam::Vec3;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use parking_lot::RwLock;
use ahash::AHashMap;
use rayon::prelude::*;

use crate::world::{ChunkPos, Chunk, Block};
use crate::constants::CHUNK_SIZE_I32;

/// Size of the voxel volume texture (in voxels per dimension)
/// 256³ = 16M voxels, covering ±128 blocks from camera (256 total)
/// Memory: 256³ × 4 bytes = ~64MB for voxel texture
pub const VOXEL_VOLUME_SIZE: u32 = 256;

/// Number of mip levels for cone tracing
/// 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 9 levels
pub const VOXEL_MIP_LEVELS: u32 = 9;

/// Voxel data format: RGBA8 where:
/// - R: opacity (0 = air, 255 = solid)
/// - G: emissive intensity (for light sources)
/// - B: material type (for future use)
/// - A: reserved
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelData {
    pub opacity: u8,
    pub emissive: u8,
    pub material: u8,
    pub flags: u8,
}

impl VoxelData {
    pub const AIR: Self = Self { opacity: 0, emissive: 0, material: 0, flags: 0 };
    pub const SOLID: Self = Self { opacity: 255, emissive: 0, material: 1, flags: 0 };

    pub fn from_block(block: Block) -> Self {
        match block {
            Block::Air => Self::AIR,
            Block::Water => Self { opacity: 64, emissive: 0, material: 2, flags: 0 },
            Block::Torch => Self { opacity: 0, emissive: 220, material: 3, flags: 0 },
            Block::Leaves => Self { opacity: 192, emissive: 0, material: 4, flags: 0 },
            _ if block.is_solid() => Self::SOLID,
            _ => Self::AIR,
        }
    }
}

/// Radiance data for cone tracing (stored in separate texture)
/// Uses R11G11B10 format for HDR in compact form
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelRadiance {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub density: f32,
}

/// GPU uniform for voxel volume parameters
/// Note: Must be 128 bytes to match WGSL alignment requirements
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVolumeUniform {
    /// World position of volume center (camera position, snapped to voxel grid)
    pub volume_center: [f32; 4],
    /// Volume dimensions in world units
    pub volume_size: [f32; 4],
    /// Inverse volume size for UV calculations
    pub inv_volume_size: [f32; 4],
    /// Sun direction for shadow rays
    pub sun_direction: [f32; 4],
    /// Sun color and intensity
    pub sun_color: [f32; 4],
    /// Ambient color
    pub ambient_color: [f32; 4],
    /// Time of day and shadow settings (vec4 for alignment)
    /// x = time, y = shadow_mode (0=hard, 1=soft), z = shadow_debug, w = padding
    pub time_of_day: [f32; 4],
    /// Player held light: [x, y, z, intensity (0-15 normalized to 0-1)]
    pub player_light: [f32; 4],
}

impl Default for VoxelVolumeUniform {
    fn default() -> Self {
        Self {
            volume_center: [0.0; 4],
            volume_size: [VOXEL_VOLUME_SIZE as f32; 4],
            inv_volume_size: [1.0 / VOXEL_VOLUME_SIZE as f32; 4],
            sun_direction: [0.5, 0.8, 0.3, 1.0],
            sun_color: [1.0, 0.95, 0.85, 1.0],
            ambient_color: [0.3, 0.35, 0.4, 1.0],
            time_of_day: [0.5, 0.0, 0.0, 0.0],
            player_light: [0.0; 4],  // No held light by default
        }
    }
}

/// A dirty block that needs incremental update
#[derive(Clone, Copy)]
struct DirtyBlock {
    /// World position of the changed block
    x: i32,
    y: i32,
    z: i32,
}

/// Maximum number of dirty blocks before triggering a full refresh
/// With incremental updates, each block only affects a small region
const MAX_DIRTY_BLOCKS: usize = 64;

/// Manages the 3D voxel volume for GI calculations
pub struct VoxelGI {
    /// 3D texture storing voxel opacity/emissive data
    pub voxel_texture: wgpu::Texture,
    pub voxel_texture_view: wgpu::TextureView,

    /// 3D texture storing accumulated radiance (for VCT)
    pub radiance_texture: wgpu::Texture,
    pub radiance_texture_view: wgpu::TextureView,
    /// Mip views for cone tracing
    pub radiance_mip_views: Vec<wgpu::TextureView>,

    /// Uniform buffer for volume parameters
    pub uniform_buffer: wgpu::Buffer,

    /// Bind group layout for voxel GI
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group containing all voxel GI resources
    pub bind_group: wgpu::BindGroup,

    /// Compute pipeline for updating voxel volume from chunks
    pub update_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for light propagation (flood fill)
    pub propagate_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for mipmap generation
    pub mipmap_pipeline: wgpu::ComputePipeline,

    /// Current volume center (world coordinates)
    volume_center: Vec3,
    /// Last update position (to detect when re-upload needed)
    /// Starts at origin, will update on first frame
    last_update_center: Vec3,
    /// Whether we've done the initial voxel fill
    initialized: bool,
    /// Force full update on next frame
    force_full_update: bool,
    /// Dirty blocks needing incremental update (world positions)
    dirty_blocks: Vec<DirtyBlock>,

    /// Staging buffer for CPU -> GPU voxel data transfer (reserved for async uploads)
    #[allow(dead_code)]
    staging_buffer: wgpu::Buffer,
    /// CPU-side voxel data cache
    voxel_cache: Vec<u8>,
}

impl VoxelGI {
    pub fn new(device: &wgpu::Device) -> Self {
        // Create 3D voxel texture (RGBA8)
        let voxel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Volume Texture"),
            size: wgpu::Extent3d {
                width: VOXEL_VOLUME_SIZE,
                height: VOXEL_VOLUME_SIZE,
                depth_or_array_layers: VOXEL_VOLUME_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let voxel_texture_view = voxel_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create radiance texture with mipmaps for cone tracing
        let radiance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Radiance Texture"),
            size: wgpu::Extent3d {
                width: VOXEL_VOLUME_SIZE,
                height: VOXEL_VOLUME_SIZE,
                depth_or_array_layers: VOXEL_VOLUME_SIZE,
            },
            mip_level_count: VOXEL_MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let radiance_texture_view = radiance_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create individual mip level views for compute shader writes
        let radiance_mip_views: Vec<_> = (0..VOXEL_MIP_LEVELS)
            .map(|mip| {
                radiance_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Radiance Mip {}", mip)),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Create uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel GI Uniform Buffer"),
            contents: bytemuck::cast_slice(&[VoxelVolumeUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create sampler for voxel textures
        let voxel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Voxel Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Voxel GI Bind Group Layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Voxel texture (for reading)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Radiance texture (for cone tracing)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel GI Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&voxel_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&radiance_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&voxel_sampler),
                },
            ],
        });

        // Create compute pipelines
        let update_pipeline = Self::create_update_pipeline(device, &bind_group_layout);
        let propagate_pipeline = Self::create_propagate_pipeline(device, &bind_group_layout);
        let mipmap_pipeline = Self::create_mipmap_pipeline(device, &bind_group_layout);

        // Create staging buffer for CPU uploads
        let staging_size = (VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE * 4) as usize;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Staging Buffer"),
            size: staging_size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        Self {
            voxel_texture,
            voxel_texture_view,
            radiance_texture,
            radiance_texture_view,
            radiance_mip_views,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            update_pipeline,
            propagate_pipeline,
            mipmap_pipeline,
            volume_center: Vec3::ZERO,
            last_update_center: Vec3::ZERO,
            initialized: false,
            force_full_update: false,
            dirty_blocks: Vec::with_capacity(MAX_DIRTY_BLOCKS),
            staging_buffer,
            voxel_cache: vec![0u8; staging_size],
        }
    }

    /// Mark the voxel volume as needing a full update
    /// Use mark_dirty_at() for single block changes (much faster)
    pub fn mark_dirty(&mut self) {
        self.force_full_update = true;
    }

    /// Mark a specific block as dirty for incremental update
    /// This is much faster than full update - only updates one voxel + light radius
    pub fn mark_dirty_at(&mut self, x: i32, y: i32, z: i32) {
        // If we have too many dirty blocks, switch to full update
        if self.dirty_blocks.len() >= MAX_DIRTY_BLOCKS {
            self.force_full_update = true;
            self.dirty_blocks.clear();
            return;
        }
        self.dirty_blocks.push(DirtyBlock { x, y, z });
    }

    fn create_update_pipeline(device: &wgpu::Device, _bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Update Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_UPDATE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Update Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Voxel Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        })
    }

    fn create_propagate_pipeline(device: &wgpu::Device, _bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Propagation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(LIGHT_PROPAGATE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Propagation Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Light Propagation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        })
    }

    fn create_mipmap_pipeline(device: &wgpu::Device, _bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Mipmap Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_MIPMAP_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Mipmap Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Voxel Mipmap Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        })
    }

    /// Update the voxel volume from chunk data
    /// Supports two modes:
    /// - Full update: When camera moves significantly or on initialization
    /// - Incremental update: When individual blocks change (much faster)
    pub fn update_from_chunks(
        &mut self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        sun_direction: Vec3,
        sun_color: Vec3,
        ambient_color: Vec3,
        time_of_day: f32,
        shadow_mode: f32,
        shadow_debug: f32,
        held_light_emission: u8,
    ) {
        // Snap camera position to voxel grid to prevent jitter
        let snapped_center = Vec3::new(
            camera_pos.x.floor(),
            camera_pos.y.floor(),
            camera_pos.z.floor(),
        );

        // Check if we need a full refresh (camera moved or first time)
        let move_threshold = 24.0;
        let distance_moved = (snapped_center - self.last_update_center).length();
        let needs_full_refresh = !self.initialized || distance_moved >= move_threshold || self.force_full_update;

        // Check for incremental updates (dirty blocks)
        let has_dirty_blocks = !self.dirty_blocks.is_empty();

        // If no updates needed, just update uniforms
        if !needs_full_refresh && !has_dirty_blocks {
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission);
            return;
        }

        // Handle INCREMENTAL updates (dirty blocks) - much faster path
        if has_dirty_blocks && !needs_full_refresh {
            let dirty_blocks: Vec<DirtyBlock> = self.dirty_blocks.drain(..).collect();
            let update_start = std::time::Instant::now();

            self.update_dirty_blocks(queue, &dirty_blocks, chunks);

            let update_time = update_start.elapsed();
            if update_time.as_secs_f64() * 1000.0 > 0.5 {
                tracing::debug!(
                    "Voxel incremental update: {:.2}ms for {} blocks",
                    update_time.as_secs_f64() * 1000.0,
                    dirty_blocks.len()
                );
            }

            // Update uniforms with current volume center
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission);
            return;
        }

        // Handle FULL refresh (camera moved or initialization)
        self.force_full_update = false;
        self.dirty_blocks.clear(); // Clear any pending dirty blocks

        if !self.initialized {
            tracing::info!("Initial voxel volume fill at {:?}", snapped_center);
            self.initialized = true;
        } else {
            tracing::debug!("Updating voxel volume: moved {:.1} blocks", distance_moved);
        }

        self.volume_center = snapped_center;
        self.last_update_center = snapped_center;

        // Update uniform buffer
        self.update_uniform(queue, snapped_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission);

        // Time the voxel cache fill
        let fill_start = std::time::Instant::now();
        self.fill_voxel_cache(chunks, time_of_day, snapped_center);
        let fill_time = fill_start.elapsed();

        // Time the GPU upload
        let upload_start = std::time::Instant::now();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.voxel_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.voxel_cache,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(VOXEL_VOLUME_SIZE * 4),
                rows_per_image: Some(VOXEL_VOLUME_SIZE),
            },
            wgpu::Extent3d {
                width: VOXEL_VOLUME_SIZE,
                height: VOXEL_VOLUME_SIZE,
                depth_or_array_layers: VOXEL_VOLUME_SIZE,
            },
        );
        let upload_time = upload_start.elapsed();

        // Log timing if significant (> 1ms total)
        let total_ms = fill_time.as_secs_f64() * 1000.0 + upload_time.as_secs_f64() * 1000.0;
        if total_ms > 1.0 {
            tracing::info!(
                "Voxel update: fill={:.2}ms, upload={:.2}ms, total={:.2}ms",
                fill_time.as_secs_f64() * 1000.0,
                upload_time.as_secs_f64() * 1000.0,
                total_ms
            );
        }
    }

    fn update_uniform(
        &self,
        queue: &wgpu::Queue,
        center: Vec3,
        sun_direction: Vec3,
        sun_color: Vec3,
        ambient_color: Vec3,
        time_of_day: f32,
        shadow_mode: f32,
        shadow_debug: f32,
        player_pos: Vec3,
        held_light_emission: u8,
    ) {
        let uniform = VoxelVolumeUniform {
            volume_center: [center.x, center.y, center.z, 0.0],
            volume_size: [VOXEL_VOLUME_SIZE as f32; 4],
            inv_volume_size: [1.0 / VOXEL_VOLUME_SIZE as f32; 4],
            // W component is sun intensity (0-1), calculated from sun height
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, sun_direction.y.max(0.0).min(1.0)],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, 1.0],
            ambient_color: [ambient_color.x, ambient_color.y, ambient_color.z, 1.0],
            time_of_day: [time_of_day, shadow_mode, shadow_debug, 0.0],
            player_light: [player_pos.x, player_pos.y, player_pos.z, held_light_emission as f32 / 15.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn fill_voxel_cache(&mut self, chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>, time_of_day: f32, center: Vec3) {
        let half_size = (VOXEL_VOLUME_SIZE / 2) as f32;
        let volume_min = self.volume_center - Vec3::splat(half_size);
        let volume_max = self.volume_center + Vec3::splat(half_size);

        // Clear cache using optimized memset (faster than writing all voxels including air)
        let clear_start = std::time::Instant::now();
        self.voxel_cache.fill(0);
        let clear_time = clear_start.elapsed();

        let chunk_start = std::time::Instant::now();

        // Step 1: Collect overlapping chunks (fast, single-threaded)
        // Use integer bounds for faster comparison
        let vol_min_xi = volume_min.x as i32;
        let vol_min_yi = volume_min.y as i32;
        let vol_min_zi = volume_min.z as i32;
        let vol_max_xi = volume_max.x as i32;
        let vol_max_yi = volume_max.y as i32;
        let vol_max_zi = volume_max.z as i32;

        let overlapping_chunks: Vec<(ChunkPos, Arc<Chunk>)> = {
            let chunks_read = chunks.read();
            chunks_read.iter()
                .filter(|(chunk_pos, _)| {
                    // Integer bounds check (faster than float)
                    let chunk_min_x = chunk_pos.x * CHUNK_SIZE_I32;
                    let chunk_min_y = chunk_pos.y * CHUNK_SIZE_I32;
                    let chunk_min_z = chunk_pos.z * CHUNK_SIZE_I32;
                    let chunk_max_x = chunk_min_x + 16;
                    let chunk_max_y = chunk_min_y + 16;
                    let chunk_max_z = chunk_min_z + 16;

                    // AABB overlap test
                    chunk_max_x > vol_min_xi && chunk_min_x < vol_max_xi &&
                    chunk_max_y > vol_min_yi && chunk_min_y < vol_max_yi &&
                    chunk_max_z > vol_min_zi && chunk_min_z < vol_max_zi
                })
                .map(|(pos, chunk)| (*pos, Arc::clone(chunk)))
                .collect()
        };
        let num_chunks = overlapping_chunks.len();

        // Step 2: Process chunks in PARALLEL - only write solid blocks (air already zeroed)
        use std::sync::atomic::{AtomicU32, Ordering};

        // Reinterpret cache as atomic u32s for faster 4-byte aligned writes
        // SAFETY: Cache is 4-byte aligned (RGBA8 format), AtomicU32 has same size
        let atomic_cache: &[AtomicU32] = unsafe {
            std::slice::from_raw_parts(
                self.voxel_cache.as_ptr() as *const AtomicU32,
                self.voxel_cache.len() / 4
            )
        };

        // Pre-compute volume bounds as integers for faster comparisons
        let vol_min_x = volume_min.x as i32;
        let vol_min_y = volume_min.y as i32;
        let vol_min_z = volume_min.z as i32;
        let vol_size = VOXEL_VOLUME_SIZE as i32;

        // Process only solid/emissive blocks and collect light sources
        let light_sources: Vec<(i32, i32, i32, u8)> = overlapping_chunks
            .par_iter()
            .flat_map(|(chunk_pos, chunk)| {
                let mut lights: Vec<(i32, i32, i32, u8)> = Vec::new();

                // Pre-compute chunk origin for this chunk
                let chunk_origin_x = chunk_pos.x * CHUNK_SIZE_I32;
                let chunk_origin_y = chunk_pos.y * CHUNK_SIZE_I32;
                let chunk_origin_z = chunk_pos.z * CHUNK_SIZE_I32;

                for lz in 0..16i32 {
                    for ly in 0..16i32 {
                        for lx in 0..16i32 {
                            let block = chunk.get_block(lx as usize, ly as usize, lz as usize);

                            // Skip air blocks (most common case - already zeroed by clear)
                            if !block.is_solid() && block.light_emission() == 0 {
                                continue;
                            }

                            // Convert to voxel coordinates (integer math)
                            let vx = chunk_origin_x + lx - vol_min_x;
                            let vy = chunk_origin_y + ly - vol_min_y;
                            let vz = chunk_origin_z + lz - vol_min_z;

                            // Skip if outside volume bounds (single comparison per axis)
                            if (vx as u32) >= vol_size as u32 ||
                               (vy as u32) >= vol_size as u32 ||
                               (vz as u32) >= vol_size as u32 {
                                continue;
                            }

                            let voxel = VoxelData::from_block(block);

                            // Pack voxel data into single u32 (RGBA8 little-endian)
                            let packed = (voxel.opacity as u32)
                                | ((voxel.emissive as u32) << 8)
                                | ((voxel.material as u32) << 16)
                                | ((voxel.flags as u32) << 24);

                            // Calculate index in cache (one u32 per voxel)
                            let idx = (vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                     + vy as u32 * VOXEL_VOLUME_SIZE
                                     + vx as u32) as usize;

                            // Single atomic write instead of 4
                            atomic_cache[idx].store(packed, Ordering::Relaxed);

                            // Track light sources for propagation
                            if voxel.emissive > 0 {
                                lights.push((vx, vy, vz, voxel.emissive));
                            }
                        }
                    }
                }
                lights
            })
            .collect();

        let chunk_time = chunk_start.elapsed();

        // Propagate light from sources to nearby voxels
        // Store propagated light in the A channel (flags -> light)
        let prop_start = std::time::Instant::now();
        self.propagate_light(&light_sources);
        let prop_time = prop_start.elapsed();

        // Determine time of day phase for context
        let time_phase = if time_of_day < 0.2 { "night" }
            else if time_of_day < 0.3 { "dawn" }
            else if time_of_day < 0.7 { "day" }
            else if time_of_day < 0.8 { "dusk" }
            else { "night" };

        // Determine environment type based on Y position
        let env_type = if center.y < 20.0 { "deep-cave" }
            else if center.y < 45.0 { "cave/underground" }
            else if center.y < 65.0 { "surface" }
            else { "sky/mountain" };

        // Detailed timing breakdown with context
        tracing::info!(
            "Voxel fill: clear={:.1}ms chunks={:.1}ms prop={:.1}ms | {} lights, {} chunks | pos=({:.0},{:.0},{:.0}) {} {}",
            clear_time.as_secs_f64() * 1000.0,
            chunk_time.as_secs_f64() * 1000.0,
            prop_time.as_secs_f64() * 1000.0,
            light_sources.len(),
            num_chunks,
            center.x, center.y, center.z,
            env_type,
            time_phase
        );
    }

    /// Propagate light from emissive sources through the volume
    /// Stores result in the A channel (index + 3) as propagated light intensity
    /// OPTIMIZED: Uses distance-squared checks, reduced radius, early bounds skipping
    fn propagate_light(&mut self, light_sources: &[(i32, i32, i32, u8)]) {
        let size = VOXEL_VOLUME_SIZE as i32;
        let max_light_radius = 10i32;  // Reduced from 14 for performance (still good coverage)
        let max_radius_sq = (max_light_radius * max_light_radius) as f32;
        let max_radius_f = max_light_radius as f32;

        for &(lx, ly, lz, emission) in light_sources {
            let emission_f = emission as f32 / 255.0;

            // Early bounds: compute actual iteration ranges
            let z_min = (lz - max_light_radius).max(0);
            let z_max = (lz + max_light_radius).min(size - 1);
            let y_min = (ly - max_light_radius).max(0);
            let y_max = (ly + max_light_radius).min(size - 1);
            let x_min = (lx - max_light_radius).max(0);
            let x_max = (lx + max_light_radius).min(size - 1);

            // Spread light in a sphere around the source
            for vz in z_min..=z_max {
                let dz = vz - lz;
                let dz_sq = dz * dz;

                // Early skip: if dz alone exceeds radius, skip this z-plane
                if dz_sq as f32 > max_radius_sq {
                    continue;
                }

                for vy in y_min..=y_max {
                    let dy = vy - ly;
                    let dy_sq = dy * dy;
                    let dz_dy_sq = (dz_sq + dy_sq) as f32;

                    // Early skip: if dz+dy exceeds radius, skip this row
                    if dz_dy_sq > max_radius_sq {
                        continue;
                    }

                    for vx in x_min..=x_max {
                        let dx = vx - lx;
                        let dist_sq = dz_dy_sq + (dx * dx) as f32;

                        // Sphere check using distance squared (no sqrt!)
                        if dist_sq > max_radius_sq {
                            continue;
                        }

                        let idx = ((vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                  + vy as u32 * VOXEL_VOLUME_SIZE
                                  + vx as u32) * 4) as usize;

                        let opacity = self.voxel_cache[idx];
                        // Skip solid voxels (unless very close)
                        if opacity > 200 && dist_sq > 2.0 {
                            continue;
                        }

                        // Compute actual distance only for voxels that pass all filters
                        // (much faster than computing sqrt for all voxels)
                        let dist = dist_sq.sqrt();

                        // Smooth falloff: (1 - dist/max)^2 for softer edges
                        let falloff = 1.0 - dist / max_radius_f;
                        let light_value = (emission_f * falloff * falloff * 255.0) as u8;

                        // Add to existing light (max blend)
                        let current = self.voxel_cache[idx + 3];
                        self.voxel_cache[idx + 3] = current.max(light_value);
                    }
                }
            }
        }
    }

    /// Get the bind group layout for use in render pipelines
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group for binding in render passes
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Incrementally update only the dirty blocks (FAST PATH)
    /// Instead of refilling entire 256³ volume, only updates affected voxels
    /// and re-propagates light in a small radius around each changed block.
    fn update_dirty_blocks(
        &mut self,
        queue: &wgpu::Queue,
        dirty_blocks: &[DirtyBlock],
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
    ) {
        let half_size = (VOXEL_VOLUME_SIZE / 2) as f32;
        let volume_min = self.last_update_center - Vec3::splat(half_size);
        let vol_min_x = volume_min.x as i32;
        let vol_min_y = volume_min.y as i32;
        let vol_min_z = volume_min.z as i32;
        let vol_size = VOXEL_VOLUME_SIZE as i32;

        let light_radius = 12i32; // Radius for light re-propagation around changed blocks

        // Collect regions that need GPU upload (bounding box of all changes)
        let mut min_vx = vol_size;
        let mut min_vy = vol_size;
        let mut min_vz = vol_size;
        let mut max_vx = 0i32;
        let mut max_vy = 0i32;
        let mut max_vz = 0i32;

        let chunks_read = chunks.read();

        for dirty in dirty_blocks {
            // Convert world position to voxel coordinates
            let vx = dirty.x - vol_min_x;
            let vy = dirty.y - vol_min_y;
            let vz = dirty.z - vol_min_z;

            // Skip if outside volume
            if vx < 0 || vx >= vol_size || vy < 0 || vy >= vol_size || vz < 0 || vz >= vol_size {
                continue;
            }

            // Look up the current block state from chunks
            let chunk_x = dirty.x.div_euclid(CHUNK_SIZE_I32);
            let chunk_y = dirty.y.div_euclid(CHUNK_SIZE_I32);
            let chunk_z = dirty.z.div_euclid(CHUNK_SIZE_I32);
            let local_x = dirty.x.rem_euclid(CHUNK_SIZE_I32) as usize;
            let local_y = dirty.y.rem_euclid(CHUNK_SIZE_I32) as usize;
            let local_z = dirty.z.rem_euclid(CHUNK_SIZE_I32) as usize;

            let chunk_pos = ChunkPos { x: chunk_x, y: chunk_y, z: chunk_z };
            let block = chunks_read
                .get(&chunk_pos)
                .map(|c| c.get_block(local_x, local_y, local_z))
                .unwrap_or(Block::Air);

            let voxel = VoxelData::from_block(block);

            // Update the voxel in cache
            let idx = ((vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                      + vy as u32 * VOXEL_VOLUME_SIZE
                      + vx as u32) * 4) as usize;

            self.voxel_cache[idx] = voxel.opacity;
            self.voxel_cache[idx + 1] = voxel.emissive;
            self.voxel_cache[idx + 2] = voxel.material;
            // Note: idx + 3 (light) will be updated in re-propagation below

            // Clear light in the affected radius (so we can re-propagate)
            let clear_min_x = (vx - light_radius).max(0);
            let clear_max_x = (vx + light_radius).min(vol_size - 1);
            let clear_min_y = (vy - light_radius).max(0);
            let clear_max_y = (vy + light_radius).min(vol_size - 1);
            let clear_min_z = (vz - light_radius).max(0);
            let clear_max_z = (vz + light_radius).min(vol_size - 1);

            for cz in clear_min_z..=clear_max_z {
                for cy in clear_min_y..=clear_max_y {
                    for cx in clear_min_x..=clear_max_x {
                        let cidx = ((cz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                   + cy as u32 * VOXEL_VOLUME_SIZE
                                   + cx as u32) * 4 + 3) as usize;
                        self.voxel_cache[cidx] = 0; // Clear light
                    }
                }
            }

            // Track bounding box for upload (expanded by light radius)
            min_vx = min_vx.min(clear_min_x);
            min_vy = min_vy.min(clear_min_y);
            min_vz = min_vz.min(clear_min_z);
            max_vx = max_vx.max(clear_max_x);
            max_vy = max_vy.max(clear_max_y);
            max_vz = max_vz.max(clear_max_z);
        }

        drop(chunks_read);

        // Early exit if nothing to update
        if min_vx > max_vx {
            return;
        }

        // Collect light sources in the affected region (need to re-propagate)
        let mut light_sources: Vec<(i32, i32, i32, u8)> = Vec::new();

        // Scan the affected region for emissive blocks
        let scan_margin = 2; // Small margin to catch lights just outside the cleared area
        let scan_min_x = (min_vx - scan_margin).max(0);
        let scan_max_x = (max_vx + scan_margin).min(vol_size - 1);
        let scan_min_y = (min_vy - scan_margin).max(0);
        let scan_max_y = (max_vy + scan_margin).min(vol_size - 1);
        let scan_min_z = (min_vz - scan_margin).max(0);
        let scan_max_z = (max_vz + scan_margin).min(vol_size - 1);

        for vz in scan_min_z..=scan_max_z {
            for vy in scan_min_y..=scan_max_y {
                for vx in scan_min_x..=scan_max_x {
                    let idx = ((vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                              + vy as u32 * VOXEL_VOLUME_SIZE
                              + vx as u32) * 4) as usize;
                    let emissive = self.voxel_cache[idx + 1];
                    if emissive > 0 {
                        light_sources.push((vx, vy, vz, emissive));
                    }
                }
            }
        }

        // Re-propagate light from sources in the affected region
        self.propagate_light_in_region(&light_sources, min_vx, min_vy, min_vz, max_vx, max_vy, max_vz);

        // Upload only the affected region to GPU (MUCH faster than full upload)
        let region_width = (max_vx - min_vx + 1) as u32;
        let region_height = (max_vy - min_vy + 1) as u32;
        let region_depth = (max_vz - min_vz + 1) as u32;

        // For small regions, upload the sub-region directly
        // For larger regions, it's more efficient to upload the full texture
        let region_voxels = region_width * region_height * region_depth;
        let full_voxels = VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE;

        if region_voxels < full_voxels / 8 {
            // Small region: extract and upload just that part
            // This avoids the 64MB full upload
            let mut region_data = Vec::with_capacity((region_width * region_height * region_depth * 4) as usize);

            for z in min_vz..=max_vz {
                for y in min_vy..=max_vy {
                    let start_idx = ((z as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                    + y as u32 * VOXEL_VOLUME_SIZE
                                    + min_vx as u32) * 4) as usize;
                    let end_idx = start_idx + (region_width * 4) as usize;
                    region_data.extend_from_slice(&self.voxel_cache[start_idx..end_idx]);
                }
            }

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.voxel_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: min_vx as u32,
                        y: min_vy as u32,
                        z: min_vz as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &region_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(region_width * 4),
                    rows_per_image: Some(region_height),
                },
                wgpu::Extent3d {
                    width: region_width,
                    height: region_height,
                    depth_or_array_layers: region_depth,
                },
            );
        } else {
            // Large region: full upload is more efficient
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.voxel_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.voxel_cache,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(VOXEL_VOLUME_SIZE * 4),
                    rows_per_image: Some(VOXEL_VOLUME_SIZE),
                },
                wgpu::Extent3d {
                    width: VOXEL_VOLUME_SIZE,
                    height: VOXEL_VOLUME_SIZE,
                    depth_or_array_layers: VOXEL_VOLUME_SIZE,
                },
            );
        }
    }

    /// Propagate light within a specific region (for incremental updates)
    fn propagate_light_in_region(
        &mut self,
        light_sources: &[(i32, i32, i32, u8)],
        min_x: i32, min_y: i32, min_z: i32,
        max_x: i32, max_y: i32, max_z: i32,
    ) {
        let size = VOXEL_VOLUME_SIZE as i32;
        let max_light_radius = 10i32;
        let max_radius_sq = (max_light_radius * max_light_radius) as f32;
        let max_radius_f = max_light_radius as f32;

        for &(lx, ly, lz, emission) in light_sources {
            let emission_f = emission as f32 / 255.0;

            // Clamp to both light radius and region bounds
            let z_min = (lz - max_light_radius).max(0).max(min_z);
            let z_max = (lz + max_light_radius).min(size - 1).min(max_z);
            let y_min = (ly - max_light_radius).max(0).max(min_y);
            let y_max = (ly + max_light_radius).min(size - 1).min(max_y);
            let x_min = (lx - max_light_radius).max(0).max(min_x);
            let x_max = (lx + max_light_radius).min(size - 1).min(max_x);

            for vz in z_min..=z_max {
                let dz = vz - lz;
                let dz_sq = dz * dz;
                if dz_sq as f32 > max_radius_sq { continue; }

                for vy in y_min..=y_max {
                    let dy = vy - ly;
                    let dz_dy_sq = (dz_sq + dy * dy) as f32;
                    if dz_dy_sq > max_radius_sq { continue; }

                    for vx in x_min..=x_max {
                        let dx = vx - lx;
                        let dist_sq = dz_dy_sq + (dx * dx) as f32;
                        if dist_sq > max_radius_sq { continue; }

                        let idx = ((vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                  + vy as u32 * VOXEL_VOLUME_SIZE
                                  + vx as u32) * 4) as usize;

                        let opacity = self.voxel_cache[idx];
                        if opacity > 200 && dist_sq > 2.0 { continue; }

                        let dist = dist_sq.sqrt();
                        let falloff = 1.0 - dist / max_radius_f;
                        let light_value = (emission_f * falloff * falloff * 255.0) as u8;

                        let current = self.voxel_cache[idx + 3];
                        self.voxel_cache[idx + 3] = current.max(light_value);
                    }
                }
            }
        }
    }
}

// Compute shader for updating voxel volume (placeholder - actual update done via CPU for now)
// Note: workgroup size 4x4x4=64 to stay within GPU limits (max 256 invocations on some GPUs)
const VOXEL_UPDATE_SHADER: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Placeholder - voxel update currently done on CPU
    // Future optimization: use compute shader for incremental updates
}
"#;

// Compute shader for GPU-based light propagation (flood fill)
const LIGHT_PROPAGATE_SHADER: &str = r#"
// GPU-based light propagation using wavefront approach
// Each iteration propagates light one step further

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Placeholder for GPU flood fill
    // This will replace the CPU BFS when fully implemented
}
"#;

// Compute shader for generating voxel mipmaps
const VOXEL_MIPMAP_SHADER: &str = r#"
// Generate mipmaps for cone tracing
// Each mip level averages 2x2x2 voxels from the previous level

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Placeholder for mipmap generation
    // Will be needed for cone tracing
}
"#;
