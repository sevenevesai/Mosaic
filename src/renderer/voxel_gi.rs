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
    /// Time of day for lighting calculations (vec4 for alignment)
    pub time_of_day: [f32; 4],  // x = time, yzw = padding
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
    /// Force update on next frame (set when blocks change)
    force_update: bool,

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
            force_update: false,
            staging_buffer,
            voxel_cache: vec![0u8; staging_size],
        }
    }

    /// Mark the voxel volume as needing an update (call when blocks change)
    pub fn mark_dirty(&mut self) {
        self.force_update = true;
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
    /// Called each frame when camera moves significantly
    pub fn update_from_chunks(
        &mut self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        sun_direction: Vec3,
        sun_color: Vec3,
        ambient_color: Vec3,
        time_of_day: f32,
        held_light_emission: u8,  // Light emission of held item (0 = none, 14 = torch)
    ) {
        // Snap camera position to voxel grid to prevent jitter
        let snapped_center = Vec3::new(
            camera_pos.x.floor(),
            camera_pos.y.floor(),
            camera_pos.z.floor(),
        );

        // Only update voxel data if camera moved more than threshold, first time, or forced
        // Threshold of 24 blocks reduces update frequency while maintaining coverage
        // (force_update still triggers immediately for block placement)
        let move_threshold = 24.0;
        let distance_moved = (snapped_center - self.last_update_center).length();
        let needs_refresh = !self.initialized || distance_moved >= move_threshold || self.force_update;

        if !needs_refresh {
            // IMPORTANT: Use last_update_center (where data IS), not snapped_center (where camera IS)
            // Otherwise shader thinks volume is at wrong position
            // But always update player light position (camera_pos) for held torch
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day, camera_pos, held_light_emission);
            return;
        }

        // Reset force flag
        self.force_update = false;

        if !self.initialized {
            tracing::info!("Initial voxel volume fill at {:?}", snapped_center);
            self.initialized = true;
        } else {
            tracing::debug!("Updating voxel volume: moved {:.1} blocks", distance_moved);
        }

        self.volume_center = snapped_center;
        self.last_update_center = snapped_center;

        // Update uniform buffer (includes player held light position)
        self.update_uniform(queue, snapped_center, sun_direction, sun_color, ambient_color, time_of_day, camera_pos, held_light_emission);

        // Time the voxel cache fill (includes light propagation)
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
            time_of_day: [time_of_day, 0.0, 0.0, 0.0],
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
