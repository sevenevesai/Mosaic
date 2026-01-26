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
    /// Additional padding to reach 128 bytes
    pub _padding: [f32; 4],
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
            _padding: [0.0; 4],
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

    /// Staging buffer for CPU -> GPU voxel data transfer
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
    ) {
        // Snap camera position to voxel grid to prevent jitter
        let snapped_center = Vec3::new(
            camera_pos.x.floor(),
            camera_pos.y.floor(),
            camera_pos.z.floor(),
        );

        // Only update voxel data if camera moved more than threshold, first time, or forced
        let move_threshold = 8.0;
        let distance_moved = (snapped_center - self.last_update_center).length();
        let needs_refresh = !self.initialized || distance_moved >= move_threshold || self.force_update;

        if !needs_refresh {
            // IMPORTANT: Use last_update_center (where data IS), not snapped_center (where camera IS)
            // Otherwise shader thinks volume is at wrong position
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day);
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

        // Update uniform buffer
        self.update_uniform(queue, snapped_center, sun_direction, sun_color, ambient_color, time_of_day);

        // Fill voxel cache from chunks
        self.fill_voxel_cache(chunks);

        // Upload to GPU
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

    fn update_uniform(
        &self,
        queue: &wgpu::Queue,
        center: Vec3,
        sun_direction: Vec3,
        sun_color: Vec3,
        ambient_color: Vec3,
        time_of_day: f32,
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
            _padding: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn fill_voxel_cache(&mut self, chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>) {
        let half_size = (VOXEL_VOLUME_SIZE / 2) as f32;
        let volume_min = self.volume_center - Vec3::splat(half_size);
        let volume_max = self.volume_center + Vec3::splat(half_size);

        // Clear cache (all air)
        self.voxel_cache.fill(0);

        // Collect light sources for propagation
        let mut light_sources: Vec<(i32, i32, i32, u8)> = Vec::new();

        let chunks_read = chunks.read();

        // OPTIMIZED: Iterate over LOADED CHUNKS only, not all voxels
        // This is O(loaded_chunks * 4096) instead of O(16 million)
        for (chunk_pos, chunk) in chunks_read.iter() {
            // Chunk world bounds
            let chunk_world_min = Vec3::new(
                (chunk_pos.x * CHUNK_SIZE_I32) as f32,
                (chunk_pos.y * CHUNK_SIZE_I32) as f32,
                (chunk_pos.z * CHUNK_SIZE_I32) as f32,
            );
            let chunk_world_max = chunk_world_min + Vec3::splat(16.0);

            // Skip chunks completely outside the volume
            if chunk_world_max.x < volume_min.x || chunk_world_min.x > volume_max.x ||
               chunk_world_max.y < volume_min.y || chunk_world_min.y > volume_max.y ||
               chunk_world_max.z < volume_min.z || chunk_world_min.z > volume_max.z {
                continue;
            }

            // Iterate over blocks in this chunk
            for lz in 0..16 {
                for ly in 0..16 {
                    for lx in 0..16 {
                        let block = chunk.get_block(lx, ly, lz);

                        // Skip air blocks (most common case)
                        if !block.is_solid() && block.light_emission() == 0 {
                            continue;
                        }

                        // World position of this block
                        let world_x = chunk_pos.x * CHUNK_SIZE_I32 + lx as i32;
                        let world_y = chunk_pos.y * CHUNK_SIZE_I32 + ly as i32;
                        let world_z = chunk_pos.z * CHUNK_SIZE_I32 + lz as i32;

                        // Convert to voxel coordinates
                        let vx = (world_x as f32 - volume_min.x) as i32;
                        let vy = (world_y as f32 - volume_min.y) as i32;
                        let vz = (world_z as f32 - volume_min.z) as i32;

                        // Skip if outside volume bounds
                        if vx < 0 || vx >= VOXEL_VOLUME_SIZE as i32 ||
                           vy < 0 || vy >= VOXEL_VOLUME_SIZE as i32 ||
                           vz < 0 || vz >= VOXEL_VOLUME_SIZE as i32 {
                            continue;
                        }

                        let voxel = VoxelData::from_block(block);

                        // Calculate index in cache (RGBA8 format)
                        let idx = ((vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                  + vy as u32 * VOXEL_VOLUME_SIZE
                                  + vx as u32) * 4) as usize;

                        self.voxel_cache[idx] = voxel.opacity;
                        self.voxel_cache[idx + 1] = voxel.emissive;
                        self.voxel_cache[idx + 2] = voxel.material;
                        self.voxel_cache[idx + 3] = voxel.flags;

                        // Track light sources for propagation
                        if voxel.emissive > 0 {
                            light_sources.push((vx, vy, vz, voxel.emissive));
                        }
                    }
                }
            }
        }

        drop(chunks_read);

        // Propagate light from sources to nearby voxels
        // Store propagated light in the A channel (flags -> light)
        self.propagate_light(&light_sources);
    }

    /// Propagate light from emissive sources through the volume
    /// Stores result in the A channel (index + 3) as propagated light intensity
    fn propagate_light(&mut self, light_sources: &[(i32, i32, i32, u8)]) {
        let size = VOXEL_VOLUME_SIZE as i32;
        let max_light_radius = 14i32;  // Torch light radius

        for &(lx, ly, lz, emission) in light_sources {
            let emission_f = emission as f32 / 255.0;

            // Spread light in a sphere around the source
            for dz in -max_light_radius..=max_light_radius {
                for dy in -max_light_radius..=max_light_radius {
                    for dx in -max_light_radius..=max_light_radius {
                        let vx = lx + dx;
                        let vy = ly + dy;
                        let vz = lz + dz;

                        // Bounds check
                        if vx < 0 || vx >= size || vy < 0 || vy >= size || vz < 0 || vz >= size {
                            continue;
                        }

                        // Calculate distance
                        let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                        let dist = dist_sq.sqrt();

                        if dist > max_light_radius as f32 {
                            continue;
                        }

                        // Check if blocked by solid voxel (simple occlusion)
                        // For performance, only check direct path for close voxels
                        let idx = ((vz as u32 * VOXEL_VOLUME_SIZE * VOXEL_VOLUME_SIZE
                                  + vy as u32 * VOXEL_VOLUME_SIZE
                                  + vx as u32) * 4) as usize;

                        let opacity = self.voxel_cache[idx];
                        if opacity > 200 && dist > 1.0 {
                            // This voxel is solid, don't light it (unless adjacent)
                            continue;
                        }

                        // Smooth falloff: (1 - dist/max)^2 for softer edges
                        let falloff = (1.0 - dist / max_light_radius as f32).max(0.0);
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
