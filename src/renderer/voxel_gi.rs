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
    /// Volume origin for toroidal wrapping (world position of voxel [0,0,0])
    pub volume_origin: [i32; 4],
    /// Player shadow box: [feet_x, feet_y, feet_z, half_width]
    /// Analytical box for smooth player shadow (no grid snapping)
    /// Box spans: center=(x, y+height/2, z), half_extents=(w, height/2, w)
    pub player_shadow_box: [f32; 4],
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
            volume_origin: [0; 4],   // Toroidal origin
            player_shadow_box: [0.0; 4], // No player shadow by default
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

/// Maximum chunks that can overlap the 256^3 voxel volume
/// Volume spans 16 chunks per axis, but can straddle boundaries → 17 max
/// Maximum chunk slots in the GPU buffer for delta uploads
/// Need headroom beyond the ~2048 visible chunks to handle churn during movement
/// Extra slots allow old chunks to coexist briefly with new chunks during transitions
const MAX_OVERLAPPING_CHUNKS: u32 = 3072; // 2048 visible + 1024 headroom for delta churn

/// GPU-side chunk metadata for voxel filling
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuChunkMetadata {
    /// World origin of chunk (chunk_pos * 16)
    world_origin: [i32; 3],
    /// Index into block data buffer (chunk_index * 4096)
    chunk_index: u32,
}

/// Uniforms for voxel fill compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VoxelFillUniforms {
    /// Center of the voxel volume in world coordinates
    volume_center: [i32; 4],  // w unused, for alignment
    /// Minimum chunk position (for lookup texture indexing)
    min_chunk_pos: [i32; 4],  // w = num_chunks
}

/// GPU Clear pipeline resources
#[allow(dead_code)]
struct VoxelClearPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

/// GPU Voxel Fill pipeline resources
#[allow(dead_code)]
struct VoxelFillPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    chunk_metadata_buffer: wgpu::Buffer,
    chunk_blocks_buffer: wgpu::Buffer,
    chunk_lookup_texture: wgpu::Texture,
    chunk_lookup_view: wgpu::TextureView,
    bind_group: Option<wgpu::BindGroup>,
}

/// GPU Light Propagation pipeline resources (jump flooding algorithm)
struct LightPropagatePipeline {
    /// Init pass: seed emissive voxels into A channel
    init_pipeline: wgpu::ComputePipeline,
    /// Propagate pass: jump flooding iterations
    propagate_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for both passes
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for step_size parameter
    uniform_buffer: wgpu::Buffer,
}

/// Uniforms for light propagation compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LightPropagateUniforms {
    /// Jump step size for current pass (128, 64, 32, 16, 8, 4, 2, 1)
    step_size: i32,
    /// Maximum light propagation radius (14 blocks for torches)
    max_light_radius: f32,
    /// Padding for alignment
    _padding: [f32; 2],
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

    /// Current volume center (world coordinates) - where voxel data is actually centered
    volume_center: Vec3,
    /// Last update position (to detect when re-upload needed)
    /// Starts at origin, will update on first frame
    last_update_center: Vec3,
    /// Smoothed center that gradually follows the camera (for smooth shadow transitions)
    /// This is where we WANT the volume to be - volume_center catches up to this
    smoothed_center: Vec3,
    /// How fast the smoothed center follows the camera (0.0 = never, 1.0 = instant)
    follow_speed: f32,
    /// Distance threshold before triggering a volume update (smaller = more frequent, smoother)
    update_threshold: f32,
    /// Volume origin for toroidal wrapping (world position of voxel [0,0,0])
    /// This is volume_center - 128, snapped to chunk boundaries for efficiency
    #[allow(dead_code)]
    volume_origin: glam::IVec3,
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

    /// GPU compute pipeline for clearing voxel volume
    clear_pipeline: VoxelClearPipeline,
    /// GPU compute pipeline for filling voxel volume from chunks
    fill_pipeline: VoxelFillPipeline,
    /// GPU compute pipeline for light propagation (jump flooding)
    light_propagate_pipeline: LightPropagatePipeline,
    /// Whether to use GPU fill (true) or CPU fill (false) - for A/B testing
    use_gpu_fill: bool,
    /// Shadow blend factor for smooth fade-in after volume updates (0.0 = no shadow, 1.0 = full shadow)
    shadow_blend: f32,
    /// How fast shadows fade in after an update (higher = faster)
    shadow_fade_speed: f32,

    // === Chunk Delta Upload System ===
    // Tracks which chunks are already uploaded to GPU to avoid redundant uploads

    /// Maps ChunkPos to slot index in chunk_blocks_buffer (for delta uploads)
    uploaded_chunks: AHashMap<ChunkPos, u32>,
    /// Free slot indices available for reuse when chunks leave the volume
    free_slots: Vec<u32>,
    /// Next slot index to allocate if free_slots is empty
    next_slot: u32,
    /// Last volume center used for chunk tracking (integer, for stability)
    last_chunk_volume_center: glam::IVec3,
    /// Whether delta upload mode is active (requires stable volume tracking)
    delta_upload_enabled: bool,

}

impl VoxelGI {
    pub fn new(device: &wgpu::Device) -> Self {
        // Create 3D voxel texture (RGBA8)
        // Note: view_formats includes Rgba8Unorm to ensure compatibility with storage texture use
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
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
        // Create view with explicit format for storage texture compatibility
        let voxel_texture_view = voxel_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Voxel Volume View"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D3),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: None,
        });

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

        // Create compute pipelines (placeholders)
        let update_pipeline = Self::create_update_pipeline(device, &bind_group_layout);
        let propagate_pipeline = Self::create_propagate_pipeline(device, &bind_group_layout);
        let mipmap_pipeline = Self::create_mipmap_pipeline(device, &bind_group_layout);

        // Create GPU clear and fill pipelines
        let clear_pipeline = Self::create_clear_pipeline(device, &voxel_texture_view);
        let fill_pipeline = Self::create_fill_pipeline(device, &voxel_texture_view);
        let light_propagate_pipeline = Self::create_light_propagate_pipeline(device, &voxel_texture_view);

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
            smoothed_center: Vec3::ZERO,
            follow_speed: 0.08,        // Smooth following - adjust for feel (0.05-0.15 range)
            update_threshold: 8.0,     // Trigger update every ~8 blocks (less frequent for better perf)
            volume_origin: glam::IVec3::ZERO,
            initialized: false,
            force_full_update: false,
            dirty_blocks: Vec::with_capacity(MAX_DIRTY_BLOCKS),
            staging_buffer,
            voxel_cache: vec![0u8; staging_size],
            clear_pipeline,
            fill_pipeline,
            light_propagate_pipeline,
            use_gpu_fill: true, // Enable GPU fill by default
            shadow_blend: 1.0,  // Start fully blended
            shadow_fade_speed: 4.0, // Fade in over ~0.25 seconds
            // Chunk delta upload system
            uploaded_chunks: AHashMap::with_capacity(MAX_OVERLAPPING_CHUNKS as usize),
            free_slots: Vec::with_capacity(256), // Pre-allocate for typical churn
            next_slot: 0,
            last_chunk_volume_center: glam::IVec3::ZERO,
            delta_upload_enabled: true, // Enable by default
        }
    }

    /// Mark the voxel volume as needing a full update
    /// Use mark_dirty_at() for single block changes (much faster)
    pub fn mark_dirty(&mut self) {
        self.force_full_update = true;
    }

    /// Mark a specific block as dirty for incremental update
    /// This is much faster than full update - only updates one voxel + light radius
    /// In GPU mode, this invalidates the chunk in delta tracking so it gets re-uploaded
    pub fn mark_dirty_at(&mut self, x: i32, y: i32, z: i32) {
        // Invalidate the chunk in delta tracking so it gets re-uploaded
        if self.delta_upload_enabled {
            let chunk_pos = ChunkPos {
                x: x.div_euclid(CHUNK_SIZE_I32),
                y: y.div_euclid(CHUNK_SIZE_I32),
                z: z.div_euclid(CHUNK_SIZE_I32),
            };
            self.invalidate_chunk(chunk_pos);
        }

        // In GPU mode, trigger full refresh since incremental GPU updates aren't supported yet
        if self.use_gpu_fill {
            self.force_full_update = true;
            return;
        }

        // If we have too many dirty blocks, switch to full update
        if self.dirty_blocks.len() >= MAX_DIRTY_BLOCKS {
            self.force_full_update = true;
            self.dirty_blocks.clear();
            return;
        }
        self.dirty_blocks.push(DirtyBlock { x, y, z });
    }

    /// Invalidate a chunk in the delta tracking system
    /// The chunk will be re-uploaded on the next frame
    pub fn invalidate_chunk(&mut self, chunk_pos: ChunkPos) {
        if let Some(slot) = self.uploaded_chunks.remove(&chunk_pos) {
            self.free_slots.push(slot);
        }
    }

    /// Invalidate all chunks (e.g., when world is reset)
    pub fn invalidate_all_chunks(&mut self) {
        self.uploaded_chunks.clear();
        self.free_slots.clear();
        self.next_slot = 0;
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
        // Note: This is a legacy placeholder pipeline that is no longer used.
        // The actual light propagation is handled by light_propagate_pipeline.
        let placeholder_shader = r#"
            @compute @workgroup_size(4, 4, 4)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                // Placeholder - not used, see light_propagate_pipeline instead
            }
        "#;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Legacy Propagation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(placeholder_shader.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Legacy Propagation Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Legacy Propagation Pipeline"),
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

    /// Create the GPU clear pipeline
    fn create_clear_pipeline(device: &wgpu::Device, voxel_texture_view: &wgpu::TextureView) -> VoxelClearPipeline {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Voxel Clear Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Clear Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(voxel_texture_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_CLEAR_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Clear Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Voxel Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        VoxelClearPipeline {
            pipeline,
            bind_group_layout,
            bind_group,
        }
    }

    /// Create the GPU voxel fill pipeline
    fn create_fill_pipeline(device: &wgpu::Device, _voxel_texture_view: &wgpu::TextureView) -> VoxelFillPipeline {
        use std::num::NonZeroU64;

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Voxel Fill Bind Group Layout"),
            entries: &[
                // Binding 0: Output voxel volume (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // Binding 1: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<VoxelFillUniforms>() as u64),
                    },
                    count: None,
                },
                // Binding 2: Chunk metadata array (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Chunk block data (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Chunk lookup texture (17x17x17)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Fill Uniforms"),
            size: std::mem::size_of::<VoxelFillUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create chunk metadata buffer (512 chunks * 16 bytes = 8KB)
        let chunk_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Metadata Buffer"),
            size: (MAX_OVERLAPPING_CHUNKS as usize * std::mem::size_of::<GpuChunkMetadata>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create chunk blocks buffer (512 chunks * 4096 bytes = 2MB)
        let chunk_blocks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chunk Blocks Buffer"),
            size: (MAX_OVERLAPPING_CHUNKS as usize * 4096) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create 17x17x17 chunk lookup texture (R32Uint)
        let chunk_lookup_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Chunk Lookup Texture"),
            size: wgpu::Extent3d {
                width: 17,
                height: 17,
                depth_or_array_layers: 17,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let chunk_lookup_view = chunk_lookup_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create shader and pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Fill Shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_FILL_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Fill Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Voxel Fill Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        VoxelFillPipeline {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            chunk_metadata_buffer,
            chunk_blocks_buffer,
            chunk_lookup_texture,
            chunk_lookup_view,
            bind_group: None, // Created when we have all resources bound
        }
    }

    /// Create the GPU light propagation pipeline (jump flooding algorithm)
    fn create_light_propagate_pipeline(device: &wgpu::Device, voxel_texture_view: &wgpu::TextureView) -> LightPropagatePipeline {
        use std::num::NonZeroU64;

        // Create bind group layout - needs read-write access to voxel texture
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Propagate Bind Group Layout"),
            entries: &[
                // Binding 0: Voxel volume (read-write for propagation)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // Binding 1: Uniforms (step_size, max_radius)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<LightPropagateUniforms>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Propagate Uniforms"),
            size: std::mem::size_of::<LightPropagateUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Propagate Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create init shader (seeds emissive voxels)
        let init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Init Shader"),
            source: wgpu::ShaderSource::Wgsl(LIGHT_INIT_SHADER.into()),
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Light Init Pipeline"),
            layout: Some(&pipeline_layout),
            module: &init_shader,
            entry_point: "main",
        });

        // Create propagate shader (jump flooding)
        let propagate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Propagate Shader"),
            source: wgpu::ShaderSource::Wgsl(LIGHT_PROPAGATE_SHADER.into()),
        });

        let propagate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Light Propagate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &propagate_shader,
            entry_point: "main",
        });

        // Note: bind_group is created at dispatch time since we need the texture view
        let _ = voxel_texture_view; // Will use in dispatch

        LightPropagatePipeline {
            init_pipeline,
            propagate_pipeline,
            bind_group_layout,
            uniform_buffer,
        }
    }

    /// Dispatch light propagation compute passes after voxel fill
    /// Uses jump flooding algorithm for O(log N) convergence
    pub fn dispatch_light_propagation(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) {
        let propagate_start = std::time::Instant::now();

        // Create bind group with current texture view
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Propagate Bind Group"),
            layout: &self.light_propagate_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.voxel_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_propagate_pipeline.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Pass 1: Initialize A channel from emissive (G channel)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Light Init Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.light_propagate_pipeline.init_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(64, 64, 64); // 256/4 = 64
        }

        // Passes 2-9: Jump flooding with decreasing step sizes
        // Each pass propagates light one "jump" further
        let step_sizes = [64, 32, 16, 8, 4, 2, 1];
        for &step in &step_sizes {
            // Update uniform with current step size
            let uniform = LightPropagateUniforms {
                step_size: step,
                max_light_radius: 14.0, // Torch light radius
                _padding: [0.0; 2],
            };
            queue.write_buffer(
                &self.light_propagate_pipeline.uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniform]),
            );

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Light Propagate Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.light_propagate_pipeline.propagate_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(64, 64, 64);
        }

        let propagate_time = propagate_start.elapsed();
        tracing::debug!("Light propagation: {:.2}ms ({} passes)", propagate_time.as_secs_f64() * 1000.0, step_sizes.len() + 1);
    }

    /// Update the voxel volume from chunk data
    /// Supports three modes:
    /// - Full update: When camera moves significantly or on initialization
    /// - Scroll update: When camera moves moderately (only update new slabs)
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
        torch_shadows: bool,
        fog_distance: f32,
        shadow_quality: u32,
        player_feet_pos: Vec3,
    ) {
        // Use smoothed_center for gradual transitions (already tracks camera smoothly)
        let snapped_center = Vec3::new(
            self.smoothed_center.x.floor(),
            self.smoothed_center.y.floor(),
            self.smoothed_center.z.floor(),
        );

        // Check if we need a full refresh - use smoothed center distance and configurable threshold
        // Smaller threshold = more frequent but smaller updates = smoother transitions
        let distance_moved = (snapped_center - self.last_update_center).length();
        let needs_full_refresh = !self.initialized || distance_moved >= self.update_threshold || self.force_full_update;

        // Check for incremental updates (dirty blocks)
        let has_dirty_blocks = !self.dirty_blocks.is_empty();

        // If no updates needed, just update uniforms
        if !needs_full_refresh && !has_dirty_blocks {
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission, torch_shadows, fog_distance, shadow_quality, player_feet_pos);
            return;
        }

        // Handle INCREMENTAL updates (dirty blocks) - much faster path
        // Skip incremental updates in GPU mode - CPU cache is not maintained
        if has_dirty_blocks && !needs_full_refresh && !self.use_gpu_fill {
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
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission, torch_shadows, fog_distance, shadow_quality, player_feet_pos);
            return;
        }

        // Handle FULL refresh (camera moved or initialization)
        // Skip CPU fill if GPU fill mode is enabled - GPU handles full refreshes
        if self.use_gpu_fill {
            self.force_full_update = false;
            self.dirty_blocks.clear(); // GPU mode doesn't support incremental updates

            // DON'T overwrite volume_center/last_update_center here!
            // gpu_fill_volume() already set these correctly when it ran.
            // We just need to update the uniform with the values gpu_fill_volume set.

            // Always use the actual volume center for uniforms (where the voxel data is)
            self.update_uniform(queue, self.last_update_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission, torch_shadows, fog_distance, shadow_quality, player_feet_pos);
            return;
        }

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
        self.update_uniform(queue, snapped_center, sun_direction, sun_color, ambient_color, time_of_day, shadow_mode, shadow_debug, camera_pos, held_light_emission, torch_shadows, fog_distance, shadow_quality, player_feet_pos);

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

        let total_ms = fill_time.as_secs_f64() * 1000.0 + upload_time.as_secs_f64() * 1000.0;
        tracing::debug!(
            "CPU voxel fill: fill={:.2}ms, upload={:.2}ms, total={:.2}ms",
            fill_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            total_ms
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
        shadow_mode: f32,
        shadow_debug: f32,
        player_pos: Vec3,
        held_light_emission: u8,
        torch_shadows: bool,
        fog_distance: f32,
        shadow_quality: u32,
        player_feet_pos: Vec3,
    ) {
        // Volume origin is the world position of voxel [0,0,0]
        // For toroidal wrapping: origin = center - half_size
        let half_size = (VOXEL_VOLUME_SIZE / 2) as i32;
        let origin = glam::IVec3::new(
            center.x.floor() as i32 - half_size,
            center.y.floor() as i32 - half_size,
            center.z.floor() as i32 - half_size,
        );

        // Pack volume_origin.w: bit 0 = torch_shadows, bits 1-3 = shadow_quality (0-4)
        let packed_w = (if torch_shadows { 1i32 } else { 0 }) | ((shadow_quality as i32) << 1);

        let uniform = VoxelVolumeUniform {
            volume_center: [center.x, center.y, center.z, fog_distance],
            volume_size: [VOXEL_VOLUME_SIZE as f32; 4],
            inv_volume_size: [1.0 / VOXEL_VOLUME_SIZE as f32; 4],
            // W component is sun intensity (0-1), calculated from sun height
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, sun_direction.y.max(0.0).min(1.0)],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, 1.0],
            ambient_color: [ambient_color.x, ambient_color.y, ambient_color.z, 1.0],
            time_of_day: [time_of_day, shadow_mode, shadow_debug, self.shadow_blend],
            player_light: [player_pos.x, player_pos.y, player_pos.z, held_light_emission as f32 / 15.0],
            // W component: bit 0 = torch_shadows, bits 1-3 = shadow_quality
            volume_origin: [origin.x, origin.y, origin.z, packed_w],
            // Player shadow box: feet position + half-width (0.3 blocks)
            // Shader builds the full box: center = (x, y+0.9, z), half = (0.3, 0.9, 0.3)
            player_shadow_box: [player_feet_pos.x, player_feet_pos.y, player_feet_pos.z, 0.3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Execute GPU voxel fill using compute shaders
    /// This replaces the CPU fill_voxel_cache path for better performance
    /// Uses smoothed_center for gradual shadow transitions
    pub fn gpu_fill_volume(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        _camera_pos: Vec3,  // Ignored - we use smoothed_center instead
    ) {
        use glam::IVec3;

        let fill_start = std::time::Instant::now();

        // Use smoothed center for gradual transitions
        // IMPORTANT: Use .floor() for consistent negative number handling (as i32 truncates toward zero)
        let volume_center = IVec3::new(
            self.smoothed_center.x.floor() as i32,
            self.smoothed_center.y.floor() as i32,
            self.smoothed_center.z.floor() as i32,
        );

        // Upload chunk data to GPU using delta tracking
        let upload_start = std::time::Instant::now();
        let (num_chunks, _min_chunk_pos, chunks_uploaded, chunks_reused) = if self.delta_upload_enabled {
            self.upload_chunk_data_for_gpu_delta(device, queue, chunks, volume_center)
        } else {
            // Fallback to full upload (for comparison/debugging)
            let (n, m) = self.upload_chunk_data_for_gpu(device, queue, chunks, volume_center);
            (n, m, n, 0)
        };
        let upload_time = upload_start.elapsed();

        if num_chunks == 0 {
            tracing::warn!("GPU voxel fill: no chunks to process");
            return;
        }

        // Calculate upload savings
        let bytes_uploaded = chunks_uploaded as usize * 4096;
        let bytes_saved = chunks_reused as usize * 4096;

        // Create bind group with all resources
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Fill Bind Group"),
            layout: &self.fill_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.voxel_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.fill_pipeline.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fill_pipeline.chunk_metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.fill_pipeline.chunk_blocks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.fill_pipeline.chunk_lookup_view),
                },
            ],
        });

        // Clear pass - create bind group at runtime to ensure we're using the current texture
        let clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Clear Bind Group Runtime"),
            layout: &self.clear_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.voxel_texture_view),
                },
            ],
        });

        // SKIP CLEAR PASS - the fill shader writes to every voxel anyway
        // This saves ~0.3ms per update. The clear_bind_group is still created above
        // but not dispatched. Can remove the creation too if this works well.
        // {
        //     let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //         label: Some("Voxel Clear Pass"),
        //         timestamp_writes: None,
        //     });
        //     compute_pass.set_pipeline(&self.clear_pipeline.pipeline);
        //     compute_pass.set_bind_group(0, &clear_bind_group, &[]);
        //     compute_pass.dispatch_workgroups(64, 64, 64);
        // }

        // Fill pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Voxel Fill Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.fill_pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // 256 / 4 = 64 workgroups per dimension
            compute_pass.dispatch_workgroups(64, 64, 64);
        }

        // Light propagation pass - propagate emissive light through the volume
        // Uses jump flooding algorithm for O(log N) convergence
        self.dispatch_light_propagation(device, encoder, queue);

        // Suppress unused variable warning
        let _ = clear_bind_group;

        // Update tracking
        self.volume_center = Vec3::new(volume_center.x as f32, volume_center.y as f32, volume_center.z as f32);
        self.last_update_center = self.volume_center;
        self.initialized = true;
        self.force_full_update = false;
        self.dirty_blocks.clear();

        // Note: shadow_blend no longer reset here - using edge-distance fade instead

        let fill_time = fill_start.elapsed();

        // Log with delta upload statistics
        if self.delta_upload_enabled && chunks_reused > 0 {
            let reuse_pct = (chunks_reused as f32 / num_chunks as f32) * 100.0;
            tracing::debug!(
                "GPU voxel fill: upload={:.2}ms, total={:.2}ms | {} chunks ({} new, {} reused={:.0}%), {}KB uploaded, {}KB saved",
                upload_time.as_secs_f64() * 1000.0,
                fill_time.as_secs_f64() * 1000.0,
                num_chunks,
                chunks_uploaded,
                chunks_reused,
                reuse_pct,
                bytes_uploaded / 1024,
                bytes_saved / 1024,
            );
        } else {
            tracing::debug!(
                "GPU voxel fill: upload={:.2}ms, total={:.2}ms | {} chunks, center=({},{},{})",
                upload_time.as_secs_f64() * 1000.0,
                fill_time.as_secs_f64() * 1000.0,
                num_chunks,
                volume_center.x, volume_center.y, volume_center.z,
            );
        }
    }

    /// Upload chunk data to GPU buffers for compute shader processing
    /// Returns (num_chunks, min_chunk_pos)
    fn upload_chunk_data_for_gpu(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        volume_center: glam::IVec3,
    ) -> (u32, glam::IVec3) {
        use glam::IVec3;

        let half_size = 128i32;
        let min_block = volume_center - IVec3::splat(half_size);
        let max_block = volume_center + IVec3::splat(half_size - 1);

        // Calculate chunk range that overlaps the volume
        let min_chunk = IVec3::new(
            min_block.x.div_euclid(CHUNK_SIZE_I32),
            min_block.y.div_euclid(CHUNK_SIZE_I32),
            min_block.z.div_euclid(CHUNK_SIZE_I32),
        );
        let max_chunk = IVec3::new(
            max_block.x.div_euclid(CHUNK_SIZE_I32),
            max_block.y.div_euclid(CHUNK_SIZE_I32),
            max_block.z.div_euclid(CHUNK_SIZE_I32),
        );

        let mut metadata_vec: Vec<GpuChunkMetadata> = Vec::with_capacity(MAX_OVERLAPPING_CHUNKS as usize);
        let mut blocks_vec: Vec<u8> = Vec::with_capacity(MAX_OVERLAPPING_CHUNKS as usize * 4096);

        // 17x17x17 lookup texture data (filled with 0xFFFFFFFF = no chunk)
        let lookup_size = 17 * 17 * 17;
        let mut lookup_data: Vec<u32> = vec![0xFFFFFFFF; lookup_size];

        let chunks_read = chunks.read();
        let mut chunk_index = 0u32;

        // Calculate center chunk position for prioritized iteration
        let center_chunk = IVec3::new(
            volume_center.x.div_euclid(CHUNK_SIZE_I32),
            volume_center.y.div_euclid(CHUNK_SIZE_I32),
            volume_center.z.div_euclid(CHUNK_SIZE_I32),
        );

        // Iterate from center outward in Y to prioritize chunks near the player
        // This ensures we capture the most important chunks even if we hit the limit
        let y_range = max_chunk.y - min_chunk.y + 1;
        let mut y_offsets: Vec<i32> = Vec::with_capacity(y_range as usize);
        for d in 0..=y_range {
            if d == 0 {
                y_offsets.push(0);
            } else {
                // Alternate positive and negative offsets from center
                y_offsets.push(d);
                y_offsets.push(-d);
            }
        }

        'outer: for dy in y_offsets {
            let cy = center_chunk.y + dy;
            if cy < min_chunk.y || cy > max_chunk.y {
                continue;
            }

            for cz in min_chunk.z..=max_chunk.z {
                for cx in min_chunk.x..=max_chunk.x {
                    let pos = ChunkPos { x: cx, y: cy, z: cz };

                    if let Some(chunk) = chunks_read.get(&pos) {
                        // Add metadata
                        let world_origin = [cx * CHUNK_SIZE_I32, cy * CHUNK_SIZE_I32, cz * CHUNK_SIZE_I32];
                        metadata_vec.push(GpuChunkMetadata {
                            world_origin,
                            chunk_index,
                        });

                        // Add block data - single memcpy instead of 4096 iterations
                        // Block is #[repr(u8)] so this is a direct byte copy
                        blocks_vec.extend_from_slice(chunk.blocks_as_bytes());

                        // Update lookup texture
                        let lx = (cx - min_chunk.x) as usize;
                        let ly = (cy - min_chunk.y) as usize;
                        let lz = (cz - min_chunk.z) as usize;
                        if lx < 17 && ly < 17 && lz < 17 {
                            let lookup_idx = lx + ly * 17 + lz * 17 * 17;
                            lookup_data[lookup_idx] = chunk_index;
                        }

                        chunk_index += 1;

                        if chunk_index >= MAX_OVERLAPPING_CHUNKS {
                            break 'outer;
                        }
                    }
                }
            }
        }
        drop(chunks_read);

        let num_chunks = metadata_vec.len() as u32;

        // Debug: find the RANGE of filled positions in lookup texture
        let mut min_x = 17i32;
        let mut max_x = -1i32;
        let mut min_y = 17i32;
        let mut max_y = -1i32;
        let mut min_z = 17i32;
        let mut max_z = -1i32;
        let mut num_filled = 0;

        for z in 0..17 {
            for y in 0..17 {
                for x in 0..17 {
                    let idx = x + y * 17 + z * 17 * 17;
                    if lookup_data[idx] != 0xFFFFFFFF {
                        num_filled += 1;
                        min_x = min_x.min(x as i32);
                        max_x = max_x.max(x as i32);
                        min_y = min_y.min(y as i32);
                        max_y = max_y.max(y as i32);
                        min_z = min_z.min(z as i32);
                        max_z = max_z.max(z as i32);
                    }
                }
            }
        }

        tracing::debug!(
            "Lookup texture: {} filled, X=[{},{}], Y=[{},{}], Z=[{},{}]",
            num_filled, min_x, max_x, min_y, max_y, min_z, max_z
        );

        // Upload uniforms
        let uniforms = VoxelFillUniforms {
            volume_center: [volume_center.x, volume_center.y, volume_center.z, 0],
            min_chunk_pos: [min_chunk.x, min_chunk.y, min_chunk.z, num_chunks as i32],
        };
        queue.write_buffer(
            &self.fill_pipeline.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Upload metadata
        if !metadata_vec.is_empty() {
            queue.write_buffer(
                &self.fill_pipeline.chunk_metadata_buffer,
                0,
                bytemuck::cast_slice(&metadata_vec),
            );
        }

        // Upload block data
        if !blocks_vec.is_empty() {
            queue.write_buffer(
                &self.fill_pipeline.chunk_blocks_buffer,
                0,
                &blocks_vec,
            );
        }

        // Upload lookup texture
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.fill_pipeline.chunk_lookup_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&lookup_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(17 * 4), // 17 pixels * 4 bytes per R32Uint
                rows_per_image: Some(17),
            },
            wgpu::Extent3d {
                width: 17,
                height: 17,
                depth_or_array_layers: 17,
            },
        );

        (num_chunks, min_chunk)
    }

    /// Upload chunk data using delta tracking - only uploads NEW chunks
    /// Returns (num_chunks, min_chunk_pos, chunks_uploaded, chunks_reused)
    fn upload_chunk_data_for_gpu_delta(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        volume_center: glam::IVec3,
    ) -> (u32, glam::IVec3, u32, u32) {
        use glam::IVec3;

        let half_size = 128i32;
        let min_block = volume_center - IVec3::splat(half_size);
        let max_block = volume_center + IVec3::splat(half_size - 1);

        // Calculate chunk range that overlaps the volume
        let min_chunk = IVec3::new(
            min_block.x.div_euclid(CHUNK_SIZE_I32),
            min_block.y.div_euclid(CHUNK_SIZE_I32),
            min_block.z.div_euclid(CHUNK_SIZE_I32),
        );
        let max_chunk = IVec3::new(
            max_block.x.div_euclid(CHUNK_SIZE_I32),
            max_block.y.div_euclid(CHUNK_SIZE_I32),
            max_block.z.div_euclid(CHUNK_SIZE_I32),
        );

        // Build set of chunks needed for current volume
        let mut needed_chunks: AHashMap<ChunkPos, ()> = AHashMap::with_capacity(MAX_OVERLAPPING_CHUNKS as usize);
        for cy in min_chunk.y..=max_chunk.y {
            for cz in min_chunk.z..=max_chunk.z {
                for cx in min_chunk.x..=max_chunk.x {
                    needed_chunks.insert(ChunkPos { x: cx, y: cy, z: cz }, ());
                }
            }
        }

        // Find chunks to remove (uploaded but no longer needed)
        let chunks_to_remove: Vec<ChunkPos> = self.uploaded_chunks
            .keys()
            .filter(|pos| !needed_chunks.contains_key(pos))
            .copied()
            .collect();

        // Free slots from removed chunks
        for pos in &chunks_to_remove {
            if let Some(slot) = self.uploaded_chunks.remove(pos) {
                self.free_slots.push(slot);
            }
        }

        // Defragment if we're getting close to the limit but have free slots
        // This resets next_slot to allow reuse of freed slots more efficiently
        if self.next_slot > MAX_OVERLAPPING_CHUNKS / 2 && self.free_slots.len() > 256 {
            // Sort free_slots to find the highest slot we can safely reset to
            self.free_slots.sort_unstable();
            // Reset next_slot to just past the highest used slot
            let max_used = self.uploaded_chunks.values().max().copied().unwrap_or(0);
            self.next_slot = max_used + 1;
            tracing::debug!(
                "Delta upload defrag: reset next_slot to {}, {} free slots",
                self.next_slot, self.free_slots.len()
            );
        }

        // Prepare metadata and lookup data
        let mut metadata_vec: Vec<GpuChunkMetadata> = Vec::with_capacity(MAX_OVERLAPPING_CHUNKS as usize);
        let lookup_size = 17 * 17 * 17;
        let mut lookup_data: Vec<u32> = vec![0xFFFFFFFF; lookup_size];

        let chunks_read = chunks.read();
        let mut chunks_uploaded = 0u32;
        let mut chunks_reused = 0u32;
        let mut total_chunks = 0u32;

        // Calculate center chunk for prioritized iteration
        let center_chunk = IVec3::new(
            volume_center.x.div_euclid(CHUNK_SIZE_I32),
            volume_center.y.div_euclid(CHUNK_SIZE_I32),
            volume_center.z.div_euclid(CHUNK_SIZE_I32),
        );

        // Iterate from center outward in Y
        let y_range = max_chunk.y - min_chunk.y + 1;
        let mut y_offsets: Vec<i32> = Vec::with_capacity(y_range as usize);
        for d in 0..=y_range {
            if d == 0 {
                y_offsets.push(0);
            } else {
                y_offsets.push(d);
                y_offsets.push(-d);
            }
        }

        'outer: for dy in y_offsets {
            let cy = center_chunk.y + dy;
            if cy < min_chunk.y || cy > max_chunk.y {
                continue;
            }

            for cz in min_chunk.z..=max_chunk.z {
                for cx in min_chunk.x..=max_chunk.x {
                    let pos = ChunkPos { x: cx, y: cy, z: cz };

                    if let Some(chunk) = chunks_read.get(&pos) {
                        // Check if chunk already has a slot
                        let slot = if let Some(&existing_slot) = self.uploaded_chunks.get(&pos) {
                            // Reuse existing slot - no upload needed
                            chunks_reused += 1;
                            existing_slot
                        } else {
                            // Need to allocate a new slot and upload
                            let new_slot = if let Some(recycled) = self.free_slots.pop() {
                                recycled
                            } else {
                                let s = self.next_slot;
                                self.next_slot += 1;
                                if self.next_slot >= MAX_OVERLAPPING_CHUNKS {
                                    // Buffer full - fall back to full upload
                                    tracing::warn!("Chunk slot buffer full, resetting delta tracking");
                                    self.uploaded_chunks.clear();
                                    self.free_slots.clear();
                                    self.next_slot = 0;
                                    drop(chunks_read);
                                    return self.upload_chunk_data_for_gpu_full_reset(queue, chunks, volume_center);
                                }
                                s
                            };

                            // Upload this chunk's block data at the specific slot offset
                            let offset = new_slot as u64 * 4096;
                            queue.write_buffer(
                                &self.fill_pipeline.chunk_blocks_buffer,
                                offset,
                                chunk.blocks_as_bytes(),
                            );

                            self.uploaded_chunks.insert(pos, new_slot);
                            chunks_uploaded += 1;
                            new_slot
                        };

                        // Add metadata (always needed for current frame)
                        let world_origin = [cx * CHUNK_SIZE_I32, cy * CHUNK_SIZE_I32, cz * CHUNK_SIZE_I32];
                        metadata_vec.push(GpuChunkMetadata {
                            world_origin,
                            chunk_index: slot,
                        });

                        // Update lookup texture
                        let lx = (cx - min_chunk.x) as usize;
                        let ly = (cy - min_chunk.y) as usize;
                        let lz = (cz - min_chunk.z) as usize;
                        if lx < 17 && ly < 17 && lz < 17 {
                            let lookup_idx = lx + ly * 17 + lz * 17 * 17;
                            lookup_data[lookup_idx] = slot;
                        }

                        total_chunks += 1;

                        if total_chunks >= MAX_OVERLAPPING_CHUNKS {
                            break 'outer;
                        }
                    }
                }
            }
        }
        drop(chunks_read);

        // Upload uniforms
        let uniforms = VoxelFillUniforms {
            volume_center: [volume_center.x, volume_center.y, volume_center.z, 0],
            min_chunk_pos: [min_chunk.x, min_chunk.y, min_chunk.z, total_chunks as i32],
        };
        queue.write_buffer(
            &self.fill_pipeline.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Upload metadata (small - always upload)
        if !metadata_vec.is_empty() {
            queue.write_buffer(
                &self.fill_pipeline.chunk_metadata_buffer,
                0,
                bytemuck::cast_slice(&metadata_vec),
            );
        }

        // Upload lookup texture (small - always upload)
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.fill_pipeline.chunk_lookup_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&lookup_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(17 * 4),
                rows_per_image: Some(17),
            },
            wgpu::Extent3d {
                width: 17,
                height: 17,
                depth_or_array_layers: 17,
            },
        );

        // Update tracking
        self.last_chunk_volume_center = volume_center;

        (total_chunks, min_chunk, chunks_uploaded, chunks_reused)
    }

    /// Full reset fallback - clears delta tracking and uploads everything
    fn upload_chunk_data_for_gpu_full_reset(
        &mut self,
        queue: &wgpu::Queue,
        chunks: &Arc<RwLock<AHashMap<ChunkPos, Arc<Chunk>>>>,
        volume_center: glam::IVec3,
    ) -> (u32, glam::IVec3, u32, u32) {
        use glam::IVec3;

        // Clear all tracking
        self.uploaded_chunks.clear();
        self.free_slots.clear();
        self.next_slot = 0;

        let half_size = 128i32;
        let min_block = volume_center - IVec3::splat(half_size);
        let max_block = volume_center + IVec3::splat(half_size - 1);

        let min_chunk = IVec3::new(
            min_block.x.div_euclid(CHUNK_SIZE_I32),
            min_block.y.div_euclid(CHUNK_SIZE_I32),
            min_block.z.div_euclid(CHUNK_SIZE_I32),
        );
        let max_chunk = IVec3::new(
            max_block.x.div_euclid(CHUNK_SIZE_I32),
            max_block.y.div_euclid(CHUNK_SIZE_I32),
            max_block.z.div_euclid(CHUNK_SIZE_I32),
        );

        let mut metadata_vec: Vec<GpuChunkMetadata> = Vec::with_capacity(MAX_OVERLAPPING_CHUNKS as usize);
        let mut blocks_vec: Vec<u8> = Vec::with_capacity(MAX_OVERLAPPING_CHUNKS as usize * 4096);
        let lookup_size = 17 * 17 * 17;
        let mut lookup_data: Vec<u32> = vec![0xFFFFFFFF; lookup_size];

        let chunks_read = chunks.read();
        let mut chunk_index = 0u32;

        for cy in min_chunk.y..=max_chunk.y {
            for cz in min_chunk.z..=max_chunk.z {
                for cx in min_chunk.x..=max_chunk.x {
                    let pos = ChunkPos { x: cx, y: cy, z: cz };

                    if let Some(chunk) = chunks_read.get(&pos) {
                        let world_origin = [cx * CHUNK_SIZE_I32, cy * CHUNK_SIZE_I32, cz * CHUNK_SIZE_I32];
                        metadata_vec.push(GpuChunkMetadata {
                            world_origin,
                            chunk_index,
                        });

                        blocks_vec.extend_from_slice(chunk.blocks_as_bytes());

                        let lx = (cx - min_chunk.x) as usize;
                        let ly = (cy - min_chunk.y) as usize;
                        let lz = (cz - min_chunk.z) as usize;
                        if lx < 17 && ly < 17 && lz < 17 {
                            let lookup_idx = lx + ly * 17 + lz * 17 * 17;
                            lookup_data[lookup_idx] = chunk_index;
                        }

                        // Track in delta system
                        self.uploaded_chunks.insert(pos, chunk_index);

                        chunk_index += 1;
                        if chunk_index >= MAX_OVERLAPPING_CHUNKS {
                            break;
                        }
                    }
                }
                if chunk_index >= MAX_OVERLAPPING_CHUNKS {
                    break;
                }
            }
            if chunk_index >= MAX_OVERLAPPING_CHUNKS {
                break;
            }
        }
        drop(chunks_read);

        self.next_slot = chunk_index;
        let num_chunks = chunk_index;

        // Upload everything
        let uniforms = VoxelFillUniforms {
            volume_center: [volume_center.x, volume_center.y, volume_center.z, 0],
            min_chunk_pos: [min_chunk.x, min_chunk.y, min_chunk.z, num_chunks as i32],
        };
        queue.write_buffer(&self.fill_pipeline.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        if !metadata_vec.is_empty() {
            queue.write_buffer(&self.fill_pipeline.chunk_metadata_buffer, 0, bytemuck::cast_slice(&metadata_vec));
        }

        if !blocks_vec.is_empty() {
            queue.write_buffer(&self.fill_pipeline.chunk_blocks_buffer, 0, &blocks_vec);
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.fill_pipeline.chunk_lookup_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&lookup_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(17 * 4),
                rows_per_image: Some(17),
            },
            wgpu::Extent3d {
                width: 17,
                height: 17,
                depth_or_array_layers: 17,
            },
        );

        self.last_chunk_volume_center = volume_center;

        (num_chunks, min_chunk, num_chunks, 0) // All uploaded, none reused
    }

    /// Update the smoothed center to gradually follow the camera
    /// Call this every frame before checking needs_gpu_refresh
    /// Returns true if the smoothed center moved (for debug logging)
    pub fn update_smoothed_center(&mut self, camera_pos: Vec3, delta_time: f32) -> bool {
        let target = Vec3::new(
            camera_pos.x.floor(),
            camera_pos.y.floor(),
            camera_pos.z.floor(),
        );

        // On first frame or if not initialized, snap to camera
        if !self.initialized {
            self.smoothed_center = target;
            return true;
        }

        // Smooth interpolation toward camera position
        // Use frame-rate independent lerp: lerp(a, b, 1 - e^(-speed * dt))
        let t = 1.0 - (-self.follow_speed * delta_time * 60.0).exp();
        let old_center = self.smoothed_center;
        self.smoothed_center = old_center.lerp(target, t);

        // Update shadow blend (fade in after volume updates)
        // shadow_blend goes from 0.0 -> 1.0 over time
        if self.shadow_blend < 1.0 {
            let blend_t = 1.0 - (-self.shadow_fade_speed * delta_time).exp();
            self.shadow_blend = (self.shadow_blend + blend_t).min(1.0);
        }

        // Return true if we moved at all (for debugging)
        (self.smoothed_center - old_center).length() > 0.01
    }

    /// Get current shadow blend factor (0.0 = no shadows, 1.0 = full shadows)
    pub fn get_shadow_blend(&self) -> f32 {
        self.shadow_blend
    }

    /// Get the current smoothed center (for GPU fill to use as target)
    pub fn get_smoothed_center(&self) -> Vec3 {
        self.smoothed_center
    }

    /// Check if GPU fill is needed based on smoothed center movement
    /// Uses smoothed_center (which follows camera) vs last_update_center (where volume is)
    pub fn needs_gpu_refresh(&self, _camera_pos: Vec3) -> bool {
        // Check distance from smoothed center to where volume actually is
        let distance_moved = (self.smoothed_center - self.last_update_center).length();
        !self.initialized || distance_moved >= self.update_threshold || self.force_full_update
    }

    /// Toggle between GPU and CPU fill modes
    pub fn set_use_gpu_fill(&mut self, use_gpu: bool) {
        self.use_gpu_fill = use_gpu;
        tracing::info!("Voxel fill mode: {}", if use_gpu { "GPU" } else { "CPU" });
    }

    /// Get current fill mode
    pub fn is_gpu_fill_enabled(&self) -> bool {
        self.use_gpu_fill
    }

    /// Toggle delta upload mode (incremental chunk uploads)
    pub fn set_delta_upload_enabled(&mut self, enabled: bool) {
        self.delta_upload_enabled = enabled;
        if !enabled {
            // Clear tracking when disabling
            self.uploaded_chunks.clear();
            self.free_slots.clear();
            self.next_slot = 0;
        }
        tracing::info!("Delta upload mode: {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Check if delta upload mode is enabled
    pub fn is_delta_upload_enabled(&self) -> bool {
        self.delta_upload_enabled
    }

    /// Get delta upload statistics
    pub fn get_delta_stats(&self) -> (usize, usize, u32) {
        (self.uploaded_chunks.len(), self.free_slots.len(), self.next_slot)
    }

    /// Set the shadow update threshold (how many blocks before triggering an update)
    pub fn set_update_threshold(&mut self, threshold: f32) {
        self.update_threshold = threshold;
        tracing::info!("Shadow update step: {} blocks", threshold);
    }

    /// Get current update threshold
    pub fn get_update_threshold(&self) -> f32 {
        self.update_threshold
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

        tracing::debug!(
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

// Compute shader for light initialization (seeds emissive voxels into A channel)
const LIGHT_INIT_SHADER: &str = r#"
// Initialize light propagation by copying emissive (G channel) to light (A channel)

@group(0) @binding(0) var voxel_volume: texture_storage_3d<rgba8unorm, read_write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check for 256^3 volume
    if (global_id.x >= 256u || global_id.y >= 256u || global_id.z >= 256u) {
        return;
    }

    let coords = vec3<i32>(global_id);
    let current = textureLoad(voxel_volume, coords);

    // Seed: emissive channel (G) -> propagated light channel (A)
    // Emissive voxels become light sources
    let light = current.g;

    // Write back with light seeded in A channel
    textureStore(voxel_volume, coords, vec4<f32>(current.r, current.g, current.b, light));
}
"#;

// Compute shader for GPU-based light propagation (jump flooding algorithm)
const LIGHT_PROPAGATE_SHADER: &str = r#"
// GPU-based light propagation using jump flooding algorithm
// Each iteration propagates light one "jump" further with decreasing step sizes
// O(log N) convergence: 8 passes for 256^3 volume

struct LightPropagateUniforms {
    step_size: i32,           // Jump distance for this pass (64, 32, 16, 8, 4, 2, 1)
    max_light_radius: f32,    // Maximum propagation radius (14 for torches)
    _padding: vec2<f32>,
}

@group(0) @binding(0) var voxel_volume: texture_storage_3d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> params: LightPropagateUniforms;

// Sample a neighbor and return its light contribution (attenuated)
fn sample_neighbor(coords: vec3<i32>, offset: vec3<i32>, attenuation: f32) -> f32 {
    let neighbor_coords = coords + offset;

    // Clamp coordinates (edges of volume)
    let clamped = clamp(neighbor_coords, vec3<i32>(0), vec3<i32>(255));

    // Skip if we went out of bounds
    if (any(clamped != neighbor_coords)) {
        return 0.0;
    }

    // Sample neighbor
    let neighbor = textureLoad(voxel_volume, clamped);

    // Skip if neighbor is solid blocker
    if (neighbor.r > 0.78 && neighbor.g < 0.1) {
        return 0.0;
    }

    return neighbor.a * attenuation;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check for 256^3 volume
    if (global_id.x >= 256u || global_id.y >= 256u || global_id.z >= 256u) {
        return;
    }

    let coords = vec3<i32>(global_id);
    let current = textureLoad(voxel_volume, coords);

    // If this voxel is solid (opacity > 0.78) and not emissive, it blocks light
    let is_solid_blocker = current.r > 0.78 && current.g < 0.1;
    if (is_solid_blocker) {
        return;
    }

    var best_light = current.a;  // Current propagated light

    // If this voxel is emissive, it's a light source (keep its brightness)
    if (current.g > 0.1) {
        best_light = max(best_light, current.g);
    }

    // Calculate attenuation based on step distance
    let step_dist = f32(params.step_size);
    let attenuation = max(0.0, 1.0 - (step_dist / params.max_light_radius));

    if (attenuation > 0.0) {
        // Sample all 6 cardinal neighbors (unrolled loop for WGSL compatibility)
        let step = params.step_size;
        best_light = max(best_light, sample_neighbor(coords, vec3<i32>(-step, 0, 0), attenuation));
        best_light = max(best_light, sample_neighbor(coords, vec3<i32>(step, 0, 0), attenuation));
        best_light = max(best_light, sample_neighbor(coords, vec3<i32>(0, -step, 0), attenuation));
        best_light = max(best_light, sample_neighbor(coords, vec3<i32>(0, step, 0), attenuation));
        best_light = max(best_light, sample_neighbor(coords, vec3<i32>(0, 0, -step), attenuation));
        best_light = max(best_light, sample_neighbor(coords, vec3<i32>(0, 0, step), attenuation));
    }

    // Write back with updated light value
    textureStore(voxel_volume, coords, vec4<f32>(current.r, current.g, current.b, best_light));
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

// Compute shader for clearing voxel volume to zero
const VOXEL_CLEAR_SHADER: &str = r#"
@group(0) @binding(0) var voxel_volume: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check for 256^3 volume
    if (global_id.x >= 256u || global_id.y >= 256u || global_id.z >= 256u) {
        return;
    }

    textureStore(voxel_volume, vec3<i32>(global_id), vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
"#;

// Compute shader for filling voxel volume from chunk data
const VOXEL_FILL_SHADER: &str = r#"
struct VoxelFillUniforms {
    volume_center: vec4<i32>,   // xyz = center, w unused
    min_chunk_pos: vec4<i32>,   // xyz = min chunk pos, w = num_chunks
}

struct ChunkMetadata {
    world_origin: vec3<i32>,
    chunk_index: u32,
}

@group(0) @binding(0) var voxel_volume: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: VoxelFillUniforms;
@group(0) @binding(2) var<storage, read> chunk_metadata: array<ChunkMetadata>;
@group(0) @binding(3) var<storage, read> chunk_blocks: array<u32>;
@group(0) @binding(4) var chunk_lookup: texture_3d<u32>;

// Block type constants (matching Block enum in chunk.rs)
const BLOCK_AIR: u32 = 0u;
const BLOCK_WATER: u32 = 6u;
const BLOCK_LEAVES: u32 = 8u;
const BLOCK_TORCH: u32 = 15u;
const BLOCK_URANIUM_ORE: u32 = 28u;
const BLOCK_JUNGLE_LEAVES: u32 = 40u;
const BLOCK_VINES: u32 = 41u;
const BLOCK_CRYSTAL_AMETHYST: u32 = 48u;
const BLOCK_CRYSTAL_QUARTZ: u32 = 49u;
const BLOCK_DEAD_BUSH: u32 = 51u;
const BLOCK_SPRUCE_LEAVES: u32 = 54u;

// Read a single byte from the packed u32 array
// Chunk blocks are packed as 4096 bytes = 1024 u32s per chunk
fn read_block(chunk_index: u32, local_x: u32, local_y: u32, local_z: u32) -> u32 {
    let block_index = local_x + local_y * 16u + local_z * 256u;
    let buffer_offset = chunk_index * 1024u + block_index / 4u;
    let byte_offset = block_index % 4u;

    let packed = chunk_blocks[buffer_offset];
    return (packed >> (byte_offset * 8u)) & 0xFFu;
}

// Convert block type to voxel data: (opacity, emissive, material, flags)
fn get_voxel_data(block_type: u32) -> vec4<f32> {
    var opacity: f32 = 0.0;
    var emissive: f32 = 0.0;
    var material: f32 = 0.0;

    // Handle special cases using if-else chain (WGSL switch doesn't support multiple case values)
    if (block_type == BLOCK_AIR) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if (block_type == BLOCK_WATER) {
        opacity = 64.0 / 255.0;
        material = 2.0 / 255.0;
    } else if (block_type == BLOCK_TORCH) {
        emissive = 220.0 / 255.0;  // Light level 14
        material = 3.0 / 255.0;
    } else if (block_type == BLOCK_LEAVES || block_type == BLOCK_JUNGLE_LEAVES ||
               block_type == BLOCK_SPRUCE_LEAVES || block_type == BLOCK_VINES ||
               block_type == BLOCK_DEAD_BUSH) {
        opacity = 192.0 / 255.0;  // Semi-transparent foliage
        material = 4.0 / 255.0;
    } else if (block_type == BLOCK_URANIUM_ORE) {
        opacity = 1.0;
        emissive = 125.0 / 255.0;  // Light level 8
        material = 1.0 / 255.0;
    } else if (block_type == BLOCK_CRYSTAL_AMETHYST) {
        opacity = 192.0 / 255.0;
        emissive = 157.0 / 255.0;  // Light level 10
        material = 5.0 / 255.0;
    } else if (block_type == BLOCK_CRYSTAL_QUARTZ) {
        opacity = 192.0 / 255.0;
        emissive = 188.0 / 255.0;  // Light level 12
        material = 6.0 / 255.0;
    } else {
        // All other blocks are solid
        opacity = 1.0;
        material = 1.0 / 255.0;
    }

    return vec4<f32>(opacity, emissive, material, 0.0);
}

// Integer division that rounds toward negative infinity (like Rust's div_euclid)
fn div_floor(a: i32, b: i32) -> i32 {
    let d = a / b;
    let r = a % b;
    // If remainder is non-zero and signs differ, subtract 1
    if (r != 0 && ((a < 0) != (b < 0))) {
        return d - 1;
    }
    return d;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_coords = vec3<i32>(global_id);

    // Bounds check for 256^3 volume
    if (voxel_coords.x >= 256 || voxel_coords.y >= 256 || voxel_coords.z >= 256) {
        return;
    }

    // Convert voxel coordinates to world coordinates
    let half_size = 128;
    let world_pos = vec3<i32>(
        voxel_coords.x + uniforms.volume_center.x - half_size,
        voxel_coords.y + uniforms.volume_center.y - half_size,
        voxel_coords.z + uniforms.volume_center.z - half_size
    );

    // Find which chunk contains this world position
    let chunk_pos = vec3<i32>(
        div_floor(world_pos.x, 16),
        div_floor(world_pos.y, 16),
        div_floor(world_pos.z, 16)
    );

    // Convert to lookup texture coordinates
    let lookup_pos = chunk_pos - uniforms.min_chunk_pos.xyz;

    // Bounds check for lookup texture (17x17x17)
    if (lookup_pos.x < 0 || lookup_pos.x >= 17 ||
        lookup_pos.y < 0 || lookup_pos.y >= 17 ||
        lookup_pos.z < 0 || lookup_pos.z >= 17) {
        // Outside lookup range - no shadow (air)
        textureStore(voxel_volume, voxel_coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Read from lookup texture
    let chunk_index = textureLoad(chunk_lookup, lookup_pos, 0).r;

    if (chunk_index == 0xFFFFFFFFu) {
        // No chunk at this position - air
        textureStore(voxel_volume, voxel_coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Calculate local position within chunk (0-15)
    let local_pos = vec3<i32>(
        world_pos.x - chunk_pos.x * 16,
        world_pos.y - chunk_pos.y * 16,
        world_pos.z - chunk_pos.z * 16
    );

    // Read block type from chunk data
    let block_type = read_block(chunk_index, u32(local_pos.x), u32(local_pos.y), u32(local_pos.z));

    // Convert to voxel data and store
    let voxel_data = get_voxel_data(block_type);
    textureStore(voxel_volume, voxel_coords, voxel_data);
}
"#;
