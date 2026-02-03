//! Rendering module
//!
//! Handles GPU initialization, pipeline management, and frame rendering.

mod bloom;
mod camera;
mod chunk_pool;
mod debug_overlay;
mod gpu_culling;
mod impostor;
mod inventory_ui;
mod item_renderer;
mod pipeline;
mod prop_renderer;
pub mod prop_registry;
mod refraction;
mod scenery_renderer;
mod ssao;
mod dof;
mod texture;
pub mod voxel_gi;
mod volumetrics;
pub mod vox_loader;
mod water;
mod clouds;

pub use bloom::Bloom;
pub use camera::{Camera, Frustum};
pub use chunk_pool::ChunkPool;
pub use debug_overlay::{DebugOverlay, DebugMetrics};
pub use impostor::{ImpostorRenderer, ImpostorInstance, ImpostorRange};
pub use item_renderer::{ItemRenderer, ItemInstanceData};
pub use prop_renderer::{PropRenderer, PropInstance};
pub use prop_registry::{PropRegistry, PropDefinition, PropId, CollisionType};
pub use scenery_renderer::{SceneryRenderer, SceneryQuality, SceneryInstance};
pub use refraction::{RefractionQuality, RefractionParams};
pub use ssao::{Ssao, SsaoQuality, SsaoParams};
pub use dof::{Dof, DofQuality, DofParams, FocusMode, NearBlur, BokehShape};
pub use texture::TextureAtlas;
pub use voxel_gi::VoxelGI;
pub use vox_loader::{VoxModel, VoxMesh};
pub use volumetrics::VolumetricLighting;
pub use water::{Water, WaterQuality};
pub use clouds::{Clouds, CloudQuality};

/// Display mode for the window
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    Windowed,
    Borderless,
    Fullscreen,
}

impl DisplayMode {
    /// Get the display mode as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            DisplayMode::Windowed => "WINDOWED",
            DisplayMode::Borderless => "BORDERLESS",
            DisplayMode::Fullscreen => "FULLSCREEN",
        }
    }

    /// Cycle to the next display mode
    pub fn next(&self) -> DisplayMode {
        match self {
            DisplayMode::Windowed => DisplayMode::Borderless,
            DisplayMode::Borderless => DisplayMode::Fullscreen,
            DisplayMode::Fullscreen => DisplayMode::Windowed,
        }
    }
}

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::constants::CHUNK_SIZE_I32;
use crate::utils::gpu_profiler::GpuProfiler;
use crate::utils::profiler::FrameProfiler;
use crate::world::WorldManager;

use pipeline::{UiVertex, LineVertex, create_voxel_lit_chunk_pipeline, create_dimmer_pipeline};
use glam::{Vec3, Mat4};

/// Render statistics for the current frame
#[derive(Default, Clone)]
pub struct RenderStats {
    pub chunks_rendered: u32,
    pub chunks_culled: u32,
    pub impostors_rendered: u32,
    pub triangles: u32,
    pub gpu_time_ms: f64,
    // Shadow pass stats (legacy, now unused)
    pub shadow_chunks_rendered: u32,
    pub shadow_chunks_culled: u32,
    pub shadow_pass_ms: f64,
    pub main_pass_ms: f64,
    // Voxel GI timing
    pub voxel_gi_ms: f64,
    /// Per-pass GPU timing breakdown (pass_name -> time_ms)
    pub pass_times: Vec<(String, f64)>,
}

/// GPU state and rendering logic
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Pipelines
    /// Chunk pipeline (uses DDA ray marching + VCT for lighting)
    chunk_pipeline: wgpu::RenderPipeline,
    /// Light bind group for chunk shader
    light_bind_group: wgpu::BindGroup,

    // Bind groups
    #[allow(dead_code)]
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    #[allow(dead_code)]
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,

    // Depth buffer
    depth_texture: wgpu::TextureView,

    // Light uniform buffer
    light_buffer: wgpu::Buffer,

    // Sky rendering
    sky_pipeline: wgpu::RenderPipeline,
    sky_buffer: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    moon_texture: wgpu::Texture,
    #[allow(dead_code)]
    moon_texture_view: wgpu::TextureView,

    // GPU profiler
    gpu_profiler: GpuProfiler,

    // UI rendering (crosshair)
    ui_pipeline: wgpu::RenderPipeline,
    crosshair_vertex_buffer: wgpu::Buffer,

    // Block outline rendering
    line_pipeline: wgpu::RenderPipeline,
    outline_vertex_buffer: wgpu::Buffer,

    // Debug overlay
    debug_overlay: DebugOverlay,

    // Chunk buffer pool for multi-draw indirect
    chunk_pool: ChunkPool,

    // GPU frustum culling (compute shader based)
    gpu_culler: gpu_culling::GpuCuller,
    /// Whether to use GPU culling (can be toggled for comparison)
    use_gpu_culling: bool,
    /// Whether multi_draw_indexed_indirect_count is available
    mdi_count_available: bool,

    // Ray-marched water rendering system
    water_system: Water,

    // Scene copy for water refraction (copied before water renders)
    refraction_source_texture: wgpu::Texture,
    refraction_source_view: wgpu::TextureView,

    // Stats from last frame
    pub stats: RenderStats,

    // Animation timing
    start_time: std::time::Instant,

    // Day/night cycle
    time_of_day: f32,      // 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
    cycle_speed: f32,      // Multiplier for cycle speed (1.0 = 10 min day, higher = faster)
    day_count: u32,        // Number of complete days elapsed (for moon phase)

    // Shadow mode: -1.0 = off, 0.0 = hard shadows, 1.0 = soft shadows
    shadow_mode: f32,
    // Shadow debug mode: 0.0 = off, 1.0 = visualize shadow data
    shadow_debug: f32,

    // Graphics settings
    bloom_enabled: bool,
    render_distance: i32,
    shadow_distance: i32,
    display_mode: DisplayMode,
    /// Flag indicating display mode change is pending (app.rs applies it)
    display_mode_pending: bool,
    /// Whether torches cast shadows (more immersive but higher GPU cost)
    torch_shadows: bool,

    // Audio settings (stored here for settings menu, synced to AudioManager in app.rs)
    /// Master audio volume (0.0 - 1.0)
    audio_master: f32,
    /// SFX audio volume (0.0 - 1.0)
    audio_sfx: f32,
    /// Ambient audio volume (0.0 - 1.0)
    audio_ambient: f32,

    // Voxel-based Global Illumination system
    // Uses DDA ray marching for shadows, VCT for indirect lighting
    voxel_gi: VoxelGI,

    // Bloom post-processing for emissive blocks
    bloom: Bloom,
    scene_texture: wgpu::Texture,
    scene_texture_view: wgpu::TextureView,

    // Volumetric light shafts (god rays)
    volumetrics: VolumetricLighting,
    volumetric_uniform_buffer: wgpu::Buffer,

    // SSAO (screen-space ambient occlusion)
    ssao: Ssao,

    // Depth of Field post-processing
    dof: Dof,

    // Procedural cloud rendering
    clouds: Clouds,

    // Settings menu dimmer overlay
    dimmer_pipeline: wgpu::RenderPipeline,

    // Inventory UI
    inventory_ui: inventory_ui::InventoryUI,

    // Item entity rendering (dropped items in world)
    item_renderer: ItemRenderer,

    // Prop rendering (voxel models like chests)
    prop_renderer: PropRenderer,

    // Scenery rendering (grass, flowers, rocks - thousands of instances)
    prop_registry: Arc<PropRegistry>,
    scenery_renderer: scenery_renderer::SceneryRenderer,

    // Impostor rendering for distant chunks
    impostor_renderer: ImpostorRenderer,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // Create instance with best available backend
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).expect("Failed to create surface");

        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        tracing::info!("Using GPU: {}", adapter.get_info().name);

        // Request device with optional features for GPU profiling, multi-draw, and storage texture read-write
        let desired_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::MULTI_DRAW_INDIRECT
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES; // For read-write storage textures in light propagation
        let available_features = adapter.features();
        let features_to_use = desired_features & available_features;

        if features_to_use.contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            tracing::info!("Multi-draw indirect enabled");
        } else {
            tracing::warn!("Multi-draw indirect not available - using fallback");
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Mosaic Device"),
                    required_features: features_to_use,
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create GPU profiler
        let gpu_profiler = GpuProfiler::new(&device, &queue);

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Choose lowest-latency present mode available
        // Mailbox = triple buffer (no tearing, low latency)
        // Immediate = no vsync (lowest latency, may tear)
        // Fifo = vsync (highest latency)
        let present_mode = if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo // Fallback, always supported
        };

        tracing::info!("Using present mode: {:?}", present_mode);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1, // Minimize input-to-display latency
        };
        surface.configure(&device, &config);

        // Create camera uniform buffer
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Load texture atlas
        let texture_atlas = TextureAtlas::load(&device, &queue);
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_atlas.sampler),
                },
            ],
        });

        // === LIGHTING SETUP ===
        // Create light uniform buffer (for sun/moon direction, colors, etc.)
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[LightUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Light bind group layout
        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Light bind group
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        // === SKY RENDERING SETUP ===
        let sky_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky Buffer"),
            contents: bytemuck::cast_slice(&[SkyUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load moon texture
        let (moon_texture, moon_texture_view, moon_sampler) = Self::load_moon_texture(&device, &queue);

        let sky_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky Bind Group"),
            layout: &sky_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sky_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&moon_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&moon_sampler),
                },
            ],
        });

        let sky_pipeline = pipeline::create_sky_pipeline(&device, surface_format, &sky_bind_group_layout);

        // === VOXEL GI SYSTEM SETUP ===
        // Create voxel GI system (provides DDA shadows + VCT indirect lighting)
        let voxel_gi = VoxelGI::new(&device);

        // Create chunk pipeline (uses voxel GI for lighting)
        let chunk_pipeline = create_voxel_lit_chunk_pipeline(
            &device,
            surface_format,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            &light_bind_group_layout,
            voxel_gi.bind_group_layout(),
        );
        tracing::info!("Voxel GI system initialized (256^3 volume, DDA shadows + VCT)");

        // Create UI pipeline for crosshair
        let ui_pipeline = pipeline::create_ui_pipeline(&device, surface_format);

        // Create crosshair vertex buffer (two thin rectangles forming a +)
        let crosshair_size = 0.02; // Size in clip space
        let crosshair_thickness = 0.003;
        let crosshair_vertices = [
            // Horizontal bar
            UiVertex { position: [-crosshair_size, -crosshair_thickness] },
            UiVertex { position: [crosshair_size, -crosshair_thickness] },
            UiVertex { position: [crosshair_size, crosshair_thickness] },
            UiVertex { position: [-crosshair_size, -crosshair_thickness] },
            UiVertex { position: [crosshair_size, crosshair_thickness] },
            UiVertex { position: [-crosshair_size, crosshair_thickness] },
            // Vertical bar
            UiVertex { position: [-crosshair_thickness, -crosshair_size] },
            UiVertex { position: [crosshair_thickness, -crosshair_size] },
            UiVertex { position: [crosshair_thickness, crosshair_size] },
            UiVertex { position: [-crosshair_thickness, -crosshair_size] },
            UiVertex { position: [crosshair_thickness, crosshair_size] },
            UiVertex { position: [-crosshair_thickness, crosshair_size] },
        ];
        let crosshair_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Crosshair Vertex Buffer"),
            contents: bytemuck::cast_slice(&crosshair_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create line pipeline for block outline
        let line_pipeline = pipeline::create_line_pipeline(&device, surface_format, &camera_bind_group_layout);

        // Create dimmer pipeline for settings menu overlay
        let dimmer_pipeline = create_dimmer_pipeline(&device, surface_format);

        // Create outline vertex buffer (will be updated each frame)
        // 12 edges * 2 vertices = 24 vertices for a cube wireframe
        let outline_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Outline Vertex Buffer"),
            size: (24 * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth texture
        let depth_texture = Self::create_depth_texture(&device, &config);

        // Create debug overlay
        let mut debug_overlay = DebugOverlay::new(&device, surface_format);
        debug_overlay.upload_font(&queue);
        debug_overlay.update_screen_size(size.width as f32, size.height as f32);

        // Create chunk pool for multi-draw indirect
        let mdi_available = features_to_use.contains(wgpu::Features::MULTI_DRAW_INDIRECT);
        let chunk_pool = ChunkPool::new(&device, mdi_available);

        // Create GPU frustum culler (compute shader based)
        let mut gpu_culler = gpu_culling::GpuCuller::new(&device, &chunk_pool.indirect_buffer);
        gpu_culler.rebuild_bind_group(&device, &chunk_pool.indirect_buffer);
        let mdi_count_available = features_to_use.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let use_gpu_culling = mdi_available && mdi_count_available; // Need both for GPU culling

        if use_gpu_culling {
            tracing::info!("GPU frustum culling enabled (compute shader)");
        } else {
            tracing::info!("Using CPU frustum culling");
        }

        // === BLOOM POST-PROCESSING ===
        // Scene texture for off-screen rendering (needed for bloom and water refraction)
        // Uses surface format for compatibility with existing pipelines
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format, // Use surface format for pipeline compatibility
            // COPY_SRC needed for water refraction (copy to refraction_source_texture)
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scene_texture_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom system
        let mut bloom = Bloom::new(&device, size.width, size.height, surface_format);
        bloom.create_extract_bind_group(&device, &scene_texture_view);

        // Create volumetric light shafts system
        let volumetrics = VolumetricLighting::new(
            &device,
            size.width,
            size.height,
            surface_format,
            voxel_gi.bind_group_layout(),
        );
        let volumetric_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Volumetric Uniform Buffer"),
            contents: bytemuck::cast_slice(&[volumetrics::VolumetricUniform {
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0; 4],
                sun_direction: [0.5, 0.8, 0.3, 1.0],
                sun_color: [1.0, 0.95, 0.85, 0.015], // a = density
                volume_center: [0.0; 4],
                screen_params: [size.width as f32, size.height as f32, 32.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create SSAO system
        let ssao = Ssao::new(&device, size.width, size.height, surface_format);
        // Upload noise texture data
        ssao.upload_noise_texture(&queue);

        // Create DoF system
        let dof = Dof::new(&device, size.width, size.height, surface_format);
        // Upload noise texture data
        dof.upload_noise_texture(&queue);

        // Create ray-marched water system
        let water_system = Water::new(&device, surface_format, &camera_bind_group_layout);

        // Create procedural cloud system
        let clouds = Clouds::new(&device, surface_format);

        // Create refraction source texture (copy of scene before water renders)
        let refraction_source_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("refraction_source_texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let refraction_source_view = refraction_source_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create inventory UI renderer
        let inventory_ui = inventory_ui::InventoryUI::new(&device, surface_format, &texture_bind_group_layout);

        // Create item entity renderer
        let item_renderer = ItemRenderer::new(
            &device,
            surface_format,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
        );

        // Create prop renderer (for voxel models like chests)
        let mut prop_renderer = PropRenderer::new(
            &device,
            surface_format,
            &camera_bind_group_layout,
        );

        // Load chest prop (model is 19x8x9 voxels, scale to 2 blocks wide)
        // At 2/19 scale: 19 * (2/19) = 2 blocks wide, 8 * (2/19) = 0.84 blocks tall
        let chest_path = std::path::Path::new("assets/props/furniture/chest/prop_chest_light_brown/prop_chest_light_brown.vox");
        if let Err(e) = prop_renderer.load_prop(&device, "chest", chest_path, 2.0 / 19.0) {
            tracing::warn!("Failed to load chest prop: {}", e);
        }

        // Create prop registry and scenery renderer for vegetation (grass, flowers, rocks)
        let prop_registry = Arc::new(PropRegistry::new(&device));
        let scenery_renderer = scenery_renderer::SceneryRenderer::new(
            &device,
            surface_format,
            &camera_bind_group_layout,
            Arc::clone(&prop_registry),
        );

        // Create impostor renderer for distant chunks
        let impostor_renderer = ImpostorRenderer::new(&device, surface_format);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            chunk_pipeline,
            light_bind_group,
            camera_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            texture_bind_group_layout,
            texture_bind_group,
            depth_texture,
            light_buffer,
            sky_pipeline,
            sky_buffer,
            sky_bind_group,
            moon_texture,
            moon_texture_view,
            gpu_profiler,
            ui_pipeline,
            crosshair_vertex_buffer,
            line_pipeline,
            outline_vertex_buffer,
            debug_overlay,
            chunk_pool,
            gpu_culler,
            use_gpu_culling,
            mdi_count_available,
            water_system,
            refraction_source_texture,
            refraction_source_view,
            stats: RenderStats::default(),
            start_time: std::time::Instant::now(),
            time_of_day: 0.35,   // Start at mid-morning
            cycle_speed: 10.0,   // 10x speed for testing (1 min = 1 day cycle)
            day_count: 0,        // Start at day 0
            shadow_mode: 0.0,    // Start with hard shadows (-1=off, 0=hard, 1=soft)
            shadow_debug: 0.0,   // Shadow debug off by default
            bloom_enabled: true,
            render_distance: crate::constants::RENDER_DISTANCE,
            shadow_distance: crate::constants::RENDER_DISTANCE,
            display_mode: DisplayMode::Windowed,
            display_mode_pending: false,
            torch_shadows: false,  // Default OFF for performance
            // Audio settings (defaults, will be overwritten by apply_settings)
            audio_master: 0.8,
            audio_sfx: 1.0,
            audio_ambient: 0.6,
            voxel_gi,
            bloom,
            scene_texture,
            scene_texture_view,
            volumetrics,
            volumetric_uniform_buffer,
            ssao,
            dof,
            clouds,
            dimmer_pipeline,
            inventory_ui,
            item_renderer,
            prop_renderer,
            prop_registry,
            scenery_renderer,
            impostor_renderer,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // Wait for all GPU work to complete before resizing
            // Poll multiple times to ensure all async callbacks complete
            for _ in 0..3 {
                self.device.poll(wgpu::Maintain::Wait);
            }

            // Force-unmap any pending GPU profiler buffers to avoid validation errors
            self.gpu_profiler.force_unmap();

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
            self.debug_overlay.update_screen_size(new_size.width as f32, new_size.height as f32);
            self.inventory_ui.update_screen_size(new_size.width as f32, new_size.height as f32);

            // Resize scene texture for bloom and water refraction
            self.scene_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scene_texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format, // Use surface format
                // COPY_SRC needed for water refraction
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.scene_texture_view = self.scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Resize bloom textures
            self.bloom.resize(&self.device, new_size.width, new_size.height);
            self.bloom.create_extract_bind_group(&self.device, &self.scene_texture_view);

            // Resize volumetrics textures
            self.volumetrics.resize(&self.device, new_size.width, new_size.height);

            // Resize SSAO textures
            self.ssao.resize(&self.device, new_size.width, new_size.height);

            // Resize DoF textures
            self.dof.resize(&self.device, new_size.width, new_size.height);

            // Resize refraction source texture
            self.refraction_source_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("refraction_source_texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.refraction_source_view = self.refraction_source_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Update water system bind group with new textures
            self.water_system.update_bind_group(
                &self.device,
                &self.depth_texture,
                &self.refraction_source_view,
            );
        }
    }

    pub fn toggle_debug_overlay(&mut self) {
        self.debug_overlay.toggle();
    }

    pub fn is_debug_overlay_enabled(&self) -> bool {
        self.debug_overlay.is_enabled()
    }

    /// Mark voxel volume as needing full update (use mark_voxel_dirty_at for single blocks)
    pub fn mark_voxel_dirty(&mut self) {
        self.voxel_gi.mark_dirty();
    }

    /// Mark a specific block position as dirty for incremental voxel update
    /// Much faster than full update - only updates the affected voxel and light radius
    pub fn mark_voxel_dirty_at(&mut self, x: i32, y: i32, z: i32) {
        self.voxel_gi.mark_dirty_at(x, y, z);
    }

    /// Update the day/night cycle time
    /// dt is delta time in seconds
    pub fn update_time(&mut self, dt: f32) {
        // Base cycle: 600 seconds (10 minutes) for a full day
        let base_cycle_duration = 600.0;
        let time_increment = dt / base_cycle_duration * self.cycle_speed;
        let old_time = self.time_of_day;
        self.time_of_day = (self.time_of_day + time_increment) % 1.0;

        // Detect day wrap (time went from high to low)
        if self.time_of_day < old_time && time_increment > 0.0 {
            self.day_count = self.day_count.wrapping_add(1);
        }
    }

    /// Get current day count (for moon phase calculation)
    pub fn day_count(&self) -> u32 {
        self.day_count
    }

    /// Toggle through cycle speed presets
    /// Returns the new speed for display
    pub fn toggle_cycle_speed(&mut self) -> f32 {
        // Cycle through: 1x -> 5x -> 20x -> 50x -> 100x -> paused -> 1x
        self.cycle_speed = match self.cycle_speed as i32 {
            0 => 1.0,
            1 => 5.0,
            5 => 20.0,
            20 => 50.0,
            50 => 100.0,
            100 => 0.0,  // Paused
            _ => 1.0,
        };
        self.cycle_speed
    }

    /// Toggle shadow update step: 2 -> 4 -> 8 -> 16 -> 32 -> 48 -> 2
    /// Returns the new step for display
    pub fn toggle_shadow_update_step(&mut self) -> f32 {
        let current = self.voxel_gi.get_update_threshold() as i32;
        let new_step = match current {
            2 => 4.0,
            4 => 8.0,
            8 => 16.0,
            16 => 32.0,
            32 => 48.0,
            48 => 2.0,
            _ => 2.0,
        };
        self.voxel_gi.set_update_threshold(new_step);
        new_step
    }

    /// Set shadow update step directly (for loading from settings)
    pub fn set_shadow_update_step(&mut self, step: f32) {
        self.voxel_gi.set_update_threshold(step);
    }

    /// Get current shadow update step
    pub fn get_shadow_update_step(&self) -> f32 {
        self.voxel_gi.get_update_threshold()
    }

    /// Cycle shadow update step backwards: 2 <- 4 <- 8 <- 16 <- 32 <- 48 <- 2
    pub fn cycle_shadow_update_step_prev(&mut self) -> f32 {
        let current = self.voxel_gi.get_update_threshold() as i32;
        let new_step = match current {
            2 => 48.0,
            4 => 2.0,
            8 => 4.0,
            16 => 8.0,
            32 => 16.0,
            48 => 32.0,
            _ => 2.0,
        };
        self.voxel_gi.set_update_threshold(new_step);
        new_step
    }

    /// Get current time of day (0.0-1.0)
    pub fn time_of_day(&self) -> f32 {
        self.time_of_day
    }

    /// Get current cycle speed multiplier
    pub fn cycle_speed(&self) -> f32 {
        self.cycle_speed
    }

    /// Set cycle speed directly
    pub fn set_cycle_speed(&mut self, speed: f32) {
        self.cycle_speed = speed;
    }

    /// Cycle speed backwards: 1x -> paused -> 100x -> 50x -> 20x -> 5x -> 1x
    pub fn cycle_speed_prev(&mut self) -> f32 {
        self.cycle_speed = match self.cycle_speed as i32 {
            0 => 100.0,
            1 => 0.0,  // Paused
            5 => 1.0,
            20 => 5.0,
            50 => 20.0,
            100 => 50.0,
            _ => 0.0,
        };
        self.cycle_speed
    }

    /// Toggle shadow mode: SOFT (1) -> HARD (0) -> OFF (-1) -> SOFT
    /// Returns the new mode for display
    pub fn toggle_shadow_mode(&mut self) -> f32 {
        self.shadow_mode = if self.shadow_mode > 0.5 {
            0.0  // SOFT -> HARD
        } else if self.shadow_mode > -0.5 {
            -1.0 // HARD -> OFF
        } else {
            1.0  // OFF -> SOFT
        };
        self.shadow_mode
    }

    /// Get current shadow mode (-1=off, 0=hard, 1=soft)
    pub fn shadow_mode(&self) -> f32 {
        self.shadow_mode
    }

    /// Get shadow mode as string
    pub fn shadow_mode_str(&self) -> &'static str {
        if self.shadow_mode < -0.5 {
            "OFF"
        } else if self.shadow_mode < 0.5 {
            "HARD"
        } else {
            "SOFT"
        }
    }

    /// Set shadow mode directly (-1=off, 0=hard, 1=soft)
    pub fn set_shadow_mode(&mut self, mode: f32) {
        self.shadow_mode = mode;
    }

    /// Toggle shadow debug visualization
    pub fn toggle_shadow_debug(&mut self) -> f32 {
        self.shadow_debug = if self.shadow_debug < 0.5 { 1.0 } else { 0.0 };
        self.shadow_debug
    }

    /// Get current shadow debug mode
    pub fn shadow_debug(&self) -> f32 {
        self.shadow_debug
    }

    /// Toggle volumetric light shafts on/off
    pub fn toggle_volumetrics(&mut self) -> bool {
        self.volumetrics.toggle()
    }

    /// Toggle GPU frustum culling on/off for A/B testing
    pub fn toggle_gpu_culling(&mut self) -> bool {
        // Only toggle if MDI count is available (required for GPU culling)
        if self.mdi_count_available {
            self.use_gpu_culling = !self.use_gpu_culling;
        }
        self.use_gpu_culling
    }

    /// Toggle GPU voxel fill on/off for A/B testing
    pub fn toggle_gpu_voxel_fill(&mut self) -> bool {
        let new_state = !self.voxel_gi.is_gpu_fill_enabled();
        self.voxel_gi.set_use_gpu_fill(new_state);
        new_state
    }

    /// Check if volumetrics are enabled
    pub fn volumetrics_enabled(&self) -> bool {
        self.volumetrics.enabled
    }

    /// Check if GPU culling is enabled
    pub fn is_gpu_culling_enabled(&self) -> bool {
        self.use_gpu_culling
    }

    /// Check if GPU voxel fill is enabled
    pub fn is_gpu_voxel_fill_enabled(&self) -> bool {
        self.voxel_gi.is_gpu_fill_enabled()
    }

    /// Check if bloom is enabled
    pub fn is_bloom_enabled(&self) -> bool {
        self.bloom_enabled
    }

    /// Toggle bloom on/off
    pub fn toggle_bloom(&mut self) -> bool {
        self.bloom_enabled = !self.bloom_enabled;
        self.bloom_enabled
    }

    /// Set bloom enabled state directly
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.bloom_enabled = enabled;
    }

    /// Set volumetrics enabled state directly
    pub fn set_volumetrics_enabled(&mut self, enabled: bool) {
        self.volumetrics.set_enabled(enabled);
    }

    /// Set GPU voxel fill enabled state directly
    pub fn set_gpu_voxel_fill_enabled(&mut self, enabled: bool) {
        self.voxel_gi.set_use_gpu_fill(enabled);
    }

    /// Set GPU culling enabled state directly
    pub fn set_gpu_culling_enabled(&mut self, enabled: bool) {
        if self.mdi_count_available {
            self.use_gpu_culling = enabled;
        }
    }

    /// Set debug overlay enabled state directly
    pub fn set_debug_overlay_enabled(&mut self, enabled: bool) {
        self.debug_overlay.set_enabled(enabled);
    }

    // ========== AUDIO VOLUME CONTROLS ==========

    /// Get master audio volume (0.0 - 1.0)
    pub fn audio_master(&self) -> f32 {
        self.audio_master
    }

    /// Cycle master audio volume: 0% -> 10% -> ... -> 100% -> 0%
    pub fn cycle_audio_master(&mut self) {
        self.audio_master = if self.audio_master >= 1.0 {
            0.0
        } else {
            (self.audio_master + 0.1).min(1.0)
        };
    }

    /// Get SFX audio volume (0.0 - 1.0)
    pub fn audio_sfx(&self) -> f32 {
        self.audio_sfx
    }

    /// Cycle SFX audio volume: 0% -> 10% -> ... -> 100% -> 0%
    pub fn cycle_audio_sfx(&mut self) {
        self.audio_sfx = if self.audio_sfx >= 1.0 {
            0.0
        } else {
            (self.audio_sfx + 0.1).min(1.0)
        };
    }

    /// Get ambient audio volume (0.0 - 1.0)
    pub fn audio_ambient(&self) -> f32 {
        self.audio_ambient
    }

    /// Cycle ambient audio volume: 0% -> 10% -> ... -> 100% -> 0%
    pub fn cycle_audio_ambient(&mut self) {
        self.audio_ambient = if self.audio_ambient >= 1.0 {
            0.0
        } else {
            (self.audio_ambient + 0.1).min(1.0)
        };
    }

    /// Set master audio volume directly (0.0-1.0)
    pub fn set_audio_master(&mut self, value: f32) {
        self.audio_master = value.clamp(0.0, 1.0);
    }

    /// Cycle master audio volume backwards
    pub fn cycle_audio_master_prev(&mut self) {
        self.audio_master = if self.audio_master <= 0.0 {
            1.0
        } else {
            (self.audio_master - 0.1).max(0.0)
        };
    }

    /// Set SFX audio volume directly (0.0-1.0)
    pub fn set_audio_sfx(&mut self, value: f32) {
        self.audio_sfx = value.clamp(0.0, 1.0);
    }

    /// Cycle SFX audio volume backwards
    pub fn cycle_audio_sfx_prev(&mut self) {
        self.audio_sfx = if self.audio_sfx <= 0.0 {
            1.0
        } else {
            (self.audio_sfx - 0.1).max(0.0)
        };
    }

    /// Set ambient audio volume directly (0.0-1.0)
    pub fn set_audio_ambient(&mut self, value: f32) {
        self.audio_ambient = value.clamp(0.0, 1.0);
    }

    /// Cycle ambient audio volume backwards
    pub fn cycle_audio_ambient_prev(&mut self) {
        self.audio_ambient = if self.audio_ambient <= 0.0 {
            1.0
        } else {
            (self.audio_ambient - 0.1).max(0.0)
        };
    }

    /// Check if audio settings have changed (for syncing to AudioManager)
    pub fn audio_settings_changed(&self, master: f32, sfx: f32, ambient: f32) -> bool {
        (self.audio_master - master).abs() > 0.001
            || (self.audio_sfx - sfx).abs() > 0.001
            || (self.audio_ambient - ambient).abs() > 0.001
    }

    /// Check if torch shadows are enabled
    pub fn is_torch_shadows_enabled(&self) -> bool {
        self.torch_shadows
    }

    /// Toggle torch shadows on/off
    pub fn toggle_torch_shadows(&mut self) -> bool {
        self.torch_shadows = !self.torch_shadows;
        tracing::info!("Torch shadows: {}", if self.torch_shadows { "ON" } else { "OFF" });
        self.torch_shadows
    }

    /// Set torch shadows enabled state directly
    pub fn set_torch_shadows_enabled(&mut self, enabled: bool) {
        self.torch_shadows = enabled;
    }

    /// Check if SSAO is enabled
    pub fn is_ssao_enabled(&self) -> bool {
        self.ssao.is_enabled()
    }

    /// Get current SSAO quality
    pub fn ssao_quality(&self) -> SsaoQuality {
        self.ssao.quality
    }

    /// Cycle SSAO quality: Off -> Low -> Medium -> High -> Ultra -> Off
    pub fn cycle_ssao_quality(&mut self) -> SsaoQuality {
        let new_quality = self.ssao.cycle_quality(&self.device, self.size.width, self.size.height);
        tracing::info!("SSAO: {}", new_quality.as_str());
        new_quality
    }

    /// Set SSAO quality directly
    pub fn set_ssao_quality(&mut self, quality: SsaoQuality) {
        self.ssao.set_quality(&self.device, self.size.width, self.size.height, quality);
    }

    /// Set SSAO quality by index (0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra)
    pub fn set_ssao_quality_index(&mut self, index: usize) {
        let quality = match index {
            0 => SsaoQuality::Off,
            1 => SsaoQuality::Low,
            2 => SsaoQuality::Medium,
            3 => SsaoQuality::High,
            _ => SsaoQuality::Ultra,
        };
        self.set_ssao_quality(quality);
    }

    /// Cycle SSAO quality backwards
    pub fn cycle_ssao_quality_prev(&mut self) -> SsaoQuality {
        let new_quality = match self.ssao.quality {
            SsaoQuality::Off => SsaoQuality::Ultra,
            SsaoQuality::Low => SsaoQuality::Off,
            SsaoQuality::Medium => SsaoQuality::Low,
            SsaoQuality::High => SsaoQuality::Medium,
            SsaoQuality::Ultra => SsaoQuality::High,
        };
        self.set_ssao_quality(new_quality);
        new_quality
    }

    /// Get mutable access to SSAO params for advanced settings
    pub fn ssao_params_mut(&mut self) -> &mut SsaoParams {
        &mut self.ssao.params
    }

    /// Check if DoF is enabled
    pub fn is_dof_enabled(&self) -> bool {
        self.dof.is_enabled()
    }

    /// Get current DoF quality
    pub fn dof_quality(&self) -> DofQuality {
        self.dof.quality
    }

    /// Cycle DoF quality: Off -> Low -> Medium -> High -> Ultra -> Off
    pub fn cycle_dof_quality(&mut self) -> DofQuality {
        let new_quality = self.dof.cycle_quality();
        tracing::info!("DoF: {}", new_quality.as_str());
        new_quality
    }

    /// Set DoF quality directly
    pub fn set_dof_quality(&mut self, quality: DofQuality) {
        self.dof.set_quality(quality);
    }

    /// Get mutable access to DoF params for advanced settings
    pub fn dof_params_mut(&mut self) -> &mut DofParams {
        &mut self.dof.params
    }

    /// Toggle DoF focus lock
    pub fn toggle_dof_focus_lock(&mut self) {
        self.dof.toggle_focus_lock();
    }

    /// Cycle DoF debug visualization mode
    pub fn cycle_dof_debug_mode(&mut self) {
        self.dof.cycle_debug_mode();
    }

    /// Get DoF focus distance
    pub fn dof_focus_distance(&self) -> f32 {
        self.dof.params.manual_focus_distance
    }

    /// Cycle DoF focus distance: 5 -> 10 -> 15 -> 20 -> 30 -> 50 -> 75 -> 100 -> 5
    pub fn cycle_dof_focus_distance(&mut self) -> f32 {
        let distances = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
        let current = self.dof.params.manual_focus_distance;
        let next = distances.iter()
            .find(|&&d| d > current + 0.1)
            .copied()
            .unwrap_or(distances[0]);
        self.dof.params.manual_focus_distance = next;
        next
    }

    /// Get DoF blur strength (aperture)
    pub fn dof_blur_strength(&self) -> f32 {
        self.dof.params.aperture
    }

    /// Cycle DoF blur strength: 10% intervals from 10% to 100%
    pub fn cycle_dof_blur_strength(&mut self) -> f32 {
        let strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let current = self.dof.params.aperture;
        let next = strengths.iter()
            .find(|&&s| s > current + 0.01)
            .copied()
            .unwrap_or(strengths[0]);
        self.dof.params.aperture = next;
        next
    }

    /// Get DoF focus range (transition width)
    pub fn dof_focus_range(&self) -> f32 {
        self.dof.params.focus_transition
    }

    /// Cycle DoF focus range: matches focus distance values
    pub fn cycle_dof_focus_range(&mut self) -> f32 {
        let ranges = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
        let current = self.dof.params.focus_transition;
        let next = ranges.iter()
            .find(|&&r| r > current + 0.1)
            .copied()
            .unwrap_or(ranges[0]);
        self.dof.params.focus_transition = next;
        next
    }

    /// Set DoF focus range directly
    pub fn set_dof_focus_range(&mut self, range: f32) {
        self.dof.params.focus_transition = range;
    }

    /// Cycle DoF focus range backwards
    pub fn cycle_dof_focus_range_prev(&mut self) -> f32 {
        let ranges = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
        let current = self.dof.params.focus_transition;
        let prev = ranges.iter().rev()
            .find(|&&r| r < current - 0.1)
            .copied()
            .unwrap_or(ranges[ranges.len() - 1]);
        self.dof.params.focus_transition = prev;
        prev
    }

    /// Set DoF blur strength directly (0.0-1.0)
    pub fn set_dof_blur_strength(&mut self, strength: f32) {
        self.dof.params.aperture = strength.clamp(0.0, 1.0);
    }

    /// Cycle DoF blur strength backwards
    pub fn cycle_dof_blur_strength_prev(&mut self) -> f32 {
        let strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let current = self.dof.params.aperture;
        let prev = strengths.iter().rev()
            .find(|&&s| s < current - 0.01)
            .copied()
            .unwrap_or(strengths[strengths.len() - 1]);
        self.dof.params.aperture = prev;
        prev
    }

    /// Set DoF focus distance directly
    pub fn set_dof_focus_distance(&mut self, distance: f32) {
        self.dof.params.manual_focus_distance = distance;
    }

    /// Cycle DoF focus distance backwards
    pub fn cycle_dof_focus_distance_prev(&mut self) -> f32 {
        let distances = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
        let current = self.dof.params.manual_focus_distance;
        let prev = distances.iter().rev()
            .find(|&&d| d < current - 0.1)
            .copied()
            .unwrap_or(distances[distances.len() - 1]);
        self.dof.params.manual_focus_distance = prev;
        prev
    }

    /// Set DoF quality by index (0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra)
    pub fn set_dof_quality_index(&mut self, index: usize) {
        let quality = match index {
            0 => DofQuality::Off,
            1 => DofQuality::Low,
            2 => DofQuality::Medium,
            3 => DofQuality::High,
            _ => DofQuality::Ultra,
        };
        self.dof.set_quality(quality);
    }

    /// Cycle DoF quality backwards
    pub fn cycle_dof_quality_prev(&mut self) -> DofQuality {
        let new_quality = match self.dof.quality {
            DofQuality::Off => DofQuality::Ultra,
            DofQuality::Low => DofQuality::Off,
            DofQuality::Medium => DofQuality::Low,
            DofQuality::High => DofQuality::Medium,
            DofQuality::Ultra => DofQuality::High,
        };
        self.dof.set_quality(new_quality);
        new_quality
    }

    // === Water Quality Settings ===

    /// Get current water quality
    pub fn water_quality(&self) -> WaterQuality {
        self.water_system.quality()
    }

    /// Cycle water quality: Off -> Low -> Medium -> High -> Ultra -> Off
    pub fn cycle_water_quality(&mut self) -> WaterQuality {
        self.water_system.cycle_quality();
        let quality = self.water_system.quality();
        tracing::info!("Water Quality: {}", quality.as_str());
        quality
    }

    /// Set water quality directly
    pub fn set_water_quality(&mut self, quality: WaterQuality) {
        self.water_system.set_quality(quality);
    }

    // Legacy refraction quality methods (now delegate to water quality)

    /// Get current water refraction quality (legacy - returns water quality)
    pub fn refraction_quality(&self) -> RefractionQuality {
        // Map water quality to refraction quality for backwards compatibility
        match self.water_system.quality() {
            WaterQuality::Off => RefractionQuality::Off,
            WaterQuality::Low => RefractionQuality::Low,
            WaterQuality::Medium => RefractionQuality::Medium,
            WaterQuality::High => RefractionQuality::High,
            WaterQuality::Ultra => RefractionQuality::Ultra,
        }
    }

    /// Cycle water refraction quality (legacy - cycles water quality)
    pub fn cycle_refraction_quality(&mut self) -> RefractionQuality {
        self.cycle_water_quality();
        self.refraction_quality()
    }

    /// Set water refraction quality (legacy - sets water quality)
    pub fn set_refraction_quality(&mut self, quality: RefractionQuality) {
        let water_quality = match quality {
            RefractionQuality::Off => WaterQuality::Off,
            RefractionQuality::Low => WaterQuality::Low,
            RefractionQuality::Medium => WaterQuality::Medium,
            RefractionQuality::High => WaterQuality::High,
            RefractionQuality::Ultra => WaterQuality::Ultra,
        };
        self.water_system.set_quality(water_quality);
    }

    /// Set refraction quality by index (0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra)
    pub fn set_refraction_quality_index(&mut self, index: usize) {
        let quality = match index {
            0 => RefractionQuality::Off,
            1 => RefractionQuality::Low,
            2 => RefractionQuality::Medium,
            3 => RefractionQuality::High,
            _ => RefractionQuality::Ultra,
        };
        self.set_refraction_quality(quality);
    }

    /// Cycle refraction quality backwards
    pub fn cycle_refraction_quality_prev(&mut self) -> RefractionQuality {
        let new_quality = match self.refraction_quality() {
            RefractionQuality::Off => RefractionQuality::Ultra,
            RefractionQuality::Low => RefractionQuality::Off,
            RefractionQuality::Medium => RefractionQuality::Low,
            RefractionQuality::High => RefractionQuality::Medium,
            RefractionQuality::Ultra => RefractionQuality::High,
        };
        self.set_refraction_quality(new_quality);
        new_quality
    }

    // === Extended View (Impostor) Settings ===

    /// Get current extended view range
    pub fn impostor_range(&self) -> ImpostorRange {
        self.impostor_renderer.range
    }

    /// Cycle extended view range: Off -> +16 -> +24 -> +32 -> +40 -> Off
    pub fn cycle_impostor_range(&mut self) -> ImpostorRange {
        self.impostor_renderer.range = self.impostor_renderer.range.next();
        tracing::info!("Extended View: {}", self.impostor_renderer.range.as_str());
        self.impostor_renderer.range
    }

    /// Set extended view range directly
    pub fn set_impostor_range(&mut self, range: ImpostorRange) {
        self.impostor_renderer.range = range;
    }

    /// Set impostor range by index (0=Off, 1=+16, 2=+24, 3=+32, 4=+40)
    pub fn set_impostor_range_index(&mut self, index: usize) {
        self.impostor_renderer.range = match index {
            0 => ImpostorRange::Off,
            1 => ImpostorRange::Plus16,
            2 => ImpostorRange::Plus24,
            3 => ImpostorRange::Plus32,
            _ => ImpostorRange::Plus40,
        };
    }

    /// Cycle impostor range backwards
    pub fn cycle_impostor_range_prev(&mut self) -> ImpostorRange {
        self.impostor_renderer.range = match self.impostor_renderer.range {
            ImpostorRange::Off => ImpostorRange::Plus40,
            ImpostorRange::Plus16 => ImpostorRange::Off,
            ImpostorRange::Plus24 => ImpostorRange::Plus16,
            ImpostorRange::Plus32 => ImpostorRange::Plus24,
            ImpostorRange::Plus40 => ImpostorRange::Plus32,
        };
        self.impostor_renderer.range
    }

    /// Get total view distance (render_distance + impostor extra)
    pub fn total_view_distance(&self) -> i32 {
        self.render_distance + self.impostor_renderer.range.extra_chunks()
    }

    // === Cloud Settings ===

    /// Get current cloud quality
    pub fn cloud_quality(&self) -> CloudQuality {
        self.clouds.quality()
    }

    /// Cycle cloud quality: Off -> Low -> Medium -> High -> Ultra -> Off
    pub fn cycle_cloud_quality(&mut self) -> CloudQuality {
        self.clouds.cycle_quality();
        let quality = self.clouds.quality();
        tracing::info!("Cloud Quality: {}", quality.as_str());
        quality
    }

    /// Cycle cloud quality backwards
    pub fn cycle_cloud_quality_prev(&mut self) -> CloudQuality {
        self.clouds.cycle_quality_prev();
        self.clouds.quality()
    }

    /// Set cloud quality directly
    pub fn set_cloud_quality(&mut self, quality: CloudQuality) {
        self.clouds.set_quality(quality);
    }

    /// Set cloud quality by index (0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra)
    pub fn set_cloud_quality_index(&mut self, index: usize) {
        let quality = match index {
            0 => CloudQuality::Off,
            1 => CloudQuality::Low,
            2 => CloudQuality::Medium,
            3 => CloudQuality::High,
            _ => CloudQuality::Ultra,
        };
        self.clouds.set_quality(quality);
    }

    /// Get cloud coverage (0.0 - 1.0)
    pub fn cloud_coverage(&self) -> f32 {
        self.clouds.coverage
    }

    /// Set cloud coverage (0.0 - 1.0)
    pub fn set_cloud_coverage(&mut self, coverage: f32) {
        self.clouds.set_coverage(coverage);
    }

    /// Get cloud speed (0.0 - 1.0)
    pub fn cloud_speed(&self) -> f32 {
        self.clouds.speed
    }

    /// Set cloud speed (0.0 - 1.0)
    pub fn set_cloud_speed(&mut self, speed: f32) {
        self.clouds.set_speed(speed);
    }

    /// Get cloud shadows enabled
    pub fn cloud_shadows_enabled(&self) -> bool {
        self.clouds.shadows_enabled
    }

    /// Set cloud shadows enabled
    pub fn set_cloud_shadows_enabled(&mut self, enabled: bool) {
        self.clouds.set_shadows_enabled(enabled);
    }

    /// Toggle cloud shadows
    pub fn toggle_cloud_shadows(&mut self) -> bool {
        self.clouds.shadows_enabled = !self.clouds.shadows_enabled;
        self.clouds.shadows_enabled
    }

    // === Scenery Settings ===

    /// Get current scenery quality
    pub fn scenery_quality(&self) -> scenery_renderer::SceneryQuality {
        self.scenery_renderer.quality()
    }

    /// Cycle scenery quality: Off -> Low -> Medium -> High -> Ultra -> Off
    pub fn cycle_scenery_quality(&mut self) -> scenery_renderer::SceneryQuality {
        let new_quality = self.scenery_renderer.quality().next();
        self.scenery_renderer.set_quality(new_quality);
        tracing::info!("Scenery Quality: {}", new_quality.as_str());
        new_quality
    }

    /// Cycle scenery quality backwards
    pub fn cycle_scenery_quality_prev(&mut self) -> scenery_renderer::SceneryQuality {
        let new_quality = self.scenery_renderer.quality().prev();
        self.scenery_renderer.set_quality(new_quality);
        new_quality
    }

    /// Set scenery quality directly
    pub fn set_scenery_quality(&mut self, quality: scenery_renderer::SceneryQuality) {
        self.scenery_renderer.set_quality(quality);
    }

    /// Set scenery quality by index (0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra)
    pub fn set_scenery_quality_index(&mut self, index: usize) {
        let quality = match index {
            0 => scenery_renderer::SceneryQuality::Off,
            1 => scenery_renderer::SceneryQuality::Low,
            2 => scenery_renderer::SceneryQuality::Medium,
            3 => scenery_renderer::SceneryQuality::High,
            _ => scenery_renderer::SceneryQuality::Ultra,
        };
        self.scenery_renderer.set_quality(quality);
    }

    /// Check if water refraction is enabled
    pub fn is_refraction_enabled(&self) -> bool {
        self.water_system.quality() != WaterQuality::Off
    }

    /// Get current render distance
    pub fn render_distance(&self) -> i32 {
        self.render_distance
    }

    /// Cycle render distance: 8 -> 12 -> 16 -> ... -> 32 -> 8
    pub fn cycle_render_distance(&mut self) -> i32 {
        self.render_distance = if self.render_distance >= 32 {
            8
        } else {
            self.render_distance + 4
        };
        self.render_distance
    }

    /// Cycle render distance backwards: 32 -> 28 -> ... -> 8 -> 32
    pub fn cycle_render_distance_prev(&mut self) -> i32 {
        self.render_distance = if self.render_distance <= 8 {
            32
        } else {
            self.render_distance - 4
        };
        self.render_distance
    }

    /// Set render distance directly
    pub fn set_render_distance(&mut self, distance: i32) {
        self.render_distance = distance.clamp(8, 32);
    }

    /// Get current shadow distance
    pub fn shadow_distance(&self) -> i32 {
        self.shadow_distance
    }

    /// Increase shadow distance (max = render distance)
    pub fn increase_shadow_distance(&mut self) -> i32 {
        self.shadow_distance = (self.shadow_distance + 4).min(self.render_distance);
        self.shadow_distance
    }

    /// Decrease shadow distance (min 4)
    pub fn decrease_shadow_distance(&mut self) -> i32 {
        self.shadow_distance = (self.shadow_distance - 4).max(4);
        self.shadow_distance
    }

    /// Get current display mode
    pub fn display_mode(&self) -> DisplayMode {
        self.display_mode
    }

    /// Toggle display mode: Windowed -> Borderless -> Fullscreen -> Windowed
    /// Sets pending flag for app.rs to apply to window
    pub fn toggle_display_mode(&mut self) -> DisplayMode {
        self.display_mode = self.display_mode.next();
        self.display_mode_pending = true;
        self.display_mode
    }

    /// Set display mode (used when applying mode from window)
    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.display_mode = mode;
    }

    /// Cycle display mode backwards: Windowed <- Fullscreen <- Borderless <- Windowed
    pub fn cycle_display_mode_prev(&mut self) -> DisplayMode {
        self.display_mode = match self.display_mode {
            DisplayMode::Windowed => DisplayMode::Fullscreen,
            DisplayMode::Borderless => DisplayMode::Windowed,
            DisplayMode::Fullscreen => DisplayMode::Borderless,
        };
        self.display_mode_pending = true;
        self.display_mode
    }

    /// Set display mode by index (0=Windowed, 1=Borderless, 2=Fullscreen)
    pub fn set_display_mode_index(&mut self, index: usize) {
        self.display_mode = match index {
            0 => DisplayMode::Windowed,
            1 => DisplayMode::Borderless,
            _ => DisplayMode::Fullscreen,
        };
        self.display_mode_pending = true;
    }

    /// Check if display mode change is pending
    pub fn is_display_mode_pending(&self) -> bool {
        self.display_mode_pending
    }

    /// Clear the pending display mode flag
    pub fn clear_display_mode_pending(&mut self) {
        self.display_mode_pending = false;
    }

    /// Apply settings from a Settings struct
    pub fn apply_settings(&mut self, settings: &crate::settings::Settings) {
        self.render_distance = settings.render_distance;
        self.shadow_mode = settings.shadow_mode;
        self.bloom_enabled = settings.bloom_enabled;
        self.volumetrics.set_enabled(settings.volumetrics_enabled);
        if self.mdi_count_available {
            self.use_gpu_culling = settings.gpu_culling;
        }
        self.voxel_gi.set_use_gpu_fill(settings.gpu_voxel_fill);
        self.cycle_speed = settings.cycle_speed;
        self.debug_overlay.set_enabled(settings.debug_overlay);
        self.display_mode = settings.display_mode;
        self.voxel_gi.set_update_threshold(settings.shadow_update_step);
        self.torch_shadows = settings.torch_shadows;
        // Apply SSAO quality setting
        let ssao_quality = match settings.ssao_quality.to_uppercase().as_str() {
            "OFF" => SsaoQuality::Off,
            "LOW" => SsaoQuality::Low,
            "MEDIUM" => SsaoQuality::Medium,
            "HIGH" => SsaoQuality::High,
            "ULTRA" => SsaoQuality::Ultra,
            _ => SsaoQuality::Medium,
        };
        self.ssao.set_quality(&self.device, self.size.width, self.size.height, ssao_quality);

        // Apply DoF quality setting
        let dof_quality = match settings.dof_quality.to_uppercase().as_str() {
            "OFF" => DofQuality::Off,
            "LOW" => DofQuality::Low,
            "MEDIUM" => DofQuality::Medium,
            "HIGH" => DofQuality::High,
            "ULTRA" => DofQuality::Ultra,
            _ => DofQuality::Low,
        };
        self.dof.set_quality(dof_quality);

        // Apply DoF sub-settings
        self.dof.params.manual_focus_distance = settings.dof_focus_distance;
        self.dof.params.aperture = settings.dof_blur_strength;
        self.dof.params.focus_transition = settings.dof_focus_range;

        // Apply water refraction quality setting
        let refraction_quality = match settings.refraction_quality.to_uppercase().as_str() {
            "OFF" => RefractionQuality::Off,
            "LOW" => RefractionQuality::Low,
            "MEDIUM" => RefractionQuality::Medium,
            "HIGH" => RefractionQuality::High,
            "ULTRA" => RefractionQuality::Ultra,
            _ => RefractionQuality::Medium,
        };
        self.set_refraction_quality(refraction_quality);

        // Apply extended view (impostor) range setting
        let impostor_range = ImpostorRange::from_str(&settings.impostor_quality);
        self.impostor_renderer.range = impostor_range;

        // Apply cloud settings
        let cloud_quality = CloudQuality::from_str(&settings.cloud_quality);
        self.clouds.set_quality(cloud_quality);
        self.clouds.set_coverage(settings.cloud_coverage);
        self.clouds.set_speed(settings.cloud_speed);
        self.clouds.set_shadows_enabled(settings.cloud_shadows);

        // Apply scenery settings
        let scenery_quality = match settings.scenery_quality.to_uppercase().as_str() {
            "OFF" => scenery_renderer::SceneryQuality::Off,
            "LOW" => scenery_renderer::SceneryQuality::Low,
            "MEDIUM" => scenery_renderer::SceneryQuality::Medium,
            "HIGH" => scenery_renderer::SceneryQuality::High,
            "ULTRA" => scenery_renderer::SceneryQuality::Ultra,
            _ => scenery_renderer::SceneryQuality::Medium,
        };
        self.scenery_renderer.set_quality(scenery_quality);

        // Apply audio settings
        self.audio_master = settings.audio_master;
        self.audio_sfx = settings.audio_sfx;
        self.audio_ambient = settings.audio_ambient;
    }

    /// Collect current settings into a Settings struct
    pub fn collect_settings(&self, windowed_size: (u32, u32)) -> crate::settings::Settings {
        crate::settings::Settings {
            render_distance: self.render_distance,
            shadow_mode: self.shadow_mode,
            bloom_enabled: self.bloom_enabled,
            volumetrics_enabled: self.volumetrics.enabled,
            gpu_culling: self.use_gpu_culling,
            gpu_voxel_fill: self.voxel_gi.is_gpu_fill_enabled(),
            cycle_speed: self.cycle_speed,
            debug_overlay: self.debug_overlay.is_enabled(),
            display_mode: self.display_mode,
            windowed_width: windowed_size.0,
            windowed_height: windowed_size.1,
            shadow_update_step: self.voxel_gi.get_update_threshold(),
            torch_shadows: self.torch_shadows,
            // Audio settings
            audio_master: self.audio_master,
            audio_sfx: self.audio_sfx,
            audio_ambient: self.audio_ambient,
            ssao_quality: self.ssao.quality.as_str().to_string(),
            dof_quality: self.dof.quality.as_str().to_string(),
            dof_focus_distance: self.dof.params.manual_focus_distance,
            dof_blur_strength: self.dof.params.aperture,
            dof_focus_range: self.dof.params.focus_transition,
            refraction_quality: self.refraction_quality().as_str().to_string(),
            impostor_quality: self.impostor_renderer.range.as_str().to_string(),
            // Cloud settings
            cloud_quality: self.clouds.quality().as_str().to_string(),
            cloud_coverage: self.clouds.coverage,
            cloud_speed: self.clouds.speed,
            cloud_shadows: self.clouds.shadows_enabled,
            // Scenery settings
            scenery_quality: self.scenery_renderer.quality().as_str().to_string(),
        }
    }

    /// Create a settings snapshot for profiler logging
    pub fn settings_snapshot(&self) -> crate::utils::profiler::SettingsSnapshot {
        crate::utils::profiler::SettingsSnapshot {
            render_distance: self.render_distance,
            ssao_quality: self.ssao.quality.as_str().to_string(),
            dof_quality: self.dof.quality.as_str().to_string(),
            water_quality: self.refraction_quality().as_str().to_string(),
            impostor_quality: self.impostor_renderer.range.as_str().to_string(),
            bloom_enabled: self.bloom_enabled,
            volumetrics_enabled: self.volumetrics.enabled,
            gpu_culling: self.use_gpu_culling,
            shadow_mode: if self.shadow_mode > 0.5 { "SOFT".to_string() } else { "HARD".to_string() },
            torch_shadows: self.torch_shadows,
            resolution: (self.size.width, self.size.height),
        }
    }

    /// Load moon texture from assets/moon-1.png for sky rendering
    fn load_moon_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        // Try to load moon texture from file
        let moon_data = Self::load_moon_image().unwrap_or_else(|| {
            log::warn!("Failed to load moon texture, using procedural fallback");
            Self::generate_procedural_moon()
        });

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Moon Texture"),
            size: wgpu::Extent3d {
                width: moon_data.0,
                height: moon_data.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &moon_data.2,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * moon_data.0),
                rows_per_image: Some(moon_data.1),
            },
            wgpu::Extent3d {
                width: moon_data.0,
                height: moon_data.1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use nearest filtering to preserve pixel art crispness
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Moon Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,  // Crisp pixels
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        log::info!("Moon texture loaded: {}x{}", moon_data.0, moon_data.1);
        (texture, view, sampler)
    }

    /// Try to load moon image from file
    fn load_moon_image() -> Option<(u32, u32, Vec<u8>)> {
        use std::path::Path;

        // Get the executable's directory for asset resolution
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // Try multiple paths
        let possible_paths = [
            Path::new("assets/moon-1.png").to_path_buf(),
            exe_dir.join("assets/moon-1.png"),
            exe_dir.join("..").join("..").join("..").join("assets/moon-1.png"),
        ];

        for path in &possible_paths {
            if let Ok(img) = image::open(path) {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                log::info!("Loaded moon texture from: {}", path.display());
                return Some((width, height, rgba.into_raw()));
            }
        }

        log::warn!("Moon texture not found. Tried paths:");
        for path in &possible_paths {
            log::warn!("  - {}", path.display());
        }
        None
    }

    /// Generate a procedural moon texture as fallback
    fn generate_procedural_moon() -> (u32, u32, Vec<u8>) {
        let size = 128u32;
        let mut data = vec![0u8; (size * size * 4) as usize];

        let center = size as f32 / 2.0;
        let radius = center - 2.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt();

                let idx = ((y * size + x) * 4) as usize;

                if dist < radius {
                    // Inside moon disc - warm amber color
                    let edge_factor = 1.0 - (dist / radius).powf(2.0);
                    let r = (255.0 * (0.9 + 0.1 * edge_factor)) as u8;
                    let g = (200.0 * (0.85 + 0.15 * edge_factor)) as u8;
                    let b = (140.0 * (0.7 + 0.3 * edge_factor)) as u8;

                    // Add some crater-like variation
                    let noise = ((x.wrapping_mul(7) ^ y.wrapping_mul(13)) % 30) as i32 - 15;
                    let r = (r as i32 + noise).clamp(0, 255) as u8;
                    let g = (g as i32 + noise / 2).clamp(0, 255) as u8;
                    let b = (b as i32 + noise / 3).clamp(0, 255) as u8;

                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                    data[idx + 3] = 255;
                } else {
                    // Outside moon - transparent
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                    data[idx + 3] = 0;
                }
            }
        }

        (size, size, data)
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        world: &mut WorldManager,
        _profiler: &FrameProfiler,
        target_block: Option<[i32; 3]>,
        debug_metrics: &DebugMetrics,
        held_light_emission: u8,  // Light emission of held item (0 = none, 14 = torch)
        settings_menu: &crate::settings_menu::SettingsMenu,
        inventory: &crate::inventory::Inventory,
    ) -> Result<(), wgpu::SurfaceError> {
        let render_start = std::time::Instant::now();

        // Sync meshes to chunk pool for MDI rendering
        // Only uploads meshes within render_distance; impostor chunks skip upload
        let sync_start = std::time::Instant::now();
        if self.chunk_pool.mdi_available {
            self.sync_meshes_to_pool(world, camera.position);
        }
        let sync_elapsed = sync_start.elapsed();

        // Poll GPU profiler for previous frame results (total GPU time only)
        self.gpu_profiler.poll_results(&self.device);

        // Update camera uniform with animation timing
        let uniforms_start = std::time::Instant::now();
        let current_time = self.start_time.elapsed().as_secs_f32();
        let spawn_time = 0.0;  // World starts loading at time 0

        // Check if camera is underwater
        let cam_block_x = camera.position.x.floor() as i32;
        let cam_block_y = camera.position.y.floor() as i32;
        let cam_block_z = camera.position.z.floor() as i32;
        let is_underwater = world.get_block(cam_block_x, cam_block_y, cam_block_z)
            .map(|b| b == crate::world::Block::Water)
            .unwrap_or(false);

        let camera_uniform = CameraUniform::from_camera_with_anim(camera, current_time, spawn_time, is_underwater);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Update light uniform with day/night cycle (center shadow map on camera)
        let light_uniform = LightUniform::from_camera_position_and_time(camera.position, self.time_of_day);
        let cycle = DayNightCycle::from_time(self.time_of_day);
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_uniform]));

        // Update sky uniform with day/night colors and moon phase
        let sky_uniform = SkyUniform::from_camera_and_time(camera, self.time_of_day, self.day_count);
        self.queue
            .write_buffer(&self.sky_buffer, 0, bytemuck::cast_slice(&[sky_uniform]));

        // Update volumetric uniform for light shafts
        let volumetric_uniform = volumetrics::VolumetricUniform {
            inv_view_proj: camera.view_projection_matrix().inverse().to_cols_array_2d(),
            camera_pos: camera.position.extend(1.0).to_array(),
            sun_direction: [cycle.sun_direction.x, cycle.sun_direction.y, cycle.sun_direction.z, cycle.sun_intensity],
            sun_color: [cycle.sun_color.x, cycle.sun_color.y, cycle.sun_color.z, 0.006], // a = density (very low for thin rays)
            volume_center: [camera.position.x, camera.position.y, camera.position.z, 256.0],
            screen_params: [self.size.width as f32, self.size.height as f32, 40.0, current_time],
        };
        self.queue
            .write_buffer(&self.volumetric_uniform_buffer, 0, bytemuck::cast_slice(&[volumetric_uniform]));

        // Update SSAO uniforms
        self.ssao.update_uniforms(
            &self.queue,
            camera.projection_matrix().inverse(),
            camera.projection_matrix(),
            0.1,   // near plane
            1000.0, // far plane
            self.size.width,
            self.size.height,
        );

        // Update DoF uniforms
        self.dof.update_uniforms(
            &self.queue,
            camera.projection_matrix().inverse(),
            0.1,   // near plane
            1000.0, // far plane
        );

        let uniforms_elapsed = uniforms_start.elapsed();

        // Build frustum for culling
        let frustum = Frustum::from_view_proj(camera.view_projection_matrix());

        // Get surface texture (can block waiting for vsync/present)
        let surface_start = std::time::Instant::now();
        let output = self.surface.get_current_texture()?;
        let surface_elapsed = surface_start.elapsed();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // === VOXEL GI UPDATE ===
        // Update the voxel volume texture from chunk data
        let voxel_gi_start = std::time::Instant::now();
        let chunks = world.chunks();

        // Update smoothed center for gradual shadow transitions
        // This makes the volume smoothly follow the camera instead of snapping
        let delta_time = debug_metrics.frame_time_ms / 1000.0; // Convert ms to seconds
        self.voxel_gi.update_smoothed_center(camera.position, delta_time);

        // Use GPU fill for full refresh if enabled, CPU for incremental/fallback
        if self.voxel_gi.is_gpu_fill_enabled() && self.voxel_gi.needs_gpu_refresh(camera.position) {
            // GPU compute path - clears and fills voxel volume via compute shaders
            self.voxel_gi.gpu_fill_volume(
                &self.device,
                &mut encoder,
                &self.queue,
                &chunks,
                camera.position,
            );
        }

        // Always call update_from_chunks for:
        // - Uniform updates (sun direction, colors, time of day)
        // - Incremental dirty block updates
        // - Full CPU refresh if GPU fill is disabled
        self.voxel_gi.update_from_chunks(
            &self.queue,
            camera.position,
            &chunks,
            cycle.sun_direction,
            cycle.sun_color,
            cycle.ambient,
            self.time_of_day,
            self.shadow_mode,
            self.shadow_debug,
            held_light_emission,
            self.torch_shadows,
            (self.total_view_distance() * CHUNK_SIZE_I32) as f32,
        );
        let voxel_gi_elapsed = voxel_gi_start.elapsed();

        // === SCENERY INSTANCE COLLECTION ===
        // Collect vegetation instances from visible chunks
        {
            let chunks_read = chunks.read();
            let visible_chunks: Vec<(crate::world::ChunkPos, &crate::world::Chunk)> = chunks_read
                .iter()
                .filter(|(pos, chunk)| {
                    if !chunk.has_props() {
                        return false;
                    }
                    let origin = pos.world_origin();
                    let min = origin;
                    let max = origin + glam::Vec3::splat(16.0);
                    frustum.test_aabb(min, max)
                })
                .map(|(pos, chunk)| (*pos, chunk.as_ref()))
                .collect();

            self.scenery_renderer.collect_instances(&visible_chunks, camera.position);
        }

        // Upload scenery instances and uniforms
        let wind_dir = glam::Vec3::new(1.0, 0.0, 0.3).normalize();
        self.scenery_renderer.upload(&self.queue, current_time, wind_dir, 1.0);

        // GPU per-pass timing with double-buffered staging buffers
        self.gpu_profiler.begin_frame();

        // Reset stats
        let mut chunks_rendered = 0u32;
        let mut chunks_culled = 0u32;
        let mut triangles = 0u32;
        // Shadow pass removed - using voxel GI with DDA ray marching instead
        let shadow_chunks_rendered = 0u32;
        let shadow_chunks_culled = 0u32;
        let shadow_elapsed = std::time::Duration::ZERO;
        let main_start = std::time::Instant::now();
        // Timing for spike detection (assigned inside render pass)
        let cull_elapsed: std::time::Duration;
        let mdi_prep_elapsed: std::time::Duration;

        // ═══════════════════════════════════════════════════════════════════════════
        // GPU FRUSTUM CULLING (compute pass - runs before render pass)
        // Updates bounds buffer only when chunks change, then dispatches cull shader
        // ═══════════════════════════════════════════════════════════════════════════
        if self.use_gpu_culling && self.chunk_pool.mdi_available {
            // Only rebuild bounds buffer when chunks have been added/removed
            // This eliminates per-frame CPU iteration over all chunks
            if self.chunk_pool.is_bounds_dirty() {
                self.gpu_culler.update_bounds(&self.queue, self.chunk_pool.slots());
                self.chunk_pool.clear_bounds_dirty();
            }

            // Extract frustum planes for GPU
            let frustum_planes = frustum.planes_as_array();

            // Run GPU culling compute shader with visibility caching
            // Skips compute dispatch if camera hasn't moved significantly
            self.gpu_culler.cull_with_cache(
                &self.device,
                &self.queue,
                &mut encoder,
                &frustum_planes,
                crate::constants::CHUNK_SIZE_F32,
                camera.position,
                camera.forward(),
            );
        }

        // Pre-generate menu lines if menu is open (before render pass borrow)
        let menu_lines = if settings_menu.is_open {
            Some(settings_menu.generate_lines(self))
        } else {
            None
        };

        // Build inventory UI vertices (before render pass borrow)
        self.inventory_ui.build_vertices(&self.queue, inventory);

        // === COLLECT AND UPLOAD IMPOSTOR DATA ===
        // Gather distant chunks that should render as impostors
        // === COLLECT IMPOSTOR DATA ===
        // Impostors render chunks BEYOND render_distance (extended view)
        // Chunks within render_distance use full mesh, beyond use impostors
        if self.impostor_renderer.range.enabled() {
            let chunk_size = crate::constants::CHUNK_SIZE_F32;
            let render_dist_blocks = self.render_distance as f32 * chunk_size;
            let total_dist_blocks = self.total_view_distance() as f32 * chunk_size;
            let mut impostor_instances = Vec::with_capacity(512);

            for chunk_mesh in world.visible_meshes() {
                // Skip empty chunks (no color data)
                if chunk_mesh.average_color[3] == 0 {
                    continue;
                }

                // Calculate distance in blocks
                let chunk_center = chunk_mesh.pos.center();
                let dist_blocks = (chunk_center - camera.position).length();

                // Chunks beyond render_distance but within total view distance = impostors
                if dist_blocks > render_dist_blocks && dist_blocks <= total_dist_blocks {
                    // Quick frustum check
                    let chunk_min = chunk_mesh.pos.world_origin();
                    let chunk_max = chunk_min + glam::Vec3::splat(chunk_size);
                    if frustum.test_aabb(chunk_min, chunk_max) {
                        impostor_instances.push(ImpostorInstance::new(
                            chunk_mesh.pos,
                            chunk_mesh.average_color,
                        ));
                    }
                }
            }

            // Update impostor uniforms and upload instances
            self.impostor_renderer.update_uniforms(
                &self.queue,
                camera.view_projection_matrix(),
                camera.position,
                self.size.width as f32,
                self.size.height as f32,
                cycle.sun_intensity,  // Use sun intensity as sky brightness
            );
            self.impostor_renderer.upload_instances(&self.queue, &impostor_instances);
        }

        // === MAIN RENDER PASS (render to scene texture for bloom) ===
        // GPU timestamp: start main pass
        if let Some(idx) = self.gpu_profiler.begin_pass("Main") {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_texture_view,  // Render to scene texture for bloom
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.75,
                            g: 0.82,
                            b: 0.92,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render sky first (at far depth, will be behind everything)
            render_pass.set_pipeline(&self.sky_pipeline);
            render_pass.set_bind_group(0, &self.sky_bind_group, &[]);
            render_pass.draw(0..3, 0..1);  // Fullscreen triangle

            // Set up chunk pipeline (uses voxel GI with DDA ray marching + VCT)
            render_pass.set_pipeline(&self.chunk_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_bind_group(2, &self.light_bind_group, &[]);
            render_pass.set_bind_group(3, self.voxel_gi.bind_group(), &[]);

            // ═══════════════════════════════════════════════════════════════════════════
            // FRUSTUM CULLING: Either GPU (compute shader) or CPU path
            // GPU path: compute shader tests AABBs and writes indirect commands
            // CPU path: iterate chunks on CPU and build command list
            // ═══════════════════════════════════════════════════════════════════════════
            let cull_start = std::time::Instant::now();
            let mut pool_chunks: Vec<crate::world::ChunkPos> = Vec::with_capacity(3000);
            let mut individual_chunks: Vec<&crate::world::ChunkMesh> = Vec::with_capacity(500);

            if self.use_gpu_culling && self.chunk_pool.mdi_available {
                // GPU MODE: Only need to find non-pool chunks for CPU fallback rendering
                // Pool chunks are handled entirely by GPU compute shader
                for chunk_mesh in world.visible_meshes() {
                    // Quick check: is this chunk NOT in pool but has GPU mesh?
                    if chunk_mesh.gpu_mesh.is_some() && !self.chunk_pool.has_chunk(&chunk_mesh.pos) {
                        // Non-pool chunk needs CPU frustum test
                        let chunk_min = chunk_mesh.pos.world_origin();
                        let chunk_max = chunk_min + glam::Vec3::splat(crate::constants::CHUNK_SIZE_F32);
                        if frustum.test_aabb(chunk_min, chunk_max) {
                            individual_chunks.push(chunk_mesh);
                        } else {
                            chunks_culled += 1;
                        }
                    }
                }
            } else {
                // CPU MODE: Test all chunks with CPU frustum culling
                for chunk_mesh in world.visible_meshes() {
                    let in_pool = self.chunk_pool.mdi_available && self.chunk_pool.get_slot(&chunk_mesh.pos).is_some();
                    let has_gpu_mesh = chunk_mesh.gpu_mesh.is_some();

                    // Skip non-renderable chunks
                    if !in_pool && !has_gpu_mesh {
                        continue;
                    }

                    // CPU frustum test
                    let chunk_min = chunk_mesh.pos.world_origin();
                    let chunk_max = chunk_min + glam::Vec3::splat(crate::constants::CHUNK_SIZE_F32);

                    if !frustum.test_aabb(chunk_min, chunk_max) {
                        chunks_culled += 1;
                        continue;
                    }

                    // Chunk is visible
                    if in_pool {
                        if let Some(slot) = self.chunk_pool.get_slot(&chunk_mesh.pos) {
                            pool_chunks.push(chunk_mesh.pos);
                            triangles += slot.index_count / 3;
                        }
                    } else if has_gpu_mesh {
                        individual_chunks.push(chunk_mesh);
                    }
                }
            }
            cull_elapsed = cull_start.elapsed();

            // MDI pass: render chunks from pool
            let mdi_prep_start = std::time::Instant::now();

            // CPU path: prepare indirect draws BEFORE setting render pass buffers (avoid borrow conflict)
            let cpu_draw_count = if self.chunk_pool.mdi_available && !self.use_gpu_culling && !pool_chunks.is_empty() {
                self.chunk_pool.prepare_indirect_draws(&self.queue, &pool_chunks)
            } else {
                0
            };

            if self.chunk_pool.mdi_available {
                render_pass.set_vertex_buffer(0, self.chunk_pool.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.chunk_pool.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                if self.use_gpu_culling {
                    // GPU path: use compute shader populated indirect buffer with count from GPU
                    let max_draws = self.gpu_culler.chunk_count();
                    if max_draws > 0 {
                        render_pass.multi_draw_indexed_indirect_count(
                            &self.chunk_pool.indirect_buffer,
                            0,
                            self.gpu_culler.count_buffer(),
                            0,
                            max_draws,
                        );
                        // Note: actual rendered count comes from GPU, we use max as estimate
                        chunks_rendered = max_draws;
                    }
                } else if cpu_draw_count > 0 {
                    // CPU path: use CPU-built command list
                    render_pass.multi_draw_indexed_indirect(
                        &self.chunk_pool.indirect_buffer,
                        0,
                        cpu_draw_count,
                    );
                    chunks_rendered += cpu_draw_count;
                }
            }
            mdi_prep_elapsed = mdi_prep_start.elapsed();

            // Individual pass: render chunks not in pool (overflow, oversized, etc.)
            for chunk_mesh in individual_chunks {
                if let Some(gpu_mesh) = &chunk_mesh.gpu_mesh {
                    render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        gpu_mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                    triangles += gpu_mesh.index_count / 3;
                    chunks_rendered += 1;
                }
            }

            // === IMPOSTOR RENDER PASS ===
            // Render distant chunks as single colored quads
            if self.impostor_renderer.range.enabled() {
                self.impostor_renderer.render(&mut render_pass);
            }

            // === ITEM ENTITY RENDER PASS ===
            // Render dropped items as 3D cubes
            self.item_renderer.render(
                &mut render_pass,
                &self.camera_bind_group,
                &self.texture_bind_group,
            );

            // === SCENERY RENDER PASS ===
            // Render vegetation (grass, flowers, rocks)
            let scenery_count = self.scenery_renderer.render(
                &mut render_pass,
                &self.camera_bind_group,
            );
            triangles += scenery_count * 10; // Rough estimate of triangles per instance

            // === PROP RENDER PASS ===
            // Render voxel props (chests, etc.)
            self.prop_renderer.render(
                &mut render_pass,
                &self.camera_bind_group,
            );

            // End opaque pass to allow scene texture copy for water refraction
        }
        // GPU timestamp: end main pass
        if let Some(idx) = self.gpu_profiler.end_pass() {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }

        // === COPY SCENE FOR WATER REFRACTION ===
        // Copy the current scene (before water) so water shader can sample it for refraction
        if self.water_system.quality() != WaterQuality::Off {
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.scene_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.refraction_source_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );

            // Update water system bind group with current textures
            self.water_system.update_bind_group(
                &self.device,
                &self.depth_texture,
                &self.refraction_source_view,
            );

            // Update water uniforms using day/night cycle values
            let inv_view = camera.view_matrix().inverse();
            let inv_proj = camera.projection_matrix().inverse();
            self.water_system.update_uniforms(
                &self.queue,
                camera.view_projection_matrix(),
                inv_view,
                inv_proj,
                camera.position,
                current_time,
                cycle.sun_direction,
                cycle.sun_intensity,
                cycle.sun_color,
                cycle.ambient,
                cycle.sun_intensity.clamp(0.0, 1.0),  // Use sun intensity as brightness
                self.size.width as f32,
                self.size.height as f32,
            );

            // Set water levels (just sea level for now)
            self.water_system.set_water_levels(&[(crate::constants::WATER_SEA_LEVEL as f32, 0.0)]);
        }

        // === WATER RENDER PASS (Ray-marched fullscreen) ===
        // Renders water as fullscreen ray-plane intersection
        if self.water_system.quality() != WaterQuality::Off {
            // GPU timestamp: start water pass
            if let Some(idx) = self.gpu_profiler.begin_pass("Water") {
                self.gpu_profiler.write_timestamp(&mut encoder, idx);
            }
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Water Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep the opaque scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, // No depth write - water samples depth texture
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.water_system.render(&mut render_pass, &self.camera_bind_group);
            // GPU timestamp: end water pass (inside block to ensure pass is dropped first)
            drop(render_pass);
            if let Some(idx) = self.gpu_profiler.end_pass() {
                self.gpu_profiler.write_timestamp(&mut encoder, idx);
            }
        }

        // === CLOUD RENDER PASS (3D Voxel Geometry) ===
        // Renders 3D voxel clouds as geometry with depth testing
        if self.clouds.quality() != CloudQuality::Off {
            // Update cloud mesh and uniforms
            let delta_time = debug_metrics.frame_time_ms / 1000.0;
            self.clouds.update(
                &self.queue,
                camera.position,
                current_time,
                delta_time,
                camera.view_projection_matrix(),
                cycle.sun_direction,
                cycle.sun_intensity,
                cycle.sun_color,
                cycle.ambient,
            );

            // GPU timestamp: start cloud pass
            if let Some(idx) = self.gpu_profiler.begin_pass("Clouds") {
                self.gpu_profiler.write_timestamp(&mut encoder, idx);
            }

            // Render clouds as geometry with depth testing
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Cloud Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                self.clouds.render(&mut render_pass);
            }

            // GPU timestamp: end cloud pass
            if let Some(idx) = self.gpu_profiler.end_pass() {
                self.gpu_profiler.write_timestamp(&mut encoder, idx);
            }
        }

        // === BLOCK OUTLINE PASS ===
        // Render block outline if targeting a block (needs depth testing)
        if let Some([bx, by, bz]) = target_block {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Outline Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Update outline vertices for this block position
            let x = bx as f32;
            let y = by as f32;
            let z = bz as f32;
            // Slight expansion to prevent z-fighting
            let e = 0.002;
            let x0 = x - e;
            let y0 = y - e;
            let z0 = z - e;
            let x1 = x + 1.0 + e;
            let y1 = y + 1.0 + e;
            let z1 = z + 1.0 + e;

            // 12 edges of a cube
            let outline_vertices = [
                // Bottom face edges
                LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x1, y0, z0] },
                LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y0, z1] },
                LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x0, y0, z1] },
                LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y0, z0] },
                // Top face edges
                LineVertex { position: [x0, y1, z0] }, LineVertex { position: [x1, y1, z0] },
                LineVertex { position: [x1, y1, z0] }, LineVertex { position: [x1, y1, z1] },
                LineVertex { position: [x1, y1, z1] }, LineVertex { position: [x0, y1, z1] },
                LineVertex { position: [x0, y1, z1] }, LineVertex { position: [x0, y1, z0] },
                // Vertical edges
                LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x0, y1, z0] },
                LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y1, z0] },
                LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x1, y1, z1] },
                LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y1, z1] },
            ];
            self.queue.write_buffer(&self.outline_vertex_buffer, 0, bytemuck::cast_slice(&outline_vertices));

            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.outline_vertex_buffer.slice(..));
            render_pass.draw(0..24, 0..1);
        }

        // === SSAO (Screen-Space Ambient Occlusion) ===
        // Compute AO from depth buffer and apply to scene
        // This runs after main pass (needs depth) and before volumetrics/bloom
        // GPU timestamp: start SSAO
        if let Some(idx) = self.gpu_profiler.begin_pass("SSAO") {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }
        self.ssao.create_bind_groups(&self.device, &self.depth_texture);
        self.ssao.render(&mut encoder, &self.scene_texture_view);
        // GPU timestamp: end SSAO
        if let Some(idx) = self.gpu_profiler.end_pass() {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }

        // === VOLUMETRIC LIGHT SHAFTS ===
        // Render god rays using raymarching through voxel volume
        // This runs at half resolution and composites onto the scene
        // GPU timestamp: start Volumetrics
        if let Some(idx) = self.gpu_profiler.begin_pass("Volumetrics") {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }
        self.volumetrics.create_bind_group(
            &self.device,
            &self.depth_texture,
            &self.volumetric_uniform_buffer,
        );
        self.volumetrics.render(
            &mut encoder,
            &self.scene_texture_view,  // Composite onto scene (before bloom)
            self.voxel_gi.bind_group(),
        );
        // GPU timestamp: end Volumetrics
        if let Some(idx) = self.gpu_profiler.end_pass() {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }

        // === DEPTH OF FIELD ===
        // Apply DoF effect: separate near/far blur layers to prevent color bleeding
        // This runs after volumetrics (needs lit scene) and before UI/bloom
        // GPU timestamp: start DoF
        if let Some(idx) = self.gpu_profiler.begin_pass("DoF") {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }
        self.dof.create_bind_groups(&self.device, &self.depth_texture, &self.scene_texture_view);
        self.dof.render(&mut encoder, &self.scene_texture_view);
        // GPU timestamp: end DoF
        if let Some(idx) = self.gpu_profiler.end_pass() {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }

        // === UI RENDER PASS (after DoF to prevent blur on text) ===
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep the post-processed scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // UI pipelines were created with depth format, so we need to provide it
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render crosshair (UI, no depth) - hide when inventory is open
            if !inventory.is_open {
                render_pass.set_pipeline(&self.ui_pipeline);
                render_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
                render_pass.draw(0..12, 0..1);
            }

            // Render inventory UI (hotbar always, full inventory when open)
            self.inventory_ui.render(&mut render_pass, &self.texture_bind_group);

            // Render debug overlay and optionally settings menu
            self.debug_overlay.render(
                &mut render_pass,
                &self.queue,
                debug_metrics,
                menu_lines.as_deref(),
                if menu_lines.is_some() { Some(&self.dimmer_pipeline) } else { None },
            );
        }

        // === BLOOM POST-PROCESSING ===
        // Apply bloom effect: extract bright pixels, blur, composite to output
        // If bloom disabled, just blits scene to output with tone mapping
        // GPU timestamp: start Bloom
        if let Some(idx) = self.gpu_profiler.begin_pass("Bloom") {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }
        self.bloom.render_or_blit(&mut encoder, &view, &self.scene_texture_view, &self.device, self.bloom_enabled);
        // GPU timestamp: end Bloom
        if let Some(idx) = self.gpu_profiler.end_pass() {
            self.gpu_profiler.write_timestamp(&mut encoder, idx);
        }

        let main_elapsed = main_start.elapsed();

        // Resolve GPU timestamps
        self.gpu_profiler.resolve(&mut encoder);

        let submit_start = std::time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        let submit_elapsed = submit_start.elapsed();

        // Begin async readback of GPU timing
        self.gpu_profiler.begin_readback();

        let render_total = render_start.elapsed();

        // Log spike breakdown when render exceeds 5ms
        if render_total.as_secs_f64() * 1000.0 > 5.0 {
            tracing::warn!(
                "RENDER SPIKE {:.1}ms: sync={:.1}ms surface={:.1}ms voxel_gi={:.1}ms main={:.1}ms submit={:.1}ms | {} chunks",
                render_total.as_secs_f64() * 1000.0,
                sync_elapsed.as_secs_f64() * 1000.0,
                surface_elapsed.as_secs_f64() * 1000.0,
                voxel_gi_elapsed.as_secs_f64() * 1000.0,
                main_elapsed.as_secs_f64() * 1000.0,
                submit_elapsed.as_secs_f64() * 1000.0,
                chunks_rendered,
            );
        }

        // Combine CPU timing (sync, voxel_gi, cull, submit) with GPU pass timing
        let mut pass_times = vec![
            ("sync".to_string(), sync_elapsed.as_secs_f64() * 1000.0),
            ("voxel_gi".to_string(), voxel_gi_elapsed.as_secs_f64() * 1000.0),
            ("cull".to_string(), cull_elapsed.as_secs_f64() * 1000.0),
            ("submit".to_string(), submit_elapsed.as_secs_f64() * 1000.0),
        ];
        // Add GPU per-pass timing (Main, Water, SSAO, Volumetrics, DoF, Bloom)
        let gpu_pass_times = self.gpu_profiler.pass_times_vec();
        if !gpu_pass_times.is_empty() {
            pass_times.push(("---GPU---".to_string(), 0.0)); // Separator
            pass_times.extend(gpu_pass_times);
        }

        // Update stats
        self.stats = RenderStats {
            chunks_rendered,
            chunks_culled,
            impostors_rendered: self.impostor_renderer.instance_count(),
            triangles,
            gpu_time_ms: self.gpu_profiler.avg_gpu_time_ms(),
            shadow_chunks_rendered,
            shadow_chunks_culled,
            shadow_pass_ms: shadow_elapsed.as_secs_f64() * 1000.0,
            main_pass_ms: main_elapsed.as_secs_f64() * 1000.0,
            voxel_gi_ms: voxel_gi_elapsed.as_secs_f64() * 1000.0,
            pass_times,
        };

        Ok(())
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Update item entity renderer with current items
    pub fn update_item_entities(&mut self, items: impl Iterator<Item = ItemInstanceData>) {
        let time = self.start_time.elapsed().as_secs_f32();
        self.item_renderer.update(&self.queue, items, time);
    }

    /// Update prop instances for rendering
    pub fn update_props(&mut self, name: &str, instances: &[PropInstance]) {
        self.prop_renderer.update_instances(&self.queue, name, instances);
    }

    /// Check if a prop type is loaded
    pub fn has_prop(&self, name: &str) -> bool {
        self.prop_renderer.has_prop(name)
    }

    /// Upload a chunk mesh to the buffer pool
    pub fn upload_chunk_to_pool(&mut self, mesh: &crate::world::ChunkMesh) {
        self.chunk_pool.upload_chunk(&self.queue, mesh);
    }

    /// Remove a chunk from the buffer pool
    pub fn remove_chunk_from_pool(&mut self, pos: crate::world::ChunkPos) {
        self.chunk_pool.remove_chunk(pos);
    }

    /// Check if multi-draw indirect is available
    pub fn is_mdi_available(&self) -> bool {
        self.chunk_pool.mdi_available
    }

    /// Check if a chunk is already in the buffer pool
    pub fn is_chunk_in_pool(&self, pos: &crate::world::ChunkPos) -> bool {
        self.chunk_pool.has_chunk(pos)
    }

    /// Sync all visible meshes to the chunk pool for MDI rendering
    /// Only uploads meshes within render_distance (not impostor range)
    /// Chunks beyond render_distance are rendered as impostors, not uploaded to GPU
    pub fn sync_meshes_to_pool(&mut self, world: &mut crate::world::WorldManager, camera_pos: glam::Vec3) {
        let chunk_size = crate::constants::CHUNK_SIZE_F32;
        let render_dist_blocks = self.render_distance as f32 * chunk_size;

        // FIRST: Upload urgent meshes (player block changes) WITHOUT throttling
        // These need to appear immediately for responsive gameplay
        let urgent = world.take_urgent_uploads();
        for pos in &urgent {
            if let Some(mesh) = world.visible_meshes().find(|m| &m.pos == pos) {
                // Only upload if within render distance
                let dist = (mesh.pos.center() - camera_pos).length();
                if dist <= render_dist_blocks {
                    self.chunk_pool.upload_chunk(&self.queue, mesh);
                }
            }
        }

        // SECOND: Upload pending meshes from worker threads (throttled)
        // This is O(pending) instead of O(all_visible) - huge optimization when stable
        // 16 uploads/frame at 60fps = 960 chunks/sec, plenty fast for smooth loading
        const MAX_UPLOADS_PER_FRAME: usize = 16;
        let pending = world.take_pending_uploads(MAX_UPLOADS_PER_FRAME);

        for pos in pending {
            // Skip urgent meshes we already uploaded
            if urgent.contains(&pos) {
                continue;
            }

            // Find the mesh for this position
            if let Some(mesh) = world.visible_meshes().find(|m| m.pos == pos) {
                if mesh.vertices.is_empty() {
                    continue;
                }

                // Skip chunks beyond render_distance - they render as impostors
                let dist = (mesh.pos.center() - camera_pos).length();
                if dist > render_dist_blocks {
                    continue;
                }

                // Upload the mesh (checking for duplicates done in upload_chunk)
                self.chunk_pool.upload_chunk(&self.queue, mesh);
            }
        }

        // Evict chunks that were unloaded from the world
        // This is O(unloaded) instead of O(all_visible) - huge optimization when stable
        let unloaded = world.take_unloaded_chunks();
        for pos in unloaded {
            self.chunk_pool.remove_chunk(pos);
        }

        // Also evict chunks that moved beyond render distance (impostor range)
        // This handles the case where chunks are still in world but shouldn't have GPU mesh
        // Check pool slots directly instead of iterating all visible meshes
        let slots_to_evict: Vec<_> = self.chunk_pool.slots()
            .iter()
            .filter(|(pos, _)| {
                let dist = (pos.center() - camera_pos).length();
                dist > render_dist_blocks
            })
            .map(|(pos, _)| *pos)
            .collect();
        for pos in slots_to_evict {
            self.chunk_pool.remove_chunk(pos);
        }
    }

    /// Get chunk pool stats for debug display
    pub fn pool_stats(&self) -> (usize, u32, bool) {
        (self.chunk_pool.chunk_count(), self.chunk_pool.draw_count, self.chunk_pool.mdi_available)
    }
}

/// Camera uniform data for shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_pos: [f32; 4],
    /// Animation data: [current_time, spawn_time, wave_speed, drop_height]
    anim_data: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view_pos: [0.0; 4],
            anim_data: [0.0; 4],
        }
    }
}

impl CameraUniform {
    fn from_camera_with_anim(camera: &Camera, current_time: f32, spawn_time: f32, is_underwater: bool) -> Self {
        Self {
            view_proj: camera.view_projection_matrix().to_cols_array_2d(),
            view_pos: camera.position.extend(1.0).to_array(),
            anim_data: [
                current_time,                      // Current time in seconds
                spawn_time,                        // When the world started loading
                80.0,                              // Wave speed: 80 blocks per second
                if is_underwater { 1.0 } else { 0.0 },  // Underwater flag
            ],
        }
    }
}

/// Day/night cycle calculations
/// time: 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
pub struct DayNightCycle {
    pub sun_direction: Vec3,
    pub moon_direction: Vec3,
    pub sun_color: Vec3,
    pub moon_color: Vec3,
    pub sun_intensity: f32,
    pub moon_intensity: f32,
    pub sky_top: Vec3,
    pub sky_horizon: Vec3,
    pub ambient: Vec3,
}

impl DayNightCycle {
    pub fn from_time(time: f32) -> Self {
        // Wrap time to 0-1 range
        let t = time.fract();
        if t < 0.0 { t + 1.0 } else { t };

        // Sun rotates around X axis: rises at 0.25, peaks at 0.5, sets at 0.75
        let sun_angle = (t - 0.25) * std::f32::consts::TAU; // Full rotation
        let sun_y = sun_angle.sin();
        let sun_xz = sun_angle.cos();
        let sun_direction = Vec3::new(sun_xz * 0.5, sun_y, sun_xz * 0.866).normalize();

        // Moon is opposite to sun
        let moon_direction = -sun_direction;

        // Calculate sun/moon intensity - stay visible until actually below horizon
        // Only fade once below the horizon line (y < 0)
        let sun_intensity = if sun_direction.y > 0.0 {
            // Above horizon - full intensity, slight dim very close to horizon
            (sun_direction.y * 4.0 + 0.5).clamp(0.5, 1.0)
        } else {
            // Below horizon - fade out quickly
            ((sun_direction.y + 0.15) * 4.0).clamp(0.0, 0.5)
        };
        // Moon - stays visible until below horizon, provides soft night lighting
        // Increased intensity for visible moonlight effect
        let moon_intensity = if moon_direction.y > 0.0 {
            (moon_direction.y * 1.5 + 0.25).clamp(0.25, 0.5)
        } else {
            ((moon_direction.y + 0.15) * 2.5).clamp(0.0, 0.25)
        };

        // Define color keyframes (warm palette)
        // Time points: midnight(0), pre-dawn(0.15), dawn(0.25), morning(0.35),
        //              noon(0.5), afternoon(0.65), dusk(0.75), twilight(0.85)

        let (sky_top, sky_horizon, sun_color, ambient) = if t < 0.20 {
            // Night (0.0 - 0.20) - deep blue/indigo night sky
            let f = t / 0.20;
            (
                // Deep indigo zenith - darker and more blue
                Self::lerp_color(Vec3::new(0.02, 0.02, 0.06), Vec3::new(0.06, 0.04, 0.12), f),
                // Warmer horizon with slight purple tint
                Self::lerp_color(Vec3::new(0.04, 0.03, 0.08), Vec3::new(0.10, 0.06, 0.14), f),
                Vec3::new(0.0, 0.0, 0.0), // sun below horizon
                // Very dark ambient - moonlight provides most light
                Self::lerp_color(Vec3::new(0.015, 0.015, 0.025), Vec3::new(0.025, 0.02, 0.035), f),
            )
        } else if t < 0.25 {
            // Pre-dawn to dawn (0.20 - 0.25) - fast brightening
            let f = (t - 0.20) / 0.05;
            (
                Self::lerp_color(Vec3::new(0.10, 0.07, 0.15), Vec3::new(1.0, 0.48, 0.33), f),  // dark -> coral (fast)
                Self::lerp_color(Vec3::new(0.15, 0.12, 0.20), Vec3::new(1.0, 0.70, 0.42), f),  // dark -> peach (fast)
                Self::lerp_color(Vec3::new(0.2, 0.08, 0.04), Vec3::new(1.0, 0.42, 0.21), f),   // dim -> orange
                Self::lerp_color(Vec3::new(0.05, 0.04, 0.07), Vec3::new(0.29, 0.21, 0.21), f), // dark -> warm ambient
            )
        } else if t < 0.35 {
            // Dawn to morning (0.25 - 0.35) - transition from warm dawn to vivid blue sky
            let f = (t - 0.25) / 0.10;
            (
                // Zenith: coral dawn -> DEEP saturated blue (more saturated!)
                Self::lerp_color(Vec3::new(1.0, 0.48, 0.33), Vec3::new(0.15, 0.42, 0.90), f),
                // Horizon: peach -> saturated but lighter blue
                Self::lerp_color(Vec3::new(1.0, 0.70, 0.42), Vec3::new(0.40, 0.60, 0.88), f),
                // Sun: orange -> bright warm white
                Self::lerp_color(Vec3::new(1.0, 0.42, 0.21), Vec3::new(1.0, 0.95, 0.85), f),
                // Ambient: warm shadows -> bright daylight
                Self::lerp_color(Vec3::new(0.29, 0.21, 0.21), Vec3::new(0.60, 0.58, 0.52), f),
            )
        } else if t < 0.5 {
            // Morning to noon (0.35 - 0.5) - vivid blue sky
            let f = (t - 0.35) / 0.15;
            (
                // Zenith: DEEP saturated blue -> rich blue at noon (very saturated!)
                Self::lerp_color(Vec3::new(0.15, 0.42, 0.90), Vec3::new(0.12, 0.38, 0.92), f),
                // Horizon: saturated blue -> warm-tinged blue at horizon
                Self::lerp_color(Vec3::new(0.40, 0.60, 0.88), Vec3::new(0.50, 0.65, 0.85), f),
                // Sun: warm white -> bright white (noon sun)
                Self::lerp_color(Vec3::new(1.0, 0.95, 0.85), Vec3::new(1.0, 0.98, 0.92), f),
                // Ambient: bright daylight
                Self::lerp_color(Vec3::new(0.60, 0.58, 0.52), Vec3::new(0.65, 0.62, 0.55), f),
            )
        } else if t < 0.65 {
            // Noon to afternoon (0.5 - 0.65) - blue sky starts warming toward golden hour
            let f = (t - 0.5) / 0.15;
            (
                // Zenith: deep blue -> warming blue
                Self::lerp_color(Vec3::new(0.12, 0.38, 0.92), Vec3::new(0.25, 0.48, 0.85), f),
                // Horizon: warm blue -> starting to warm with golden tones
                Self::lerp_color(Vec3::new(0.50, 0.65, 0.85), Vec3::new(0.70, 0.70, 0.68), f),
                // Sun: noon white -> warming toward gold
                Self::lerp_color(Vec3::new(1.0, 0.98, 0.92), Vec3::new(1.0, 0.88, 0.65), f),
                // Ambient: bright -> warming
                Self::lerp_color(Vec3::new(0.65, 0.62, 0.55), Vec3::new(0.65, 0.58, 0.48), f),
            )
        } else if t < 0.75 {
            // Afternoon to dusk (0.65 - 0.75) - golden hour into sunset (keep warm!)
            let f = (t - 0.65) / 0.10;
            (
                // Zenith: warm blue -> burnt orange sunset
                Self::lerp_color(Vec3::new(0.35, 0.55, 0.82), Vec3::new(1.0, 0.42, 0.29), f),
                // Horizon: golden -> deep amber
                Self::lerp_color(Vec3::new(0.75, 0.75, 0.70), Vec3::new(1.0, 0.67, 0.36), f),
                // Sun: gold -> deep orange
                Self::lerp_color(Vec3::new(1.0, 0.88, 0.65), Vec3::new(1.0, 0.34, 0.13), f),
                // Ambient: warm -> dim warm
                Self::lerp_color(Vec3::new(0.70, 0.62, 0.52), Vec3::new(0.50, 0.31, 0.25), f),
            )
        } else if t < 0.80 {
            // Dusk to twilight (0.75 - 0.80) - faster transition to dark
            let f = (t - 0.75) / 0.05;
            (
                Self::lerp_color(Vec3::new(1.0, 0.42, 0.29), Vec3::new(0.15, 0.10, 0.22), f),  // burnt orange -> dark purple
                Self::lerp_color(Vec3::new(1.0, 0.67, 0.36), Vec3::new(0.20, 0.15, 0.25), f),  // amber -> dark mauve
                Self::lerp_color(Vec3::new(1.0, 0.34, 0.13), Vec3::new(0.1, 0.03, 0.01), f),   // deep orange -> very dim
                Self::lerp_color(Vec3::new(0.50, 0.31, 0.25), Vec3::new(0.08, 0.06, 0.10), f), // fast to dark ambient
            )
        } else {
            // Night (0.80 - 1.0) - deep blue/indigo night
            let f = (t - 0.80) / 0.20;
            (
                // Transition to deep indigo
                Self::lerp_color(Vec3::new(0.12, 0.07, 0.18), Vec3::new(0.02, 0.02, 0.06), f),
                Self::lerp_color(Vec3::new(0.16, 0.10, 0.20), Vec3::new(0.04, 0.03, 0.08), f),
                Vec3::new(0.0, 0.0, 0.0), // sun below horizon
                // Dark ambient - moonlight is primary source
                Self::lerp_color(Vec3::new(0.05, 0.035, 0.06), Vec3::new(0.015, 0.015, 0.025), f),
            )
        };

        // Warm amber moon color (not cold white)
        let moon_color = Vec3::new(1.0, 0.85, 0.61);

        Self {
            sun_direction,
            moon_direction,
            sun_color,
            moon_color,
            sun_intensity,
            moon_intensity,
            sky_top,
            sky_horizon,
            ambient,
        }
    }

    fn lerp_color(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        a + (b - a) * t
    }
}

/// Light uniform data for shadows and lighting
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    view_proj: [[f32; 4]; 4],
    sun_direction: [f32; 4],    // xyz = direction, w = intensity
    sun_color: [f32; 4],        // rgb = color, a = unused
    moon_direction: [f32; 4],   // xyz = direction, w = intensity
    moon_color: [f32; 4],       // rgb = warm amber, a = unused
    ambient_color: [f32; 4],    // rgb = ambient, a = time_of_day
}

/// Sky uniform data for sky rendering
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniform {
    inv_view_proj: [[f32; 4]; 4],
    sun_direction: [f32; 4],     // xyz = direction, w = intensity
    moon_direction: [f32; 4],    // xyz = direction, w = intensity
    sky_top_color: [f32; 4],     // rgb = color, a = time_of_day
    sky_horizon_color: [f32; 4], // rgb = color, a = unused
}

impl Default for SkyUniform {
    fn default() -> Self {
        let cycle = DayNightCycle::from_time(0.35); // Default to morning
        Self {
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            sun_direction: [cycle.sun_direction.x, cycle.sun_direction.y, cycle.sun_direction.z, cycle.sun_intensity],
            moon_direction: [cycle.moon_direction.x, cycle.moon_direction.y, cycle.moon_direction.z, cycle.moon_intensity],
            sky_top_color: [cycle.sky_top.x, cycle.sky_top.y, cycle.sky_top.z, 0.35],
            sky_horizon_color: [cycle.sky_horizon.x, cycle.sky_horizon.y, cycle.sky_horizon.z, 0.0],
        }
    }
}

impl SkyUniform {
    fn from_camera_and_time(camera: &Camera, time_of_day: f32, day_count: u32) -> Self {
        let cycle = DayNightCycle::from_time(time_of_day);

        // Use rotation-only view matrix for sky rendering (no translation)
        // This prevents floating-point precision issues at large world coordinates
        // which caused moon/star jittering when camera position changed
        let view_rotation_only = Mat4::look_to_rh(Vec3::ZERO, camera.forward(), Vec3::Y);
        let sky_view_proj = camera.projection_matrix() * view_rotation_only;

        // Moon phase: based on day count, not continuous time
        // Phase stays constant within a single night, changes between days
        // 0.0 = new moon, 0.5 = full moon, 1.0 = new moon again
        let lunar_cycle_days = 8; // 8 game days per full lunar cycle
        let moon_phase = (day_count % lunar_cycle_days) as f32 / lunar_cycle_days as f32;

        Self {
            inv_view_proj: sky_view_proj.inverse().to_cols_array_2d(),
            sun_direction: [cycle.sun_direction.x, cycle.sun_direction.y, cycle.sun_direction.z, cycle.sun_intensity],
            moon_direction: [cycle.moon_direction.x, cycle.moon_direction.y, cycle.moon_direction.z, cycle.moon_intensity],
            sky_top_color: [cycle.sky_top.x, cycle.sky_top.y, cycle.sky_top.z, time_of_day],
            sky_horizon_color: [cycle.sky_horizon.x, cycle.sky_horizon.y, cycle.sky_horizon.z, moon_phase],
        }
    }
}

impl Default for LightUniform {
    fn default() -> Self {
        // Initialize with a proper shadow matrix at origin to prevent bright faces on first frame
        Self::from_camera_position_and_time(Vec3::ZERO, 0.35)
    }
}

impl LightUniform {
    /// Create light uniform from camera position and time of day
    /// Uses shadow map stabilization to prevent shadow edge flickering/swimming
    fn from_camera_position_and_time(camera_pos: Vec3, time_of_day: f32) -> Self {
        let cycle = DayNightCycle::from_time(time_of_day);

        // Always use sun direction for shadow map - shader fades shadows at night
        // This prevents jarring shadow direction flips during transition
        let shadow_dir = cycle.sun_direction;

        // Create orthographic projection for shadow map
        // Coverage of 280 units matches ~17 chunks for better shadow stability
        // Larger coverage reduces edge artifacts at cost of shadow resolution
        let shadow_distance = 200.0;  // Increased for larger coverage
        let shadow_size = 280.0;

        // === LIGHT VIEW STABILIZATION ===
        // Note: view_proj is kept for shader compatibility but not used for shadow sampling
        // (voxel GI uses DDA ray marching instead)
        const SHADOW_MAP_SIZE: u32 = 4096;
        let texels_per_unit = SHADOW_MAP_SIZE as f32 / (shadow_size * 2.0);
        let units_per_texel = 1.0 / texels_per_unit;

        let up = Vec3::Y;
        // Handle case when shadow_dir is nearly vertical
        let light_right = if shadow_dir.y.abs() > 0.99 {
            Vec3::X
        } else {
            shadow_dir.cross(up).normalize()
        };
        let light_up = light_right.cross(shadow_dir).normalize();

        let cam_light_x = camera_pos.dot(light_right);
        let cam_light_y = camera_pos.dot(light_up);
        let cam_light_z = camera_pos.dot(shadow_dir);

        let snapped_x = (cam_light_x / units_per_texel).floor() * units_per_texel;
        let snapped_y = (cam_light_y / units_per_texel).floor() * units_per_texel;

        let snapped_center = light_right * snapped_x + light_up * snapped_y + shadow_dir * cam_light_z;
        let light_pos = snapped_center + shadow_dir * shadow_distance;

        let view = Mat4::look_at_rh(light_pos, snapped_center, Vec3::Y);
        let proj = Mat4::orthographic_rh(
            -shadow_size, shadow_size,
            -shadow_size, shadow_size,
            0.1, shadow_distance * 2.5,
        );

        Self {
            view_proj: (proj * view).to_cols_array_2d(),
            sun_direction: [cycle.sun_direction.x, cycle.sun_direction.y, cycle.sun_direction.z, cycle.sun_intensity],
            sun_color: [cycle.sun_color.x, cycle.sun_color.y, cycle.sun_color.z, 0.0],
            moon_direction: [cycle.moon_direction.x, cycle.moon_direction.y, cycle.moon_direction.z, cycle.moon_intensity],
            moon_color: [cycle.moon_color.x, cycle.moon_color.y, cycle.moon_color.z, 0.0],
            ambient_color: [cycle.ambient.x, cycle.ambient.y, cycle.ambient.z, time_of_day],
        }
    }
}
