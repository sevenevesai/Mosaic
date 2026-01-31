//! Rendering module
//!
//! Handles GPU initialization, pipeline management, and frame rendering.

mod bloom;
mod camera;
mod chunk_pool;
mod debug_overlay;
mod gpu_culling;
mod pipeline;
mod texture;
pub mod voxel_gi;
mod volumetrics;

pub use bloom::Bloom;
pub use camera::{Camera, Frustum};
pub use chunk_pool::ChunkPool;
pub use debug_overlay::{DebugOverlay, DebugMetrics};
pub use texture::TextureAtlas;
pub use voxel_gi::VoxelGI;
pub use volumetrics::VolumetricLighting;

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

use crate::utils::gpu_profiler::GpuProfiler;
use crate::utils::profiler::FrameProfiler;
use crate::world::WorldManager;

use pipeline::{UiVertex, LineVertex, WaterUniform, create_water_bind_group_layout, create_water_pipeline, create_voxel_lit_chunk_pipeline, create_dimmer_pipeline};
use glam::{Vec3, Mat4};

/// Render statistics for the current frame
#[derive(Default, Clone)]
pub struct RenderStats {
    pub chunks_rendered: u32,
    pub chunks_culled: u32,
    pub triangles: u32,
    pub gpu_time_ms: f64,
    // Shadow pass stats (legacy, now unused)
    pub shadow_chunks_rendered: u32,
    pub shadow_chunks_culled: u32,
    pub shadow_pass_ms: f64,
    pub main_pass_ms: f64,
    // Voxel GI timing
    pub voxel_gi_ms: f64,
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

    // Water rendering
    water_pipeline: wgpu::RenderPipeline,
    water_buffer: wgpu::Buffer,
    water_bind_group: wgpu::BindGroup,
    water_light_bind_group: wgpu::BindGroup,

    // Stats from last frame
    pub stats: RenderStats,

    // Animation timing
    start_time: std::time::Instant,

    // Day/night cycle
    time_of_day: f32,      // 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
    cycle_speed: f32,      // Multiplier for cycle speed (1.0 = 10 min day, higher = faster)

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

    // Settings menu dimmer overlay
    dimmer_pipeline: wgpu::RenderPipeline,
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

        // === WATER RENDERING SETUP ===
        // Water needs light uniform visible in both vertex and fragment shaders
        let water_light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Light Bind Group Layout"),
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
        let water_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Light Bind Group"),
            layout: &water_light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        // === BLOOM POST-PROCESSING ===
        // Scene texture for off-screen rendering (needed for bloom)
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

        // Create water uniform bind group layout and buffer
        let water_bind_group_layout = create_water_bind_group_layout(&device);
        let water_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Uniform Buffer"),
            contents: bytemuck::cast_slice(&[WaterUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Bind Group"),
            layout: &water_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: water_buffer.as_entire_binding(),
            }],
        });
        // Create water pipeline (uses water_light_bind_group_layout for lighting)
        let water_pipeline = create_water_pipeline(
            &device,
            surface_format,
            &camera_bind_group_layout,
            &water_light_bind_group_layout,
            &water_bind_group_layout,
        );

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
            water_pipeline,
            water_buffer,
            water_bind_group,
            water_light_bind_group,
            stats: RenderStats::default(),
            start_time: std::time::Instant::now(),
            time_of_day: 0.35,   // Start at mid-morning
            cycle_speed: 10.0,   // 10x speed for testing (1 min = 1 day cycle)
            shadow_mode: 0.0,    // Start with hard shadows (-1=off, 0=hard, 1=soft)
            shadow_debug: 0.0,   // Shadow debug off by default
            bloom_enabled: true,
            render_distance: crate::constants::RENDER_DISTANCE,
            shadow_distance: crate::constants::RENDER_DISTANCE,
            display_mode: DisplayMode::Windowed,
            display_mode_pending: false,
            torch_shadows: false,  // Default OFF for performance
            voxel_gi,
            bloom,
            scene_texture,
            scene_texture_view,
            volumetrics,
            volumetric_uniform_buffer,
            dimmer_pipeline,
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

            // Resize scene texture for bloom
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.scene_texture_view = self.scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Resize bloom textures
            self.bloom.resize(&self.device, new_size.width, new_size.height);
            self.bloom.create_extract_bind_group(&self.device, &self.scene_texture_view);

            // Resize volumetrics textures
            self.volumetrics.resize(&self.device, new_size.width, new_size.height);
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
        self.time_of_day = (self.time_of_day + time_increment) % 1.0;
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

    /// Get current time of day (0.0-1.0)
    pub fn time_of_day(&self) -> f32 {
        self.time_of_day
    }

    /// Get current cycle speed multiplier
    pub fn cycle_speed(&self) -> f32 {
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
    ) -> Result<(), wgpu::SurfaceError> {
        let render_start = std::time::Instant::now();

        // Sync meshes to chunk pool for MDI rendering
        let sync_start = std::time::Instant::now();
        if self.chunk_pool.mdi_available {
            self.sync_meshes_to_pool(world);
        }
        let sync_elapsed = sync_start.elapsed();

        // Poll GPU profiler for previous frame results
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

        // Update water uniform with current time for wave animation
        let water_uniform = WaterUniform {
            time: current_time,
            ..WaterUniform::default()
        };
        self.queue
            .write_buffer(&self.water_buffer, 0, bytemuck::cast_slice(&[water_uniform]));

        // Update light uniform with day/night cycle (center shadow map on camera)
        let light_uniform = LightUniform::from_camera_position_and_time(camera.position, self.time_of_day);
        let cycle = DayNightCycle::from_time(self.time_of_day);
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_uniform]));

        // Update sky uniform with day/night colors and moon phase
        let sky_uniform = SkyUniform::from_camera_and_time(camera, self.time_of_day, current_time);
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

        let uniforms_elapsed = uniforms_start.elapsed();

        // Build frustum for culling
        let frustum = Frustum::from_view_proj(camera.view_projection_matrix());

        // Get surface texture
        let output = self.surface.get_current_texture()?;
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
        );
        let voxel_gi_elapsed = voxel_gi_start.elapsed();

        // Prepare timestamp writes for GPU profiling
        let timestamp_writes = self.gpu_profiler.query_set().map(|qs| {
            wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }
        });

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

            // Run GPU culling compute shader
            self.gpu_culler.cull(
                &self.device,
                &self.queue,
                &mut encoder,
                &frustum_planes,
                crate::constants::CHUNK_SIZE_F32,
            );
        }

        // Pre-generate menu lines if menu is open (before render pass borrow)
        let menu_lines = if settings_menu.is_open {
            Some(settings_menu.generate_lines(self))
        } else {
            None
        };

        // === MAIN RENDER PASS (render to scene texture for bloom) ===
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
                timestamp_writes,
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

            // === WATER RENDER PASS ===
            // Render water meshes with transparency (after opaque geometry)
            render_pass.set_pipeline(&self.water_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.water_light_bind_group, &[]);
            render_pass.set_bind_group(2, &self.water_bind_group, &[]);

            for water_mesh in world.visible_water_meshes() {
                let chunk_min = water_mesh.pos.world_origin();
                let chunk_max = chunk_min + glam::Vec3::splat(crate::constants::CHUNK_SIZE_F32);

                // Frustum cull water chunks
                if !frustum.test_aabb(chunk_min, chunk_max) {
                    continue;
                }

                if let Some(gpu_mesh) = &water_mesh.gpu_mesh {
                    render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        gpu_mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                    triangles += gpu_mesh.index_count / 3;
                }
            }

            // Render block outline if targeting a block
            if let Some([bx, by, bz]) = target_block {
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

            // Render crosshair (UI, no depth)
            render_pass.set_pipeline(&self.ui_pipeline);
            render_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
            render_pass.draw(0..12, 0..1);

            // Render debug overlay and optionally settings menu
            self.debug_overlay.render(
                &mut render_pass,
                &self.queue,
                debug_metrics,
                menu_lines.as_deref(),
                if menu_lines.is_some() { Some(&self.dimmer_pipeline) } else { None },
            );
        }

        // === VOLUMETRIC LIGHT SHAFTS ===
        // Render god rays using raymarching through voxel volume
        // This runs at half resolution and composites onto the scene
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

        // === BLOOM POST-PROCESSING ===
        // Apply bloom effect: extract bright pixels, blur, composite to output
        // If bloom disabled, just blits scene to output with tone mapping
        self.bloom.render_or_blit(&mut encoder, &view, &self.scene_texture_view, &self.device, self.bloom_enabled);

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
                "RENDER SPIKE {:.1}ms: sync={:.1}ms uniforms={:.1}ms voxel_gi={:.1}ms cull={:.1}ms mdi={:.1}ms submit={:.1}ms | {} chunks",
                render_total.as_secs_f64() * 1000.0,
                sync_elapsed.as_secs_f64() * 1000.0,
                uniforms_elapsed.as_secs_f64() * 1000.0,
                voxel_gi_elapsed.as_secs_f64() * 1000.0,
                cull_elapsed.as_secs_f64() * 1000.0,
                mdi_prep_elapsed.as_secs_f64() * 1000.0,
                submit_elapsed.as_secs_f64() * 1000.0,
                chunks_rendered,
            );
        }

        // Update stats
        self.stats = RenderStats {
            chunks_rendered,
            chunks_culled,
            triangles,
            gpu_time_ms: self.gpu_profiler.avg_gpu_time_ms(),
            shadow_chunks_rendered,
            shadow_chunks_culled,
            shadow_pass_ms: shadow_elapsed.as_secs_f64() * 1000.0,
            main_pass_ms: main_elapsed.as_secs_f64() * 1000.0,
            voxel_gi_ms: voxel_gi_elapsed.as_secs_f64() * 1000.0,
        };

        Ok(())
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
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
    /// Only uploads meshes that aren't already in the pool (or have changed size)
    pub fn sync_meshes_to_pool(&mut self, world: &mut crate::world::WorldManager) {
        // FIRST: Upload urgent meshes (player block changes) WITHOUT throttling
        // These need to appear immediately for responsive gameplay
        let urgent = world.take_urgent_uploads();
        for pos in &urgent {
            if let Some(mesh) = world.visible_meshes().find(|m| &m.pos == pos) {
                self.chunk_pool.upload_chunk(&self.queue, mesh);
            }
        }

        // Throttle regular uploads to prevent frame spikes when many chunks arrive at once
        // 16 uploads/frame at 60fps = 960 chunks/sec, plenty fast for smooth loading
        const MAX_UPLOADS_PER_FRAME: usize = 16;
        let mut uploads_this_frame = 0;

        // Only upload meshes not already in pool (or with different size indicating regeneration)
        for mesh in world.visible_meshes() {
            if mesh.vertices.is_empty() {
                continue;
            }

            // Skip urgent meshes we already uploaded
            if urgent.contains(&mesh.pos) {
                continue;
            }

            // Check if already in pool with same data
            if let Some(existing) = self.chunk_pool.get_slot(&mesh.pos) {
                // Only re-upload if vertex count changed (mesh was regenerated)
                if existing.vertex_count == mesh.vertices.len() as u32 {
                    continue;
                }
            }

            // Upload new or changed mesh (throttled)
            self.chunk_pool.upload_chunk(&self.queue, mesh);
            uploads_this_frame += 1;

            if uploads_this_frame >= MAX_UPLOADS_PER_FRAME {
                break; // Continue next frame
            }
        }

        // Evict non-visible chunks if pool is getting full (only when not busy uploading)
        if uploads_this_frame == 0 && self.chunk_pool.is_nearly_full() {
            let visible: Vec<_> = world.visible_meshes()
                .map(|m| m.pos)
                .collect();
            self.chunk_pool.evict_non_visible(&visible);
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
            // Dawn to morning (0.25 - 0.35)
            let f = (t - 0.25) / 0.10;
            (
                Self::lerp_color(Vec3::new(1.0, 0.48, 0.33), Vec3::new(0.53, 0.64, 0.79), f),  // coral -> warm blue
                Self::lerp_color(Vec3::new(1.0, 0.70, 0.42), Vec3::new(1.0, 0.82, 0.52), f),  // peach -> cream
                Self::lerp_color(Vec3::new(1.0, 0.42, 0.21), Vec3::new(1.0, 0.78, 0.34), f),  // orange -> gold
                Self::lerp_color(Vec3::new(0.29, 0.21, 0.21), Vec3::new(0.63, 0.50, 0.38), f),
            )
        } else if t < 0.5 {
            // Morning to noon (0.35 - 0.5)
            let f = (t - 0.35) / 0.15;
            (
                Self::lerp_color(Vec3::new(0.53, 0.64, 0.79), Vec3::new(0.35, 0.56, 0.74), f), // warm blue -> clear sky
                Self::lerp_color(Vec3::new(1.0, 0.82, 0.52), Vec3::new(0.79, 0.86, 0.91), f), // cream -> soft white
                Self::lerp_color(Vec3::new(1.0, 0.78, 0.34), Vec3::new(1.0, 0.96, 0.88), f),  // gold -> warm white
                Self::lerp_color(Vec3::new(0.63, 0.50, 0.38), Vec3::new(0.82, 0.75, 0.63), f),
            )
        } else if t < 0.65 {
            // Noon to afternoon (0.5 - 0.65)
            let f = (t - 0.5) / 0.15;
            (
                Self::lerp_color(Vec3::new(0.35, 0.56, 0.74), Vec3::new(0.49, 0.64, 0.77), f), // clear sky -> soft blue
                Self::lerp_color(Vec3::new(0.79, 0.86, 0.91), Vec3::new(0.96, 0.84, 0.63), f), // soft white -> golden
                Self::lerp_color(Vec3::new(1.0, 0.96, 0.88), Vec3::new(1.0, 0.80, 0.40), f),   // warm white -> gold
                Self::lerp_color(Vec3::new(0.82, 0.75, 0.63), Vec3::new(0.75, 0.63, 0.50), f),
            )
        } else if t < 0.75 {
            // Afternoon to dusk (0.65 - 0.75)
            let f = (t - 0.65) / 0.10;
            (
                Self::lerp_color(Vec3::new(0.49, 0.64, 0.77), Vec3::new(1.0, 0.42, 0.29), f),  // soft blue -> burnt orange
                Self::lerp_color(Vec3::new(0.96, 0.84, 0.63), Vec3::new(1.0, 0.67, 0.36), f),  // golden -> amber
                Self::lerp_color(Vec3::new(1.0, 0.80, 0.40), Vec3::new(1.0, 0.34, 0.13), f),   // gold -> deep orange
                Self::lerp_color(Vec3::new(0.75, 0.63, 0.50), Vec3::new(0.50, 0.31, 0.25), f),
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
    fn from_camera_and_time(camera: &Camera, time_of_day: f32, elapsed_time: f32) -> Self {
        let cycle = DayNightCycle::from_time(time_of_day);

        // Use rotation-only view matrix for sky rendering (no translation)
        // This prevents floating-point precision issues at large world coordinates
        // which caused moon/star jittering when camera position changed
        let view_rotation_only = Mat4::look_to_rh(Vec3::ZERO, camera.forward(), Vec3::Y);
        let sky_view_proj = camera.projection_matrix() * view_rotation_only;

        // Moon phase: cycles over ~5 minutes (300 seconds) for testing
        // 0.0 = new moon, 0.5 = full moon, 1.0 = new moon again
        // In production, tie this to in-game days (e.g., 29.5 day cycle)
        let moon_phase_cycle = 300.0; // seconds per full lunar cycle
        let moon_phase = (elapsed_time / moon_phase_cycle).fract();

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
