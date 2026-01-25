//! Rendering module
//!
//! Handles GPU initialization, pipeline management, and frame rendering.

mod camera;
mod chunk_pool;
mod debug_overlay;
mod pipeline;
mod texture;

pub use camera::{Camera, Frustum};
pub use chunk_pool::ChunkPool;
pub use debug_overlay::{DebugOverlay, DebugMetrics};
pub use texture::TextureAtlas;

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::utils::gpu_profiler::GpuProfiler;
use crate::utils::profiler::FrameProfiler;
use crate::world::WorldManager;

use pipeline::{UiVertex, LineVertex, SHADOW_MAP_SIZE, WaterUniform, create_water_bind_group_layout, create_water_pipeline};
use glam::{Vec3, Mat4};

/// Render statistics for the current frame
#[derive(Default, Clone)]
pub struct RenderStats {
    pub chunks_rendered: u32,
    pub chunks_culled: u32,
    pub triangles: u32,
    pub gpu_time_ms: f64,
    // Shadow pass stats
    pub shadow_chunks_rendered: u32,
    pub shadow_chunks_culled: u32,
    pub shadow_pass_ms: f64,
    pub main_pass_ms: f64,
}

/// GPU state and rendering logic
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Pipelines
    chunk_pipeline: wgpu::RenderPipeline,

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

    // Shadow mapping
    shadow_texture: wgpu::TextureView,
    shadow_pipeline: wgpu::RenderPipeline,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,      // For shadow pass (just light uniform)
    shadow_bind_group: wgpu::BindGroup,     // For main pass (light + shadow texture)

    // Sky rendering
    sky_pipeline: wgpu::RenderPipeline,
    sky_buffer: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,

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

        // Request device with optional features for GPU profiling and multi-draw
        let desired_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::MULTI_DRAW_INDIRECT
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
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
                    label: Some("Neolith Device"),
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

        // === SHADOW MAPPING SETUP ===
        // Create shadow map depth texture
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_texture_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create light uniform buffer
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[LightUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Light bind group layout (for shadow pass - just the light uniform)
        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Light bind group for shadow pass
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        // Shadow bind group layout (for main pass - light uniform + shadow texture)
        let shadow_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        // Shadow sampler for comparison sampling
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Shadow bind group for main pass
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        // Create shadow pipeline
        let shadow_pipeline = pipeline::create_shadow_pipeline(&device, &light_bind_group_layout);

        // === SKY RENDERING SETUP ===
        let sky_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky Buffer"),
            contents: bytemuck::cast_slice(&[SkyUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sky_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky Bind Group Layout"),
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

        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky Bind Group"),
            layout: &sky_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_buffer.as_entire_binding(),
            }],
        });

        let sky_pipeline = pipeline::create_sky_pipeline(&device, surface_format, &sky_bind_group_layout);

        // Create chunk rendering pipeline (with shadow support)
        let chunk_pipeline = pipeline::create_chunk_pipeline(
            &device,
            surface_format,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            &shadow_bind_group_layout,
        );

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
            camera_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            texture_bind_group_layout,
            texture_bind_group,
            depth_texture,
            shadow_texture: shadow_texture_view,
            shadow_pipeline,
            light_buffer,
            light_bind_group,
            shadow_bind_group,
            sky_pipeline,
            sky_buffer,
            sky_bind_group,
            gpu_profiler,
            ui_pipeline,
            crosshair_vertex_buffer,
            line_pipeline,
            outline_vertex_buffer,
            debug_overlay,
            chunk_pool,
            water_pipeline,
            water_buffer,
            water_bind_group,
            water_light_bind_group,
            stats: RenderStats::default(),
            start_time: std::time::Instant::now(),
            time_of_day: 0.35,   // Start at mid-morning
            cycle_speed: 10.0,   // 10x speed for testing (1 min = 1 day cycle)
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
            self.debug_overlay.update_screen_size(new_size.width as f32, new_size.height as f32);
        }
    }

    pub fn toggle_debug_overlay(&mut self) {
        self.debug_overlay.toggle();
    }

    pub fn is_debug_overlay_enabled(&self) -> bool {
        self.debug_overlay.is_enabled()
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

    /// Get current time of day (0.0-1.0)
    pub fn time_of_day(&self) -> f32 {
        self.time_of_day
    }

    /// Get current cycle speed multiplier
    pub fn cycle_speed(&self) -> f32 {
        self.cycle_speed
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
        world: &WorldManager,
        _profiler: &FrameProfiler,
        target_block: Option<[i32; 3]>,
        debug_metrics: &DebugMetrics,
    ) -> Result<(), wgpu::SurfaceError> {
        // Sync meshes to chunk pool for MDI rendering
        if self.chunk_pool.mdi_available {
            self.sync_meshes_to_pool(world);
        }

        // Poll GPU profiler for previous frame results
        self.gpu_profiler.poll_results(&self.device);

        // Update camera uniform with animation timing
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
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_uniform]));

        // Update sky uniform with day/night colors
        let sky_uniform = SkyUniform::from_camera_and_time(camera, self.time_of_day);
        self.queue
            .write_buffer(&self.sky_buffer, 0, bytemuck::cast_slice(&[sky_uniform]));

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
        let mut shadow_chunks_rendered = 0u32;
        let mut shadow_chunks_culled = 0u32;

        // Calculate shadow frustum bounds for culling
        // Shadow map covers ±120 units from camera in X/Z plane
        let shadow_size = 120.0f32;
        let shadow_min = Vec3::new(
            camera.position.x - shadow_size,
            -100.0, // Include all terrain heights
            camera.position.z - shadow_size,
        );
        let shadow_max = Vec3::new(
            camera.position.x + shadow_size,
            500.0, // Include tall mountains
            camera.position.z + shadow_size,
        );

        let shadow_start = std::time::Instant::now();

        // === SHADOW PASS ===
        // Render scene from light's perspective to shadow map
        {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],  // No color output
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.light_bind_group, &[]);

            // Build lists for shadow pass (pool vs individual)
            let mut shadow_pool: Vec<crate::world::ChunkPos> = Vec::with_capacity(2000);
            let mut shadow_individual: Vec<&crate::world::ChunkMesh> = Vec::with_capacity(500);

            for chunk_mesh in world.visible_meshes() {
                let chunk_min = chunk_mesh.pos.world_origin();
                let chunk_max = chunk_min + glam::Vec3::splat(crate::constants::CHUNK_SIZE_F32);

                // AABB intersection test with shadow bounds
                if chunk_max.x < shadow_min.x || chunk_min.x > shadow_max.x ||
                   chunk_max.y < shadow_min.y || chunk_min.y > shadow_max.y ||
                   chunk_max.z < shadow_min.z || chunk_min.z > shadow_max.z {
                    shadow_chunks_culled += 1;
                    continue;
                }

                if self.chunk_pool.mdi_available && self.chunk_pool.has_chunk(&chunk_mesh.pos) {
                    shadow_pool.push(chunk_mesh.pos);
                } else if chunk_mesh.gpu_mesh.is_some() {
                    shadow_individual.push(chunk_mesh);
                }
            }

            // MDI shadow pass
            if self.chunk_pool.mdi_available && !shadow_pool.is_empty() {
                let draw_count = self.chunk_pool.prepare_indirect_draws(&self.queue, &shadow_pool);
                if draw_count > 0 {
                    shadow_pass.set_vertex_buffer(0, self.chunk_pool.vertex_buffer.slice(..));
                    shadow_pass.set_index_buffer(
                        self.chunk_pool.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    shadow_pass.multi_draw_indexed_indirect(
                        &self.chunk_pool.indirect_buffer,
                        0,
                        draw_count,
                    );
                    shadow_chunks_rendered += draw_count;
                }
            }

            // Individual shadow pass (fallback)
            for chunk_mesh in shadow_individual {
                if let Some(gpu_mesh) = &chunk_mesh.gpu_mesh {
                    shadow_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                    shadow_pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    shadow_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                    shadow_chunks_rendered += 1;
                }
            }
        }

        let shadow_elapsed = shadow_start.elapsed();
        let main_start = std::time::Instant::now();

        // === MAIN RENDER PASS ===
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            // Switch to chunk pipeline
            render_pass.set_pipeline(&self.chunk_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_bind_group(2, &self.shadow_bind_group, &[]);

            // Build list of visible chunk positions (frustum culled)
            // Separate chunks that are in the pool vs need individual draws
            let mut pool_chunks: Vec<crate::world::ChunkPos> = Vec::with_capacity(3000);
            let mut individual_chunks: Vec<&crate::world::ChunkMesh> = Vec::with_capacity(500);

            for chunk_mesh in world.visible_meshes() {
                let chunk_min = chunk_mesh.pos.world_origin();
                let chunk_max = chunk_min + glam::Vec3::splat(crate::constants::CHUNK_SIZE_F32);

                if !frustum.test_aabb(chunk_min, chunk_max) {
                    chunks_culled += 1;
                    continue;
                }

                // Check if chunk is in the pool
                if self.chunk_pool.mdi_available {
                    if let Some(slot) = self.chunk_pool.get_slot(&chunk_mesh.pos) {
                        pool_chunks.push(chunk_mesh.pos);
                        triangles += slot.index_count / 3;
                    } else if chunk_mesh.gpu_mesh.is_some() {
                        // Not in pool but has individual buffer - use fallback
                        individual_chunks.push(chunk_mesh);
                    }
                } else if chunk_mesh.gpu_mesh.is_some() {
                    individual_chunks.push(chunk_mesh);
                }
            }

            // MDI pass: render all chunks that are in the pool
            if self.chunk_pool.mdi_available && !pool_chunks.is_empty() {
                let draw_count = self.chunk_pool.prepare_indirect_draws(&self.queue, &pool_chunks);

                if draw_count > 0 {
                    render_pass.set_vertex_buffer(0, self.chunk_pool.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.chunk_pool.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.multi_draw_indexed_indirect(
                        &self.chunk_pool.indirect_buffer,
                        0,
                        draw_count,
                    );
                    chunks_rendered += draw_count;
                }
            }

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

            // Render debug overlay
            self.debug_overlay.render(&mut render_pass, &self.queue, debug_metrics);
        }

        let main_elapsed = main_start.elapsed();

        // Resolve GPU timestamps
        self.gpu_profiler.resolve(&mut encoder);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Begin async readback of GPU timing
        self.gpu_profiler.begin_readback();

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
    pub fn sync_meshes_to_pool(&mut self, world: &crate::world::WorldManager) {
        // Only upload meshes not already in pool (or with different size indicating regeneration)
        for mesh in world.visible_meshes() {
            if mesh.vertices.is_empty() {
                continue;
            }

            // Check if already in pool with same data
            if let Some(existing) = self.chunk_pool.get_slot(&mesh.pos) {
                // Only re-upload if vertex count changed (mesh was regenerated)
                if existing.vertex_count == mesh.vertices.len() as u32 {
                    continue;
                }
            }

            // Upload new or changed mesh
            self.chunk_pool.upload_chunk(&self.queue, mesh);
        }

        // Evict non-visible chunks if pool is getting full
        if self.chunk_pool.is_nearly_full() {
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

        // Calculate sun/moon intensity based on height
        let sun_intensity = (sun_direction.y * 2.0).clamp(0.0, 1.0);
        // Moon is much dimmer than sun - subtle illumination
        let moon_intensity = (moon_direction.y * 1.2).clamp(0.0, 0.25);

        // Define color keyframes (warm palette)
        // Time points: midnight(0), pre-dawn(0.15), dawn(0.25), morning(0.35),
        //              noon(0.5), afternoon(0.65), dusk(0.75), twilight(0.85)

        let (sky_top, sky_horizon, sun_color, ambient) = if t < 0.20 {
            // Night (0.0 - 0.20) - longer dark period before dawn
            let f = t / 0.20;
            (
                Self::lerp_color(Vec3::new(0.04, 0.04, 0.08), Vec3::new(0.10, 0.07, 0.15), f), // very dark -> slightly lighter
                Self::lerp_color(Vec3::new(0.08, 0.08, 0.14), Vec3::new(0.15, 0.12, 0.20), f), // dark purple -> hint of color
                Vec3::new(0.0, 0.0, 0.0), // sun below horizon
                Self::lerp_color(Vec3::new(0.03, 0.03, 0.05), Vec3::new(0.05, 0.04, 0.07), f), // very dark ambient
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
            // Night (0.80 - 1.0) - longer dark period
            let f = (t - 0.80) / 0.20;
            (
                Self::lerp_color(Vec3::new(0.15, 0.10, 0.22), Vec3::new(0.04, 0.04, 0.08), f), // dark purple -> very dark
                Self::lerp_color(Vec3::new(0.20, 0.15, 0.25), Vec3::new(0.08, 0.08, 0.14), f), // dark mauve -> darker
                Vec3::new(0.0, 0.0, 0.0), // sun below horizon
                Self::lerp_color(Vec3::new(0.08, 0.06, 0.10), Vec3::new(0.03, 0.03, 0.05), f), // very dark ambient
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
    fn from_camera_and_time(camera: &Camera, time_of_day: f32) -> Self {
        let cycle = DayNightCycle::from_time(time_of_day);
        Self {
            inv_view_proj: camera.view_projection_matrix().inverse().to_cols_array_2d(),
            sun_direction: [cycle.sun_direction.x, cycle.sun_direction.y, cycle.sun_direction.z, cycle.sun_intensity],
            moon_direction: [cycle.moon_direction.x, cycle.moon_direction.y, cycle.moon_direction.z, cycle.moon_intensity],
            sky_top_color: [cycle.sky_top.x, cycle.sky_top.y, cycle.sky_top.z, time_of_day],
            sky_horizon_color: [cycle.sky_horizon.x, cycle.sky_horizon.y, cycle.sky_horizon.z, 0.0],
        }
    }
}

impl Default for LightUniform {
    fn default() -> Self {
        let cycle = DayNightCycle::from_time(0.35);
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            sun_direction: [cycle.sun_direction.x, cycle.sun_direction.y, cycle.sun_direction.z, cycle.sun_intensity],
            sun_color: [cycle.sun_color.x, cycle.sun_color.y, cycle.sun_color.z, 0.0],
            moon_direction: [cycle.moon_direction.x, cycle.moon_direction.y, cycle.moon_direction.z, cycle.moon_intensity],
            moon_color: [cycle.moon_color.x, cycle.moon_color.y, cycle.moon_color.z, 0.0],
            ambient_color: [cycle.ambient.x, cycle.ambient.y, cycle.ambient.z, 0.35],
        }
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
        let shadow_distance = 100.0;
        let shadow_size = 120.0;

        // === SHADOW MAP STABILIZATION ===
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
