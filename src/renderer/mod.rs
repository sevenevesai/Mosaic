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

use pipeline::{UiVertex, LineVertex, SHADOW_MAP_SIZE};
use glam::{Vec3, Mat4};

/// Render statistics for the current frame
#[derive(Default, Clone)]
pub struct RenderStats {
    pub chunks_rendered: u32,
    pub chunks_culled: u32,
    pub triangles: u32,
    pub gpu_time_ms: f64,
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
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

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

    // Stats from last frame
    pub stats: RenderStats,
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
                    label: Some("Blocks Device"),
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
            stats: RenderStats::default(),
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
        // Poll GPU profiler for previous frame results
        self.gpu_profiler.poll_results(&self.device);

        // Update camera uniform
        let camera_uniform = CameraUniform::from_camera(camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Update light uniform (center shadow map on camera)
        let light_uniform = LightUniform::from_camera_position(camera.position);
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_uniform]));

        // Update sky uniform
        let sky_uniform = SkyUniform::from_camera(camera);
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

            // Render chunks to shadow map (no frustum culling for shadows - render all visible)
            for chunk_mesh in world.visible_meshes() {
                if let Some(gpu_mesh) = &chunk_mesh.gpu_mesh {
                    shadow_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                    shadow_pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    shadow_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                }
            }
        }

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

            // Use original per-chunk rendering (stable, working)
            // MDI implementation needs more work - disabled for now
            for chunk_mesh in world.visible_meshes() {
                // Frustum culling - skip chunks outside camera view
                let chunk_min = chunk_mesh.pos.world_origin();
                let chunk_max = chunk_min + glam::Vec3::splat(crate::constants::CHUNK_SIZE_F32);

                if !frustum.test_aabb(chunk_min, chunk_max) {
                    chunks_culled += 1;
                    continue;
                }

                if let Some(gpu_mesh) = &chunk_mesh.gpu_mesh {
                    render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                    triangles += gpu_mesh.index_count / 3;
                    chunks_rendered += 1;
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
}

/// Camera uniform data for shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_pos: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view_pos: [0.0; 4],
        }
    }
}

impl CameraUniform {
    fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_projection_matrix().to_cols_array_2d(),
            view_pos: camera.position.extend(1.0).to_array(),
        }
    }
}

/// Light uniform data for shadows
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    view_proj: [[f32; 4]; 4],
    direction: [f32; 4],  // xyz = direction, w = unused
}

/// Sky uniform data for sky rendering
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniform {
    inv_view_proj: [[f32; 4]; 4],
    sun_direction: [f32; 4],
    screen_size: [f32; 4],  // xy = screen size, zw = unused
}

impl Default for SkyUniform {
    fn default() -> Self {
        Self {
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            sun_direction: [0.6, 0.8, 0.4, 0.0],
            screen_size: [1920.0, 1080.0, 0.0, 0.0],
        }
    }
}

impl SkyUniform {
    fn from_camera(camera: &Camera) -> Self {
        let sun_dir = Vec3::new(0.6, 0.8, 0.4).normalize();
        Self {
            inv_view_proj: camera.view_projection_matrix().inverse().to_cols_array_2d(),
            sun_direction: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            screen_size: [1920.0, 1080.0, 0.0, 0.0],  // Not used in current shader
        }
    }
}

impl Default for LightUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            direction: [0.6, 0.8, 0.4, 0.0],
        }
    }
}

impl LightUniform {
    /// Create light uniform from camera position (for shadow map centering)
    /// Uses shadow map stabilization to prevent shadow edge flickering/swimming
    fn from_camera_position(camera_pos: Vec3) -> Self {
        // Sun direction (same as in shader)
        let sun_dir = Vec3::new(0.6, 0.8, 0.4).normalize();

        // Create orthographic projection for shadow map
        // Center shadow map around camera for best coverage
        let shadow_distance = 100.0;  // How far shadows extend from camera
        let shadow_size = 120.0;       // Width/height of shadow frustum

        // === SHADOW MAP STABILIZATION ===
        // Snap the shadow map to texel boundaries to prevent shadow edge swimming/flickering
        // This ensures shadow texels stay aligned with world geometry as camera moves

        // Calculate world units per shadow texel
        let texels_per_unit = SHADOW_MAP_SIZE as f32 / (shadow_size * 2.0);
        let units_per_texel = 1.0 / texels_per_unit;

        // Create a stable "look at" point by snapping to shadow texel grid
        // We snap in the plane perpendicular to the light direction
        let up = Vec3::Y;
        let light_right = sun_dir.cross(up).normalize();
        let light_up = light_right.cross(sun_dir).normalize();

        // Project camera position onto light's coordinate system
        let cam_light_x = camera_pos.dot(light_right);
        let cam_light_y = camera_pos.dot(light_up);
        let cam_light_z = camera_pos.dot(sun_dir);

        // Snap X and Y to texel boundaries (these cause the swimming)
        let snapped_x = (cam_light_x / units_per_texel).floor() * units_per_texel;
        let snapped_y = (cam_light_y / units_per_texel).floor() * units_per_texel;

        // Reconstruct the snapped position in world space
        let snapped_center = light_right * snapped_x + light_up * snapped_y + sun_dir * cam_light_z;

        // Light position: snapped center offset along sun direction
        let light_pos = snapped_center + sun_dir * shadow_distance;

        // Look at the snapped center position
        let view = Mat4::look_at_rh(light_pos, snapped_center, Vec3::Y);

        // Orthographic projection for directional light
        let proj = Mat4::orthographic_rh(
            -shadow_size, shadow_size,
            -shadow_size, shadow_size,
            0.1, shadow_distance * 2.5,
        );

        Self {
            view_proj: (proj * view).to_cols_array_2d(),
            direction: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
        }
    }
}
