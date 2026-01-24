//! Debug overlay for performance monitoring
//!
//! Renders text-based debug information using a simple bitmap font.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

/// Font character width and height in pixels
const CHAR_WIDTH: u32 = 6;
const CHAR_HEIGHT: u32 = 10;
const FONT_COLS: u32 = 16;
const FONT_ROWS: u32 = 6;

/// Vertex for text rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl TextVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Performance metrics for display
#[derive(Default, Clone)]
pub struct DebugMetrics {
    // Frame timing
    pub fps: f32,
    pub frame_time_ms: f32,
    pub world_update_ms: f32,
    pub render_ms: f32,

    // CPU/Threading
    pub chunks_pending_gen: u32,
    pub chunks_pending_mesh: u32,

    // GPU/Render
    pub triangles: u32,
    pub draw_calls: u32,
    pub chunks_rendered: u32,
    pub chunks_culled: u32,
    pub gpu_time_ms: f64,

    // Memory
    pub loaded_chunks: u32,
    pub total_vertices: u64,

    // Position
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub yaw: f32,
    pub pitch: f32,
}

/// Debug overlay rendering system
pub struct DebugOverlay {
    enabled: bool,
    pipeline: wgpu::RenderPipeline,
    font_texture: wgpu::Texture,
    font_view: wgpu::TextureView,
    font_sampler: wgpu::Sampler,
    font_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    max_chars: usize,

    // Frame time history for smoothing
    frame_times: VecDeque<f32>,

    // Screen dimensions for layout
    screen_width: f32,
    screen_height: f32,

    // Performance logging
    session_start: Instant,
    last_log_time: Instant,
    log_count: u32,
    max_logs_per_session: u32,
    stabilization_delay_secs: f32,
    log_interval_secs: f32,
}

const TEXT_SHADER: &str = r#"
@group(0) @binding(0)
var t_font: texture_2d<f32>;
@group(0) @binding(1)
var s_font: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(t_font, s_font, in.uv);
    // White text with alpha from texture
    if tex.r < 0.5 {
        discard;
    }
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

impl DebugOverlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // Generate bitmap font texture
        let font_data = Self::generate_font_texture();
        let font_width = FONT_COLS * CHAR_WIDTH;
        let font_height = FONT_ROWS * CHAR_HEIGHT;

        let font_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Debug Font Texture"),
            size: wgpu::Extent3d {
                width: font_width,
                height: font_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let font_view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let font_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Debug Font Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout and bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Debug Font Bind Group Layout"),
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

        let font_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Debug Font Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&font_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&font_sampler),
                },
            ],
        });

        // Create pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Debug Text Shader"),
            source: wgpu::ShaderSource::Wgsl(TEXT_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Debug Text Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Text Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[TextVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,  // Match main pipeline (no MSAA)
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create vertex buffer (sized for max characters)
        let max_chars = 2000;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Text Vertex Buffer"),
            size: (max_chars * 6 * std::mem::size_of::<TextVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let now = Instant::now();
        Self {
            enabled: false,
            pipeline,
            font_texture,
            font_view,
            font_sampler,
            font_bind_group,
            vertex_buffer,
            max_chars,
            frame_times: VecDeque::with_capacity(60),
            screen_width: 1280.0,
            screen_height: 720.0,
            session_start: now,
            last_log_time: now,
            log_count: 0,
            max_logs_per_session: 5,
            stabilization_delay_secs: 10.0, // Wait 10s for initial load
            log_interval_secs: 30.0,        // Log every 30s after that
        }
    }

    /// Generate a simple bitmap font texture (ASCII 32-127)
    fn generate_font_texture() -> Vec<u8> {
        let width = (FONT_COLS * CHAR_WIDTH) as usize;
        let height = (FONT_ROWS * CHAR_HEIGHT) as usize;
        let mut data = vec![0u8; width * height];

        // Simple 6x10 bitmap font for ASCII 32-127 (space to DEL)
        // Each character is defined as a series of bytes representing rows
        let font_data: &[&[u8]] = &[
            // Space (32)
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            // ! (33)
            &[0x00, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x00, 0x00],
            // " (34)
            &[0x00, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            // # (35)
            &[0x00, 0x0A, 0x1F, 0x0A, 0x0A, 0x1F, 0x0A, 0x00, 0x00, 0x00],
            // $ (36)
            &[0x04, 0x0E, 0x15, 0x14, 0x0E, 0x05, 0x15, 0x0E, 0x04, 0x00],
            // % (37)
            &[0x00, 0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13, 0x00, 0x00],
            // & (38)
            &[0x00, 0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0D, 0x00, 0x00],
            // ' (39)
            &[0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            // ( (40)
            &[0x00, 0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02, 0x00, 0x00],
            // ) (41)
            &[0x00, 0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08, 0x00, 0x00],
            // * (42)
            &[0x00, 0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00, 0x00, 0x00],
            // + (43)
            &[0x00, 0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00, 0x00, 0x00],
            // , (44)
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08, 0x00],
            // - (45)
            &[0x00, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00],
            // . (46)
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00],
            // / (47)
            &[0x00, 0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10, 0x00, 0x00],
            // 0 (48)
            &[0x00, 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00, 0x00],
            // 1 (49)
            &[0x00, 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00, 0x00],
            // 2 (50)
            &[0x00, 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F, 0x00, 0x00],
            // 3 (51)
            &[0x00, 0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E, 0x00, 0x00],
            // 4 (52)
            &[0x00, 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02, 0x00, 0x00],
            // 5 (53)
            &[0x00, 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E, 0x00, 0x00],
            // 6 (54)
            &[0x00, 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E, 0x00, 0x00],
            // 7 (55)
            &[0x00, 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00, 0x00],
            // 8 (56)
            &[0x00, 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00, 0x00],
            // 9 (57)
            &[0x00, 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C, 0x00, 0x00],
            // : (58)
            &[0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00],
            // ; (59)
            &[0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x04, 0x04, 0x08, 0x00],
            // < (60)
            &[0x00, 0x01, 0x02, 0x04, 0x08, 0x04, 0x02, 0x01, 0x00, 0x00],
            // = (61)
            &[0x00, 0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00],
            // > (62)
            &[0x00, 0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00],
            // ? (63)
            &[0x00, 0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04, 0x00, 0x00],
            // @ (64)
            &[0x00, 0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E, 0x00, 0x00],
            // A (65)
            &[0x00, 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00, 0x00],
            // B (66)
            &[0x00, 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E, 0x00, 0x00],
            // C (67)
            &[0x00, 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E, 0x00, 0x00],
            // D (68)
            &[0x00, 0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C, 0x00, 0x00],
            // E (69)
            &[0x00, 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F, 0x00, 0x00],
            // F (70)
            &[0x00, 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x00, 0x00],
            // G (71)
            &[0x00, 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F, 0x00, 0x00],
            // H (72)
            &[0x00, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00, 0x00],
            // I (73)
            &[0x00, 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00, 0x00],
            // J (74)
            &[0x00, 0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C, 0x00, 0x00],
            // K (75)
            &[0x00, 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11, 0x00, 0x00],
            // L (76)
            &[0x00, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F, 0x00, 0x00],
            // M (77)
            &[0x00, 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11, 0x00, 0x00],
            // N (78)
            &[0x00, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11, 0x00, 0x00],
            // O (79)
            &[0x00, 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],
            // P (80)
            &[0x00, 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10, 0x00, 0x00],
            // Q (81)
            &[0x00, 0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D, 0x00, 0x00],
            // R (82)
            &[0x00, 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11, 0x00, 0x00],
            // S (83)
            &[0x00, 0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E, 0x00, 0x00],
            // T (84)
            &[0x00, 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00],
            // U (85)
            &[0x00, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00],
            // V (86)
            &[0x00, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04, 0x00, 0x00],
            // W (87)
            &[0x00, 0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11, 0x00, 0x00],
            // X (88)
            &[0x00, 0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11, 0x00, 0x00],
            // Y (89)
            &[0x00, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00],
            // Z (90)
            &[0x00, 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F, 0x00, 0x00],
            // [ (91)
            &[0x00, 0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E, 0x00, 0x00],
            // \ (92)
            &[0x00, 0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01, 0x00, 0x00],
            // ] (93)
            &[0x00, 0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E, 0x00, 0x00],
            // ^ (94)
            &[0x00, 0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            // _ (95)
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00],
        ];

        // Render each character to the texture
        for (i, char_data) in font_data.iter().enumerate() {
            let char_x = (i as u32 % FONT_COLS) * CHAR_WIDTH;
            let char_y = (i as u32 / FONT_COLS) * CHAR_HEIGHT;

            for (row, &row_data) in char_data.iter().enumerate() {
                for col in 0..CHAR_WIDTH {
                    let pixel_x = char_x + col;
                    let pixel_y = char_y + row as u32;
                    let idx = (pixel_y as usize * width) + pixel_x as usize;

                    // Check if bit is set (from right to left)
                    let bit = (row_data >> (CHAR_WIDTH - 1 - col)) & 1;
                    data[idx] = if bit == 1 { 255 } else { 0 };
                }
            }
        }

        data
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        tracing::info!("Debug overlay: {}", if self.enabled { "ON" } else { "OFF" });
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn update_screen_size(&mut self, width: f32, height: f32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Check if we should log metrics and do so if needed
    pub fn maybe_log_metrics(&mut self, metrics: &DebugMetrics) {
        if self.log_count >= self.max_logs_per_session {
            return;
        }

        let elapsed = self.session_start.elapsed().as_secs_f32();

        // Wait for stabilization
        if elapsed < self.stabilization_delay_secs {
            return;
        }

        let since_last_log = self.last_log_time.elapsed().as_secs_f32();
        let first_log = self.log_count == 0;

        // Log immediately after stabilization, then at intervals
        if first_log || since_last_log >= self.log_interval_secs {
            self.log_metrics_to_file(metrics, elapsed);
            self.last_log_time = Instant::now();
            self.log_count += 1;
        }
    }

    fn log_metrics_to_file(&self, metrics: &DebugMetrics, elapsed_secs: f32) {
        let log_path = "performance_log.txt";

        let log_entry = format!(
            "[Session +{:>5.0}s] FPS: {:>4.0} | Frame: {:>5.2}ms | GPU: {:>5.2}ms | \
Tris: {:>5}K | Chunks: {:>4} loaded {:>3} rendered | \
Pending: {:>3} gen {:>3} mesh | Verts: {:>6}K\n",
            elapsed_secs,
            metrics.fps,
            metrics.frame_time_ms,
            metrics.gpu_time_ms,
            metrics.triangles / 1000,
            metrics.loaded_chunks,
            metrics.chunks_rendered,
            metrics.chunks_pending_gen,
            metrics.chunks_pending_mesh,
            metrics.total_vertices / 1000,
        );

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            let _ = file.write_all(log_entry.as_bytes());
            tracing::info!("Performance logged to {}", log_path);
        }
    }

    pub fn upload_font(&self, queue: &wgpu::Queue) {
        let font_data = Self::generate_font_texture();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.font_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(FONT_COLS * CHAR_WIDTH),
                rows_per_image: Some(FONT_ROWS * CHAR_HEIGHT),
            },
            wgpu::Extent3d {
                width: FONT_COLS * CHAR_WIDTH,
                height: FONT_ROWS * CHAR_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Render the debug overlay
    pub fn render<'a>(
        &'a mut self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        metrics: &DebugMetrics,
    ) {
        // Always check for periodic logging (even if overlay disabled)
        self.maybe_log_metrics(metrics);

        if !self.enabled {
            return;
        }

        // Build text lines
        let lines = vec![
            format!("FPS: {:.0} ({:.2}ms)", metrics.fps, metrics.frame_time_ms),
            format!("GPU: {:.2}ms", metrics.gpu_time_ms),
            format!("World: {:.2}ms  Render: {:.2}ms", metrics.world_update_ms, metrics.render_ms),
            String::new(),
            format!("Triangles: {}K", metrics.triangles / 1000),
            format!("Draw calls: {}", metrics.draw_calls),
            format!("Chunks: {} rendered, {} culled", metrics.chunks_rendered, metrics.chunks_culled),
            format!("Loaded: {} chunks", metrics.loaded_chunks),
            String::new(),
            format!("Pending: {} gen, {} mesh", metrics.chunks_pending_gen, metrics.chunks_pending_mesh),
            format!("Vertices: {}K", metrics.total_vertices / 1000),
            String::new(),
            format!("XYZ: {:.1}, {:.1}, {:.1}", metrics.pos_x, metrics.pos_y, metrics.pos_z),
            format!("Chunk: {}, {}, {}", metrics.chunk_x, metrics.chunk_y, metrics.chunk_z),
            format!("Look: {:.1}, {:.1}", metrics.yaw.to_degrees(), metrics.pitch.to_degrees()),
        ];

        // Generate vertices for all text
        let mut vertices = Vec::new();
        let scale = 2.0; // 2x scale for readability
        let char_w = (CHAR_WIDTH as f32 * scale) / self.screen_width * 2.0;
        let char_h = (CHAR_HEIGHT as f32 * scale) / self.screen_height * 2.0;
        let line_height = char_h * 1.2;

        let start_x = -0.98; // Left edge with padding
        let mut y = 0.95; // Top edge with padding

        for line in &lines {
            let mut x = start_x;
            for ch in line.chars() {
                if let Some((u0, v0, u1, v1)) = self.char_uvs(ch) {
                    // Two triangles for quad
                    vertices.push(TextVertex { position: [x, y - char_h], uv: [u0, v1] });
                    vertices.push(TextVertex { position: [x + char_w, y - char_h], uv: [u1, v1] });
                    vertices.push(TextVertex { position: [x + char_w, y], uv: [u1, v0] });

                    vertices.push(TextVertex { position: [x, y - char_h], uv: [u0, v1] });
                    vertices.push(TextVertex { position: [x + char_w, y], uv: [u1, v0] });
                    vertices.push(TextVertex { position: [x, y], uv: [u0, v0] });
                }
                x += char_w;
            }
            y -= line_height;
        }

        if vertices.is_empty() {
            return;
        }

        // Upload vertices
        let vertex_count = vertices.len().min(self.max_chars * 6);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices[..vertex_count]));

        // Render
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.font_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..vertex_count as u32, 0..1);
    }

    fn char_uvs(&self, ch: char) -> Option<(f32, f32, f32, f32)> {
        let code = ch as u32;
        if code < 32 || code > 95 + 32 {
            return None;
        }

        let idx = code - 32;
        let col = idx % FONT_COLS;
        let row = idx / FONT_COLS;

        let u0 = col as f32 / FONT_COLS as f32;
        let v0 = row as f32 / FONT_ROWS as f32;
        let u1 = (col + 1) as f32 / FONT_COLS as f32;
        let v1 = (row + 1) as f32 / FONT_ROWS as f32;

        Some((u0, v0, u1, v1))
    }
}
