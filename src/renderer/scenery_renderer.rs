//! Scenery Renderer - High-performance vegetation rendering
//!
//! Renders thousands of grass, flower, and rock instances using GPU instancing.
//! Collects instances from visible chunks and batches by prop type for minimal draw calls.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;
use wgpu::util::DeviceExt;

use super::prop_registry::{PropId, PropRegistry};
use super::vox_loader::{VoxMesh, VoxVertex};
use crate::world::{Chunk, ChunkPos};

/// Maximum instances per prop type (can render ~8000 grass tufts)
const MAX_INSTANCES_PER_PROP: usize = 8192;

/// Maximum draw distance for vegetation (blocks)
const MAX_DRAW_DISTANCE: f32 = 48.0;

/// Distance at which alpha fade begins (blocks)
const FADE_START_DISTANCE: f32 = 36.0;

/// Instance data for scenery rendering
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneryInstance {
    /// World position
    pub position: [f32; 3],
    /// Rotation around Y axis (radians)
    pub rotation: f32,
    /// Scale multiplier
    pub scale: f32,
    /// Sway strength (0.0 = no sway, 1.0 = full wind)
    pub sway_strength: f32,
    /// Distance fade (0.0 = invisible, 1.0 = fully visible)
    pub alpha: f32,
    /// Padding for alignment
    _padding: f32,
}

impl SceneryInstance {
    pub fn new(position: Vec3, rotation: f32, scale: f32, sway_strength: f32, alpha: f32) -> Self {
        Self {
            position: position.into(),
            rotation,
            scale,
            sway_strength,
            alpha,
            _padding: 0.0,
        }
    }

    /// Vertex buffer layout for instance data
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneryInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // rotation
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                // scale
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
                // sway_strength
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                // alpha
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Per-prop-type rendering data
struct PropBatch {
    mesh: Arc<VoxMesh>,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
}

/// Scenery uniform buffer
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneryUniforms {
    /// Current time for wind animation
    pub time: f32,
    /// Wind direction X
    pub wind_dir_x: f32,
    /// Wind direction Z
    pub wind_dir_z: f32,
    /// Wind strength multiplier
    pub wind_strength: f32,
}

impl Default for SceneryUniforms {
    fn default() -> Self {
        Self {
            time: 0.0,
            wind_dir_x: 1.0,
            wind_dir_z: 0.3,
            wind_strength: 1.0,
        }
    }
}

/// High-performance scenery renderer
pub struct SceneryRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Instance buffers per prop type
    batches: HashMap<PropId, PropBatch>,
    /// Reference to prop registry
    registry: Arc<PropRegistry>,
    /// CPU-side instance staging
    staging: HashMap<PropId, Vec<SceneryInstance>>,
    /// Whether rendering is enabled
    enabled: bool,
    /// Quality settings
    quality: SceneryQuality,
}

/// Scenery quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SceneryQuality {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl SceneryQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            SceneryQuality::Off => "OFF",
            SceneryQuality::Low => "LOW",
            SceneryQuality::Medium => "MEDIUM",
            SceneryQuality::High => "HIGH",
            SceneryQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> SceneryQuality {
        match self {
            SceneryQuality::Off => SceneryQuality::Low,
            SceneryQuality::Low => SceneryQuality::Medium,
            SceneryQuality::Medium => SceneryQuality::High,
            SceneryQuality::High => SceneryQuality::Ultra,
            SceneryQuality::Ultra => SceneryQuality::Off,
        }
    }

    pub fn prev(&self) -> SceneryQuality {
        match self {
            SceneryQuality::Off => SceneryQuality::Ultra,
            SceneryQuality::Low => SceneryQuality::Off,
            SceneryQuality::Medium => SceneryQuality::Low,
            SceneryQuality::High => SceneryQuality::Medium,
            SceneryQuality::Ultra => SceneryQuality::High,
        }
    }

    /// Density multiplier
    pub fn density(&self) -> f32 {
        match self {
            SceneryQuality::Off => 0.0,
            SceneryQuality::Low => 0.3,
            SceneryQuality::Medium => 0.6,
            SceneryQuality::High => 1.0,
            SceneryQuality::Ultra => 1.5,
        }
    }

    /// Max draw distance
    pub fn draw_distance(&self) -> f32 {
        match self {
            SceneryQuality::Off => 0.0,
            SceneryQuality::Low => 24.0,
            SceneryQuality::Medium => 36.0,
            SceneryQuality::High => 48.0,
            SceneryQuality::Ultra => 64.0,
        }
    }

    /// Whether wind animation is enabled
    pub fn wind_enabled(&self) -> bool {
        !matches!(self, SceneryQuality::Off | SceneryQuality::Low)
    }
}

impl SceneryRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        registry: Arc<PropRegistry>,
    ) -> Self {
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scenery Shader"),
            source: wgpu::ShaderSource::Wgsl(SCENERY_SHADER.into()),
        });

        // Uniform buffer for wind/time
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scenery Uniform Buffer"),
            contents: bytemuck::cast_slice(&[SceneryUniforms::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scenery Uniform Bind Group Layout"),
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scenery Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scenery Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex layout for vox vertices
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        // Create pipeline with alpha blending for fade
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scenery Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout, SceneryInstance::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create instance buffers for each prop type
        let mut batches = HashMap::new();
        for prop_def in registry.all() {
            let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Scenery Instance Buffer: {:?}", prop_def.id)),
                size: (MAX_INSTANCES_PER_PROP * std::mem::size_of::<SceneryInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            batches.insert(prop_def.id, PropBatch {
                mesh: Arc::clone(&prop_def.mesh),
                instance_buffer,
                instance_count: 0,
            });
        }

        Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            batches,
            registry,
            staging: HashMap::new(),
            enabled: true,
            quality: SceneryQuality::default(),
        }
    }

    /// Set quality preset
    pub fn set_quality(&mut self, quality: SceneryQuality) {
        self.quality = quality;
        self.enabled = quality != SceneryQuality::Off;
    }

    /// Get current quality
    pub fn quality(&self) -> SceneryQuality {
        self.quality
    }

    /// Collect instances from visible chunks
    pub fn collect_instances(
        &mut self,
        chunks: &[(ChunkPos, &Chunk)],
        camera_pos: Vec3,
    ) {
        if !self.enabled {
            return;
        }

        // Clear staging buffers
        for staging in self.staging.values_mut() {
            staging.clear();
        }

        let max_dist = self.quality.draw_distance();
        let max_dist_sq = max_dist * max_dist;
        let fade_start_sq = (max_dist * 0.75).powi(2);

        for (chunk_pos, chunk) in chunks {
            // Skip chunks without props
            if !chunk.has_props() {
                continue;
            }

            let chunk_origin = chunk_pos.world_origin();

            // Quick distance check for entire chunk
            let chunk_center = chunk_origin + Vec3::splat(8.0);
            let chunk_dist_sq = (chunk_center - camera_pos).length_squared();
            if chunk_dist_sq > (max_dist + 16.0).powi(2) {
                continue;
            }

            // Process each prop in chunk
            for prop in chunk.iter_props() {
                let world_pos = prop.world_position(chunk_origin);

                // Distance check
                let dist_sq = (world_pos - camera_pos).length_squared();
                if dist_sq > max_dist_sq {
                    continue;
                }

                // Calculate alpha fade
                let alpha = if dist_sq > fade_start_sq {
                    let t = (dist_sq - fade_start_sq) / (max_dist_sq - fade_start_sq);
                    (1.0_f32 - t).max(0.0)
                } else {
                    1.0
                };

                // Get prop definition for sway strength
                let prop_id = PropId::try_from(prop.prop_id);
                let sway_strength = prop_id
                    .and_then(|id| self.registry.get(id))
                    .map(|def| def.sway_strength)
                    .unwrap_or(0.0);

                // Disable sway if quality doesn't support it
                let effective_sway = if self.quality.wind_enabled() {
                    sway_strength
                } else {
                    0.0
                };

                let instance = SceneryInstance::new(
                    world_pos,
                    prop.rotation,
                    prop.scale,
                    effective_sway,
                    alpha,
                );

                // Add to staging buffer for this prop type
                if let Some(id) = prop_id {
                    self.staging
                        .entry(id)
                        .or_insert_with(Vec::new)
                        .push(instance);
                }
            }
        }
    }

    /// Upload staged instances to GPU
    pub fn upload(&mut self, queue: &wgpu::Queue, time: f32, wind_dir: Vec3, wind_strength: f32) {
        if !self.enabled {
            return;
        }

        // Update uniforms
        let uniforms = SceneryUniforms {
            time,
            wind_dir_x: wind_dir.x,
            wind_dir_z: wind_dir.z,
            wind_strength,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Upload instance data for each prop type
        for (prop_id, batch) in &mut self.batches {
            if let Some(instances) = self.staging.get(prop_id) {
                let count = instances.len().min(MAX_INSTANCES_PER_PROP);
                batch.instance_count = count as u32;

                if count > 0 {
                    queue.write_buffer(
                        &batch.instance_buffer,
                        0,
                        bytemuck::cast_slice(&instances[..count]),
                    );
                }
            } else {
                batch.instance_count = 0;
            }
        }
    }

    /// Render all scenery
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) -> u32 {
        if !self.enabled {
            return 0;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);

        let mut total_instances = 0;

        for (_prop_id, batch) in &self.batches {
            if batch.instance_count == 0 {
                continue;
            }

            // Use mesh stored in batch (Arc ensures it lives long enough)
            render_pass.set_vertex_buffer(0, batch.mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
            render_pass.set_index_buffer(batch.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..batch.mesh.index_count, 0, 0..batch.instance_count);

            total_instances += batch.instance_count;
        }

        total_instances
    }

    /// Get total instance count for stats
    pub fn instance_count(&self) -> u32 {
        self.batches.values().map(|b| b.instance_count).sum()
    }
}

impl PropId {
    fn try_from(value: u8) -> Option<Self> {
        match value {
            0 => Some(PropId::Grass01),
            1 => Some(PropId::Grass02),
            2 => Some(PropId::ForestGrass01),
            3 => Some(PropId::ForestGrass02),
            4 => Some(PropId::DeadGrass),
            10 => Some(PropId::Flower01),
            11 => Some(PropId::Flower02),
            12 => Some(PropId::Flower03),
            13 => Some(PropId::DeadFlower),
            14 => Some(PropId::WaterFlower01),
            15 => Some(PropId::WaterFlower02),
            16 => Some(PropId::FlowerSmallPink),
            17 => Some(PropId::FlowerSmallRed),
            18 => Some(PropId::FlowerSmallYellow),
            19 => Some(PropId::FlowerSmallSkyBlue),
            30 => Some(PropId::Stone01),
            31 => Some(PropId::Stone02),
            32 => Some(PropId::Stone03),
            33 => Some(PropId::ForestStone01),
            34 => Some(PropId::ForestStone02),
            35 => Some(PropId::ForestStone03),
            36 => Some(PropId::SandStone),
            37 => Some(PropId::RockFlat),
            38 => Some(PropId::Rock),
            _ => None,
        }
    }
}

/// Scenery shader with wind animation and alpha fade
const SCENERY_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
}

struct SceneryUniform {
    time: f32,
    wind_dir_x: f32,
    wind_dir_z: f32,
    wind_strength: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> scenery: SceneryUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct InstanceInput {
    @location(3) world_position: vec3<f32>,
    @location(4) rotation: f32,
    @location(5) scale: f32,
    @location(6) sway_strength: f32,
    @location(7) alpha: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) alpha: f32,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Rotate vertex position around Y axis
    let cos_r = cos(instance.rotation);
    let sin_r = sin(instance.rotation);
    let rotated_x = vertex.position.x * cos_r - vertex.position.z * sin_r;
    let rotated_z = vertex.position.x * sin_r + vertex.position.z * cos_r;
    var rotated_pos = vec3<f32>(rotated_x, vertex.position.y, rotated_z);

    // Wind animation (height-based sway)
    if instance.sway_strength > 0.0 {
        // Height factor: base is fixed, top sways most
        let model_height = 1.0; // Approximate model height
        let height_factor = saturate(rotated_pos.y / model_height);

        // Wind phase with position-based offset for variety
        let phase_offset = instance.world_position.x * 0.5 + instance.world_position.z * 0.3;
        let wind_phase = scenery.time * 2.5 + phase_offset;

        // Multi-frequency sway for natural look
        let primary_sway = sin(wind_phase) * 0.7;
        let secondary_sway = sin(wind_phase * 2.3 + 1.0) * 0.3;
        let sway = (primary_sway + secondary_sway) * instance.sway_strength * scenery.wind_strength * 0.12;

        // Apply sway in wind direction
        rotated_pos.x += sway * height_factor * scenery.wind_dir_x;
        rotated_pos.z += sway * height_factor * scenery.wind_dir_z * 0.5;
    }

    // Apply scale and world position
    let world_pos = instance.world_position + rotated_pos * instance.scale;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vertex.color;
    out.world_pos = world_pos;
    out.alpha = instance.alpha;

    // Rotate normal
    let rotated_nx = vertex.normal.x * cos_r - vertex.normal.z * sin_r;
    let rotated_nz = vertex.normal.x * sin_r + vertex.normal.z * cos_r;
    out.world_normal = vec3<f32>(rotated_nx, vertex.normal.y, rotated_nz);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Discard fully transparent
    if in.color.a < 0.01 || in.alpha < 0.01 {
        discard;
    }

    // Simple directional lighting (sun direction)
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let ndotl = max(dot(normalize(in.world_normal), light_dir), 0.0);

    // Ambient + diffuse
    let ambient = 0.45;
    let diffuse = 0.55;
    let brightness = ambient + diffuse * ndotl;

    let final_alpha = in.color.a * in.alpha;

    return vec4<f32>(in.color.rgb * brightness, final_alpha);
}
"#;
