//! Prop/voxel model renderer
//!
//! Renders voxel models (like chests) at various positions in the world using instancing.

use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;

use super::vox_loader::{VoxMesh, VoxModel, VoxVertex};

/// Maximum instances per prop type
const MAX_INSTANCES: usize = 1024;

/// Instance data for a placed prop
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PropInstance {
    /// World position
    pub position: [f32; 3],
    /// Rotation around Y axis (radians)
    pub rotation: f32,
    /// Scale (1.0 = normal)
    pub scale: f32,
    /// Padding for alignment
    _padding: [f32; 3],
}

impl PropInstance {
    pub fn new(position: Vec3, rotation: f32, scale: f32) -> Self {
        Self {
            position: position.into(),
            rotation,
            scale,
            _padding: [0.0; 3],
        }
    }
}

/// A loaded prop type with its mesh
struct PropType {
    mesh: VoxMesh,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
}

/// Renders props (voxel models) in the world
pub struct PropRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// Loaded prop types by name
    props: HashMap<String, PropType>,
}

impl PropRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Prop Shader"),
            source: wgpu::ShaderSource::Wgsl(PROP_SHADER.into()),
        });

        // Empty bind group for prop-specific uniforms
        let prop_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Prop Bind Group Layout"),
            entries: &[],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prop Bind Group"),
            layout: &prop_bind_group_layout,
            entries: &[],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Prop Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &prop_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex layout for vox vertices
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        // Instance layout
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PropInstance>() as u64,
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
            ],
        };

        // Create pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Prop Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout, instance_layout],
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

        Self {
            pipeline,
            bind_group,
            props: HashMap::new(),
        }
    }

    /// Load a prop from a .vox file
    pub fn load_prop(&mut self, device: &wgpu::Device, name: &str, path: &Path, scale: f32) -> Result<(), String> {
        let model = VoxModel::load(path)?;
        let mesh = VoxMesh::from_model(device, &model, scale);

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Prop Instance Buffer: {}", name)),
            size: (MAX_INSTANCES * std::mem::size_of::<PropInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.props.insert(name.to_string(), PropType {
            mesh,
            instance_buffer,
            instance_count: 0,
        });

        tracing::info!("Loaded prop '{}' from {:?}", name, path);
        Ok(())
    }

    /// Update instances for a prop type
    pub fn update_instances(&mut self, queue: &wgpu::Queue, name: &str, instances: &[PropInstance]) {
        if let Some(prop) = self.props.get_mut(name) {
            let count = instances.len().min(MAX_INSTANCES);
            prop.instance_count = count as u32;

            if count > 0 {
                queue.write_buffer(
                    &prop.instance_buffer,
                    0,
                    bytemuck::cast_slice(&instances[..count]),
                );
            }
        }
    }

    /// Render all props
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);

        for prop in self.props.values() {
            if prop.instance_count == 0 {
                continue;
            }

            render_pass.set_vertex_buffer(0, prop.mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, prop.instance_buffer.slice(..));
            render_pass.set_index_buffer(prop.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..prop.mesh.index_count, 0, 0..prop.instance_count);
        }
    }

    /// Check if a prop is loaded
    pub fn has_prop(&self, name: &str) -> bool {
        self.props.contains_key(name)
    }

    /// Get the center offset for a prop (for positioning)
    pub fn prop_center(&self, name: &str) -> Option<Vec3> {
        self.props.get(name).map(|p| p.mesh.center)
    }
}

/// Prop shader with instancing and simple lighting
const PROP_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct InstanceInput {
    @location(3) world_position: vec3<f32>,
    @location(4) rotation: f32,
    @location(5) scale: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Rotate vertex position around Y axis
    let cos_r = cos(instance.rotation);
    let sin_r = sin(instance.rotation);
    let rotated_x = vertex.position.x * cos_r - vertex.position.z * sin_r;
    let rotated_z = vertex.position.x * sin_r + vertex.position.z * cos_r;
    let rotated_pos = vec3<f32>(rotated_x, vertex.position.y, rotated_z);

    // Apply scale and world position
    let world_pos = instance.world_position + rotated_pos * instance.scale;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vertex.color;
    out.world_pos = world_pos;

    // Rotate normal
    let rotated_nx = vertex.normal.x * cos_r - vertex.normal.z * sin_r;
    let rotated_nz = vertex.normal.x * sin_r + vertex.normal.z * cos_r;
    out.world_normal = vec3<f32>(rotated_nx, vertex.normal.y, rotated_nz);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Discard fully transparent
    if in.color.a < 0.01 {
        discard;
    }

    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let ndotl = max(dot(normalize(in.world_normal), light_dir), 0.0);

    // Ambient + diffuse
    let ambient = 0.4;
    let diffuse = 0.6;
    let brightness = ambient + diffuse * ndotl;

    return vec4<f32>(in.color.rgb * brightness, in.color.a);
}
"#;
