//! Item entity rendering
//!
//! Renders dropped items as small 3D cubes with proper face textures.

use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::inventory::Item;
use crate::world::ItemEntity;

/// Maximum items to render per frame
const MAX_ITEMS: usize = 256;

/// Size of the cube (0.25 = quarter block)
const CUBE_SIZE: f32 = 0.25;

/// Cube vertex with position, UV, and face ID
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    /// Local position relative to cube center
    position: [f32; 3],
    /// UV coordinates for texture sampling
    uv: [f32; 2],
    /// Face ID: 0=top, 1=bottom, 2=left(-X), 3=right(+X), 4=front(-Z), 5=back(+Z)
    face_id: u32,
}

/// Item instance data for GPU
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ItemInstance {
    /// World position (center of item)
    position: [f32; 3],
    /// Scale (for flash animation)
    scale: f32,
    /// Rotation angle (radians around Y axis)
    rotation: f32,
    /// Texture indices: [top, bottom, sides, unused]
    textures: [u32; 4],
}

/// Renders item entities as 3D cubes
pub struct ItemRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    item_count: u32,
    index_count: u32,
}

impl ItemRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Item Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(ITEM_CUBE_SHADER.into()),
        });

        // Empty bind group for item-specific uniforms (future use)
        let item_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Item Bind Group Layout"),
            entries: &[],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Item Bind Group"),
            layout: &item_bind_group_layout,
            entries: &[],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Item Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                texture_bind_group_layout,
                &item_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Vertex layout
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // uv
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // face_id
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        // Instance buffer layout
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ItemInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // scale
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                // rotation
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
                // textures (as uvec4)
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Uint32x4,
                },
            ],
        };

        // Create pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Item Cube Pipeline"),
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
                cull_mode: Some(wgpu::Face::Back), // Backface culling for solid cube
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

        // Generate cube mesh
        let (vertices, indices) = generate_cube_mesh(CUBE_SIZE);
        let index_count = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Item Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Item Cube Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Item Instance Buffer"),
            size: (MAX_ITEMS * std::mem::size_of::<ItemInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            instance_buffer,
            vertex_buffer,
            index_buffer,
            bind_group,
            item_count: 0,
            index_count,
        }
    }

    /// Update instance buffer with current item entities
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        items: impl Iterator<Item = ItemInstanceData>,
        time: f32,
    ) {
        let mut instances = Vec::with_capacity(MAX_ITEMS);

        for item in items.take(MAX_ITEMS) {
            // Bob animation
            let bob = (time * 2.0 + item.position.x + item.position.z).sin() * 0.1;

            // Rotation animation
            let rotation = time * 0.8 + item.id as f32 * 0.5;

            // Flash when about to despawn
            let scale = if item.is_flashing && ((time * 8.0) as i32 % 2 == 0) {
                0.0 // Invisible during flash
            } else {
                1.0 // Normal size
            };

            instances.push(ItemInstance {
                position: [item.position.x, item.position.y + bob + 0.25, item.position.z],
                scale,
                rotation,
                textures: [item.texture_top, item.texture_bottom, item.texture_side, 0],
            });
        }

        self.item_count = instances.len() as u32;

        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }
    }

    /// Render item entities
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        texture_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.item_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, texture_bind_group, &[]);
        render_pass.set_bind_group(2, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.index_count, 0, 0..self.item_count);
    }
}

/// Data needed to render an item
pub struct ItemInstanceData {
    pub id: u64,
    pub position: Vec3,
    pub texture_top: u32,
    pub texture_bottom: u32,
    pub texture_side: u32,
    pub is_flashing: bool,
}

impl ItemInstanceData {
    pub fn from_entity(entity: &ItemEntity) -> Self {
        let textures = match entity.item {
            Item::Block(block) => block.texture_indices(),
        };

        Self {
            id: entity.id,
            position: entity.position,
            texture_top: textures[0],
            texture_bottom: textures[1],
            texture_side: textures[2], // Use left face as representative side
            is_flashing: entity.is_flashing(),
        }
    }
}

/// Generate a cube mesh centered at origin
fn generate_cube_mesh(half_size: f32) -> (Vec<CubeVertex>, Vec<u16>) {
    let h = half_size;
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    // Face definitions: (normal direction, 4 corners, face_id)
    // Corners are in CCW order when looking at the face from outside

    // Top face (Y+) - face_id 0
    let base = vertices.len() as u16;
    vertices.push(CubeVertex { position: [-h,  h, -h], uv: [0.0, 0.0], face_id: 0 });
    vertices.push(CubeVertex { position: [-h,  h,  h], uv: [0.0, 1.0], face_id: 0 });
    vertices.push(CubeVertex { position: [ h,  h,  h], uv: [1.0, 1.0], face_id: 0 });
    vertices.push(CubeVertex { position: [ h,  h, -h], uv: [1.0, 0.0], face_id: 0 });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

    // Bottom face (Y-) - face_id 1
    let base = vertices.len() as u16;
    vertices.push(CubeVertex { position: [-h, -h,  h], uv: [0.0, 0.0], face_id: 1 });
    vertices.push(CubeVertex { position: [-h, -h, -h], uv: [0.0, 1.0], face_id: 1 });
    vertices.push(CubeVertex { position: [ h, -h, -h], uv: [1.0, 1.0], face_id: 1 });
    vertices.push(CubeVertex { position: [ h, -h,  h], uv: [1.0, 0.0], face_id: 1 });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

    // Left face (X-) - face_id 2
    let base = vertices.len() as u16;
    vertices.push(CubeVertex { position: [-h, -h, -h], uv: [0.0, 1.0], face_id: 2 });
    vertices.push(CubeVertex { position: [-h, -h,  h], uv: [1.0, 1.0], face_id: 2 });
    vertices.push(CubeVertex { position: [-h,  h,  h], uv: [1.0, 0.0], face_id: 2 });
    vertices.push(CubeVertex { position: [-h,  h, -h], uv: [0.0, 0.0], face_id: 2 });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

    // Right face (X+) - face_id 3
    let base = vertices.len() as u16;
    vertices.push(CubeVertex { position: [ h, -h,  h], uv: [0.0, 1.0], face_id: 3 });
    vertices.push(CubeVertex { position: [ h, -h, -h], uv: [1.0, 1.0], face_id: 3 });
    vertices.push(CubeVertex { position: [ h,  h, -h], uv: [1.0, 0.0], face_id: 3 });
    vertices.push(CubeVertex { position: [ h,  h,  h], uv: [0.0, 0.0], face_id: 3 });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

    // Front face (Z-) - face_id 4
    let base = vertices.len() as u16;
    vertices.push(CubeVertex { position: [-h, -h, -h], uv: [1.0, 1.0], face_id: 4 });
    vertices.push(CubeVertex { position: [-h,  h, -h], uv: [1.0, 0.0], face_id: 4 });
    vertices.push(CubeVertex { position: [ h,  h, -h], uv: [0.0, 0.0], face_id: 4 });
    vertices.push(CubeVertex { position: [ h, -h, -h], uv: [0.0, 1.0], face_id: 4 });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

    // Back face (Z+) - face_id 5
    let base = vertices.len() as u16;
    vertices.push(CubeVertex { position: [ h, -h,  h], uv: [1.0, 1.0], face_id: 5 });
    vertices.push(CubeVertex { position: [ h,  h,  h], uv: [1.0, 0.0], face_id: 5 });
    vertices.push(CubeVertex { position: [-h,  h,  h], uv: [0.0, 0.0], face_id: 5 });
    vertices.push(CubeVertex { position: [-h, -h,  h], uv: [0.0, 1.0], face_id: 5 });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

    (vertices, indices)
}

/// Item 3D cube shader
const ITEM_CUBE_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var texture_atlas: texture_2d<f32>;
@group(1) @binding(1)
var texture_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) face_id: u32,
}

struct InstanceInput {
    @location(3) world_position: vec3<f32>,
    @location(4) scale: f32,
    @location(5) rotation: f32,
    @location(6) textures: vec4<u32>, // top, bottom, sides, unused
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) face_id: u32,
    @location(2) @interpolate(flat) textures: vec4<u32>,
    @location(3) world_normal: vec3<f32>,
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
    out.uv = vertex.uv;
    out.face_id = vertex.face_id;
    out.textures = instance.textures;

    // Compute rotated normal for lighting
    var normal: vec3<f32>;
    switch vertex.face_id {
        case 0u: { normal = vec3<f32>(0.0, 1.0, 0.0); }  // Top
        case 1u: { normal = vec3<f32>(0.0, -1.0, 0.0); } // Bottom
        case 2u: { normal = vec3<f32>(-1.0, 0.0, 0.0); } // Left
        case 3u: { normal = vec3<f32>(1.0, 0.0, 0.0); }  // Right
        case 4u: { normal = vec3<f32>(0.0, 0.0, -1.0); } // Front
        default: { normal = vec3<f32>(0.0, 0.0, 1.0); }  // Back
    }

    // Rotate normal around Y axis
    let rotated_nx = normal.x * cos_r - normal.z * sin_r;
    let rotated_nz = normal.x * sin_r + normal.z * cos_r;
    out.world_normal = vec3<f32>(rotated_nx, normal.y, rotated_nz);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Select texture based on face
    var tile_index: u32;
    switch in.face_id {
        case 0u: { tile_index = in.textures.x; } // Top
        case 1u: { tile_index = in.textures.y; } // Bottom
        default: { tile_index = in.textures.z; } // Sides (left, right, front, back)
    }

    // Atlas is 8x8 tiles (64 tiles), each tile is 256x256 in a 2048x2048 atlas
    let tiles_per_row = 8u;
    let tile_size = 1.0 / f32(tiles_per_row);
    let atlas_size = 2048.0;
    let tile_pixels = atlas_size / f32(tiles_per_row); // 256 pixels per tile

    // Half-texel inset to prevent bleeding at edges
    let half_texel = 0.5 / atlas_size;
    let inset = half_texel * 2.0; // A bit more margin for safety

    // Clamp UVs away from tile edges
    let safe_uv = clamp(in.uv, vec2<f32>(inset), vec2<f32>(1.0 - inset));

    let tile_x = tile_index % tiles_per_row;
    let tile_y = tile_index / tiles_per_row;

    let atlas_uv = vec2<f32>(
        (f32(tile_x) + safe_uv.x) * tile_size,
        (f32(tile_y) + safe_uv.y) * tile_size
    );

    let color = textureSample(texture_atlas, texture_sampler, atlas_uv);

    // Discard transparent pixels
    if color.a < 0.5 {
        discard;
    }

    // Simple directional lighting for depth perception
    // Light from above-front-right (typical isometric view light)
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let ndotl = max(dot(in.world_normal, light_dir), 0.0);

    // Ambient + diffuse lighting
    let ambient = 0.5;
    let diffuse = 0.5;
    let brightness = ambient + diffuse * ndotl;

    return vec4<f32>(color.rgb * brightness, color.a);
}
"#;
