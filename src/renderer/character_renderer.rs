//! Character entity rendering with skeletal animation support.
//!
//! Renders skinned characters using a storage buffer for bone matrices.
//! Phase 1: static T-pose rendering. Animation comes in Phase 2.

use glam::{Mat4, Vec3, Quat};
use wgpu::util::DeviceExt;

use crate::entity::loader::CharacterVertex;

/// Maximum characters to render simultaneously.
const MAX_CHARACTERS: usize = 64;

/// Maximum bones per character skeleton.
pub const MAX_BONES_PER_CHARACTER: usize = 128;

/// Total bone matrix storage (all characters).
const MAX_BONE_MATRICES: usize = MAX_CHARACTERS * MAX_BONES_PER_CHARACTER;

/// Per-instance data sent to the GPU via vertex buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CharacterInstance {
    /// Model matrix column 0.
    pub model_c0: [f32; 4],
    /// Model matrix column 1.
    pub model_c1: [f32; 4],
    /// Model matrix column 2.
    pub model_c2: [f32; 4],
    /// Model matrix column 3 (translation).
    pub model_c3: [f32; 4],
    /// Offset into the bone matrix storage buffer (in number of matrices).
    pub bone_offset: u32,
    /// Padding to align to 16 bytes.
    pub _pad: [u32; 3],
}

impl CharacterInstance {
    pub fn new(position: Vec3, rotation: Quat, scale: f32, bone_offset: u32) -> Self {
        let model = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            rotation,
            position,
        );
        let cols = model.to_cols_array_2d();
        Self {
            model_c0: cols[0],
            model_c1: cols[1],
            model_c2: cols[2],
            model_c3: cols[3],
            bone_offset,
            _pad: [0; 3],
        }
    }
}

/// Mesh slot for a loaded character mesh on the GPU.
pub struct CharacterMeshSlot {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub vertex_count: u32,
}

/// Renders skinned character entities.
pub struct CharacterRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Bone matrices storage buffer (all entities share this).
    bone_buffer: wgpu::Buffer,
    /// Per-character instance buffer.
    instance_buffer: wgpu::Buffer,
    /// Bind group containing bone matrices + light data.
    character_bind_group: wgpu::BindGroup,
    /// Loaded character meshes.
    mesh_slots: Vec<CharacterMeshSlot>,
    /// Current instance count for rendering.
    instance_count: u32,
    /// Bind group layout (needed for texture bind group creation).
    pub character_bind_group_layout: wgpu::BindGroupLayout,
    /// Character texture bind group (single texture for Phase 1).
    texture_bind_group: Option<wgpu::BindGroup>,
    texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl CharacterRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer: &wgpu::Buffer,
        voxel_gi_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Character Shader"),
            source: wgpu::ShaderSource::Wgsl(CHARACTER_SHADER.into()),
        });

        // Bind group layout for bone matrices + light uniforms
        let character_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Character Bind Group Layout"),
            entries: &[
                // Bone matrices storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Light uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Texture bind group layout (character texture array)
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Character Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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

        // Pipeline layout: camera(0), character_data(1), texture(2), voxel_gi(3)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Character Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                &character_bind_group_layout,
                &texture_bind_group_layout,
                voxel_gi_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Vertex layout (per-vertex data from FBX)
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CharacterVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal: vec3<f32>
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // tex_coords: vec2<f32>
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // bone_indices: vec4<u8> (as uint)
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint8x4,
                },
                // bone_weights: vec4<u8> (as unorm -> 0.0-1.0)
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
                // texture_layer: u32
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        // Instance layout (per-character)
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CharacterInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // model_c0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // model_c1
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // model_c2
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // model_c3
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // bone_offset
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Character Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_character",
                buffers: &[vertex_layout, instance_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_character",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Disabled for debugging - render both sides
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

        // Bone matrices storage buffer
        let bone_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Character Bone Matrix Buffer"),
            size: (MAX_BONE_MATRICES * std::mem::size_of::<[f32; 16]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Character Instance Buffer"),
            size: (MAX_CHARACTERS * std::mem::size_of::<CharacterInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Character bind group
        let character_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Character Bind Group"),
            layout: &character_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bone_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bone_buffer,
            instance_buffer,
            character_bind_group,
            mesh_slots: Vec::new(),
            instance_count: 0,
            character_bind_group_layout,
            texture_bind_group: None,
            texture_bind_group_layout,
        }
    }

    /// Upload a character mesh to the GPU, returns the mesh slot index.
    pub fn upload_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[CharacterVertex],
        indices: &[u32],
    ) -> usize {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Character Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Character Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let slot_index = self.mesh_slots.len();
        self.mesh_slots.push(CharacterMeshSlot {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
        });

        slot_index
    }

    /// Replace an existing mesh slot with new data (for live editing).
    pub fn replace_mesh(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        vertices: &[CharacterVertex],
        indices: &[u32],
    ) {
        if slot >= self.mesh_slots.len() {
            return;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Character Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Character Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.mesh_slots[slot] = CharacterMeshSlot {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
        };
    }

    /// Load multiple character textures into a texture array.
    /// All textures must be the same size. If sizes differ, they will be resized to match the first.
    pub fn load_texture_array(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        paths: &[std::path::PathBuf],
    ) -> Result<(), String> {
        if paths.is_empty() {
            return Err("No texture paths provided".to_string());
        }

        // Load all images and determine the target size from the first one
        let mut images: Vec<image::RgbaImage> = Vec::with_capacity(paths.len());
        let mut target_width = 0u32;
        let mut target_height = 0u32;

        for (i, path) in paths.iter().enumerate() {
            let img_data = std::fs::read(path)
                .map_err(|e| format!("Failed to read texture '{}': {}", path.display(), e))?;
            let img = image::load_from_memory(&img_data)
                .map_err(|e| format!("Failed to decode texture '{}': {}", path.display(), e))?;
            let rgba = img.to_rgba8();

            if i == 0 {
                target_width = rgba.width();
                target_height = rgba.height();
            }

            // Resize if needed to match first texture
            let final_rgba = if rgba.width() != target_width || rgba.height() != target_height {
                image::imageops::resize(&rgba, target_width, target_height, image::imageops::FilterType::Lanczos3)
            } else {
                rgba
            };

            images.push(final_rgba);
        }

        let layer_count = images.len() as u32;

        // Create texture array
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Character Texture Array"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each layer
        for (i, rgba) in images.iter().enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                rgba.as_raw(),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * target_width),
                    rows_per_image: Some(target_height),
                },
                wgpu::Extent3d {
                    width: target_width,
                    height: target_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Character Texture Array Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Character Texture Array Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        self.texture_bind_group = Some(bind_group);
        tracing::info!("Loaded character texture array: {} layers at {}x{}", layer_count, target_width, target_height);
        Ok(())
    }

    /// Create a placeholder texture array for untextured models.
    pub fn create_placeholder_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Character Placeholder Texture Array"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1, // Single layer placeholder
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Light gray placeholder
        let pixels: Vec<u8> = vec![200, 200, 200, 255].repeat(16);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Character Placeholder Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.texture_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Character Placeholder Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        }));
    }

    /// Upload bone matrices for all active entities.
    pub fn update_bones(
        &self,
        queue: &wgpu::Queue,
        matrices: &[Mat4],
    ) {
        if !matrices.is_empty() {
            let data: Vec<[f32; 16]> = matrices.iter()
                .map(|m| m.to_cols_array())
                .collect();
            queue.write_buffer(&self.bone_buffer, 0, bytemuck::cast_slice(&data));
        }
    }

    /// Upload per-character instance data.
    pub fn update_instances(
        &mut self,
        queue: &wgpu::Queue,
        instances: &[CharacterInstance],
    ) {
        self.instance_count = instances.len().min(MAX_CHARACTERS) as u32;
        if !instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances[..self.instance_count as usize]),
            );
        }
    }

    /// Render all character entities.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        voxel_gi_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.instance_count == 0 || self.mesh_slots.is_empty() {
            return;
        }

        let texture_bg = match &self.texture_bind_group {
            Some(bg) => bg,
            None => return,
        };

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.character_bind_group, &[]);
        render_pass.set_bind_group(2, texture_bg, &[]);
        render_pass.set_bind_group(3, voxel_gi_bind_group, &[]);

        // Render each mesh slot (Phase 1: typically just one mesh)
        for slot in &self.mesh_slots {
            render_pass.set_vertex_buffer(0, slot.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(slot.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..slot.index_count, 0, 0..self.instance_count);
        }
    }

    /// Get number of loaded mesh slots.
    pub fn mesh_slot_count(&self) -> usize {
        self.mesh_slots.len()
    }

    /// Check if any meshes are loaded.
    pub fn has_meshes(&self) -> bool {
        !self.mesh_slots.is_empty()
    }
}

/// WGSL shader for skinned character rendering.
const CHARACTER_SHADER: &str = r#"
// Camera uniforms (shared with all passes)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
}

// Light uniforms (shared with chunk pass)
struct LightUniform {
    shadow_view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    moon_direction: vec4<f32>,
    moon_color: vec4<f32>,
    ambient_color: vec4<f32>,
}

// Voxel volume uniform (same as chunk shader)
struct VoxelVolumeUniform {
    volume_center: vec4<f32>,
    volume_size: vec4<f32>,
    inv_volume_size: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    time_of_day: vec4<f32>,
    player_light: vec4<f32>,
    volume_origin: vec4<i32>,
    player_shadow_box: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> bone_matrices: array<mat4x4<f32>>;

@group(1) @binding(1)
var<uniform> light: LightUniform;

@group(2) @binding(0)
var char_texture: texture_2d_array<f32>;
@group(2) @binding(1)
var char_sampler: sampler;

// Voxel GI group (shadows)
@group(3) @binding(0)
var<uniform> voxel_volume: VoxelVolumeUniform;
@group(3) @binding(1)
var t_voxel: texture_3d<f32>;
@group(3) @binding(2)
var t_radiance: texture_3d<f32>;
@group(3) @binding(3)
var s_voxel: sampler;

// Per-vertex attributes
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) bone_indices: vec4<u32>,
    @location(4) bone_weights: vec4<f32>,  // Unorm8x4 auto-converts to 0.0-1.0
    @location(10) texture_layer: u32,
}

// Per-instance attributes
struct InstanceInput {
    @location(5) model_c0: vec4<f32>,
    @location(6) model_c1: vec4<f32>,
    @location(7) model_c2: vec4<f32>,
    @location(8) model_c3: vec4<f32>,
    @location(9) bone_offset: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) @interpolate(flat) texture_layer: u32,
}

// === VOXEL GI SHADOW FUNCTIONS ===

fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    let center = voxel_volume.volume_center.xyz;
    let half_size = 128.0;
    let local = world_pos - center + vec3<f32>(half_size);
    return local / 256.0;
}

fn voxel_in_bounds(v: vec3<i32>) -> bool {
    return all(v >= vec3<i32>(0)) && all(v < vec3<i32>(256));
}

fn char_ray_march_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let max_steps = 96;
    let center = voxel_volume.volume_center.xyz;

    // Start offset to escape the character's own voxels
    let start = world_pos + light_dir * 1.2;
    let local_start = start - center + vec3<f32>(128.0);

    var voxel = vec3<i32>(floor(local_start));
    let step_sign = vec3<i32>(sign(light_dir));
    let abs_dir = max(abs(light_dir), vec3<f32>(1e-8));
    let t_delta = vec3<f32>(1.0) / abs_dir;

    let frac = local_start - vec3<f32>(floor(local_start));
    var t_max: vec3<f32>;
    if (light_dir.x >= 0.0) { t_max.x = (1.0 - frac.x) * t_delta.x; } else { t_max.x = frac.x * t_delta.x; }
    if (light_dir.y >= 0.0) { t_max.y = (1.0 - frac.y) * t_delta.y; } else { t_max.y = frac.y * t_delta.y; }
    if (light_dir.z >= 0.0) { t_max.z = (1.0 - frac.z) * t_delta.z; } else { t_max.z = frac.z * t_delta.z; }

    for (var i = 0; i < max_steps; i++) {
        if (!voxel_in_bounds(voxel)) { break; }

        // Distance check using max-component (no sqrt)
        let vd = abs(vec3<f32>(voxel) - vec3<f32>(127.5));
        let center_dist = max(vd.x, max(vd.y, vd.z));
        if (center_dist > 120.0) { break; }

        // Integer textureLoad: exact voxel value, no interpolation
        let voxel_data = textureLoad(t_voxel, voxel, 0);

        if (voxel_data.r > 0.5) {
            return 0.2;  // Hit solid = in shadow
        }

        // Amanatides & Woo step
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                voxel.x += step_sign.x;
                t_max.x += t_delta.x;
            } else {
                voxel.z += step_sign.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                voxel.y += step_sign.y;
                t_max.y += t_delta.y;
            } else {
                voxel.z += step_sign.z;
                t_max.z += t_delta.z;
            }
        }
    }

    return 1.0;  // No occlusion
}

// === VERTEX SHADER ===

@vertex
fn vs_character(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Reconstruct model matrix from instance columns
    let model = mat4x4<f32>(
        instance.model_c0,
        instance.model_c1,
        instance.model_c2,
        instance.model_c3,
    );

    // Skinning: blend bone matrices weighted by vertex bone weights
    let j = vertex.bone_indices;
    let w = vertex.bone_weights;
    let o = instance.bone_offset;

    let skin_mat =
        bone_matrices[o + j.x] * w.x +
        bone_matrices[o + j.y] * w.y +
        bone_matrices[o + j.z] * w.z +
        bone_matrices[o + j.w] * w.w;

    // Apply skinning then model transform
    let skinned_pos = skin_mat * vec4<f32>(vertex.position, 1.0);
    let world_pos = model * skinned_pos;

    out.clip_position = camera.view_proj * world_pos;
    out.tex_coords = vertex.tex_coords;
    out.world_position = world_pos.xyz;
    out.texture_layer = vertex.texture_layer;

    // Transform normal (using upper-left 3x3 of skin * model)
    let skin_normal = (skin_mat * vec4<f32>(vertex.normal, 0.0)).xyz;
    let model_normal = (model * vec4<f32>(skin_normal, 0.0)).xyz;
    out.world_normal = normalize(model_normal);

    return out;
}

// === FRAGMENT SHADER ===

@fragment
fn fs_character(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample character texture from array using per-vertex layer index
    let albedo = textureSample(char_texture, char_sampler, in.tex_coords, in.texture_layer);

    // Discard transparent pixels
    if albedo.a < 0.5 {
        discard;
    }

    // Day/night aware lighting
    let normal = normalize(in.world_normal);
    let sun_dir = normalize(light.sun_direction.xyz);
    let sun_intensity = light.sun_direction.w;
    let moon_dir = normalize(light.moon_direction.xyz);
    let moon_intensity = light.moon_direction.w;
    let ambient = light.ambient_color.rgb;

    // Shadow from voxel GI (only when sun is up)
    var shadow = 1.0;
    let shadow_blend = voxel_volume.time_of_day.w;
    if (sun_intensity > 0.05 && shadow_blend > 0.0) {
        let voxel_sun_dir = normalize(voxel_volume.sun_direction.xyz);
        shadow = char_ray_march_shadow(in.world_position, voxel_sun_dir);
        shadow = mix(1.0, shadow, shadow_blend);
    }

    // Diffuse lighting from sun (modulated by shadow)
    let sun_ndotl = max(dot(normal, sun_dir), 0.0);
    let sun_diffuse = light.sun_color.rgb * sun_ndotl * sun_intensity * shadow;

    // Diffuse lighting from moon (dimmer, no shadows)
    let moon_ndotl = max(dot(normal, moon_dir), 0.0);
    let moon_diffuse = light.moon_color.rgb * moon_ndotl * moon_intensity * 0.3;

    // Hemisphere ambient (brighter from above)
    let up_factor = dot(normal, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    let hemisphere_ambient = ambient * mix(0.6, 1.0, up_factor);

    let lighting = hemisphere_ambient + sun_diffuse + moon_diffuse;

    return vec4<f32>(albedo.rgb * lighting, albedo.a);
}
"#;
