//! Volumetric Light Shafts (God Rays)
//!
//! Implements world-space raymarching for visible light beams through openings.
//! Uses the existing voxel volume for shadow queries.
//!
//! Phase 1: Debug visualization (bright colors where light beams should appear)
//! Phase 2: Proper in-scattering with Henyey-Greenstein phase function
//! Phase 3: Performance optimizations (half-res, dithering, bilateral blur)

// Constants for volumetric lighting (reserved for future use)
#[allow(dead_code)]
const VOLUMETRIC_STEPS: u32 = 32;
#[allow(dead_code)]
const VOLUMETRIC_DENSITY: f32 = 0.015;

/// Volumetric lighting system
pub struct VolumetricLighting {
    /// Half-resolution texture for volumetric accumulation
    volumetric_texture: wgpu::Texture,
    volumetric_view: wgpu::TextureView,

    /// Non-filtering sampler for depth texture
    depth_sampler: wgpu::Sampler,

    /// Linear sampler for volumetric texture compositing
    sampler: wgpu::Sampler,

    /// Bind group layout for volumetric pass
    bind_group_layout: wgpu::BindGroupLayout,

    /// Bind group (created dynamically with depth texture)
    bind_group: Option<wgpu::BindGroup>,

    /// Pipeline for volumetric raymarching
    raymarch_pipeline: wgpu::RenderPipeline,

    /// Pipeline for compositing volumetrics onto scene (additive blend)
    composite_pipeline: wgpu::RenderPipeline,

    /// Composite bind group layout (just volumetric texture + sampler)
    composite_bind_group_layout: wgpu::BindGroupLayout,

    /// Composite bind group (volumetric texture + sampler)
    composite_bind_group: wgpu::BindGroup,

    /// Current dimensions (half of screen size)
    width: u32,
    height: u32,

    /// Whether volumetrics are enabled
    pub enabled: bool,
}

/// Uniform data for volumetric raymarching
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricUniform {
    /// Inverse view-projection matrix for world position reconstruction
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera position in world space
    pub camera_pos: [f32; 4],
    /// Sun direction (xyz) and intensity (w)
    pub sun_direction: [f32; 4],
    /// Sun color (rgb) and density (a)
    pub sun_color: [f32; 4],
    /// Voxel volume center (xyz) and size (w)
    pub volume_center: [f32; 4],
    /// Screen dimensions (xy) and step count (z), time (w)
    pub screen_params: [f32; 4],
}

impl VolumetricLighting {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        voxel_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Use half resolution for performance
        let vol_width = width / 2;
        let vol_height = height / 2;

        // Create volumetric accumulation texture
        let volumetric_texture = Self::create_volumetric_texture(device, vol_width, vol_height);
        let volumetric_view = volumetric_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Non-filtering sampler for depth texture (required for depth textures)
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetric_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Linear sampler for volumetric texture compositing
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetric_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group layout for raymarch pass:
        // - Group 0: Volumetric uniform + depth texture + sampler
        // - Group 1: Voxel GI (reused from existing system)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_bind_group_layout"),
            entries: &[
                // Volumetric uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Depth texture
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
                // Sampler for depth (non-filtering required for depth textures)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Composite bind group layout (just volumetric texture + sampler for additive blend)
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_composite_bind_group_layout"),
            entries: &[
                // Volumetric texture
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
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create composite bind group
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_composite_bind_group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&volumetric_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create pipelines
        let raymarch_pipeline = Self::create_raymarch_pipeline(
            device,
            &bind_group_layout,
            voxel_bind_group_layout,
        );
        let composite_pipeline = Self::create_composite_pipeline(
            device,
            &composite_bind_group_layout,
            surface_format,
        );

        Self {
            volumetric_texture,
            volumetric_view,
            depth_sampler,
            sampler,
            bind_group_layout,
            bind_group: None,
            raymarch_pipeline,
            composite_pipeline,
            composite_bind_group_layout,
            composite_bind_group,
            width: vol_width,
            height: vol_height,
            enabled: false,  // Default OFF - toggle with V key
        }
    }

    fn create_volumetric_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volumetric_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_raymarch_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        voxel_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_raymarch_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_RAYMARCH_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_raymarch_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout, voxel_bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_raymarch_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    fn create_composite_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_COMPOSITE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_composite_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_composite_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Additive blending: output = src + dst
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    /// Create bind group with depth texture (called each frame or on resize)
    pub fn create_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));
    }


    /// Render volumetric light shafts
    /// output_view: where to composite the volumetrics (usually scene_texture_view)
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        voxel_bind_group: &wgpu::BindGroup,
    ) {
        if !self.enabled {
            return;
        }

        let Some(bind_group) = &self.bind_group else {
            return;
        };

        // Pass 1: Raymarch volumetrics at half resolution
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_raymarch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.volumetric_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.raymarch_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_bind_group(1, voxel_bind_group, &[]);
            pass.draw(0..3, 0..1); // Fullscreen triangle
        }

        // Pass 2: Composite volumetrics onto scene using additive blending
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Keep existing scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Resize volumetric textures
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let vol_width = width / 2;
        let vol_height = height / 2;

        if vol_width == self.width && vol_height == self.height {
            return;
        }

        self.width = vol_width;
        self.height = vol_height;

        self.volumetric_texture = Self::create_volumetric_texture(device, vol_width, vol_height);
        self.volumetric_view = self.volumetric_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate composite bind group with new volumetric texture view
        self.composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_composite_bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.volumetric_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Clear raymarch bind group (needs to be recreated with new depth texture)
        self.bind_group = None;
    }

    /// Toggle volumetrics on/off
    pub fn toggle(&mut self) -> bool {
        self.enabled = !self.enabled;
        self.enabled
    }

    /// Set volumetrics enabled state
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

// =============================================================================
// VOLUMETRIC RAYMARCH SHADER
// =============================================================================
// Phase 1: Debug visualization - bright magenta where light beams should appear
// This validates the raymarching logic before we add proper in-scattering
// =============================================================================

const VOLUMETRIC_RAYMARCH_SHADER: &str = r#"
// Volumetric uniform
struct VolumetricUniform {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_direction: vec4<f32>,  // xyz = direction, w = intensity
    sun_color: vec4<f32>,      // rgb = color, a = density
    volume_center: vec4<f32>,  // xyz = center, w = size
    screen_params: vec4<f32>,  // xy = dimensions, z = step_count, w = time
};

// Voxel volume uniform (from voxel_gi)
struct VoxelVolumeUniform {
    volume_center: vec4<f32>,
    volume_size: vec4<f32>,
    inv_volume_size: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,
    time_of_day: vec4<f32>,
    player_light: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> volumetric: VolumetricUniform;
@group(0) @binding(1)
var t_depth: texture_depth_2d;
@group(0) @binding(2)
var s_depth: sampler;

// Voxel GI bindings (group 1)
@group(1) @binding(0)
var<uniform> voxel_volume: VoxelVolumeUniform;
@group(1) @binding(1)
var t_voxel: texture_3d<f32>;
@group(1) @binding(2)
var t_radiance: texture_3d<f32>;
@group(1) @binding(3)
var s_voxel: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Convert world position to voxel UV coordinates
fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    let volume_center = voxel_volume.volume_center.xyz;
    let volume_size = voxel_volume.volume_size.x;
    let half_size = volume_size * 0.5;

    // Convert world position to volume-local coordinates
    let local_pos = world_pos - volume_center + vec3<f32>(half_size);

    // Convert to UV (0-1 range)
    return local_pos / volume_size;
}

// Check if position is inside voxel volume
fn is_in_volume(uv: vec3<f32>) -> bool {
    return all(uv >= vec3<f32>(0.0)) && all(uv <= vec3<f32>(1.0));
}

// Sample voxel opacity at world position
fn sample_voxel_opacity(world_pos: vec3<f32>) -> f32 {
    let uv = world_to_voxel_uv(world_pos);
    if (!is_in_volume(uv)) {
        return 0.0; // Outside volume = air
    }
    let voxel = textureSampleLevel(t_voxel, s_voxel, uv, 0.0);
    return voxel.r; // R channel = opacity
}

// Simple shadow test: march toward sun and check for occlusion
// Returns 1.0 if in light, 0.0 if in shadow
// Uses snapped positions for world-stable results
fn shadow_test(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let max_distance = 48.0;  // Maximum shadow ray distance
    let num_steps = 12;       // Shadow ray steps (reduced for performance)
    let step_size = max_distance / f32(num_steps);
    let snap_grid = 1.0;      // Snap to 1-block grid for stability

    var pos = world_pos + sun_dir * 1.0; // Start slightly offset to avoid self-shadowing

    for (var i = 0; i < num_steps; i++) {
        // Snap to world grid for stable shadow queries
        let snapped = floor(pos / snap_grid + 0.5) * snap_grid;
        let opacity = sample_voxel_opacity(snapped);
        if (opacity > 0.5) {
            return 0.0; // Hit solid voxel = in shadow
        }
        pos += sun_dir * step_size;
    }

    return 1.0; // No occlusion = in light
}

// Henyey-Greenstein phase function for anisotropic scattering
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(denom, 1.5));
}

// Integer hash for deterministic noise
fn hash_int(n: u32) -> u32 {
    var x = n;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// World-space 3D noise for stable dithering
// Uses integer world position so rays don't jitter as camera moves
fn world_space_noise(world_pos: vec3<i32>) -> f32 {
    // Use integer hash based on world grid cell
    let hash_val = hash_int(u32(world_pos.x + 7919) ^ hash_int(u32(world_pos.y + 4967) ^ hash_int(u32(world_pos.z + 6971))));
    return f32(hash_val & 0xFFFFu) / 65535.0;
}

// Check if there's any occlusion above a point (for outdoor suppression)
// Returns 0.0 = open sky, 1.0 = fully occluded
fn sky_occlusion(world_pos: vec3<f32>) -> f32 {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let check_height = 32.0;
    let num_checks = 8;
    let step_h = check_height / f32(num_checks);

    var pos = world_pos;
    for (var i = 0; i < num_checks; i++) {
        pos += up * step_h;
        let opacity = sample_voxel_opacity(pos);
        if (opacity > 0.5) {
            return 1.0; // Found occlusion above
        }
    }
    return 0.0; // Open sky
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample depth
    let depth = textureSampleLevel(t_depth, s_depth, in.uv, 0.0);

    // Skip sky pixels (depth = 1.0)
    if (depth >= 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Reconstruct world position from depth
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_h = volumetric.inv_view_proj * ndc;
    let world_pos = world_h.xyz / world_h.w;

    // Camera position
    let camera_pos = volumetric.camera_pos.xyz;

    // Ray from camera to world position
    let ray_dir = normalize(world_pos - camera_pos);
    let ray_length = length(world_pos - camera_pos);

    // Sun direction and intensity
    let sun_dir = normalize(volumetric.sun_direction.xyz);
    let sun_intensity = volumetric.sun_direction.w;

    // Skip if sun is below horizon (no god rays at night)
    if (sun_intensity < 0.1) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Raymarch parameters
    let num_steps = 40;
    let max_ray_dist = min(ray_length, 80.0);
    let step_size = max_ray_dist / f32(num_steps);
    let density = volumetric.sun_color.a;

    // Accumulate in-scattering with EDGE DETECTION for sharp beams
    var accumulated_light = vec3<f32>(0.0);
    var pos = camera_pos;
    var prev_in_light = 0.0;
    var found_shadow = false;  // Track if we've seen any shadow

    // Phase function - very mild forward scattering
    let cos_theta = dot(ray_dir, sun_dir);
    let phase = henyey_greenstein(cos_theta, 0.2);

    // World-space noise dithering anchored to surface position (stable as camera moves)
    let noise = world_space_noise(vec3<i32>(floor(world_pos)));
    pos += ray_dir * step_size * noise;

    for (var i = 0; i < num_steps; i++) {
        pos += ray_dir * step_size;

        // Snap sample position to world grid for stability
        // This ensures the same world locations are sampled regardless of camera angle
        let snap_grid = 0.5;
        let snapped_pos = floor(pos / snap_grid + 0.5) * snap_grid;

        // Check if we've passed the surface
        let dist_from_cam = length(pos - camera_pos);
        if (dist_from_cam > max_ray_dist) {
            break;
        }

        // Sample voxel opacity at snapped position for stable shadows
        let opacity = sample_voxel_opacity(snapped_pos);

        // Only process in air
        if (opacity < 0.3) {
            // Check if this point can see the sun (use snapped position for stability)
            let in_light = shadow_test(snapped_pos, sun_dir);

            // Track if we've encountered shadow
            if (in_light < 0.5) {
                found_shadow = true;
            }

            // === BEAM EDGE DETECTION ===
            // Only show strong scattering near shadow/light transitions
            // This creates DEFINED beam edges instead of uniform wash
            let light_delta = abs(in_light - prev_in_light);
            let edge_factor = light_delta * 3.0;  // Boost at transitions

            // Also allow contribution if we're in light AND have seen shadow
            // This shows the beam interior, not just edges
            var contribution = 0.0;
            if (in_light > 0.5 && found_shadow) {
                // In lit region after seeing shadow = inside a beam
                contribution = phase * density * step_size * (0.3 + edge_factor);
                accumulated_light += volumetric.sun_color.rgb * contribution * sun_intensity;
            }

            prev_in_light = in_light;
        }
    }

    // If we never found shadow, we're in open sunlight - no beams
    if (!found_shadow) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Very subtle clamp - keep rays thin and defined
    accumulated_light = min(accumulated_light, vec3<f32>(0.6));

    return vec4<f32>(accumulated_light, 1.0);
}
"#;

// =============================================================================
// VOLUMETRIC COMPOSITE SHADER
// =============================================================================

const VOLUMETRIC_COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0)
var t_volumetric: texture_2d<f32>;
@group(0) @binding(1)
var s_tex: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample volumetric texture (half resolution, linear interpolation)
    let volumetric_color = textureSample(t_volumetric, s_tex, in.uv).rgb;

    // Output for additive blending onto scene
    return vec4<f32>(volumetric_color, 1.0);
}
"#;
