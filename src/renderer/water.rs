//! Ray-Marched Water Rendering System
//!
//! Renders water as ray-plane intersections rather than per-chunk meshes.
//! This eliminates chunk-boundary seams, camera jitter, and mesh generation overhead.
//!
//! Features:
//! - Multiple water levels (sea, underground lakes, lava pools)
//! - Procedural wave animation (no vertex displacement)
//! - Fresnel-based reflections with skybox fallback
//! - Screen-space reflections (SSR) for nearby geometry
//! - Depth-based refraction with chromatic aberration
//! - Shore foam at water-geometry intersections
//!
//! Pipeline:
//! 1. Fullscreen pass after opaque geometry
//! 2. For each pixel, intersect ray with water planes
//! 3. Depth test against scene to determine visibility
//! 4. Apply all effects in fragment shader

use wgpu::util::DeviceExt;

/// Maximum number of distinct water levels supported
pub const MAX_WATER_LEVELS: usize = 16;

/// Water rendering quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaterQuality {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl WaterQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            WaterQuality::Off => "OFF",
            WaterQuality::Low => "LOW",
            WaterQuality::Medium => "MEDIUM",
            WaterQuality::High => "HIGH",
            WaterQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> WaterQuality {
        match self {
            WaterQuality::Off => WaterQuality::Low,
            WaterQuality::Low => WaterQuality::Medium,
            WaterQuality::Medium => WaterQuality::High,
            WaterQuality::High => WaterQuality::Ultra,
            WaterQuality::Ultra => WaterQuality::Off,
        }
    }

    /// SSR ray march step count
    pub fn ssr_steps(&self) -> u32 {
        match self {
            WaterQuality::Off => 0,
            WaterQuality::Low => 8,
            WaterQuality::Medium => 16,
            WaterQuality::High => 24,
            WaterQuality::Ultra => 32,
        }
    }

    /// Wave noise octaves
    pub fn wave_octaves(&self) -> u32 {
        match self {
            WaterQuality::Off => 1,
            WaterQuality::Low => 2,
            WaterQuality::Medium => 3,
            WaterQuality::High => 3,
            WaterQuality::Ultra => 4,
        }
    }

    /// Refraction distortion scale
    pub fn refraction_distortion(&self) -> f32 {
        match self {
            WaterQuality::Off => 0.0,
            WaterQuality::Low => 0.02,
            WaterQuality::Medium => 0.04,
            WaterQuality::High => 0.06,
            WaterQuality::Ultra => 0.08,
        }
    }

    /// Chromatic aberration strength
    pub fn chromatic_aberration(&self) -> f32 {
        match self {
            WaterQuality::Off | WaterQuality::Low | WaterQuality::Medium => 0.0,
            WaterQuality::High => 0.003,
            WaterQuality::Ultra => 0.006,
        }
    }
}

/// Water parameters for a single water level
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterLevel {
    /// Y height of this water surface
    pub height: f32,
    /// Water type: 0=water, 1=lava (affects color/effects)
    pub water_type: f32,
    /// Reserved for future use
    pub _padding: [f32; 2],
}

impl Default for WaterLevel {
    fn default() -> Self {
        Self {
            height: -1000.0, // Inactive by default
            water_type: 0.0,
            _padding: [0.0; 2],
        }
    }
}

/// Uniform buffer for water rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniforms {
    // Camera data (duplicated for standalone water pass)
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],       // xyz = position, w = time

    // Sun/sky data
    pub sun_direction: [f32; 4],    // xyz = direction, w = intensity
    pub sun_color: [f32; 4],        // rgb = color, a = unused
    pub ambient_color: [f32; 4],    // rgb = ambient, a = brightness

    // Water visual parameters
    pub shallow_color: [f32; 4],    // rgb = color, a = alpha
    pub deep_color: [f32; 4],       // rgb = color, a = alpha

    // Wave parameters
    pub wave_speed: f32,
    pub wave_strength: f32,
    pub fresnel_power: f32,
    pub foam_threshold: f32,

    // Refraction parameters
    pub refraction_distortion: f32,
    pub chromatic_aberration: f32,
    pub refraction_depth_fade: f32,

    // SSR parameters
    pub ssr_steps: u32,
    pub ssr_max_distance: f32,
    pub ssr_thickness: f32,

    // Water level count and quality
    pub water_level_count: u32,
    pub wave_octaves: u32,

    // Screen dimensions for UV calculations
    pub screen_size: [f32; 2],

    // Padding to 16-byte alignment
    pub _padding: [f32; 2],

    // Water levels array (up to MAX_WATER_LEVELS)
    pub water_levels: [WaterLevel; MAX_WATER_LEVELS],
}

impl Default for WaterUniforms {
    fn default() -> Self {
        let mut levels = [WaterLevel::default(); MAX_WATER_LEVELS];
        // Default: single sea-level water at y=32
        levels[0] = WaterLevel {
            height: 32.0,
            water_type: 0.0,
            _padding: [0.0; 2],
        };

        Self {
            view_proj: [[0.0; 4]; 4],
            inv_view_proj: [[0.0; 4]; 4],
            inv_view: [[0.0; 4]; 4],
            inv_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 4],
            sun_direction: [0.0, 1.0, 0.0, 1.0],
            sun_color: [1.0, 0.95, 0.9, 1.0],
            ambient_color: [0.3, 0.35, 0.4, 1.0],
            shallow_color: [0.15, 0.45, 0.50, 0.6],
            deep_color: [0.02, 0.12, 0.20, 0.9],
            wave_speed: 0.4,
            wave_strength: 0.08,
            fresnel_power: 5.0,
            foam_threshold: 0.6,
            refraction_distortion: 0.04,
            chromatic_aberration: 0.0,
            refraction_depth_fade: 12.0,
            ssr_steps: 16,
            ssr_max_distance: 50.0,
            ssr_thickness: 0.5,
            water_level_count: 1,
            wave_octaves: 3,
            screen_size: [1920.0, 1080.0],
            _padding: [0.0; 2],
            water_levels: levels,
        }
    }
}

/// Water rendering system
pub struct Water {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    quality: WaterQuality,
    uniforms: WaterUniforms,
}

impl Water {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout for water-specific resources
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Bind Group Layout"),
            entries: &[
                // binding 0: Water uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: Depth texture (for occlusion and SSR)
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
                // binding 2: Scene texture (for refraction)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create uniform buffer
        let uniforms = WaterUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(WATER_SHADER.into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline (fullscreen triangle, no vertex buffer)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffer - fullscreen triangle from vertex index
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
            // No depth write - water is transparent and composited over scene
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group: None,
            sampler,
            quality: WaterQuality::default(),
            uniforms,
        }
    }

    /// Update bind group with current textures
    pub fn update_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_texture: &wgpu::TextureView,
        scene_texture: &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(scene_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }

    /// Update uniforms from camera and lighting state
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        inv_view: glam::Mat4,
        inv_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        time: f32,
        sun_direction: glam::Vec3,
        sun_intensity: f32,
        sun_color: glam::Vec3,
        ambient_color: glam::Vec3,
        brightness: f32,
        screen_width: f32,
        screen_height: f32,
    ) {
        let inv_view_proj = (view_proj).inverse();

        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.inv_view_proj = inv_view_proj.to_cols_array_2d();
        self.uniforms.inv_view = inv_view.to_cols_array_2d();
        self.uniforms.inv_proj = inv_proj.to_cols_array_2d();
        self.uniforms.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, time];
        self.uniforms.sun_direction = [sun_direction.x, sun_direction.y, sun_direction.z, sun_intensity];
        self.uniforms.sun_color = [sun_color.x, sun_color.y, sun_color.z, 1.0];
        self.uniforms.ambient_color = [ambient_color.x, ambient_color.y, ambient_color.z, brightness];
        self.uniforms.screen_size = [screen_width, screen_height];

        // Apply quality settings
        self.uniforms.ssr_steps = self.quality.ssr_steps();
        self.uniforms.wave_octaves = self.quality.wave_octaves();
        self.uniforms.refraction_distortion = self.quality.refraction_distortion();
        self.uniforms.chromatic_aberration = self.quality.chromatic_aberration();

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    /// Set water levels (for multiple water bodies at different heights)
    pub fn set_water_levels(&mut self, levels: &[(f32, f32)]) {
        // Clear all levels
        for level in &mut self.uniforms.water_levels {
            level.height = -1000.0;
            level.water_type = 0.0;
        }

        // Set active levels
        let count = levels.len().min(MAX_WATER_LEVELS);
        for (i, (height, water_type)) in levels.iter().take(count).enumerate() {
            self.uniforms.water_levels[i] = WaterLevel {
                height: *height,
                water_type: *water_type,
                _padding: [0.0; 2],
            };
        }
        self.uniforms.water_level_count = count as u32;
    }

    /// Get quality setting
    pub fn quality(&self) -> WaterQuality {
        self.quality
    }

    /// Set quality setting
    pub fn set_quality(&mut self, quality: WaterQuality) {
        self.quality = quality;
    }

    /// Cycle through quality presets
    pub fn cycle_quality(&mut self) {
        self.quality = self.quality.next();
    }

    /// Render water (fullscreen pass)
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.quality == WaterQuality::Off {
            return;
        }

        if let Some(ref bind_group) = self.bind_group {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, bind_group, &[]);
            // Draw fullscreen triangle (3 vertices, no buffer)
            render_pass.draw(0..3, 0..1);
        }
    }
}

// Water shader - ray-marched infinite planes with SSR
const WATER_SHADER: &str = r#"
// Maximum water levels (must match Rust constant)
const MAX_WATER_LEVELS: u32 = 16u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Water level data
struct WaterLevel {
    height: f32,
    water_type: f32,  // 0 = water, 1 = lava
    _padding: vec2<f32>,
}

// Main uniform buffer
struct WaterUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,       // xyz = position, w = time

    sun_direction: vec4<f32>,    // xyz = direction, w = intensity
    sun_color: vec4<f32>,
    ambient_color: vec4<f32>,    // rgb = ambient, a = brightness

    shallow_color: vec4<f32>,
    deep_color: vec4<f32>,

    wave_speed: f32,
    wave_strength: f32,
    fresnel_power: f32,
    foam_threshold: f32,

    refraction_distortion: f32,
    chromatic_aberration: f32,
    refraction_depth_fade: f32,

    ssr_steps: u32,
    ssr_max_distance: f32,
    ssr_thickness: f32,

    water_level_count: u32,
    wave_octaves: u32,

    screen_size: vec2<f32>,
    _padding: vec2<f32>,

    // Note: array access in WGSL requires fixed size
    water_levels: array<WaterLevel, 16>,
}

// Camera uniform (shared with main render)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    anim_data: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> water: WaterUniforms;
@group(1) @binding(1) var depth_texture: texture_depth_2d;
@group(1) @binding(2) var scene_texture: texture_2d<f32>;
@group(1) @binding(3) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle from vertex index
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Hash function for noise
fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth noise
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

// Calculate animated wave normal - subtle, calm water
fn get_wave_normal(pos: vec2<f32>, time: f32) -> vec3<f32> {
    let wave_speed = water.wave_speed * 0.5; // Slower waves
    let wave_strength = water.wave_strength * 0.4; // Much gentler

    var dx = 0.0;
    var dz = 0.0;

    // Layer 1: Large slow swells (very gentle)
    let p1 = pos * 0.015 + vec2<f32>(time * wave_speed * 0.3, time * wave_speed * 0.2);
    dx += sin(p1.x * 1.5 + p1.y * 1.0) * 0.4;
    dz += sin(p1.y * 1.5 - p1.x * 0.8) * 0.4;

    // Layer 2: Medium ripples
    let p2 = pos * 0.04 + vec2<f32>(time * wave_speed * 0.5, -time * wave_speed * 0.3);
    dx += sin(p2.x * 2.0 + p2.y * 1.8) * 0.3;
    dz += sin(p2.y * 2.0 - p2.x * 1.5) * 0.3;

    // Layer 3: Small detail (if quality allows)
    if (water.wave_octaves >= 3u) {
        let p3 = pos * 0.08 + vec2<f32>(-time * wave_speed * 0.4, time * wave_speed * 0.5);
        dx += sin(p3.x * 3.0 + p3.y * 2.5) * 0.2;
        dz += sin(p3.y * 2.8 - p3.x * 3.0) * 0.2;
    }

    // Layer 4: Fine detail (Ultra only)
    if (water.wave_octaves >= 4u) {
        let p4 = pos * 0.12 + vec2<f32>(time * wave_speed * 0.6, time * wave_speed * 0.4);
        dx += sin(p4.x * 4.0 + p4.y * 3.5) * 0.1;
        dz += sin(p4.y * 3.5 - p4.x * 4.0) * 0.1;
    }

    return normalize(vec3<f32>(-dx * wave_strength, 1.0, -dz * wave_strength));
}

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_pos = water.inv_view_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Compute procedural sky color for reflection
fn compute_sky_color(dir: vec3<f32>) -> vec3<f32> {
    let brightness = water.ambient_color.a;
    let sun_dir = normalize(water.sun_direction.xyz);
    let sun_intensity = water.sun_direction.w;

    // Sky gradient based on height
    let sky_factor = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    // Base sky colors (warm palette matching the engine's style)
    let horizon_color = vec3<f32>(0.7, 0.75, 0.85) * brightness;
    let zenith_color = vec3<f32>(0.4, 0.55, 0.8) * brightness;
    let night_horizon = vec3<f32>(0.15, 0.12, 0.2) * brightness;
    let night_zenith = vec3<f32>(0.05, 0.05, 0.12) * brightness;

    // Blend day/night based on sun intensity
    let day_sky = mix(horizon_color, zenith_color, sky_factor);
    let night_sky = mix(night_horizon, night_zenith, sky_factor);
    var sky_color = mix(night_sky, day_sky, sun_intensity);

    // Add sun glow
    if (sun_intensity > 0.0) {
        let sun_dot = max(dot(dir, sun_dir), 0.0);
        let sun_glow = pow(sun_dot, 32.0) * sun_intensity * 0.3;
        sky_color += water.sun_color.rgb * sun_glow;
    }

    return sky_color;
}

// Schlick fresnel approximation
fn fresnel(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, water.fresnel_power);
}

// Screen-space reflection ray march
fn trace_ssr(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    if (water.ssr_steps == 0u) {
        return vec4<f32>(0.0);
    }

    let step_count = water.ssr_steps;
    let max_dist = water.ssr_max_distance;
    let thickness = water.ssr_thickness;

    let end_pos = origin + dir * max_dist;

    // Project start and end to screen space
    let start_clip = water.view_proj * vec4<f32>(origin, 1.0);
    let end_clip = water.view_proj * vec4<f32>(end_pos, 1.0);

    let start_ndc = start_clip.xyz / start_clip.w;
    let end_ndc = end_clip.xyz / end_clip.w;

    let start_uv = vec2<f32>(start_ndc.x * 0.5 + 0.5, 0.5 - start_ndc.y * 0.5);
    let end_uv = vec2<f32>(end_ndc.x * 0.5 + 0.5, 0.5 - end_ndc.y * 0.5);

    let delta_uv = (end_uv - start_uv) / f32(step_count);
    let delta_z = (end_ndc.z - start_ndc.z) / f32(step_count);

    var uv = start_uv;
    var ray_z = start_ndc.z;

    for (var i = 0u; i < step_count; i++) {
        uv += delta_uv;
        ray_z += delta_z;

        // Bounds check
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        // Sample depth buffer
        let sample_depth = textureSampleLevel(depth_texture, tex_sampler, uv, 0.0);

        // Check for intersection
        if (ray_z > sample_depth && ray_z - sample_depth < thickness) {
            // Binary search refinement
            var lo_uv = uv - delta_uv;
            var hi_uv = uv;
            var lo_z = ray_z - delta_z;
            var hi_z = ray_z;

            for (var j = 0u; j < 4u; j++) {
                let mid_uv = (lo_uv + hi_uv) * 0.5;
                let mid_z = (lo_z + hi_z) * 0.5;
                let mid_depth = textureSampleLevel(depth_texture, tex_sampler, mid_uv, 0.0);

                if (mid_z > mid_depth) {
                    hi_uv = mid_uv;
                    hi_z = mid_z;
                } else {
                    lo_uv = mid_uv;
                    lo_z = mid_z;
                }
            }

            let hit_uv = (lo_uv + hi_uv) * 0.5;

            // Smooth edge fade - wider margin for cleaner transitions
            let edge_dist = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
            let edge_fade = smoothstep(0.0, 0.15, edge_dist);

            // Distance fade - SSR fades out with distance
            let dist_factor = f32(i) / f32(step_count);
            let dist_fade = 1.0 - smoothstep(0.6, 1.0, dist_factor);

            // Sample scene at hit point
            let hit_color = textureSample(scene_texture, tex_sampler, hit_uv).rgb;
            return vec4<f32>(hit_color, edge_fade * dist_fade * 0.85);
        }
    }

    return vec4<f32>(0.0);
}

// DEBUG MODE:
// 0 = normal water
// 1 = magenta overlay (verify pass runs)
// 2 = diagnostic colors (red=depth fail, green=hit, blue=no intersection)
const DEBUG_MODE: u32 = 0u;

// Main fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // DEBUG: Show magenta overlay to verify shader runs
    if (DEBUG_MODE == 1u) {
        return vec4<f32>(1.0, 0.0, 1.0, 0.5);
    }

    let time = water.camera_pos.w;
    let cam_pos = water.camera_pos.xyz;

    // Reconstruct view ray
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let far_point = water.inv_view_proj * ndc;
    let ray_dir = normalize(far_point.xyz / far_point.w - cam_pos);

    // Sample scene depth
    let scene_depth = textureSampleLevel(depth_texture, tex_sampler, in.uv, 0.0);
    let scene_world_pos = reconstruct_world_pos(in.uv, scene_depth);

    // Find closest water level that intersects this ray
    var hit_t = 1e10;
    var hit_level = -1;
    var hit_height = 0.0;
    var hit_type = 0.0;
    var depth_test_failed = false;

    for (var i = 0u; i < water.water_level_count; i++) {
        let level = water.water_levels[i];
        if (level.height < -999.0) {
            continue; // Inactive level
        }

        // Ray-plane intersection: y = level.height
        // cam_pos.y + t * ray_dir.y = level.height
        // t = (level.height - cam_pos.y) / ray_dir.y

        if (abs(ray_dir.y) < 0.0001) {
            continue; // Ray parallel to water plane
        }

        let t = (level.height - cam_pos.y) / ray_dir.y;

        if (t > 0.0 && t < hit_t) {
            let hit_pos = cam_pos + ray_dir * t;

            // === DEPTH TEST 1: Terrain height check ===
            // If terrain is clearly above water, don't draw water on it
            // Use tighter threshold (0.1 instead of 0.5) for better precision
            if (scene_world_pos.y > level.height + 0.1) {
                depth_test_failed = true;
                continue;
            }

            // === DEPTH TEST 2: NDC depth comparison (most reliable) ===
            // Calculate where water WOULD be in NDC depth space
            let water_clip = water.view_proj * vec4<f32>(hit_pos, 1.0);
            let water_ndc_depth = water_clip.z / water_clip.w;

            // If scene is in front of water in depth buffer, don't render water
            // Use small bias to handle precision issues
            if (scene_depth < water_ndc_depth - 0.0001) {
                depth_test_failed = true;
                continue;
            }

            // === DEPTH TEST 3: Distance comparison (backup) ===
            let water_dist = length(hit_pos - cam_pos);
            let scene_dist = length(scene_world_pos - cam_pos);

            // Water is visible if it's closer than scene geometry
            let dist_bias = max(water_dist, scene_dist) * 0.005;
            if (water_dist < scene_dist + dist_bias) {
                hit_t = t;
                hit_level = i32(i);
                hit_height = level.height;
                hit_type = level.water_type;
            } else {
                depth_test_failed = true;
            }
        }
    }

    // DEBUG MODE 2: Diagnostic colors
    if (DEBUG_MODE == 2u) {
        if (hit_level >= 0) {
            // Hit and passed depth test - green
            return vec4<f32>(0.0, 1.0, 0.0, 0.7);
        } else if (depth_test_failed) {
            // Hit but depth test failed - red
            return vec4<f32>(1.0, 0.0, 0.0, 0.7);
        } else {
            // No intersection (ray parallel or behind camera) - discard to show scene
            discard;
        }
    }

    // No water hit - discard
    if (hit_level < 0) {
        discard;
    }

    // Calculate water surface position
    let water_pos = cam_pos + ray_dir * hit_t;
    let view_dir = -ray_dir;

    // Skip drawing water on vertical shore block faces
    // If scene geometry is at water level (within 1 block) and we're looking at it
    // from a steep horizontal angle, don't draw water overlay on the vertical face
    let scene_at_water_level = abs(scene_world_pos.y - hit_height) < 1.2;
    let looking_horizontal = abs(ray_dir.y) < 0.4; // Steep viewing angle
    if (scene_at_water_level && looking_horizontal) {
        discard;
    }

    // Calculate wave normal
    let wave_normal = get_wave_normal(water_pos.xz, time);

    // Distance-based normal fade (reduce waves at distance for stability)
    let dist = length(water_pos - cam_pos);
    let normal_fade = 1.0 - smoothstep(50.0, 200.0, dist);
    // Blend to flat normal at distance for stable reflections
    let final_normal = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), wave_normal, normal_fade * 0.7));

    // Calculate water depth (vertical distance to underwater geometry)
    let depth_diff = max(0.0, hit_height - scene_world_pos.y);

    // Fresnel - use mostly flat normal for stable reflections
    // Limit the "looking straight down = transparent" effect
    let flat_normal = vec3<f32>(0.0, 1.0, 0.0);
    let fresnel_normal = normalize(mix(flat_normal, final_normal, 0.3));
    let n_dot_v = max(dot(fresnel_normal, view_dir), 0.001);
    // Clamp n_dot_v to prevent extreme transparency when looking straight down
    let clamped_n_dot_v = min(n_dot_v, 0.85);
    let f = fresnel(clamped_n_dot_v, 0.04); // Higher F0 = more reflective baseline

    // Reflection direction
    let reflect_dir = reflect(-view_dir, final_normal);

    // SSR
    let ssr_result = trace_ssr(water_pos + final_normal * 0.1, reflect_dir);
    var reflection_color = compute_sky_color(reflect_dir);
    reflection_color = mix(reflection_color, ssr_result.rgb, ssr_result.a);

    // Refraction (sample scene with distortion)
    var refracted_color = vec3<f32>(0.0);
    if (water.refraction_distortion > 0.001) {
        let distortion = final_normal.xz * water.refraction_distortion;
        let depth_fade = 1.0 - smoothstep(0.0, water.refraction_depth_fade, depth_diff);
        let offset = distortion * depth_fade;

        if (water.chromatic_aberration > 0.001) {
            // Chromatic aberration
            let r_uv = clamp(in.uv + offset * (1.0 + water.chromatic_aberration * 30.0), vec2<f32>(0.0), vec2<f32>(1.0));
            let g_uv = clamp(in.uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
            let b_uv = clamp(in.uv + offset * (1.0 - water.chromatic_aberration * 30.0), vec2<f32>(0.0), vec2<f32>(1.0));
            refracted_color = vec3<f32>(
                textureSample(scene_texture, tex_sampler, r_uv).r,
                textureSample(scene_texture, tex_sampler, g_uv).g,
                textureSample(scene_texture, tex_sampler, b_uv).b
            );
        } else {
            let refract_uv = clamp(in.uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
            refracted_color = textureSample(scene_texture, tex_sampler, refract_uv).rgb;
        }
    } else {
        refracted_color = textureSample(scene_texture, tex_sampler, in.uv).rgb;
    }

    // Depth-based water color
    let brightness = water.ambient_color.a;
    let shallow = water.shallow_color.rgb * brightness;
    let deep = water.deep_color.rgb * brightness;
    let max_depth = 24.0;
    let depth_factor = clamp(depth_diff / max_depth, 0.0, 1.0);
    let water_tint = mix(shallow, deep, depth_factor);

    // Blend refracted scene with water tint
    let underwater_visibility = exp(-depth_diff * 0.2);
    var water_color = mix(water_tint, refracted_color * 0.8 + water_tint * 0.2, underwater_visibility);

    // Final composition: blend water color with reflection based on Fresnel
    // Higher minimum reflection (35%) prevents flat/transparent look when looking down
    let reflection_strength = clamp(f * 1.2 + 0.25, 0.35, 0.90);
    var final_color = mix(water_color, reflection_color, reflection_strength);

    // Specular highlight from sun - subtle, focused
    let sun_dir = normalize(water.sun_direction.xyz);
    let sun_intensity = water.sun_direction.w;
    if (sun_dir.y > 0.0 && sun_intensity > 0.0) {
        let half_vec = normalize(sun_dir + view_dir);
        // Higher power = tighter, smaller highlight
        let spec = pow(max(dot(final_normal, half_vec), 0.0), 256.0);
        // Narrower threshold and reduced intensity
        let sun_spec = smoothstep(0.7, 0.85, spec) * sun_intensity * brightness;
        final_color += water.sun_color.rgb * sun_spec * 0.3;
    }

    // Shore foam
    let shore_foam_intensity = smoothstep(1.5, 0.0, depth_diff) * 0.4;
    let foam_noise = noise2(water_pos.xz * 0.3 + vec2<f32>(time * 0.1, time * 0.07));
    let foam = shore_foam_intensity * smoothstep(water.foam_threshold - 0.1, water.foam_threshold, foam_noise);
    let foam_color = vec3<f32>(0.9, 0.92, 0.95) * brightness;
    final_color = mix(final_color, foam_color, foam);

    // Alpha: water should always be visible, not see-through
    // Base opacity from water depth
    var alpha = mix(0.75, 0.95, depth_factor); // Higher base opacity
    alpha = mix(alpha, 1.0, foam);

    // Shore fade - based on water depth
    let shore_fade = smoothstep(0.0, 2.0, depth_diff);

    // High minimum alpha - water should never be transparent/invisible
    let min_alpha = 0.7;
    let final_alpha = max(min_alpha, alpha) * shore_fade;

    // Ensure water is always at least somewhat visible (except at very edge)
    let final_alpha_clamped = max(final_alpha, shore_fade * 0.6);

    return vec4<f32>(final_color, final_alpha_clamped);
}
"#;
