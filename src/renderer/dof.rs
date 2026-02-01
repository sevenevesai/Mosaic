//! Depth of Field (DoF) Post-Processing
//!
//! Implements physically-motivated DoF with separate near/far blur layers
//! to prevent color bleeding across depth discontinuities.
//!
//! Pipeline:
//! 1. CoC pass: Calculate Circle of Confusion (signed - negative=near, positive=far)
//! 2. Downsample: Half-res color and CoC with proper filtering
//! 3. Far blur: Disc blur for pixels behind focus plane
//! 4. Near blur: Disc blur for pixels in front of focus plane (with CoC dilation)
//! 5. Composite: Blend blurred layers back to full-res using full-res CoC

// Note: DeviceExt not needed for this module

/// DoF quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DofQuality {
    Off,
    #[default]
    Low,
    Medium,
    High,
    Ultra,
}

impl DofQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            DofQuality::Off => "OFF",
            DofQuality::Low => "LOW",
            DofQuality::Medium => "MEDIUM",
            DofQuality::High => "HIGH",
            DofQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> DofQuality {
        match self {
            DofQuality::Off => DofQuality::Low,
            DofQuality::Low => DofQuality::Medium,
            DofQuality::Medium => DofQuality::High,
            DofQuality::High => DofQuality::Ultra,
            DofQuality::Ultra => DofQuality::Off,
        }
    }

    /// Get blur sample count for this quality level
    pub fn sample_count(&self) -> u32 {
        match self {
            DofQuality::Off => 0,
            DofQuality::Low => 16,
            DofQuality::Medium => 24,
            DofQuality::High => 32,
            DofQuality::Ultra => 48,
        }
    }

    /// Get maximum CoC radius in pixels for this quality level
    pub fn max_coc_radius(&self) -> f32 {
        match self {
            DofQuality::Off => 0.0,
            DofQuality::Low => 10.0,
            DofQuality::Medium => 15.0,
            DofQuality::High => 18.0,
            DofQuality::Ultra => 22.0,
        }
    }

    /// Whether to enable near-field blur for this quality level
    pub fn near_blur_enabled(&self) -> bool {
        match self {
            DofQuality::Off | DofQuality::Low => false,
            DofQuality::Medium | DofQuality::High | DofQuality::Ultra => true,
        }
    }

    /// Whether to use hexagonal bokeh (vs circular)
    pub fn hexagonal_bokeh(&self) -> bool {
        match self {
            DofQuality::Off | DofQuality::Low | DofQuality::Medium => false,
            DofQuality::High | DofQuality::Ultra => true,
        }
    }

    /// Whether to enable highlight boost
    pub fn highlight_boost_enabled(&self) -> bool {
        matches!(self, DofQuality::Ultra)
    }
}

/// Focus mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusMode {
    #[default]
    Auto,
    Manual,
    Locked,
}

impl FocusMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            FocusMode::Auto => "AUTO",
            FocusMode::Manual => "MANUAL",
            FocusMode::Locked => "LOCKED",
        }
    }

    pub fn next(&self) -> FocusMode {
        match self {
            FocusMode::Auto => FocusMode::Manual,
            FocusMode::Manual => FocusMode::Locked,
            FocusMode::Locked => FocusMode::Auto,
        }
    }
}

/// Near blur intensity setting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NearBlur {
    Off,
    #[default]
    Subtle,
    Full,
}

impl NearBlur {
    pub fn as_str(&self) -> &'static str {
        match self {
            NearBlur::Off => "OFF",
            NearBlur::Subtle => "SUBTLE",
            NearBlur::Full => "FULL",
        }
    }

    pub fn next(&self) -> NearBlur {
        match self {
            NearBlur::Off => NearBlur::Subtle,
            NearBlur::Subtle => NearBlur::Full,
            NearBlur::Full => NearBlur::Off,
        }
    }

    /// Get near CoC scale factor
    pub fn coc_scale(&self) -> f32 {
        match self {
            NearBlur::Off => 0.0,
            NearBlur::Subtle => 0.5,
            NearBlur::Full => 1.0,
        }
    }
}

/// Bokeh shape
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BokehShape {
    #[default]
    Circle,
    Hexagon,
}

impl BokehShape {
    pub fn as_str(&self) -> &'static str {
        match self {
            BokehShape::Circle => "CIRCLE",
            BokehShape::Hexagon => "HEXAGON",
        }
    }

    pub fn next(&self) -> BokehShape {
        match self {
            BokehShape::Circle => BokehShape::Hexagon,
            BokehShape::Hexagon => BokehShape::Circle,
        }
    }
}

/// DoF parameters (runtime adjustable)
#[derive(Debug, Clone, Copy)]
pub struct DofParams {
    /// Focus mode (Auto/Manual/Locked)
    pub focus_mode: FocusMode,
    /// Manual focus distance in blocks (1-500)
    pub manual_focus_distance: f32,
    /// Aperture / blur strength (0.0-1.0)
    pub aperture: f32,
    /// Focus transition width in blocks (1.0-30.0)
    pub focus_transition: f32,
    /// Maximum blur radius in pixels (4-24)
    pub max_blur_radius: f32,
    /// Near blur setting
    pub near_blur: NearBlur,
    /// Bokeh shape
    pub bokeh_shape: BokehShape,
    /// Highlight boost (0.0-0.5)
    pub highlight_boost: f32,
    /// Autofocus smoothing time constant in seconds (0.05-0.5)
    pub autofocus_speed: f32,
    /// Autofocus sample area size (1, 3, 5, 7, or 9)
    pub autofocus_area: u32,
}

impl Default for DofParams {
    fn default() -> Self {
        Self {
            focus_mode: FocusMode::Manual, // Start with manual so user can see effect
            manual_focus_distance: 15.0,   // Focus at 15 blocks
            aperture: 0.8,                 // Strong blur
            focus_transition: 20.0,        // 20 blocks to reach full blur
            max_blur_radius: 12.0,
            near_blur: NearBlur::Subtle,
            bokeh_shape: BokehShape::Circle,
            highlight_boost: 0.1,
            autofocus_speed: 0.2,
            autofocus_area: 5,
        }
    }
}

/// Uniform data for CoC calculation shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CocUniform {
    /// Inverse projection matrix for view-space reconstruction
    pub inv_proj: [[f32; 4]; 4],
    /// Focus distance in view space (negative Z)
    pub focus_distance: f32,
    /// Aperture / blur strength
    pub aperture: f32,
    /// Transition sharpness (1 / focus_transition)
    pub transition_sharpness: f32,
    /// Maximum CoC radius in pixels
    pub max_coc_radius: f32,
    /// Near CoC scale factor (from near_blur setting)
    pub near_coc_scale: f32,
    /// Near plane distance
    pub near_plane: f32,
    /// Far plane distance
    pub far_plane: f32,
    /// Padding
    pub _padding: f32,
}

/// Uniform data for downsample shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DownsampleUniform {
    /// 1.0 / source_width, 1.0 / source_height
    pub texel_size: [f32; 2],
    /// Padding
    pub _padding: [f32; 2],
}

/// Uniform data for blur shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurUniform {
    /// 1.0 / texture_width, 1.0 / texture_height
    pub texel_size: [f32; 2],
    /// Sample count
    pub sample_count: f32,
    /// Max CoC radius in pixels
    pub max_coc_radius: f32,
    /// Bokeh shape (0 = circle, 1 = hexagon)
    pub bokeh_shape: f32,
    /// Highlight boost factor
    pub highlight_boost: f32,
    /// Is far field pass (1.0) or near field (0.0)
    pub is_far_field: f32,
    /// Padding
    pub _padding: f32,
}

/// Uniform data for composite shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeUniform {
    /// 1.0 / texture_width, 1.0 / texture_height (full res)
    pub texel_size: [f32; 2],
    /// Max CoC radius in pixels
    pub max_coc_radius: f32,
    /// Debug visualization mode (0=off, 1=coc, 2=focus_plane, 3=layers, 4=samples)
    pub debug_mode: f32,
}

/// Autofocus state (CPU-side)
#[derive(Debug, Clone, Copy)]
pub struct AutofocusState {
    /// Current smoothed focus distance
    pub current_focus: f32,
    /// Target focus distance (from depth sample)
    pub target_focus: f32,
    /// Whether autofocus is currently locked
    pub is_locked: bool,
    /// Locked focus distance
    pub locked_focus: f32,
}

impl Default for AutofocusState {
    fn default() -> Self {
        Self {
            current_focus: 20.0,
            target_focus: 20.0,
            is_locked: false,
            locked_focus: 20.0,
        }
    }
}

impl AutofocusState {
    /// Update autofocus with temporal smoothing
    pub fn update(&mut self, target_depth: f32, delta_time: f32, smoothing_time: f32) {
        self.target_focus = target_depth.max(1.0).min(500.0);

        // Exponential moving average: focus = lerp(prev, target, 1 - exp(-dt / tau))
        let alpha = 1.0 - (-delta_time / smoothing_time).exp();
        self.current_focus = self.current_focus + (self.target_focus - self.current_focus) * alpha;
    }

    /// Lock focus at current distance
    pub fn lock(&mut self) {
        self.is_locked = true;
        self.locked_focus = self.current_focus;
    }

    /// Unlock focus
    pub fn unlock(&mut self) {
        self.is_locked = false;
    }

    /// Get the effective focus distance
    pub fn effective_focus(&self) -> f32 {
        if self.is_locked {
            self.locked_focus
        } else {
            self.current_focus
        }
    }
}

/// DoF post-processing system
pub struct Dof {
    /// Quality preset
    pub quality: DofQuality,
    /// Runtime parameters
    pub params: DofParams,
    /// Autofocus state
    pub autofocus: AutofocusState,

    // === Textures ===
    /// CoC texture (R16Float, signed: negative=near, positive=far)
    coc_texture: wgpu::Texture,
    coc_view: wgpu::TextureView,

    /// Half-res color texture (for blur input)
    half_color_texture: wgpu::Texture,
    half_color_view: wgpu::TextureView,

    /// Half-res CoC texture
    half_coc_texture: wgpu::Texture,
    half_coc_view: wgpu::TextureView,

    /// Far blur result texture
    far_blur_texture: wgpu::Texture,
    far_blur_view: wgpu::TextureView,

    /// Near blur result texture
    near_blur_texture: wgpu::Texture,
    near_blur_view: wgpu::TextureView,

    /// Output texture (composite writes here, then blit to scene)
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,

    /// Noise texture for sample rotation
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,

    // === Samplers ===
    nearest_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,

    // === Bind group layouts ===
    coc_bind_group_layout: wgpu::BindGroupLayout,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group_layout: wgpu::BindGroupLayout,

    // === Bind groups (some created per-frame) ===
    coc_bind_group: Option<wgpu::BindGroup>,
    downsample_bind_group: Option<wgpu::BindGroup>,
    far_blur_bind_group: Option<wgpu::BindGroup>,
    near_blur_bind_group: Option<wgpu::BindGroup>,
    composite_bind_group: Option<wgpu::BindGroup>,
    blit_bind_group: Option<wgpu::BindGroup>,

    // === Pipelines ===
    coc_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,

    // === Uniform buffers ===
    coc_uniform_buffer: wgpu::Buffer,
    downsample_uniform_buffer: wgpu::Buffer,
    far_blur_uniform_buffer: wgpu::Buffer,
    near_blur_uniform_buffer: wgpu::Buffer,
    composite_uniform_buffer: wgpu::Buffer,

    // === Dimensions ===
    full_width: u32,
    full_height: u32,
    half_width: u32,
    half_height: u32,
    surface_format: wgpu::TextureFormat,

    // === Debug ===
    pub debug_mode: u32,
}

impl Dof {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let quality = DofQuality::default();
        let params = DofParams::default();

        let half_width = width / 2;
        let half_height = height / 2;

        // === Create textures ===
        let coc_texture = Self::create_coc_texture(device, width, height, "dof_coc");
        let coc_view = coc_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let half_color_texture = Self::create_color_texture(device, half_width, half_height, "dof_half_color");
        let half_color_view = half_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let half_coc_texture = Self::create_coc_texture(device, half_width, half_height, "dof_half_coc");
        let half_coc_view = half_coc_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let far_blur_texture = Self::create_color_texture(device, half_width, half_height, "dof_far_blur");
        let far_blur_view = far_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let near_blur_texture = Self::create_color_texture(device, half_width, half_height, "dof_near_blur");
        let near_blur_view = near_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (noise_texture, noise_view) = Self::create_noise_texture(device);

        // Output texture for composite (same format as scene)
        let output_texture = Self::create_output_texture(device, width, height, surface_format, "dof_output");
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // === Create samplers ===
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof_noise_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // === Create bind group layouts ===
        let coc_bind_group_layout = Self::create_coc_bind_group_layout(device);
        let downsample_bind_group_layout = Self::create_downsample_bind_group_layout(device);
        let blur_bind_group_layout = Self::create_blur_bind_group_layout(device);
        let composite_bind_group_layout = Self::create_composite_bind_group_layout(device);
        let blit_bind_group_layout = Self::create_blit_bind_group_layout(device);

        // === Create uniform buffers ===
        let coc_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_coc_uniform"),
            size: std::mem::size_of::<CocUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let downsample_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_downsample_uniform"),
            size: std::mem::size_of::<DownsampleUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let far_blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_far_blur_uniform"),
            size: std::mem::size_of::<BlurUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let near_blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_near_blur_uniform"),
            size: std::mem::size_of::<BlurUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_composite_uniform"),
            size: std::mem::size_of::<CompositeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Create pipelines ===
        let coc_pipeline = Self::create_coc_pipeline(device, &coc_bind_group_layout);
        let downsample_pipeline = Self::create_downsample_pipeline(device, &downsample_bind_group_layout);
        let blur_pipeline = Self::create_blur_pipeline(device, &blur_bind_group_layout);
        let composite_pipeline = Self::create_composite_pipeline(device, &composite_bind_group_layout, surface_format);
        let blit_pipeline = Self::create_blit_pipeline(device, &blit_bind_group_layout, surface_format);

        Self {
            quality,
            params,
            autofocus: AutofocusState::default(),
            coc_texture,
            coc_view,
            half_color_texture,
            half_color_view,
            half_coc_texture,
            half_coc_view,
            far_blur_texture,
            far_blur_view,
            near_blur_texture,
            near_blur_view,
            output_texture,
            output_view,
            noise_texture,
            noise_view,
            nearest_sampler,
            linear_sampler,
            noise_sampler,
            coc_bind_group_layout,
            downsample_bind_group_layout,
            blur_bind_group_layout,
            composite_bind_group_layout,
            blit_bind_group_layout,
            coc_bind_group: None,
            downsample_bind_group: None,
            far_blur_bind_group: None,
            near_blur_bind_group: None,
            composite_bind_group: None,
            blit_bind_group: None,
            coc_pipeline,
            downsample_pipeline,
            blur_pipeline,
            composite_pipeline,
            blit_pipeline,
            coc_uniform_buffer,
            downsample_uniform_buffer,
            far_blur_uniform_buffer,
            near_blur_uniform_buffer,
            composite_uniform_buffer,
            full_width: width,
            full_height: height,
            half_width,
            half_height,
            surface_format,
            debug_mode: 0,
        }
    }

    fn create_coc_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Signed CoC: negative = near, positive = far
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_color_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // RGBA16Float for HDR color with alpha for near-field coverage
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_output_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_noise_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        // 4x4 noise texture for sample rotation
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dof_noise_texture"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Upload noise texture data
    pub fn upload_noise_texture(&self, queue: &wgpu::Queue) {
        let mut noise_data = [0u8; 4 * 4 * 2]; // 4x4 pixels, RG8

        // Use LCG for deterministic noise
        let mut seed: u32 = 0xDEADBEEF;
        for i in 0..16 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let angle = (seed as f32 / u32::MAX as f32) * std::f32::consts::TAU;

            // Store rotation as normalized (cos, sin) in [0, 255]
            let cos_val = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
            let sin_val = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;

            noise_data[i * 2] = cos_val;
            noise_data[i * 2 + 1] = sin_val;
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * 2),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
    }

    // === Bind group layouts ===

    fn create_coc_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_coc_bind_group_layout"),
            entries: &[
                // Uniform buffer
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
                // Depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
    }

    fn create_downsample_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_downsample_bind_group_layout"),
            entries: &[
                // Uniform buffer
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
                // Scene color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // CoC texture
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
                // Linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_blur_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_blur_bind_group_layout"),
            entries: &[
                // Uniform buffer
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
                // Color texture (half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // CoC texture (half-res)
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
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Noise texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Noise sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
    }

    fn create_composite_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_composite_bind_group_layout"),
            entries: &[
                // Uniform buffer
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
                // Scene color (original full-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Far blur (half-res)
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
                // Near blur (half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Full-res CoC
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_blit_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_blit_bind_group_layout"),
            entries: &[
                // Source texture
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
        })
    }

    // === Pipeline creation ===

    fn create_coc_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_coc_shader"),
            source: wgpu::ShaderSource::Wgsl(COC_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_coc_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_coc_pipeline"),
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
                    format: wgpu::TextureFormat::R16Float,
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

    fn create_downsample_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_downsample_shader"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_downsample_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    // Half-res color
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Half-res CoC
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
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

    fn create_blur_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_blur_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_blur_pipeline"),
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
            label: Some("dof_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_composite_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_composite_pipeline"),
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
                    blend: None, // Full replacement, no blending
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

    fn create_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_blit_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_blit_pipeline"),
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

    /// Create bind groups with actual textures (called per frame)
    pub fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
    ) {
        // CoC bind group
        self.coc_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_coc_bind_group"),
            layout: &self.coc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.coc_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        }));

        // Downsample bind group
        self.downsample_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_downsample_bind_group"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.downsample_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        }));

        // Far blur bind group
        self.far_blur_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_far_blur_bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.far_blur_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.half_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.half_coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
            ],
        }));

        // Near blur bind group
        self.near_blur_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_near_blur_bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.near_blur_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.half_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.half_coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
            ],
        }));

        // Composite bind group (reads from scene_view, writes to output_view)
        self.composite_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_composite_bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.composite_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.far_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.near_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        }));

        // Blit bind group (copies output_view back to scene_view)
        self.blit_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_blit_bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        }));
    }

    /// Update uniform buffers
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        inv_proj: glam::Mat4,
        near: f32,
        far: f32,
    ) {
        // Get effective focus distance
        let focus_dist = match self.params.focus_mode {
            FocusMode::Auto => self.autofocus.effective_focus(),
            FocusMode::Manual => self.params.manual_focus_distance,
            FocusMode::Locked => self.autofocus.locked_focus,
        };

        // CoC uniform
        let coc_uniform = CocUniform {
            inv_proj: inv_proj.to_cols_array_2d(),
            focus_distance: focus_dist,
            aperture: self.params.aperture,
            transition_sharpness: 1.0 / self.params.focus_transition.max(0.1),
            max_coc_radius: self.params.max_blur_radius,
            near_coc_scale: self.params.near_blur.coc_scale(),
            near_plane: near,
            far_plane: far,
            _padding: 0.0,
        };
        queue.write_buffer(&self.coc_uniform_buffer, 0, bytemuck::cast_slice(&[coc_uniform]));

        // Downsample uniform
        let downsample_uniform = DownsampleUniform {
            texel_size: [1.0 / self.full_width as f32, 1.0 / self.full_height as f32],
            _padding: [0.0; 2],
        };
        queue.write_buffer(&self.downsample_uniform_buffer, 0, bytemuck::cast_slice(&[downsample_uniform]));

        // Far blur uniform
        let far_blur_uniform = BlurUniform {
            texel_size: [1.0 / self.half_width as f32, 1.0 / self.half_height as f32],
            sample_count: self.quality.sample_count() as f32,
            max_coc_radius: self.params.max_blur_radius * 0.5, // Half-res
            bokeh_shape: if self.params.bokeh_shape == BokehShape::Hexagon { 1.0 } else { 0.0 },
            highlight_boost: self.params.highlight_boost,
            is_far_field: 1.0,
            _padding: 0.0,
        };
        queue.write_buffer(&self.far_blur_uniform_buffer, 0, bytemuck::cast_slice(&[far_blur_uniform]));

        // Near blur uniform
        let near_blur_uniform = BlurUniform {
            texel_size: [1.0 / self.half_width as f32, 1.0 / self.half_height as f32],
            sample_count: self.quality.sample_count() as f32,
            max_coc_radius: self.params.max_blur_radius * 0.5, // Half-res
            bokeh_shape: if self.params.bokeh_shape == BokehShape::Hexagon { 1.0 } else { 0.0 },
            highlight_boost: self.params.highlight_boost,
            is_far_field: 0.0,
            _padding: 0.0,
        };
        queue.write_buffer(&self.near_blur_uniform_buffer, 0, bytemuck::cast_slice(&[near_blur_uniform]));

        // Composite uniform
        let composite_uniform = CompositeUniform {
            texel_size: [1.0 / self.full_width as f32, 1.0 / self.full_height as f32],
            max_coc_radius: self.params.max_blur_radius,
            debug_mode: self.debug_mode as f32,
        };
        queue.write_buffer(&self.composite_uniform_buffer, 0, bytemuck::cast_slice(&[composite_uniform]));
    }

    /// Update autofocus with center depth sample
    pub fn update_autofocus(&mut self, center_depth: f32, delta_time: f32) {
        if self.params.focus_mode == FocusMode::Auto && !self.autofocus.is_locked {
            self.autofocus.update(center_depth, delta_time, self.params.autofocus_speed);
        }
    }

    /// Render DoF effect
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
    ) {
        if self.quality == DofQuality::Off {
            return;
        }

        let Some(coc_bind_group) = &self.coc_bind_group else { return; };
        let Some(downsample_bind_group) = &self.downsample_bind_group else { return; };
        let Some(far_blur_bind_group) = &self.far_blur_bind_group else { return; };
        let Some(near_blur_bind_group) = &self.near_blur_bind_group else { return; };
        let Some(composite_bind_group) = &self.composite_bind_group else { return; };
        let Some(blit_bind_group) = &self.blit_bind_group else { return; };

        // Pass 1: CoC calculation
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_coc_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.coc_view,
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

            pass.set_pipeline(&self.coc_pipeline);
            pass.set_bind_group(0, coc_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Downsample to half-res (writes to both half_color and half_coc)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_downsample_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.half_color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.half_coc_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, downsample_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3: Far-field blur
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_far_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.far_blur_view,
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

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, far_blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 4: Near-field blur (only if enabled)
        if self.quality.near_blur_enabled() && self.params.near_blur != NearBlur::Off {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_near_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.near_blur_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, near_blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 5: Composite to output texture (reads from scene, writes to output)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output_view,
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

            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 6: Blit output back to scene
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
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

            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Resize textures when window size changes
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.full_width && height == self.full_height {
            return;
        }

        self.full_width = width;
        self.full_height = height;
        self.half_width = width / 2;
        self.half_height = height / 2;

        // Recreate all textures
        self.coc_texture = Self::create_coc_texture(device, width, height, "dof_coc");
        self.coc_view = self.coc_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.half_color_texture = Self::create_color_texture(device, self.half_width, self.half_height, "dof_half_color");
        self.half_color_view = self.half_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.half_coc_texture = Self::create_coc_texture(device, self.half_width, self.half_height, "dof_half_coc");
        self.half_coc_view = self.half_coc_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.far_blur_texture = Self::create_color_texture(device, self.half_width, self.half_height, "dof_far_blur");
        self.far_blur_view = self.far_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.near_blur_texture = Self::create_color_texture(device, self.half_width, self.half_height, "dof_near_blur");
        self.near_blur_view = self.near_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.output_texture = Self::create_output_texture(device, width, height, self.surface_format, "dof_output");
        self.output_view = self.output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Clear bind groups (will be recreated next frame)
        self.coc_bind_group = None;
        self.downsample_bind_group = None;
        self.far_blur_bind_group = None;
        self.near_blur_bind_group = None;
        self.composite_bind_group = None;
        self.blit_bind_group = None;
    }

    /// Set quality preset
    pub fn set_quality(&mut self, quality: DofQuality) {
        self.quality = quality;
        // Update params based on quality
        self.params.max_blur_radius = quality.max_coc_radius();
        self.params.bokeh_shape = if quality.hexagonal_bokeh() { BokehShape::Hexagon } else { BokehShape::Circle };
        self.params.highlight_boost = if quality.highlight_boost_enabled() { 0.2 } else { 0.1 };
    }

    /// Cycle to next quality level
    pub fn cycle_quality(&mut self) -> DofQuality {
        self.set_quality(self.quality.next());
        self.quality
    }

    /// Check if DoF is enabled
    pub fn is_enabled(&self) -> bool {
        self.quality != DofQuality::Off
    }

    /// Toggle focus lock
    pub fn toggle_focus_lock(&mut self) {
        if self.autofocus.is_locked {
            self.autofocus.unlock();
        } else {
            self.autofocus.lock();
        }
    }

    /// Cycle debug visualization mode
    pub fn cycle_debug_mode(&mut self) {
        self.debug_mode = (self.debug_mode + 1) % 5;
    }
}


// =============================================================================
// CoC CALCULATION SHADER
// =============================================================================

const COC_SHADER: &str = r#"
struct CocUniform {
    inv_proj: mat4x4<f32>,
    focus_distance: f32,
    aperture: f32,
    transition_sharpness: f32,
    max_coc_radius: f32,
    near_coc_scale: f32,
    near_plane: f32,
    far_plane: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: CocUniform;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_depth: sampler;

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

// Linearize depth from NDC to view space distance
fn linearize_depth(depth: f32) -> f32 {
    let near = uniforms.near_plane;
    let far = uniforms.far_plane;
    return near * far / (far - depth * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let depth = textureSampleLevel(t_depth, s_depth, in.uv, 0.0);

    // Sky pixels (depth = 1.0) get zero CoC (will be handled separately)
    if (depth >= 0.9999) {
        return 0.0;
    }

    // Convert to linear view-space distance
    let view_dist = linearize_depth(depth);

    // Calculate signed distance from focus plane
    let focus_dist = uniforms.focus_distance;
    let signed_dist = view_dist - focus_dist;

    // Calculate CoC using distance-based falloff
    // CoC grows from 0 at focus to max_radius at focus_transition distance
    let dist_factor = abs(signed_dist) / max(1.0 / uniforms.transition_sharpness, 1.0);
    let raw_coc = uniforms.aperture * uniforms.max_coc_radius * smoothstep(0.0, 1.0, dist_factor);

    // Clamp to max radius
    var coc = min(raw_coc, uniforms.max_coc_radius);

    // Apply sign: negative for near-field, positive for far-field
    if (signed_dist < 0.0) {
        coc = -coc * uniforms.near_coc_scale;
    }

    return coc;
}
"#;


// =============================================================================
// DOWNSAMPLE SHADER (Half-res with CoC-aware filtering)
// =============================================================================

const DOWNSAMPLE_SHADER: &str = r#"
struct DownsampleUniform {
    texel_size: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: DownsampleUniform;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_coc: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) coc: f32,
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
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    // Sample 2x2 quad from full-res textures
    let offset = uniforms.texel_size * 0.5;

    let uv00 = in.uv + vec2<f32>(-offset.x, -offset.y);
    let uv10 = in.uv + vec2<f32>( offset.x, -offset.y);
    let uv01 = in.uv + vec2<f32>(-offset.x,  offset.y);
    let uv11 = in.uv + vec2<f32>( offset.x,  offset.y);

    // Sample colors
    let c00 = textureSample(t_color, s_linear, uv00);
    let c10 = textureSample(t_color, s_linear, uv10);
    let c01 = textureSample(t_color, s_linear, uv01);
    let c11 = textureSample(t_color, s_linear, uv11);

    // Sample CoC values
    let coc00 = textureSample(t_coc, s_linear, uv00).r;
    let coc10 = textureSample(t_coc, s_linear, uv10).r;
    let coc01 = textureSample(t_coc, s_linear, uv01).r;
    let coc11 = textureSample(t_coc, s_linear, uv11).r;

    // For CoC: take MAX absolute value (preserves blur extent)
    let abs_coc00 = abs(coc00);
    let abs_coc10 = abs(coc10);
    let abs_coc01 = abs(coc01);
    let abs_coc11 = abs(coc11);

    let max_abs_coc = max(max(abs_coc00, abs_coc10), max(abs_coc01, abs_coc11));

    // Determine dominant sign (which field has more blur)
    let near_sum = select(0.0, abs_coc00, coc00 < 0.0) +
                   select(0.0, abs_coc10, coc10 < 0.0) +
                   select(0.0, abs_coc01, coc01 < 0.0) +
                   select(0.0, abs_coc11, coc11 < 0.0);
    let far_sum = select(0.0, abs_coc00, coc00 > 0.0) +
                  select(0.0, abs_coc10, coc10 > 0.0) +
                  select(0.0, abs_coc01, coc01 > 0.0) +
                  select(0.0, abs_coc11, coc11 > 0.0);

    // Output CoC with dominant sign
    out.coc = select(max_abs_coc, -max_abs_coc, near_sum > far_sum);

    // For color: simple bilinear average (standard box filter)
    out.color = (c00 + c10 + c01 + c11) * 0.25;

    return out;
}
"#;


// =============================================================================
// DISC BLUR SHADER (Far-field and Near-field)
// =============================================================================

const BLUR_SHADER: &str = r#"
struct BlurUniform {
    texel_size: vec2<f32>,
    sample_count: f32,
    max_coc_radius: f32,
    bokeh_shape: f32,      // 0 = circle, 1 = hexagon
    highlight_boost: f32,
    is_far_field: f32,     // 1 = far, 0 = near
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: BlurUniform;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_coc: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var t_noise: texture_2d<f32>;
@group(0) @binding(5) var s_noise: sampler;

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

// Golden angle spiral for uniform disc sampling
fn spiral_sample(index: u32, total: u32, rotation: f32) -> vec2<f32> {
    let golden_angle = 2.39996323; // PI * (3 - sqrt(5))
    let angle = f32(index) * golden_angle + rotation * 6.28318;
    let radius = sqrt((f32(index) + 0.5) / f32(total));
    return vec2<f32>(cos(angle), sin(angle)) * radius;
}

// Hexagonal bokeh weight
fn hex_weight(p: vec2<f32>) -> f32 {
    let abs_p = abs(p);
    // Hexagon inscribed in unit circle
    let hex_dist = max(abs_p.x * 0.866 + abs_p.y * 0.5, abs_p.y);
    return smoothstep(1.0, 0.9, hex_dist);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_coc = textureSample(t_coc, s_linear, in.uv).r;
    let center_color = textureSample(t_color, s_linear, in.uv);

    let is_far = uniforms.is_far_field > 0.5;

    // Determine if this pixel should be processed
    // Far field: coc > 0, Near field: coc < 0
    let should_process = select(center_coc < -0.5, center_coc > 0.5, is_far);

    if (!should_process) {
        // For pixels not in this field, output with zero alpha (for compositing)
        return vec4<f32>(center_color.rgb, 0.0);
    }

    // Get blur radius for this pixel
    let blur_radius = min(abs(center_coc), uniforms.max_coc_radius);

    // Skip if blur radius is negligible
    if (blur_radius < 0.5) {
        return vec4<f32>(center_color.rgb, 0.0);
    }

    // Sample noise for rotation
    let noise_uv = in.uv * vec2<f32>(textureLoad(t_color, vec2<i32>(0, 0), 0).r + 1.0); // Hacky size fetch
    let noise_scale = 64.0; // Tile noise
    let noise = textureSample(t_noise, s_noise, in.uv * noise_scale);
    let rotation = (noise.r * 2.0 - 1.0) * 3.14159;

    let sample_count = u32(uniforms.sample_count);

    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    var max_brightness = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        // Get sample offset
        var offset = spiral_sample(i, sample_count, rotation);

        // Scale by blur radius
        offset *= blur_radius * uniforms.texel_size;

        let sample_uv = in.uv + offset;

        // Sample color and CoC
        let sample_color = textureSample(t_color, s_linear, sample_uv);
        let sample_coc = textureSample(t_coc, s_linear, sample_uv).r;

        // Calculate sample weight
        var weight = 1.0;

        // CoC-aware weighting: reject samples from wrong field
        if (is_far) {
            // Far field: reject near-field samples (they shouldn't bleed into far blur)
            // Also weight by sample's own blur amount
            let sample_blur = max(sample_coc, 0.0);
            weight *= smoothstep(0.0, 2.0, sample_blur);
        } else {
            // Near field: accept all samples (near blurs over everything)
            let sample_blur = max(-sample_coc, 0.0);
            weight *= smoothstep(0.0, 2.0, sample_blur);
        }

        // Bokeh shape weighting
        if (uniforms.bokeh_shape > 0.5) {
            let normalized_offset = offset / (blur_radius * uniforms.texel_size + 0.0001);
            weight *= hex_weight(normalized_offset);
        }

        // Accumulate
        color_sum += sample_color.rgb * weight;
        weight_sum += weight;

        // Track max brightness for highlight boost
        let brightness = dot(sample_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        max_brightness = max(max_brightness, brightness * weight);
    }

    // Normalize
    var result = color_sum / max(weight_sum, 0.001);

    // Apply highlight boost (subtle bokeh glow)
    if (uniforms.highlight_boost > 0.0) {
        let avg_brightness = dot(result, vec3<f32>(0.299, 0.587, 0.114));
        let boost = (max_brightness - avg_brightness) * uniforms.highlight_boost;
        result += result * boost;
    }

    // Output with alpha = blur amount (for compositing)
    let alpha = smoothstep(0.0, 3.0, blur_radius);

    return vec4<f32>(result, alpha);
}
"#;


// =============================================================================
// COMPOSITE SHADER
// =============================================================================

const COMPOSITE_SHADER: &str = r#"
struct CompositeUniform {
    texel_size: vec2<f32>,
    max_coc_radius: f32,
    debug_mode: f32,  // 0=off, 1=coc viz, 2=focus plane, 3=layers, 4=samples
};

@group(0) @binding(0) var<uniform> uniforms: CompositeUniform;
@group(0) @binding(1) var t_scene: texture_2d<f32>;
@group(0) @binding(2) var t_far_blur: texture_2d<f32>;
@group(0) @binding(3) var t_near_blur: texture_2d<f32>;
@group(0) @binding(4) var t_coc: texture_2d<f32>;
@group(0) @binding(5) var s_linear: sampler;

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
    let scene = textureSample(t_scene, s_linear, in.uv);
    let coc = textureSample(t_coc, s_linear, in.uv).r;
    let far_blur = textureSample(t_far_blur, s_linear, in.uv);
    let near_blur = textureSample(t_near_blur, s_linear, in.uv);

    // Debug visualizations
    if (uniforms.debug_mode > 0.5) {
        let debug_mode = i32(uniforms.debug_mode);

        if (debug_mode == 1) {
            // CoC visualization: blue=near, red=far, black=focus
            let near_vis = max(-coc, 0.0) / uniforms.max_coc_radius;
            let far_vis = max(coc, 0.0) / uniforms.max_coc_radius;
            return vec4<f32>(far_vis, 0.0, near_vis, 1.0);
        }

        if (debug_mode == 2) {
            // Focus plane indicator: highlight near-focus region
            let focus_band = 1.0 - smoothstep(0.0, 2.0, abs(coc));
            return vec4<f32>(scene.rgb * (1.0 - focus_band * 0.5) + vec3<f32>(0.0, focus_band * 0.5, 0.0), 1.0);
        }

        if (debug_mode == 3) {
            // Layer visualization
            if (coc < -0.5) {
                return vec4<f32>(near_blur.rgb, 1.0);
            } else if (coc > 0.5) {
                return vec4<f32>(far_blur.rgb, 1.0);
            } else {
                return vec4<f32>(scene.rgb, 1.0);
            }
        }

        if (debug_mode == 4) {
            // Sample count heatmap (based on CoC)
            let sample_intensity = abs(coc) / uniforms.max_coc_radius;
            let heat = vec3<f32>(
                smoothstep(0.0, 0.5, sample_intensity),
                smoothstep(0.25, 0.75, sample_intensity) * (1.0 - smoothstep(0.75, 1.0, sample_intensity)),
                1.0 - smoothstep(0.5, 1.0, sample_intensity)
            );
            return vec4<f32>(heat, 1.0);
        }
    }

    // Normal compositing
    var result = scene.rgb;

    // Blend in far blur based on FULL-RES CoC (preserves sharp foreground edges)
    let far_blend = smoothstep(0.0, 3.0, max(coc, 0.0)) * far_blur.a;
    result = mix(result, far_blur.rgb, far_blend);

    // Composite near blur ON TOP (near obscures everything behind it)
    let near_blend = smoothstep(0.0, 3.0, max(-coc, 0.0)) * near_blur.a;
    result = mix(result, near_blur.rgb, near_blend);

    return vec4<f32>(result, 1.0);
}
"#;


// =============================================================================
// BLIT SHADER (Copy output texture to scene)
// =============================================================================

const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_linear: sampler;

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
    return textureSample(t_source, s_linear, in.uv);
}
"#;
