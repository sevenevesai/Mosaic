//! Inventory UI rendering
//!
//! Renders hotbar and full inventory grid with:
//! - Slot backgrounds with selection highlight
//! - Item icons from block texture atlas
//! - Stack count numbers

use crate::inventory::{Inventory, HOTBAR_SIZE, layout};
use crate::inventory::item::Item;
use wgpu::util::DeviceExt;

/// Vertex for inventory UI elements (slots, icons)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InventoryVertex {
    /// Screen position in pixels
    pub position: [f32; 2],
    /// UV coordinates (for texture sampling)
    pub uv: [f32; 2],
    /// Color tint (RGBA)
    pub color: [f32; 4],
    /// Texture mode: 0 = solid color, 1 = sample from atlas
    pub tex_mode: f32,
    /// Texture index in atlas (when tex_mode = 1)
    pub tex_index: f32,
}

impl InventoryVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InventoryVertex>() as wgpu::BufferAddress,
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
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Inventory UI shader
const INVENTORY_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) tex_mode: f32,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) tex_mode: f32,
    @location(3) tex_index: f32,
};

struct ScreenUniform {
    inv_size: vec2<f32>,  // 1.0 / screen_size
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

@group(1) @binding(0)
var t_atlas: texture_2d<f32>;

@group(1) @binding(1)
var s_atlas: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Convert from pixel coordinates to clip space
    // Screen (0,0) is top-left, clip (-1,-1) is bottom-left
    let ndc = in.position * screen.inv_size * 2.0 - 1.0;
    out.clip_position = vec4<f32>(ndc.x, -ndc.y, 0.5, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.tex_mode = in.tex_mode;
    out.tex_index = in.tex_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.tex_mode < 0.5) {
        // Solid color mode
        return in.color;
    } else {
        // Texture sampling mode
        // Atlas is 8x8 tiles
        let tiles_per_row = 8.0;
        let tile_size = 1.0 / tiles_per_row;

        let tile_x = in.tex_index % tiles_per_row;
        let tile_y = floor(in.tex_index / tiles_per_row);

        // Calculate UV within tile, with half-texel inset to prevent bleeding
        let inset = 0.5 / 256.0; // Assuming 256px tile size
        let tile_uv = vec2<f32>(
            tile_x * tile_size + in.uv.x * (tile_size - 2.0 * inset) + inset,
            tile_y * tile_size + in.uv.y * (tile_size - 2.0 * inset) + inset
        );

        let tex_color = textureSample(t_atlas, s_atlas, tile_uv);
        // Apply color tint
        return tex_color * in.color;
    }
}
"#;

/// Inventory UI renderer
pub struct InventoryUI {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]  // Kept for potential future dynamic layout changes
    screen_bind_group_layout: wgpu::BindGroupLayout,
    screen_buffer: wgpu::Buffer,
    screen_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    screen_size: (f32, f32),
}

impl InventoryUI {
    /// Maximum vertices in vertex buffer (enough for full inventory + held item)
    const MAX_VERTICES: usize = 4096;

    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Screen uniform bind group layout
        let screen_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Inventory Screen Bind Group Layout"),
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

        // Screen uniform buffer
        let screen_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Inventory Screen Buffer"),
            contents: bytemuck::cast_slice(&[1.0f32 / 1920.0, 1.0 / 1080.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Inventory Screen Bind Group"),
            layout: &screen_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_buffer.as_entire_binding(),
            }],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Inventory Pipeline Layout"),
            bind_group_layouts: &[&screen_bind_group_layout, texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Inventory Shader"),
            source: wgpu::ShaderSource::Wgsl(INVENTORY_SHADER.into()),
        });

        // Pipeline with alpha blending
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Inventory Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[InventoryVertex::desc()],
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
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Vertex buffer (pre-allocated)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Inventory Vertex Buffer"),
            size: (Self::MAX_VERTICES * std::mem::size_of::<InventoryVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            screen_bind_group_layout,
            screen_buffer,
            screen_bind_group,
            vertex_buffer,
            vertex_count: 0,
            screen_size: (1920.0, 1080.0),
        }
    }

    /// Update screen size for coordinate conversion
    pub fn update_screen_size(&mut self, width: f32, height: f32) {
        self.screen_size = (width, height);
    }

    /// Build vertices for inventory rendering
    pub fn build_vertices(&mut self, queue: &wgpu::Queue, inventory: &Inventory) {
        let mut vertices: Vec<InventoryVertex> = Vec::with_capacity(Self::MAX_VERTICES);
        let (screen_width, screen_height) = self.screen_size;

        // Update screen uniform
        queue.write_buffer(
            &self.screen_buffer,
            0,
            bytemuck::cast_slice(&[1.0 / screen_width, 1.0 / screen_height, 0.0f32, 0.0]),
        );

        // Calculate hotbar position
        let slot_size = layout::SLOT_SIZE;
        let padding = layout::SLOT_PADDING;
        let hotbar_width = (HOTBAR_SIZE as f32 * slot_size) + ((HOTBAR_SIZE - 1) as f32 * padding);
        let hotbar_x = (screen_width - hotbar_width) / 2.0;
        let hotbar_y = screen_height - layout::MARGIN_BOTTOM - slot_size;

        // Slot background color
        let slot_bg = [0.1, 0.1, 0.1, 0.85];
        let slot_border = [0.3, 0.3, 0.3, 1.0];
        let selected_border = [1.0, 1.0, 0.5, 1.0];
        let hovered_bg = [0.2, 0.2, 0.2, 0.9];

        let selected_slot = inventory.selected_index();
        let hovered_slot = inventory.hovered_slot();

        // Render hotbar slots
        for i in 0..HOTBAR_SIZE {
            let x = hotbar_x + (i as f32 * (slot_size + padding));
            let y = hotbar_y;
            let is_selected = i == selected_slot;
            let is_hovered = hovered_slot == Some(i);

            // Slot background
            let bg_color = if is_hovered { hovered_bg } else { slot_bg };
            self.add_rect(&mut vertices, x, y, slot_size, slot_size, bg_color);

            // Border (thicker for selected)
            let border_color = if is_selected { selected_border } else { slot_border };
            let border_width = if is_selected { layout::SELECTION_WIDTH } else { layout::BORDER_WIDTH };
            self.add_rect_outline(&mut vertices, x, y, slot_size, slot_size, border_width, border_color);

            // Item icon and count
            if let Some(stack) = inventory.get_slot(i) {
                let icon_padding = 4.0;
                let icon_size = slot_size - icon_padding * 2.0;
                let icon_x = x + icon_padding;
                let icon_y = y + icon_padding;

                self.add_item_icon(&mut vertices, &stack.item, icon_x, icon_y, icon_size, [1.0, 1.0, 1.0, 1.0]);

                // Stack count (if > 1, skip for tools which are always 1)
                if stack.count > 1 {
                    self.add_stack_count(&mut vertices, x + slot_size - 4.0, y + slot_size - 4.0, stack.count);
                }

                // Durability bar for tools
                if let Item::Tool { durability, material, .. } = &stack.item {
                    let max_dur = material.durability();
                    self.add_durability_bar(&mut vertices, x + 4.0, y + slot_size - 7.0, slot_size - 8.0, *durability, max_dur);
                }

                // Lock indicator
                if stack.locked {
                    self.add_lock_indicator(&mut vertices, x + 2.0, y + 2.0);
                }
            }
        }

        // Render main inventory if open
        if inventory.is_open {
            let main_inv_y = hotbar_y - layout::INVENTORY_GAP - (3.0 * (slot_size + padding));

            // Semi-transparent background for inventory area
            let inv_bg_width = hotbar_width + 20.0;
            let inv_bg_height = 3.0 * (slot_size + padding) + layout::INVENTORY_GAP + slot_size + 20.0;
            let inv_bg_x = (screen_width - inv_bg_width) / 2.0;
            let inv_bg_y = main_inv_y - 10.0;
            self.add_rect(&mut vertices, inv_bg_x, inv_bg_y, inv_bg_width, inv_bg_height, [0.05, 0.05, 0.05, 0.9]);

            // Main inventory slots (3 rows of 9)
            for row in 0..3 {
                for col in 0..9 {
                    let slot_index = 9 + row * 9 + col;
                    let x = hotbar_x + (col as f32 * (slot_size + padding));
                    let y = main_inv_y + ((2 - row) as f32 * (slot_size + padding));
                    let is_hovered = hovered_slot == Some(slot_index);

                    // Slot background
                    let bg_color = if is_hovered { hovered_bg } else { slot_bg };
                    self.add_rect(&mut vertices, x, y, slot_size, slot_size, bg_color);

                    // Border
                    self.add_rect_outline(&mut vertices, x, y, slot_size, slot_size, layout::BORDER_WIDTH, slot_border);

                    // Item icon and count
                    if let Some(stack) = inventory.get_slot(slot_index) {
                        let icon_padding = 4.0;
                        let icon_size = slot_size - icon_padding * 2.0;
                        let icon_x = x + icon_padding;
                        let icon_y = y + icon_padding;

                        self.add_item_icon(&mut vertices, &stack.item, icon_x, icon_y, icon_size, [1.0, 1.0, 1.0, 1.0]);

                        if stack.count > 1 {
                            self.add_stack_count(&mut vertices, x + slot_size - 4.0, y + slot_size - 4.0, stack.count);
                        }

                        if let Item::Tool { durability, material, .. } = &stack.item {
                            let max_dur = material.durability();
                            self.add_durability_bar(&mut vertices, x + 4.0, y + slot_size - 7.0, slot_size - 8.0, *durability, max_dur);
                        }

                        if stack.locked {
                            self.add_lock_indicator(&mut vertices, x + 2.0, y + 2.0);
                        }
                    }
                }
            }

            // Render held item following mouse
            if let Some(held) = inventory.held_item() {
                let (mx, my) = inventory.mouse_pos();
                let icon_size = slot_size - 8.0;
                let icon_x = mx - icon_size / 2.0;
                let icon_y = my - icon_size / 2.0;

                self.add_item_icon(&mut vertices, &held.item, icon_x, icon_y, icon_size, [1.0, 1.0, 1.0, 0.9]);

                if held.count > 1 {
                    self.add_stack_count(&mut vertices, mx + icon_size / 2.0 - 4.0, my + icon_size / 2.0 - 4.0, held.count);
                }
            }
        }

        // Upload vertices
        self.vertex_count = vertices.len() as u32;
        if !vertices.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
    }

    /// Add a solid colored rectangle
    fn add_rect(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let v = |px: f32, py: f32| InventoryVertex {
            position: [px, py],
            uv: [0.0, 0.0],
            color,
            tex_mode: 0.0,
            tex_index: 0.0,
        };

        vertices.push(v(x, y));
        vertices.push(v(x + w, y));
        vertices.push(v(x + w, y + h));
        vertices.push(v(x, y));
        vertices.push(v(x + w, y + h));
        vertices.push(v(x, y + h));
    }

    /// Add a rectangle outline
    fn add_rect_outline(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, w: f32, h: f32, thickness: f32, color: [f32; 4]) {
        // Top
        self.add_rect(vertices, x, y, w, thickness, color);
        // Bottom
        self.add_rect(vertices, x, y + h - thickness, w, thickness, color);
        // Left
        self.add_rect(vertices, x, y + thickness, thickness, h - thickness * 2.0, color);
        // Right
        self.add_rect(vertices, x + w - thickness, y + thickness, thickness, h - thickness * 2.0, color);
    }

    /// Add a textured rectangle (for item icons)
    fn add_textured_rect(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, w: f32, h: f32, tex_index: f32, tint: [f32; 4]) {
        let v = |px: f32, py: f32, u: f32, v_coord: f32| InventoryVertex {
            position: [px, py],
            uv: [u, v_coord],
            color: tint,
            tex_mode: 1.0,
            tex_index,
        };

        vertices.push(v(x, y, 0.0, 0.0));
        vertices.push(v(x + w, y, 1.0, 0.0));
        vertices.push(v(x + w, y + h, 1.0, 1.0));
        vertices.push(v(x, y, 0.0, 0.0));
        vertices.push(v(x + w, y + h, 1.0, 1.0));
        vertices.push(v(x, y + h, 0.0, 1.0));
    }

    /// Add stack count display using 7-segment style digits
    fn add_stack_count(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, count: u8) {
        let count_str = count.to_string();
        let digit_width = 7.0;
        let digit_height = 10.0;
        let spacing = 2.0;

        let total_width = count_str.len() as f32 * (digit_width + spacing) - spacing;
        let start_x = x - total_width;
        let start_y = y - digit_height;

        // Dark background for readability
        self.add_rect(vertices, start_x - 2.0, start_y - 2.0, total_width + 4.0, digit_height + 4.0, [0.0, 0.0, 0.0, 0.7]);

        // Render each digit
        for (i, ch) in count_str.chars().enumerate() {
            let dx = start_x + i as f32 * (digit_width + spacing);
            let digit = ch.to_digit(10).unwrap_or(0) as u8;
            self.add_digit(vertices, dx, start_y, digit_width, digit_height, digit);
        }
    }

    /// Render a single digit using 7-segment style
    /// Segment layout:
    ///   AAA
    ///  F   B
    ///   GGG
    ///  E   C
    ///   DDD
    fn add_digit(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, w: f32, h: f32, digit: u8) {
        let color = [1.0, 1.0, 1.0, 1.0]; // White digits
        let seg_thickness = 2.0;
        let seg_len_h = w - 2.0;  // Horizontal segment length
        let seg_len_v = (h - 3.0 * seg_thickness) / 2.0 + seg_thickness; // Vertical segment length

        // Segment definitions: which segments are on for each digit (A,B,C,D,E,F,G)
        // 0=off, 1=on
        let segments: [[bool; 7]; 10] = [
            [true,  true,  true,  true,  true,  true,  false], // 0
            [false, true,  true,  false, false, false, false], // 1
            [true,  true,  false, true,  true,  false, true],  // 2
            [true,  true,  true,  true,  false, false, true],  // 3
            [false, true,  true,  false, false, true,  true],  // 4
            [true,  false, true,  true,  false, true,  true],  // 5
            [true,  false, true,  true,  true,  true,  true],  // 6
            [true,  true,  true,  false, false, false, false], // 7
            [true,  true,  true,  true,  true,  true,  true],  // 8
            [true,  true,  true,  true,  false, true,  true],  // 9
        ];

        let segs = segments[digit as usize % 10];

        // A - top horizontal
        if segs[0] {
            self.add_rect(vertices, x + 1.0, y, seg_len_h, seg_thickness, color);
        }
        // B - top right vertical
        if segs[1] {
            self.add_rect(vertices, x + w - seg_thickness, y + seg_thickness, seg_thickness, seg_len_v, color);
        }
        // C - bottom right vertical
        if segs[2] {
            self.add_rect(vertices, x + w - seg_thickness, y + h / 2.0, seg_thickness, seg_len_v, color);
        }
        // D - bottom horizontal
        if segs[3] {
            self.add_rect(vertices, x + 1.0, y + h - seg_thickness, seg_len_h, seg_thickness, color);
        }
        // E - bottom left vertical
        if segs[4] {
            self.add_rect(vertices, x, y + h / 2.0, seg_thickness, seg_len_v, color);
        }
        // F - top left vertical
        if segs[5] {
            self.add_rect(vertices, x, y + seg_thickness, seg_thickness, seg_len_v, color);
        }
        // G - middle horizontal
        if segs[6] {
            self.add_rect(vertices, x + 1.0, y + h / 2.0 - seg_thickness / 2.0, seg_len_h, seg_thickness, color);
        }
    }

    /// Add an item icon (handles both blocks and tools)
    fn add_item_icon(&self, vertices: &mut Vec<InventoryVertex>, item: &Item, x: f32, y: f32, size: f32, tint: [f32; 4]) {
        match item {
            Item::Block(_) => {
                let tex_index = item.texture_index() as f32;
                self.add_textured_rect(vertices, x, y, size, size, tex_index, tint);
            }
            Item::Tool { kind, material, .. } => {
                self.add_tool_icon(vertices, *kind, *material, x, y, size, tint[3]);
            }
        }
    }

    /// Render a tool icon: colored rectangle with tool letter overlay
    fn add_tool_icon(
        &self,
        vertices: &mut Vec<InventoryVertex>,
        kind: crate::inventory::item::ToolKind,
        material: crate::inventory::item::ToolMaterial,
        x: f32, y: f32, size: f32, alpha: f32,
    ) {
        // Material-colored background
        let mut color = material.color();
        color[3] = alpha;
        self.add_rect(vertices, x, y, size, size, color);

        // Darker border for definition
        let mut border_color = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, alpha];
        border_color[3] = alpha;
        self.add_rect_outline(vertices, x, y, size, size, 2.0, border_color);

        // Tool letter (P/A/S/W) rendered using 7-segment style
        let letter = kind.letter();
        let letter_size = size * 0.5;
        let letter_x = x + (size - letter_size) / 2.0;
        let letter_y = y + (size - letter_size * 1.4) / 2.0;
        self.add_tool_letter(vertices, letter_x, letter_y, letter_size, letter);
    }

    /// Render a tool letter using simple rect-based strokes
    fn add_tool_letter(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, size: f32, letter: char) {
        let color = [1.0, 1.0, 1.0, 0.9]; // White letter
        let t = (size * 0.18).max(2.0); // Stroke thickness
        let h = size * 1.4; // Letter height
        let w = size; // Letter width

        match letter {
            'P' => {
                // Vertical left stroke
                self.add_rect(vertices, x, y, t, h, color);
                // Top horizontal
                self.add_rect(vertices, x, y, w, t, color);
                // Middle horizontal
                self.add_rect(vertices, x, y + h * 0.5, w, t, color);
                // Right top vertical (upper half only)
                self.add_rect(vertices, x + w - t, y, t, h * 0.5 + t, color);
            }
            'A' => {
                // Left vertical
                self.add_rect(vertices, x, y, t, h, color);
                // Right vertical
                self.add_rect(vertices, x + w - t, y, t, h, color);
                // Top horizontal
                self.add_rect(vertices, x, y, w, t, color);
                // Middle horizontal
                self.add_rect(vertices, x, y + h * 0.5, w, t, color);
            }
            'S' => {
                // Top horizontal
                self.add_rect(vertices, x, y, w, t, color);
                // Middle horizontal
                self.add_rect(vertices, x, y + h * 0.5 - t / 2.0, w, t, color);
                // Bottom horizontal
                self.add_rect(vertices, x, y + h - t, w, t, color);
                // Top-left vertical
                self.add_rect(vertices, x, y, t, h * 0.5, color);
                // Bottom-right vertical
                self.add_rect(vertices, x + w - t, y + h * 0.5, t, h * 0.5, color);
            }
            'W' => {
                // Left vertical
                self.add_rect(vertices, x, y, t, h, color);
                // Right vertical
                self.add_rect(vertices, x + w - t, y, t, h, color);
                // Center vertical (lower half)
                self.add_rect(vertices, x + w / 2.0 - t / 2.0, y + h * 0.4, t, h * 0.6, color);
                // Bottom horizontal
                self.add_rect(vertices, x, y + h - t, w, t, color);
            }
            _ => {}
        }
    }

    /// Render a durability bar below the item icon
    fn add_durability_bar(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32, width: f32, durability: u16, max_durability: u16) {
        if durability >= max_durability {
            return; // Don't show bar when full durability
        }

        let ratio = durability as f32 / max_durability as f32;
        let bar_height = 3.0;

        // Background (dark)
        self.add_rect(vertices, x, y, width, bar_height, [0.1, 0.1, 0.1, 0.8]);

        // Durability fill (green -> yellow -> red gradient)
        let fill_color = if ratio > 0.5 {
            // Green to yellow
            let t = (ratio - 0.5) * 2.0;
            [1.0 - t, 0.8 + t * 0.2, 0.0, 0.9]
        } else {
            // Yellow to red
            let t = ratio * 2.0;
            [1.0, t * 0.8, 0.0, 0.9]
        };

        let fill_width = width * ratio;
        if fill_width > 0.5 {
            self.add_rect(vertices, x, y, fill_width, bar_height, fill_color);
        }
    }

    /// Add lock indicator (small padlock icon)
    fn add_lock_indicator(&self, vertices: &mut Vec<InventoryVertex>, x: f32, y: f32) {
        // Simple lock icon: small yellow/gold square
        self.add_rect(vertices, x, y, 8.0, 8.0, [0.8, 0.7, 0.2, 0.9]);
    }

    /// Render the inventory UI
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        texture_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.vertex_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.screen_bind_group, &[]);
        render_pass.set_bind_group(1, texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }
}
