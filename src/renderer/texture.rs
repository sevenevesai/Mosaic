//! Texture atlas handling for voxel blocks

use crate::constants::TEXTURE_TILE_SIZE;

/// Texture atlas for block textures
pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub tiles_per_row: u32,
}

impl TextureAtlas {
    /// Load or generate texture atlas
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // For PoC, generate a simple procedural texture atlas
        // In production, load from assets/textures/atlas.png
        let atlas_data = Self::generate_procedural_atlas();
        let tiles_per_row = 16;
        let atlas_size = TEXTURE_TILE_SIZE * tiles_per_row;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Block Texture Atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * atlas_size),
                rows_per_image: Some(atlas_size),
            },
            wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use nearest-neighbor sampling for pixel art
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Block Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest, // Pixel-perfect!
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tiles_per_row,
        }
    }

    /// Generate a procedural texture atlas for testing
    fn generate_procedural_atlas() -> Vec<u8> {
        let tiles_per_row = 16;
        let atlas_size = (TEXTURE_TILE_SIZE * tiles_per_row) as usize;
        let mut data = vec![0u8; atlas_size * atlas_size * 4];

        // Generate textures for each tile
        for tile_y in 0..tiles_per_row {
            for tile_x in 0..tiles_per_row {
                let tile_index = tile_y * tiles_per_row + tile_x;
                Self::generate_tile(&mut data, tile_x, tile_y, tiles_per_row, tile_index);
            }
        }

        data
    }

    fn generate_tile(
        data: &mut [u8],
        tile_x: u32,
        tile_y: u32,
        tiles_per_row: u32,
        tile_index: u32,
    ) {
        let tile_size = TEXTURE_TILE_SIZE as usize;
        let atlas_width = (tiles_per_row * TEXTURE_TILE_SIZE) as usize;

        // Base colors for different block types
        let (base_r, base_g, base_b) = match tile_index {
            0 => (100, 80, 60),    // Dirt
            1 => (80, 160, 60),    // Grass top
            2 => (100, 80, 60),    // Grass side (dirt with grass edge)
            3 => (128, 128, 128),  // Stone
            4 => (180, 180, 180),  // Cobblestone
            5 => (160, 130, 80),   // Sand
            6 => (60, 100, 180),   // Water
            7 => (80, 60, 40),     // Wood (bark)
            8 => (180, 150, 100),  // Wood (cross-section)
            9 => (60, 130, 50),    // Leaves
            10 => (50, 50, 50),    // Coal ore
            11 => (180, 160, 100), // Iron ore
            12 => (220, 200, 50),  // Gold ore
            13 => (100, 180, 220), // Diamond ore
            14 => (200, 50, 50),   // Redstone ore
            15 => (60, 60, 80),    // Bedrock
            _ => (255, 0, 255),    // Missing texture (magenta)
        };

        // Simple pseudo-random for texture variation
        let seed = tile_index.wrapping_mul(2654435761);

        for py in 0..tile_size {
            for px in 0..tile_size {
                let x = (tile_x as usize * tile_size) + px;
                let y = (tile_y as usize * tile_size) + py;
                let idx = (y * atlas_width + x) * 4;

                // Add noise/variation
                let noise = ((px.wrapping_mul(7) ^ py.wrapping_mul(13) ^ seed as usize) % 32) as i32 - 16;

                let r = (base_r as i32 + noise).clamp(0, 255) as u8;
                let g = (base_g as i32 + noise).clamp(0, 255) as u8;
                let b = (base_b as i32 + noise).clamp(0, 255) as u8;

                // Special case: grass side with dirt and grass top edge
                let (r, g, b) = if tile_index == 2 && py < 4 {
                    let noise2 = ((px.wrapping_mul(3) ^ py.wrapping_mul(7)) % 20) as i32 - 10;
                    (
                        (80 + noise2).clamp(0, 255) as u8,
                        (160 + noise2).clamp(0, 255) as u8,
                        (60 + noise2).clamp(0, 255) as u8,
                    )
                } else {
                    (r, g, b)
                };

                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
                data[idx + 3] = 255; // Alpha
            }
        }
    }

    /// Get UV coordinates for a tile index
    pub fn tile_uvs(tile_index: u32, tiles_per_row: u32) -> (f32, f32, f32, f32) {
        let tile_x = tile_index % tiles_per_row;
        let tile_y = tile_index / tiles_per_row;

        let tile_size = 1.0 / tiles_per_row as f32;

        let u_min = tile_x as f32 * tile_size;
        let v_min = tile_y as f32 * tile_size;
        let u_max = u_min + tile_size;
        let v_max = v_min + tile_size;

        (u_min, v_min, u_max, v_max)
    }
}
