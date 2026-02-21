//! Texture atlas handling for voxel blocks
//!
//! Loads block textures from assets/blocks/ using the block registry.
//! Atlas is TILES_PER_ROW x TILES_PER_ROW grid of TEXTURE_TILE_SIZE px tiles.

use crate::constants::{TEXTURE_TILE_SIZE, TILES_PER_ROW};
use crate::world::block_registry::{self, BlockRegistry};
use image::{ImageBuffer, Rgba};

/// Texture atlas for block textures
pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub tiles_per_row: u32,
}

impl TextureAtlas {
    /// Load texture atlas from assets/blocks/ with full mipmap chain and anisotropic filtering
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let tiles_per_row = TILES_PER_ROW;
        let atlas_size = TEXTURE_TILE_SIZE * tiles_per_row;

        let mip_level_count = (atlas_size as f32).log2().floor() as u32 + 1;
        log::info!("Creating texture atlas {}x{} ({} tiles, {}px each, {} mip levels)",
            atlas_size, atlas_size, tiles_per_row * tiles_per_row, TEXTURE_TILE_SIZE, mip_level_count);

        let reg = block_registry::registry();
        let atlas_data = match Self::load_atlas_from_registry(reg, tiles_per_row) {
            Ok(data) => {
                log::info!("Loaded {} textures into atlas from assets/blocks/", reg.texture_count());
                data
            }
            Err(e) => {
                log::warn!("Failed to load textures: {}. Using procedural fallback.", e);
                Self::generate_procedural_atlas(tiles_per_row)
            }
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Block Texture Atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
            mip_level_count,
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

        Self::generate_mipmaps(queue, &texture, &atlas_data, atlas_size, mip_level_count);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Block Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            tiles_per_row,
        }
    }

    /// Generate and upload mipmap chain using high-quality Lanczos3 downsampling
    fn generate_mipmaps(queue: &wgpu::Queue, texture: &wgpu::Texture, base_data: &[u8], base_size: u32, mip_count: u32) {
        let mut current_data = base_data.to_vec();
        let mut current_size = base_size;

        for mip_level in 1..mip_count {
            let new_size = (current_size / 2).max(1);

            let current_image: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(
                current_size,
                current_size,
                current_data.clone(),
            ).expect("Failed to create image buffer for mipmap");

            let resized = image::imageops::resize(
                &current_image,
                new_size,
                new_size,
                image::imageops::FilterType::Lanczos3,
            );

            let mip_data: Vec<u8> = resized.into_raw();

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mip_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * new_size),
                    rows_per_image: Some(new_size),
                },
                wgpu::Extent3d {
                    width: new_size,
                    height: new_size,
                    depth_or_array_layers: 1,
                },
            );

            current_data = mip_data;
            current_size = new_size;
        }

        log::info!("Generated {} mipmap levels", mip_count - 1);
    }

    /// Load textures from assets/blocks/ using the registry's texture name list.
    fn load_atlas_from_registry(reg: &BlockRegistry, tiles_per_row: u32) -> Result<Vec<u8>, String> {
        let tile_size = TEXTURE_TILE_SIZE;
        let atlas_size = tile_size * tiles_per_row;

        let mut atlas: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::new(atlas_size, atlas_size);

        // Fill with dark color to blend if UV bleeding occurs
        for pixel in atlas.pixels_mut() {
            *pixel = Rgba([20, 25, 20, 255]);
        }

        let texture_dir = BlockRegistry::texture_dir();
        let mut loaded_count = 0u32;

        for (index, name) in reg.texture_names.iter().enumerate() {
            let tile_x = (index as u32) % tiles_per_row;
            let tile_y = (index as u32) / tiles_per_row;

            if BlockRegistry::is_procedural(name) {
                Self::generate_procedural_tile(&mut atlas, tile_x, tile_y, name);
                loaded_count += 1;
                continue;
            }

            // Try to load from assets/blocks/{name}.png
            let loaded = if let Some(dir) = &texture_dir {
                let file_path = dir.join(format!("{}.png", name));
                match image::open(&file_path) {
                    Ok(img) => {
                        let img = if img.width() != tile_size || img.height() != tile_size {
                            img.resize_exact(tile_size, tile_size, image::imageops::FilterType::Nearest)
                        } else {
                            img
                        };
                        let rgba = img.to_rgba8();
                        let base_x = tile_x * tile_size;
                        let base_y = tile_y * tile_size;
                        for y in 0..tile_size {
                            for x in 0..tile_size {
                                let pixel = rgba.get_pixel(x, y);
                                atlas.put_pixel(base_x + x, base_y + y, *pixel);
                            }
                        }
                        true
                    }
                    Err(e) => {
                        log::warn!("Failed to load {}.png: {}", name, e);
                        false
                    }
                }
            } else {
                false
            };

            if !loaded {
                // Gray placeholder for missing textures
                Self::generate_procedural_tile(&mut atlas, tile_x, tile_y, "_placeholder");
            }
            loaded_count += 1;
        }

        if loaded_count == 0 {
            return Err("No textures could be loaded".to_string());
        }

        log::info!("Loaded {}/{} textures into atlas", loaded_count, reg.texture_count());
        Ok(atlas.into_raw())
    }

    /// Generate a procedural tile for special textures (torch, water, placeholders)
    fn generate_procedural_tile(
        atlas: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
        tile_x: u32,
        tile_y: u32,
        name: &str,
    ) {
        let tile_size = TEXTURE_TILE_SIZE;
        let base_x = tile_x * tile_size;
        let base_y = tile_y * tile_size;

        for py in 0..tile_size {
            for px in 0..tile_size {
                let (r, g, b) = match name {
                    "_torch" => {
                        // Warm yellow lamp with glowing center
                        let cx = tile_size / 2;
                        let cy = tile_size / 2;
                        let dx = (px as f32) - (cx as f32);
                        let dy = (py as f32) - (cy as f32);
                        let dist = (dx * dx + dy * dy).sqrt() / (tile_size as f32 * 0.45);
                        let glow = (1.0 - dist).max(0.0).powf(0.7);
                        let noise_val = ((px.wrapping_mul(17) ^ py.wrapping_mul(31)) % 16) as f32 / 64.0;
                        let r = ((255.0 * glow + 160.0 * (1.0 - glow)) * (1.0 + noise_val)).min(255.0) as u8;
                        let g = ((220.0 * glow + 80.0 * (1.0 - glow)) * (1.0 + noise_val * 0.5)).min(255.0) as u8;
                        let b = ((120.0 * glow + 30.0 * (1.0 - glow)) * (1.0 + noise_val * 0.3)).min(255.0) as u8;
                        (r, g, b)
                    }
                    "_water" => {
                        // Blue water with subtle wave noise
                        let noise = ((px.wrapping_mul(7) ^ py.wrapping_mul(13)) % 24) as i32 - 12;
                        (
                            (60 + noise).clamp(0, 255) as u8,
                            (100 + noise).clamp(0, 255) as u8,
                            (180 + noise).clamp(0, 255) as u8,
                        )
                    }
                    _ => {
                        // Gray placeholder
                        let noise = ((px.wrapping_mul(7) ^ py.wrapping_mul(13)) % 32) as i32 - 16;
                        let v = (80 + noise).clamp(0, 255) as u8;
                        (v, v, v)
                    }
                };
                atlas.put_pixel(base_x + px, base_y + py, Rgba([r, g, b, 255]));
            }
        }
    }

    /// Generate a full procedural atlas (fallback when no files found)
    fn generate_procedural_atlas(tiles_per_row: u32) -> Vec<u8> {
        let atlas_size = (TEXTURE_TILE_SIZE * tiles_per_row) as usize;
        let mut atlas: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::new(atlas_size as u32, atlas_size as u32);

        for pixel in atlas.pixels_mut() {
            *pixel = Rgba([20, 25, 20, 255]);
        }

        // Generate a few colored procedural tiles
        for i in 0..tiles_per_row * tiles_per_row {
            let tile_x = i % tiles_per_row;
            let tile_y = i / tiles_per_row;
            Self::generate_procedural_tile(&mut atlas, tile_x, tile_y, "_placeholder");
        }

        atlas.into_raw()
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
