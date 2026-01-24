//! Texture atlas handling for voxel blocks

use crate::constants::TEXTURE_TILE_SIZE;
use image::{GenericImageView, ImageBuffer, Rgba};
use std::path::Path;

/// Texture atlas for block textures
pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub tiles_per_row: u32,
}

/// Maps tile indices to texture file names
/// Order matches the procedural atlas for compatibility:
/// 0: Dirt, 1: Grass top, 2: Grass side, 3: Stone, 4: Cobblestone,
/// 5: Sand, 6: Water, 7: Wood bark, 8: Wood planks, 9: Leaves,
/// 10: Coal ore, 11: Iron ore, 12: Gold ore, 13: Diamond ore, 14: Redstone ore, 15: Bedrock
const TEXTURE_FILES: &[&str] = &[
    "256_Dirt Pebbles 01.png",      // 0: Dirt
    "256_Grass 01 Simple.png",      // 1: Grass top
    "256_Grass 03 Mud.png",         // 2: Grass side (grass with mud/dirt)
    "256_Stone Rubble 01.png",      // 3: Stone
    "256_Brick 02.png",             // 4: Cobblestone (using brick as substitute)
    "256_Sand 01.png",              // 5: Sand
    "256_Water Shallow 01.png",     // 6: Water
    "256_Wood Branches 01.png",     // 7: Wood bark
    "256_Wood Planks 02.png",       // 8: Wood planks (cross-section)
    "256_Grass 02 Blades.png",      // 9: Leaves (grass blades as leaf substitute)
    "256_Stone Dirt 01.png",        // 10: Coal ore (dark stone)
    "256_Cement 01.png",            // 11: Iron ore (gray cement)
    "256_Marble 02.png",            // 12: Gold ore (light marble)
    "256_Marble 01.png",            // 13: Diamond ore (marble)
    "256_Tile Pattern 03 Red.png",  // 14: Redstone ore (red tiles)
    "256_Dirt Rocks 01.png",        // 15: Bedrock (dirt with rocks)
];

const TEXTURE_DIR: &str = "assets/textures/256x256 Pixel Textures";

impl TextureAtlas {
    /// Load texture atlas from PNG files
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let tiles_per_row = 4u32; // 4x4 grid = 16 tiles
        let atlas_size = TEXTURE_TILE_SIZE * tiles_per_row; // 256 * 4 = 1024

        // Try to load from files, fall back to procedural if files not found
        let atlas_data = match Self::load_atlas_from_files(tiles_per_row) {
            Ok(data) => {
                log::info!("Loaded texture atlas from PNG files ({} tiles)", TEXTURE_FILES.len());
                data
            }
            Err(e) => {
                log::warn!("Failed to load texture files: {}. Using procedural textures.", e);
                Self::generate_procedural_atlas()
            }
        };

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

    /// Load textures from PNG files and combine into atlas
    fn load_atlas_from_files(tiles_per_row: u32) -> Result<Vec<u8>, String> {
        let tile_size = TEXTURE_TILE_SIZE as u32;
        let atlas_size = tile_size * tiles_per_row;

        // Create empty atlas (RGBA)
        let mut atlas: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::new(atlas_size, atlas_size);

        // Fill with magenta (missing texture indicator)
        for pixel in atlas.pixels_mut() {
            *pixel = Rgba([255, 0, 255, 255]);
        }

        // Get the executable's directory for asset resolution
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // Try multiple base paths
        let possible_paths = [
            Path::new(TEXTURE_DIR).to_path_buf(),  // Relative to CWD
            exe_dir.join(TEXTURE_DIR),              // Relative to exe
            exe_dir.join("..").join("..").join("..").join(TEXTURE_DIR), // cargo run from target/release
        ];

        let base_path = possible_paths.iter()
            .find(|p| p.exists())
            .cloned();

        let base_path = match base_path {
            Some(p) => {
                log::info!("Found texture directory: {}", p.display());
                p
            }
            None => {
                log::warn!("Texture directory not found. Tried:");
                for p in &possible_paths {
                    log::warn!("  - {}", p.display());
                }
                return Err("Texture directory not found".to_string());
            }
        };

        let mut loaded_count = 0;
        for (index, filename) in TEXTURE_FILES.iter().enumerate() {
            let file_path = base_path.join(filename);

            match image::open(&file_path) {
                Ok(img) => {
                    // Resize if necessary (should already be 256x256)
                    let img = if img.width() != tile_size || img.height() != tile_size {
                        img.resize_exact(tile_size, tile_size, image::imageops::FilterType::Nearest)
                    } else {
                        img
                    };

                    let rgba = img.to_rgba8();

                    // Calculate position in atlas
                    let tile_x = (index as u32) % tiles_per_row;
                    let tile_y = (index as u32) / tiles_per_row;
                    let base_x = tile_x * tile_size;
                    let base_y = tile_y * tile_size;

                    // Copy pixels to atlas
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let pixel = rgba.get_pixel(x, y);
                            atlas.put_pixel(base_x + x, base_y + y, *pixel);
                        }
                    }

                    loaded_count += 1;
                }
                Err(e) => {
                    log::warn!("Failed to load {}: {}", filename, e);
                }
            }
        }

        if loaded_count == 0 {
            return Err("No textures could be loaded".to_string());
        }

        log::info!("Loaded {}/{} textures into atlas", loaded_count, TEXTURE_FILES.len());
        Ok(atlas.into_raw())
    }

    /// Generate a procedural texture atlas for testing (fallback)
    fn generate_procedural_atlas() -> Vec<u8> {
        let tiles_per_row = 4u32;
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
                let (r, g, b) = if tile_index == 2 && py < tile_size / 4 {
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
