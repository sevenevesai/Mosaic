//! Texture atlas handling for voxel blocks

use crate::constants::TEXTURE_TILE_SIZE;
use image::{ImageBuffer, Rgba};
use std::path::Path;

/// Texture atlas for block textures
pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub tiles_per_row: u32,
}

/// Maps tile indices to texture file names
/// Atlas is 8x8 = 64 tiles total
/// 0-31: Original blocks + ores + bark
/// 32-56: New biome blocks (Phase 3)
const TEXTURE_FILES: &[&str] = &[
    "256_Dirt Pebbles 01.png",      // 0: Dirt
    "256_Grass 01 Simple.png",      // 1: Grass top
    "256_Grass 03 Mud.png",         // 2: Grass side (grass with mud/dirt)
    "256_Stone 2.png",              // 3: Stone (main cave/underground stone)
    "256_Stone Dirt 01.png",        // 4: StoneDirt (dirty stone variant for variation)
    "256_Brick 06.png",             // 5: Brick (placeable building material)
    "256_Water Shallow 01.png",     // 6: Water
    "256_Wood Planks 01.png",       // 7: Wood bark (log sides)
    "256_Wood Planks 02.png",       // 8: Wood planks (log top/bottom)
    "64_Leaves_Grass_Dense.png",    // 9: Leaves (dense grass leaves for foliage)
    "256_Stone Rubble 02.png",      // 10: StoneRubble2 (stone variant)
    "64_Ore_Iron.png",              // 11: Iron ore
    "64_Ore_Gold.png",              // 12: Gold ore
    "64_Ore_Diamond.png",           // 13: Diamond ore
    "64_Ore_Redstone.png",          // 14: Redstone ore
    "256_Dirt Rocks 01.png",        // 15: Bedrock (dirt with rocks)
    "",                             // 16: Torch (procedural only - orange glow)
    "256_Sand 01.png",              // 17: Sand (beach/desert)
    // Ores (18-29)
    "64_Ore_Coal.png",              // 18: Coal ore
    "64_Ore_Copper.png",            // 19: Copper ore
    "64_Ore_Emerald.png",           // 20: Emerald ore
    "64_Ore_Sapphire.png",          // 21: Sapphire ore
    "64_Ore_Quartz.png",            // 22: Quartz ore
    "64_Ore_Amethyst.png",          // 23: Amethyst ore
    "64_Ore_Silver.png",            // 24: Silver ore
    "64_Ore_Tin.png",               // 25: Tin ore
    "64_Ore_Nickel.png",            // 26: Nickel ore
    "64_Ore_Lead.png",              // 27: Lead ore
    "64_Ore_Titanium.png",          // 28: Titanium ore
    "64_Ore_Uranium.png",           // 29: Uranium ore (glows!)
    // Tree bark variants (30-31)
    "64_Bark_Oak_Dark.png",         // 30: Oak bark (dark variant)
    "64_Bark_Birch.png",            // 31: Birch bark (white)

    // === NEW BIOME TEXTURES (32-56) ===
    // Using existing textures where possible, procedural fallback otherwise

    // Desert/Badlands (32-33)
    "256_Sand 02.png",              // 32: Sandstone (sand variant)
    "256_Tile Pattern 03 Red.png",  // 33: Red sandstone (red tile pattern)

    // Tundra/Snow (34-37)
    "64_Snow_2.png",                // 34: Snow top
    "64_Ice_Block_2.png",           // 35: Snowy stone/rocks
    "64_Ice_Block_1.png",           // 36: Ice block
    "64_Ice_Block_1.png",           // 37: Packed ice (same as ice for now)

    // Swamp (38-39)
    "256_Dirt Muddy 01.png",        // 38: Mud (muddy dirt)
    "256_Dirt Cracked 01.png",      // 39: Peat (cracked dirt)

    // Jungle (40-43)
    "256_Grass 02 Blades.png",      // 40: Jungle grass top (grass blades)
    "256_Roots 01 Mud.png",         // 41: Jungle grass side (roots)
    "64_Leaves_Grass_Dense.png",    // 42: Jungle leaves (same as regular leaves)
    "256_Roots 02 Mud.png",         // 43: Vines (roots texture)

    // Mountain (44)
    "256_Stone Rubble 01.png",      // 44: Gravel (stone rubble)

    // Structure blocks (45-47)
    "256_Roots 01 Mud.png",         // 45: Mossy stone (roots for moss effect)
    "256_Dirt Cracked 02.png",      // 46: Cracked stone (cracked dirt)
    "256_Cement 03.png",            // 47: Claystone (cement variant)

    // Cave decoration (48-50)
    "256_Stone 2.png",              // 48: Stalactite/stalagmite (stone)
    "256_Tiles Circle 08.png",      // 49: Crystal amethyst (decorative tile)
    "256_Cement 01.png",            // 50: Crystal quartz (white cement)

    // Desert vegetation (51-53)
    "256_Grass 02 Blades.png",      // 51: Cactus side (green grass blades)
    "256_Mound 01.png",             // 52: Cactus top (mound)
    "256_Dirt Cracked 01.png",      // 53: Dead bush (cracked earth)

    // Additional trees (54-56)
    "256_Dirt Rocks 02.png",        // 54: Pine bark (dark rocky)
    "256_Roots 02 Mud.png",         // 55: Jungle bark (roots)
    "64_Leaves_Grass_Dense.png",    // 56: Spruce leaves (same as leaves)
];

const TEXTURE_DIR: &str = "assets/textures/256x256 Pixel Textures";

impl TextureAtlas {
    /// Load texture atlas from PNG files with full mipmap chain and anisotropic filtering
    /// This provides smooth, high-quality textures at all distances
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let tiles_per_row = 8u32; // 8x8 grid = 64 tiles (using 57)
        let atlas_size = TEXTURE_TILE_SIZE * tiles_per_row; // 256 * 8 = 2048

        // Calculate mip levels: log2(1024) = 10 levels (1024 -> 512 -> 256 -> ... -> 1)
        let mip_level_count = (atlas_size as f32).log2().floor() as u32 + 1;
        log::info!("Creating texture atlas {}x{} with {} mip levels", atlas_size, atlas_size, mip_level_count);

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
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload base mip level (level 0)
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

        // Generate and upload mipmap chain
        Self::generate_mipmaps(queue, &texture, &atlas_data, atlas_size, mip_level_count);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // High-quality sampler with anisotropic filtering
        // Uses trilinear filtering with 16x anisotropy for smooth textures at all angles
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Block Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            // Trilinear filtering: linear between texels and between mip levels
            mag_filter: wgpu::FilterMode::Linear,  // Smooth when close
            min_filter: wgpu::FilterMode::Linear,  // Smooth when far
            mipmap_filter: wgpu::FilterMode::Linear, // Smooth mip transitions
            // Anisotropic filtering for sharp textures at oblique angles
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

            // Create image from current level
            let current_image: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(
                current_size,
                current_size,
                current_data.clone(),
            ).expect("Failed to create image buffer for mipmap");

            // Downsample using high-quality Lanczos3 filter
            // This provides excellent quality with minimal aliasing
            let resized = image::imageops::resize(
                &current_image,
                new_size,
                new_size,
                image::imageops::FilterType::Lanczos3,
            );

            let mip_data: Vec<u8> = resized.into_raw();

            // Upload this mip level
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
            // Calculate position in atlas
            let tile_x = (index as u32) % tiles_per_row;
            let tile_y = (index as u32) / tiles_per_row;
            let base_x = tile_x * tile_size;
            let base_y = tile_y * tile_size;

            // If filename is empty, generate procedural texture
            if filename.is_empty() {
                Self::generate_tile_to_atlas(&mut atlas, tile_x, tile_y, tiles_per_row, index as u32);
                loaded_count += 1;
                continue;
            }

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
                    log::warn!("Failed to load {}: {}, using procedural fallback", filename, e);
                    // Generate procedural texture as fallback
                    Self::generate_tile_to_atlas(&mut atlas, tile_x, tile_y, tiles_per_row, index as u32);
                    loaded_count += 1;
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
        let tiles_per_row = 8u32;
        let atlas_size = (TEXTURE_TILE_SIZE * tiles_per_row) as usize;
        let mut data = vec![0u8; atlas_size * atlas_size * 4];

        // Generate textures for each tile (0-29 used for blocks and ores)
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
            3 => (128, 128, 128),  // Stone (rubble)
            4 => (110, 100, 90),   // StoneDirt (dirty stone variant)
            5 => (180, 140, 100),  // Brick (warm tan brick)
            6 => (60, 100, 180),   // Water
            7 => (80, 60, 40),     // Wood (bark)
            8 => (180, 150, 100),  // Wood (cross-section)
            9 => (60, 130, 50),    // Leaves
            10 => (140, 140, 140), // StoneRubble2 (lighter stone variant)
            11 => (180, 160, 100), // Iron ore
            12 => (220, 200, 50),  // Gold ore
            13 => (100, 180, 220), // Diamond ore
            14 => (200, 50, 50),   // Redstone ore
            15 => (60, 60, 80),    // Bedrock
            16 => (255, 180, 80),  // Torch (warm orange glow)
            17 => (194, 178, 128), // Sand (beach/desert)
            _ => (80, 80, 80),     // Unused slots (dark gray)
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

    /// Generate a procedural tile directly into an ImageBuffer atlas
    fn generate_tile_to_atlas(
        atlas: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
        tile_x: u32,
        tile_y: u32,
        tiles_per_row: u32,
        tile_index: u32,
    ) {
        let tile_size = TEXTURE_TILE_SIZE as u32;

        // Base colors for different block types
        let (base_r, base_g, base_b) = match tile_index {
            0 => (100, 80, 60),    // Dirt
            1 => (80, 160, 60),    // Grass top
            2 => (100, 80, 60),    // Grass side (dirt with grass edge)
            3 => (128, 128, 128),  // Stone (rubble)
            4 => (110, 100, 90),   // StoneDirt (dirty stone variant)
            5 => (180, 140, 100),  // Brick (warm tan brick)
            6 => (60, 100, 180),   // Water
            7 => (80, 60, 40),     // Wood (bark)
            8 => (180, 150, 100),  // Wood (cross-section)
            9 => (60, 130, 50),    // Leaves
            10 => (140, 140, 140), // StoneRubble2 (lighter stone variant)
            11 => (180, 160, 100), // Iron ore
            12 => (220, 200, 50),  // Gold ore
            13 => (100, 180, 220), // Diamond ore
            14 => (200, 50, 50),   // Redstone ore
            15 => (60, 60, 80),    // Bedrock
            16 => (255, 180, 80),  // Torch (warm orange glow)
            17 => (194, 178, 128), // Sand (beach/desert)
            _ => (80, 80, 80),     // Unused slots (dark gray)
        };

        // Simple pseudo-random for texture variation
        let seed = tile_index.wrapping_mul(2654435761);
        let base_x = tile_x * tile_size;
        let base_y = tile_y * tile_size;

        for py in 0..tile_size {
            for px in 0..tile_size {
                // Add noise/variation
                let noise = ((px.wrapping_mul(7) ^ py.wrapping_mul(13) ^ seed) % 32) as i32 - 16;

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
                } else if tile_index == 16 {
                    // Torch texture: warm yellow lamp style with glowing center
                    let cx = tile_size / 2;
                    let cy = tile_size / 2;
                    let dx = (px as i32 - cx as i32) as f32;
                    let dy = (py as i32 - cy as i32) as f32;
                    let dist = (dx * dx + dy * dy).sqrt() / (tile_size as f32 * 0.45);
                    let glow = (1.0 - dist).max(0.0).powf(0.7);

                    // Add subtle noise for texture
                    let noise_val = ((px.wrapping_mul(17) ^ py.wrapping_mul(31)) % 16) as f32 / 64.0;

                    // Warm yellow center (like tavern lamp) fading to amber/brown edges
                    // Center: bright warm yellow (255, 220, 120)
                    // Edge: darker amber (180, 100, 40)
                    let center_r = 255.0;
                    let center_g = 220.0;
                    let center_b = 120.0;
                    let edge_r = 160.0;
                    let edge_g = 80.0;
                    let edge_b = 30.0;

                    let r = ((center_r * glow + edge_r * (1.0 - glow)) * (1.0 + noise_val)).min(255.0) as u8;
                    let g = ((center_g * glow + edge_g * (1.0 - glow)) * (1.0 + noise_val * 0.5)).min(255.0) as u8;
                    let b = ((center_b * glow + edge_b * (1.0 - glow)) * (1.0 + noise_val * 0.3)).min(255.0) as u8;
                    (r, g, b)
                } else {
                    (r, g, b)
                };

                atlas.put_pixel(base_x + px, base_y + py, Rgba([r, g, b, 255]));
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
