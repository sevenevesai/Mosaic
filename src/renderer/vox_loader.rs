//! MagicaVoxel .vox file loader
//!
//! Loads .vox files and converts them to renderable meshes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use glam::Vec3;
use wgpu::util::DeviceExt;

/// A single voxel with position and color index
#[derive(Debug, Clone, Copy)]
pub struct Voxel {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub color_index: u8,
}

/// Loaded voxel model data
#[derive(Debug, Clone)]
pub struct VoxModel {
    /// Model dimensions
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    /// Voxel data
    pub voxels: Vec<Voxel>,
    /// Color palette (256 RGBA colors)
    pub palette: [[u8; 4]; 256],
}

impl VoxModel {
    /// Load a .vox file from disk
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Read magic "VOX "
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| format!("Failed to read magic: {}", e))?;
        if &magic != b"VOX " {
            return Err("Invalid .vox file: wrong magic".to_string());
        }

        // Read version (should be 150)
        let mut version = [0u8; 4];
        reader.read_exact(&mut version).map_err(|e| format!("Failed to read version: {}", e))?;
        let _version = u32::from_le_bytes(version);

        // Read MAIN chunk header
        let mut chunk_id = [0u8; 4];
        reader.read_exact(&mut chunk_id).map_err(|e| format!("Failed to read MAIN: {}", e))?;
        if &chunk_id != b"MAIN" {
            return Err("Expected MAIN chunk".to_string());
        }

        // Skip MAIN chunk content size and children size
        let mut sizes = [0u8; 8];
        reader.read_exact(&mut sizes).map_err(|e| format!("Failed to read sizes: {}", e))?;

        // Parse child chunks
        let mut size_x = 0u32;
        let mut size_y = 0u32;
        let mut size_z = 0u32;
        let mut voxels = Vec::new();
        let mut palette = Self::default_palette();

        loop {
            // Try to read next chunk
            let mut chunk_id = [0u8; 4];
            if reader.read_exact(&mut chunk_id).is_err() {
                break; // End of file
            }

            let mut content_size = [0u8; 4];
            reader.read_exact(&mut content_size).map_err(|e| format!("Failed to read content size: {}", e))?;
            let content_size = u32::from_le_bytes(content_size);

            let mut children_size = [0u8; 4];
            reader.read_exact(&mut children_size).map_err(|e| format!("Failed to read children size: {}", e))?;

            match &chunk_id {
                b"SIZE" => {
                    let mut data = [0u8; 12];
                    reader.read_exact(&mut data).map_err(|e| format!("Failed to read SIZE: {}", e))?;
                    size_x = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    size_y = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    size_z = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                }
                b"XYZI" => {
                    let mut num_voxels = [0u8; 4];
                    reader.read_exact(&mut num_voxels).map_err(|e| format!("Failed to read voxel count: {}", e))?;
                    let num_voxels = u32::from_le_bytes(num_voxels);

                    for _ in 0..num_voxels {
                        let mut voxel_data = [0u8; 4];
                        reader.read_exact(&mut voxel_data).map_err(|e| format!("Failed to read voxel: {}", e))?;
                        voxels.push(Voxel {
                            x: voxel_data[0],
                            y: voxel_data[1],
                            z: voxel_data[2],
                            color_index: voxel_data[3],
                        });
                    }
                }
                b"RGBA" => {
                    for i in 0..256 {
                        let mut color = [0u8; 4];
                        reader.read_exact(&mut color).map_err(|e| format!("Failed to read palette: {}", e))?;
                        // Palette index 0 is unused, colors are 1-255
                        let idx = if i == 255 { 0 } else { i + 1 };
                        palette[idx] = color;
                    }
                }
                _ => {
                    // Skip unknown chunks
                    let mut skip = vec![0u8; content_size as usize];
                    reader.read_exact(&mut skip).ok();
                }
            }
        }

        tracing::info!(
            "Loaded .vox model: {}x{}x{} with {} voxels",
            size_x, size_y, size_z, voxels.len()
        );

        Ok(Self {
            size_x,
            size_y,
            size_z,
            voxels,
            palette,
        })
    }

    /// Default MagicaVoxel palette
    fn default_palette() -> [[u8; 4]; 256] {
        let mut palette = [[255, 255, 255, 255]; 256];
        // MagicaVoxel default palette (simplified - first few colors)
        palette[0] = [0, 0, 0, 0]; // Index 0 is empty/transparent
        palette
    }

    /// Get voxel at position (returns None if empty)
    pub fn get_voxel(&self, x: u8, y: u8, z: u8) -> Option<u8> {
        self.voxels.iter()
            .find(|v| v.x == x && v.y == y && v.z == z)
            .map(|v| v.color_index)
    }

    /// Get color for a palette index
    pub fn get_color(&self, index: u8) -> [u8; 4] {
        self.palette[index as usize]
    }
}

/// Vertex for voxel model rendering
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// GPU-ready voxel mesh
pub struct VoxMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Center offset for the model
    pub center: Vec3,
}

impl VoxMesh {
    /// Create a mesh from a voxel model
    pub fn from_model(device: &wgpu::Device, model: &VoxModel, scale: f32) -> Self {
        let (vertices, indices) = Self::generate_mesh(model, scale);

        // Calculate center offset
        let center = Vec3::new(
            model.size_x as f32 * scale * 0.5,
            model.size_y as f32 * scale * 0.5,
            model.size_z as f32 * scale * 0.5,
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vox Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vox Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            center,
        }
    }

    /// Generate mesh vertices and indices from voxel model
    fn generate_mesh(model: &VoxModel, scale: f32) -> (Vec<VoxVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Build a lookup set for fast neighbor checking
        let mut voxel_set: HashMap<(u8, u8, u8), u8> = HashMap::new();
        for voxel in &model.voxels {
            voxel_set.insert((voxel.x, voxel.y, voxel.z), voxel.color_index);
        }

        // For each voxel, add faces where there's no neighbor
        for voxel in &model.voxels {
            let x = voxel.x;
            let y = voxel.y;
            let z = voxel.z;
            let color = model.get_color(voxel.color_index);
            let color_f = [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
                color[3] as f32 / 255.0,
            ];

            // Check each face
            let faces = [
                ([0, 1, 0], y.checked_add(1).map(|ny| voxel_set.contains_key(&(x, ny, z))).unwrap_or(false)), // +Y
                ([0, -1, 0], y.checked_sub(1).map(|ny| voxel_set.contains_key(&(x, ny, z))).unwrap_or(false)), // -Y
                ([1, 0, 0], x.checked_add(1).map(|nx| voxel_set.contains_key(&(nx, y, z))).unwrap_or(false)), // +X
                ([-1, 0, 0], x.checked_sub(1).map(|nx| voxel_set.contains_key(&(nx, y, z))).unwrap_or(false)), // -X
                ([0, 0, 1], z.checked_add(1).map(|nz| voxel_set.contains_key(&(x, y, nz))).unwrap_or(false)), // +Z
                ([0, 0, -1], z.checked_sub(1).map(|nz| voxel_set.contains_key(&(x, y, nz))).unwrap_or(false)), // -Z
            ];

            for (normal, has_neighbor) in faces {
                if !has_neighbor {
                    Self::add_face(
                        &mut vertices,
                        &mut indices,
                        x as f32 * scale,
                        y as f32 * scale,
                        z as f32 * scale,
                        scale,
                        normal,
                        color_f,
                    );
                }
            }
        }

        tracing::debug!("Generated vox mesh: {} vertices, {} indices", vertices.len(), indices.len());
        (vertices, indices)
    }

    /// Add a single face to the mesh
    fn add_face(
        vertices: &mut Vec<VoxVertex>,
        indices: &mut Vec<u32>,
        x: f32,
        y: f32,
        z: f32,
        size: f32,
        normal: [i32; 3],
        color: [f32; 4],
    ) {
        let base_index = vertices.len() as u32;
        let n = [normal[0] as f32, normal[1] as f32, normal[2] as f32];

        // Generate 4 vertices for the face based on normal direction
        let (v0, v1, v2, v3) = match normal {
            [0, 1, 0] => ( // +Y (top)
                [x, y + size, z],
                [x, y + size, z + size],
                [x + size, y + size, z + size],
                [x + size, y + size, z],
            ),
            [0, -1, 0] => ( // -Y (bottom)
                [x, y, z + size],
                [x, y, z],
                [x + size, y, z],
                [x + size, y, z + size],
            ),
            [1, 0, 0] => ( // +X (right)
                [x + size, y, z + size],
                [x + size, y, z],
                [x + size, y + size, z],
                [x + size, y + size, z + size],
            ),
            [-1, 0, 0] => ( // -X (left)
                [x, y, z],
                [x, y, z + size],
                [x, y + size, z + size],
                [x, y + size, z],
            ),
            [0, 0, 1] => ( // +Z (back)
                [x + size, y, z + size],
                [x + size, y + size, z + size],
                [x, y + size, z + size],
                [x, y, z + size],
            ),
            [0, 0, -1] => ( // -Z (front)
                [x, y, z],
                [x, y + size, z],
                [x + size, y + size, z],
                [x + size, y, z],
            ),
            _ => return,
        };

        vertices.push(VoxVertex { position: v0, normal: n, color });
        vertices.push(VoxVertex { position: v1, normal: n, color });
        vertices.push(VoxVertex { position: v2, normal: n, color });
        vertices.push(VoxVertex { position: v3, normal: n, color });

        // Two triangles (CCW winding)
        indices.push(base_index);
        indices.push(base_index + 1);
        indices.push(base_index + 2);
        indices.push(base_index);
        indices.push(base_index + 2);
        indices.push(base_index + 3);
    }
}
