//! Item entity persistence
//!
//! Saves and loads dropped items (ItemEntity) to items.dat

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use super::item_entity::{ItemEntityManager, SavedItemEntity};

/// Magic bytes for items.dat file: "ITMS"
const ITEMS_MAGIC: [u8; 4] = *b"ITMS";

/// Current format version
const ITEMS_VERSION: u16 = 1;

/// Header size (fixed)
const HEADER_SIZE: usize = 16;

/// Per-item data size
const ITEM_ENTRY_SIZE: usize = 32;

/// Error type for item data operations
#[derive(Debug)]
pub enum ItemDataError {
    Io(std::io::Error),
    InvalidFormat(String),
}

impl From<std::io::Error> for ItemDataError {
    fn from(e: std::io::Error) -> Self {
        ItemDataError::Io(e)
    }
}

impl std::fmt::Display for ItemDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ItemDataError::Io(e) => write!(f, "IO error: {}", e),
            ItemDataError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

/// Item data file format:
///
/// Header (16 bytes):
///   [0-3]   Magic "ITMS"
///   [4-5]   Version (u16)
///   [6-9]   Item count (u32)
///   [10-15] Reserved
///
/// Per item (32 bytes each):
///   [0-11]  Position (3x f32)
///   [12-23] Velocity (3x f32)
///   [24]    Item ID (u8)
///   [25]    Count (u8)
///   [26-29] Age (f32)
///   [30]    Flags: bit 0 = no_despawn
///   [31]    Reserved

/// Save item entities to file
pub fn save_items(path: &Path, manager: &ItemEntityManager) -> Result<(), ItemDataError> {
    let items = manager.save_data();
    let item_count = items.len() as u32;

    // Calculate total size
    let total_size = HEADER_SIZE + items.len() * ITEM_ENTRY_SIZE;
    let mut data = vec![0u8; total_size];

    // Write header
    data[0..4].copy_from_slice(&ITEMS_MAGIC);
    data[4..6].copy_from_slice(&ITEMS_VERSION.to_le_bytes());
    data[6..10].copy_from_slice(&item_count.to_le_bytes());
    // [10-15] reserved, already zero

    // Write items
    for (i, item) in items.iter().enumerate() {
        let offset = HEADER_SIZE + i * ITEM_ENTRY_SIZE;

        // Position
        data[offset..offset+4].copy_from_slice(&item.position[0].to_le_bytes());
        data[offset+4..offset+8].copy_from_slice(&item.position[1].to_le_bytes());
        data[offset+8..offset+12].copy_from_slice(&item.position[2].to_le_bytes());

        // Velocity
        data[offset+12..offset+16].copy_from_slice(&item.velocity[0].to_le_bytes());
        data[offset+16..offset+20].copy_from_slice(&item.velocity[1].to_le_bytes());
        data[offset+20..offset+24].copy_from_slice(&item.velocity[2].to_le_bytes());

        // Item ID and count
        data[offset+24] = item.item_id;
        data[offset+25] = item.count;

        // Age
        data[offset+26..offset+30].copy_from_slice(&item.age.to_le_bytes());

        // Flags
        let flags = if item.no_despawn { 0x01 } else { 0x00 };
        data[offset+30] = flags;
        // [31] reserved
    }

    // Atomic write
    let tmp_path = path.with_extension("dat.tmp");
    let mut file = File::create(&tmp_path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    fs::rename(&tmp_path, path)?;

    tracing::info!("Saved {} item entities to {:?}", item_count, path);
    Ok(())
}

/// Load item entities from file
pub fn load_items(path: &Path, manager: &mut ItemEntityManager) -> Result<(), ItemDataError> {
    if !path.exists() {
        tracing::debug!("No items.dat file found at {:?}", path);
        return Ok(());
    }

    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    // Verify minimum size for header
    if data.len() < HEADER_SIZE {
        return Err(ItemDataError::InvalidFormat(
            "File too small for header".to_string()
        ));
    }

    // Verify magic
    if data[0..4] != ITEMS_MAGIC {
        return Err(ItemDataError::InvalidFormat(
            "Invalid items.dat magic bytes".to_string()
        ));
    }

    // Check version
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version > ITEMS_VERSION {
        return Err(ItemDataError::InvalidFormat(format!(
            "Unsupported items version {} (max: {})",
            version, ITEMS_VERSION
        )));
    }

    // Read item count
    let item_count = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;

    // Verify file size
    let expected_size = HEADER_SIZE + item_count * ITEM_ENTRY_SIZE;
    if data.len() < expected_size {
        return Err(ItemDataError::InvalidFormat(format!(
            "File too small: expected {} bytes for {} items, got {}",
            expected_size, item_count, data.len()
        )));
    }

    // Parse items
    let mut items = Vec::with_capacity(item_count);
    for i in 0..item_count {
        let offset = HEADER_SIZE + i * ITEM_ENTRY_SIZE;

        let position = [
            f32::from_le_bytes(data[offset..offset+4].try_into().unwrap()),
            f32::from_le_bytes(data[offset+4..offset+8].try_into().unwrap()),
            f32::from_le_bytes(data[offset+8..offset+12].try_into().unwrap()),
        ];

        let velocity = [
            f32::from_le_bytes(data[offset+12..offset+16].try_into().unwrap()),
            f32::from_le_bytes(data[offset+16..offset+20].try_into().unwrap()),
            f32::from_le_bytes(data[offset+20..offset+24].try_into().unwrap()),
        ];

        let item_id = data[offset+24];
        let count = data[offset+25];
        let age = f32::from_le_bytes(data[offset+26..offset+30].try_into().unwrap());
        let flags = data[offset+30];
        let no_despawn = (flags & 0x01) != 0;

        items.push(SavedItemEntity {
            position,
            velocity,
            item_id,
            count,
            age,
            no_despawn,
        });
    }

    manager.load_data(items);
    tracing::info!("Loaded {} item entities from {:?}", item_count, path);
    Ok(())
}

/// Check if items.dat file exists
pub fn items_exist(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::Item;
    use crate::world::Block;
    use glam::Vec3;
    use tempfile::NamedTempFile;

    #[test]
    fn test_items_roundtrip() {
        let mut manager = ItemEntityManager::new();

        // Spawn some items
        manager.spawn(
            Vec3::new(10.0, 50.0, -30.0),
            Vec3::new(1.0, 2.0, 0.5),
            Item::Block(Block::Stone),
            32,
        );
        manager.spawn(
            Vec3::new(-5.0, 40.0, 20.0),
            Vec3::ZERO,
            Item::Block(Block::Dirt),
            16,
        );

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();

        // Save
        save_items(path, &manager).unwrap();

        // Load into new manager
        let mut loaded_manager = ItemEntityManager::new();
        load_items(path, &mut loaded_manager).unwrap();

        assert_eq!(loaded_manager.count(), 2);
    }

    #[test]
    fn test_empty_items() {
        let manager = ItemEntityManager::new();

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();

        // Save empty
        save_items(path, &manager).unwrap();

        // Load
        let mut loaded = ItemEntityManager::new();
        load_items(path, &mut loaded).unwrap();

        assert_eq!(loaded.count(), 0);
    }
}
