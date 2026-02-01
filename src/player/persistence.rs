//! Player data persistence
//!
//! Saves and loads player state (position, rotation, inventory) to player.dat

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use crate::inventory::{Inventory, Item, ItemStack, TOTAL_SLOTS};
use crate::world::Block;

/// Magic bytes for player.dat file: "PLYR"
const PLAYER_MAGIC: [u8; 4] = *b"PLYR";

/// Current format version
const PLAYER_VERSION: u16 = 1;

/// Fixed header size in bytes
const PLAYER_HEADER_SIZE: usize = 256;

/// Error type for player data operations
#[derive(Debug)]
pub enum PlayerDataError {
    Io(std::io::Error),
    InvalidFormat(String),
}

impl From<std::io::Error> for PlayerDataError {
    fn from(e: std::io::Error) -> Self {
        PlayerDataError::Io(e)
    }
}

impl std::fmt::Display for PlayerDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlayerDataError::Io(e) => write!(f, "IO error: {}", e),
            PlayerDataError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

/// Saved player data
#[derive(Debug, Clone)]
pub struct PlayerData {
    /// Player camera position
    pub position: [f32; 3],
    /// Player camera rotation (yaw, pitch)
    pub rotation: [f32; 2],
    /// Selected hotbar slot (0-8)
    pub selected_slot: u8,
    /// Fly mode enabled
    pub fly_mode: bool,
    /// Inventory slots: (item_id, count, flags)
    /// item_id 0 = empty slot, otherwise Block id + 1
    pub inventory: [(u16, u8, u8); TOTAL_SLOTS],
}

impl Default for PlayerData {
    fn default() -> Self {
        Self {
            position: [0.0, 80.0, 0.0],
            rotation: [0.0, 0.0],
            selected_slot: 0,
            fly_mode: false,
            inventory: [(0, 0, 0); TOTAL_SLOTS],
        }
    }
}

impl PlayerData {
    /// Create PlayerData from current player/inventory state
    pub fn from_player(
        position: [f32; 3],
        rotation: [f32; 2],
        selected_slot: u8,
        fly_mode: bool,
        inventory: &Inventory,
    ) -> Self {
        let mut inv_data = [(0u16, 0u8, 0u8); TOTAL_SLOTS];

        for (i, slot) in inventory.all_slots().iter().enumerate() {
            if let Some(stack) = slot {
                // Encode item - for Block items, use block_id + 1 (0 = empty)
                let item_id = match stack.item {
                    Item::Block(block) => (block as u8 as u16) + 1,
                };
                let flags = if stack.locked { 0x01 } else { 0x00 };
                inv_data[i] = (item_id, stack.count, flags);
            }
        }

        Self {
            position,
            rotation,
            selected_slot,
            fly_mode,
            inventory: inv_data,
        }
    }

    /// Apply saved data to an inventory
    pub fn apply_to_inventory(&self, inventory: &mut Inventory) {
        for (i, &(item_id, count, flags)) in self.inventory.iter().enumerate() {
            if item_id == 0 || count == 0 {
                // Empty slot
                if let Some(slot) = inventory.slot_mut(i) {
                    *slot = None;
                }
            } else {
                // Block item: id - 1 = block enum value
                let block_id = (item_id - 1) as u8;
                let block = Block::from(block_id);
                let item = Item::Block(block);
                let locked = (flags & 0x01) != 0;

                if let Some(slot) = inventory.slot_mut(i) {
                    let mut stack = ItemStack::new(item, count);
                    if locked {
                        stack.locked = true;
                    }
                    *slot = Some(stack);
                }
            }
        }

        // Restore selected slot
        inventory.select_slot(self.selected_slot as usize);
    }

    /// Save player data to file (atomic write)
    pub fn save(&self, path: &Path) -> Result<(), PlayerDataError> {
        let mut data = [0u8; PLAYER_HEADER_SIZE];

        // Magic (0-3)
        data[0..4].copy_from_slice(&PLAYER_MAGIC);

        // Version (4-5)
        data[4..6].copy_from_slice(&PLAYER_VERSION.to_le_bytes());

        // Flags (6-7) - bit 0: fly_mode
        let flags: u16 = if self.fly_mode { 0x01 } else { 0x00 };
        data[6..8].copy_from_slice(&flags.to_le_bytes());

        // Position (8-19)
        data[8..12].copy_from_slice(&self.position[0].to_le_bytes());
        data[12..16].copy_from_slice(&self.position[1].to_le_bytes());
        data[16..20].copy_from_slice(&self.position[2].to_le_bytes());

        // Rotation (20-27)
        data[20..24].copy_from_slice(&self.rotation[0].to_le_bytes());
        data[24..28].copy_from_slice(&self.rotation[1].to_le_bytes());

        // Selected slot (28)
        data[28] = self.selected_slot;

        // Reserved (29-31)
        // Already zero

        // Inventory (32-175): 36 slots × 4 bytes each
        let inv_start = 32;
        for (i, &(item_id, count, flags)) in self.inventory.iter().enumerate() {
            let offset = inv_start + i * 4;
            data[offset..offset + 2].copy_from_slice(&item_id.to_le_bytes());
            data[offset + 2] = count;
            data[offset + 3] = flags;
        }

        // Remaining bytes (176-255) are reserved, already zero

        // Atomic write: write to temp file, then rename
        let tmp_path = path.with_extension("dat.tmp");
        let mut file = File::create(&tmp_path)?;
        file.write_all(&data)?;
        file.sync_all()?;
        fs::rename(&tmp_path, path)?;

        tracing::info!("Saved player data to {:?}", path);
        Ok(())
    }

    /// Load player data from file
    pub fn load(path: &Path) -> Result<Self, PlayerDataError> {
        let mut file = File::open(path)?;
        let mut data = [0u8; PLAYER_HEADER_SIZE];
        file.read_exact(&mut data)?;

        // Verify magic
        if data[0..4] != PLAYER_MAGIC {
            return Err(PlayerDataError::InvalidFormat(
                "Invalid player.dat magic bytes".to_string(),
            ));
        }

        // Check version
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version > PLAYER_VERSION {
            return Err(PlayerDataError::InvalidFormat(format!(
                "Unsupported player version {} (max: {})",
                version, PLAYER_VERSION
            )));
        }

        // Parse flags
        let flags = u16::from_le_bytes([data[6], data[7]]);
        let fly_mode = (flags & 0x01) != 0;

        // Parse position
        let position = [
            f32::from_le_bytes(data[8..12].try_into().unwrap()),
            f32::from_le_bytes(data[12..16].try_into().unwrap()),
            f32::from_le_bytes(data[16..20].try_into().unwrap()),
        ];

        // Parse rotation
        let rotation = [
            f32::from_le_bytes(data[20..24].try_into().unwrap()),
            f32::from_le_bytes(data[24..28].try_into().unwrap()),
        ];

        // Selected slot
        let selected_slot = data[28].min(8); // Clamp to valid range

        // Parse inventory
        let mut inventory = [(0u16, 0u8, 0u8); TOTAL_SLOTS];
        let inv_start = 32;
        for i in 0..TOTAL_SLOTS {
            let offset = inv_start + i * 4;
            let item_id = u16::from_le_bytes([data[offset], data[offset + 1]]);
            let count = data[offset + 2];
            let flags = data[offset + 3];
            inventory[i] = (item_id, count, flags);
        }

        tracing::info!("Loaded player data from {:?}", path);
        Ok(Self {
            position,
            rotation,
            selected_slot,
            fly_mode,
            inventory,
        })
    }

    /// Check if a player.dat file exists
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_player_data_roundtrip() {
        let mut data = PlayerData::default();
        data.position = [10.0, 50.0, -30.0];
        data.rotation = [1.5, -0.5];
        data.selected_slot = 3;
        data.fly_mode = true;
        data.inventory[0] = (4, 32, 0x01); // Stone (3+1), 32 count, locked
        data.inventory[5] = (16, 64, 0x00); // Torch (15+1), 64 count

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();

        // Save
        data.save(path).unwrap();

        // Load
        let loaded = PlayerData::load(path).unwrap();

        assert_eq!(data.position, loaded.position);
        assert_eq!(data.rotation, loaded.rotation);
        assert_eq!(data.selected_slot, loaded.selected_slot);
        assert_eq!(data.fly_mode, loaded.fly_mode);
        assert_eq!(data.inventory[0], loaded.inventory[0]);
        assert_eq!(data.inventory[5], loaded.inventory[5]);
    }

    #[test]
    fn test_inventory_apply() {
        let mut data = PlayerData::default();
        // Add stone (Block::Stone = 3, so item_id = 4)
        data.inventory[0] = (4, 32, 0x00);
        // Add torch (Block::Torch = 15, so item_id = 16)
        data.inventory[1] = (16, 10, 0x01); // locked

        let mut inv = Inventory::new_empty();
        data.apply_to_inventory(&mut inv);

        let slot0 = inv.get_slot(0).unwrap();
        assert_eq!(slot0.item, Item::Block(Block::Stone));
        assert_eq!(slot0.count, 32);
        assert!(!slot0.locked);

        let slot1 = inv.get_slot(1).unwrap();
        assert_eq!(slot1.item, Item::Block(Block::Torch));
        assert_eq!(slot1.count, 10);
        assert!(slot1.locked);
    }
}
