//! Chest management
//!
//! Tracks chest positions and their inventories in the world.

use std::collections::HashMap;

use glam::IVec3;

use crate::inventory::{Item, ItemStack};

/// Number of slots in a chest
pub const CHEST_SLOTS: usize = 27;

/// Simple container for chest storage
#[derive(Clone)]
pub struct Container {
    slots: Vec<Option<ItemStack>>,
}

impl Container {
    pub fn new(size: usize) -> Self {
        Self {
            slots: vec![None; size],
        }
    }

    pub fn size(&self) -> usize {
        self.slots.len()
    }

    pub fn get_slot(&self, index: usize) -> Option<&ItemStack> {
        self.slots.get(index).and_then(|s| s.as_ref())
    }

    pub fn set_slot(&mut self, index: usize, stack: Option<ItemStack>) {
        if index < self.slots.len() {
            self.slots[index] = stack;
        }
    }

    pub fn take_slot(&mut self, index: usize) -> Option<ItemStack> {
        if index < self.slots.len() {
            self.slots[index].take()
        } else {
            None
        }
    }

    /// Add an item, returning any overflow
    pub fn add_item(&mut self, item: Item, mut count: u8) -> u8 {
        let max_stack = item.max_stack_size();

        // First try to stack with existing items
        for slot in &mut self.slots {
            if count == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item == item && stack.count < max_stack {
                    let space = max_stack - stack.count;
                    let add = count.min(space);
                    stack.count += add;
                    count -= add;
                }
            }
        }

        // Then fill empty slots
        for slot in &mut self.slots {
            if count == 0 {
                break;
            }
            if slot.is_none() {
                let add = count.min(max_stack);
                *slot = Some(ItemStack::new(item, add));
                count -= add;
            }
        }

        count // Return overflow
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, Option<&ItemStack>)> {
        self.slots.iter().enumerate().map(|(i, s)| (i, s.as_ref()))
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_none())
    }
}

/// A chest's inventory and metadata
#[derive(Clone)]
pub struct ChestData {
    /// Inventory contents (27 slots)
    pub container: Container,
    /// Rotation facing direction (radians)
    pub rotation: f32,
}

impl ChestData {
    pub fn new(rotation: f32) -> Self {
        Self {
            container: Container::new(CHEST_SLOTS),
            rotation,
        }
    }
}

/// Manages all chests in the world
pub struct ChestManager {
    /// Chests indexed by block position
    chests: HashMap<IVec3, ChestData>,
}

impl ChestManager {
    pub fn new() -> Self {
        Self {
            chests: HashMap::new(),
        }
    }

    /// Add a chest at the given position
    pub fn add_chest(&mut self, pos: IVec3, rotation: f32) {
        self.chests.insert(pos, ChestData::new(rotation));
        tracing::debug!("Added chest at {:?} with rotation {}", pos, rotation);
    }

    /// Remove a chest at the given position, returning its contents
    pub fn remove_chest(&mut self, pos: IVec3) -> Option<ChestData> {
        let chest = self.chests.remove(&pos);
        if chest.is_some() {
            tracing::debug!("Removed chest at {:?}", pos);
        }
        chest
    }

    /// Get a chest at the given position
    pub fn get_chest(&self, pos: IVec3) -> Option<&ChestData> {
        self.chests.get(&pos)
    }

    /// Get a mutable reference to a chest at the given position
    pub fn get_chest_mut(&mut self, pos: IVec3) -> Option<&mut ChestData> {
        self.chests.get_mut(&pos)
    }

    /// Check if there's a chest at the given position
    pub fn has_chest(&self, pos: IVec3) -> bool {
        self.chests.contains_key(&pos)
    }

    /// Get all chest positions for rendering
    pub fn positions(&self) -> impl Iterator<Item = (&IVec3, &ChestData)> {
        self.chests.iter()
    }

    /// Number of chests
    pub fn count(&self) -> usize {
        self.chests.len()
    }

    /// Clear all chests
    pub fn clear(&mut self) {
        self.chests.clear();
    }

    /// Get chest data for saving
    pub fn save_data(&self) -> Vec<SavedChest> {
        self.chests
            .iter()
            .map(|(pos, data)| SavedChest {
                position: [pos.x, pos.y, pos.z],
                rotation: data.rotation,
                items: data
                    .container
                    .iter()
                    .filter_map(|(slot, stack)| {
                        stack.map(|s| SavedChestItem {
                            slot: slot as u8,
                            item_id: match s.item {
                                Item::Block(block) => block as u8,
                            },
                            count: s.count,
                        })
                    })
                    .collect(),
            })
            .collect()
    }

    /// Load chests from saved data
    pub fn load_data(&mut self, data: Vec<SavedChest>) {
        self.chests.clear();
        for saved in data {
            let pos = IVec3::new(saved.position[0], saved.position[1], saved.position[2]);
            let mut chest = ChestData::new(saved.rotation);

            for item in saved.items {
                let block = crate::world::Block::from(item.item_id);
                if !matches!(block, crate::world::Block::Air) {
                    chest.container.set_slot(
                        item.slot as usize,
                        Some(ItemStack::new(Item::Block(block), item.count)),
                    );
                }
            }

            self.chests.insert(pos, chest);
        }
        tracing::info!("Loaded {} chests", self.chests.len());
    }
}

impl Default for ChestManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Saved chest data for persistence
#[derive(Clone)]
pub struct SavedChest {
    pub position: [i32; 3],
    pub rotation: f32,
    pub items: Vec<SavedChestItem>,
}

/// Saved item in a chest
#[derive(Clone)]
pub struct SavedChestItem {
    pub slot: u8,
    pub item_id: u8,
    pub count: u8,
}
