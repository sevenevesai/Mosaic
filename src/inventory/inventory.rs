//! Player inventory management

use super::{Item, ItemStack};
use crate::world::Block;

/// Number of hotbar slots
pub const HOTBAR_SIZE: usize = 9;
/// Number of main inventory slots
pub const MAIN_INVENTORY_SIZE: usize = 27;
/// Total inventory slots (hotbar + main)
pub const TOTAL_SLOTS: usize = HOTBAR_SIZE + MAIN_INVENTORY_SIZE;

/// Player inventory with hotbar and main storage
/// Layout: slots 0-8 = hotbar, slots 9-35 = main inventory
pub struct Inventory {
    /// All inventory slots
    slots: [Option<ItemStack>; TOTAL_SLOTS],
    /// Currently selected hotbar slot (0-8)
    selected_slot: usize,
    /// Whether the inventory UI is open
    pub is_open: bool,
    /// Item held by cursor during drag operations
    held_item: Option<ItemStack>,
}

impl Inventory {
    /// Create a new inventory with starter items
    pub fn new() -> Self {
        let mut slots: [Option<ItemStack>; TOTAL_SLOTS] = [None; TOTAL_SLOTS];

        // Initialize hotbar with common blocks
        slots[0] = Some(ItemStack::new(Item::Block(Block::Stone), 64));
        slots[1] = Some(ItemStack::new(Item::Block(Block::Dirt), 64));
        slots[2] = Some(ItemStack::new(Item::Block(Block::Grass), 64));
        slots[3] = Some(ItemStack::new(Item::Block(Block::Brick), 64));
        slots[4] = Some(ItemStack::new(Item::Block(Block::Wood), 64));
        slots[5] = Some(ItemStack::new(Item::Block(Block::OakLog), 64));
        slots[6] = Some(ItemStack::new(Item::Block(Block::Leaves), 64));
        slots[7] = Some(ItemStack::new(Item::Block(Block::Torch), 64));
        slots[8] = Some(ItemStack::new(Item::Block(Block::Sand), 64));

        Self {
            slots,
            selected_slot: 0,
            is_open: false,
            held_item: None,
        }
    }

    /// Get the currently selected hotbar slot index
    pub fn selected_index(&self) -> usize {
        self.selected_slot
    }

    /// Get the currently selected item stack
    pub fn selected_item(&self) -> Option<&ItemStack> {
        self.slots[self.selected_slot].as_ref()
    }

    /// Get the block to place (if any)
    pub fn selected_block(&self) -> Option<Block> {
        self.selected_item().and_then(|stack| stack.item.as_block())
    }

    /// Select a hotbar slot (0-8)
    pub fn select_slot(&mut self, slot: usize) {
        if slot < HOTBAR_SIZE {
            self.selected_slot = slot;
        }
    }

    /// Cycle to next hotbar slot
    pub fn next_slot(&mut self) {
        self.selected_slot = (self.selected_slot + 1) % HOTBAR_SIZE;
    }

    /// Cycle to previous hotbar slot
    pub fn prev_slot(&mut self) {
        self.selected_slot = (self.selected_slot + HOTBAR_SIZE - 1) % HOTBAR_SIZE;
    }

    /// Consume one of the selected item (after placing block)
    /// Returns true if item was consumed
    pub fn consume_selected(&mut self) -> bool {
        if let Some(stack) = &mut self.slots[self.selected_slot] {
            stack.count = stack.count.saturating_sub(1);
            if stack.count == 0 {
                self.slots[self.selected_slot] = None;
            }
            true
        } else {
            false
        }
    }

    /// Add an item to inventory, returns overflow count
    pub fn add_item(&mut self, item: Item, count: u8) -> u8 {
        let mut remaining = count;

        // First try to stack with existing same-item stacks
        for slot in &mut self.slots {
            if remaining == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item == item {
                    remaining = stack.try_add(remaining);
                }
            }
        }

        // Then try empty slots
        for slot in &mut self.slots {
            if remaining == 0 {
                break;
            }
            if slot.is_none() {
                let to_add = remaining.min(item.max_stack_size());
                *slot = Some(ItemStack::new(item, to_add));
                remaining -= to_add;
            }
        }

        remaining
    }

    /// Get hotbar slots (0-8) for rendering
    pub fn hotbar_slots(&self) -> &[Option<ItemStack>] {
        &self.slots[..HOTBAR_SIZE]
    }

    /// Get all slots for inventory grid rendering
    pub fn all_slots(&self) -> &[Option<ItemStack>] {
        &self.slots
    }

    /// Get a mutable reference to a slot
    pub fn slot_mut(&mut self, index: usize) -> Option<&mut Option<ItemStack>> {
        self.slots.get_mut(index)
    }

    /// Get the held item (for cursor dragging)
    pub fn held_item(&self) -> Option<&ItemStack> {
        self.held_item.as_ref()
    }

    /// Set the held item
    pub fn set_held_item(&mut self, item: Option<ItemStack>) {
        self.held_item = item;
    }

    /// Swap held item with a slot
    pub fn swap_with_slot(&mut self, slot_index: usize) {
        if slot_index < TOTAL_SLOTS {
            std::mem::swap(&mut self.held_item, &mut self.slots[slot_index]);
        }
    }

    /// Toggle inventory open/closed
    pub fn toggle(&mut self) {
        self.is_open = !self.is_open;
        // Drop held item back into inventory when closing
        if !self.is_open {
            if let Some(held) = self.held_item.take() {
                self.add_item(held.item, held.count);
            }
        }
    }

    /// Handle a click at screen position when inventory is open
    /// Returns true if a slot was clicked
    pub fn handle_click(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.is_open {
            return false;
        }

        if let Some(slot_index) = self.get_slot_at(mouse_x, mouse_y, screen_width, screen_height) {
            self.swap_with_slot(slot_index);
            true
        } else {
            false
        }
    }

    /// Get the slot index at a screen position, if any
    fn get_slot_at(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> Option<usize> {
        const SLOT_SIZE: f32 = 40.0;
        const SLOT_PADDING: f32 = 4.0;
        const MARGIN_BOTTOM: f32 = 20.0;
        const INVENTORY_GAP: f32 = 8.0;

        let grid_width = (HOTBAR_SIZE as f32 * SLOT_SIZE) + ((HOTBAR_SIZE - 1) as f32 * SLOT_PADDING);
        let start_x = (screen_width - grid_width) / 2.0;
        let hotbar_y = screen_height - MARGIN_BOTTOM - SLOT_SIZE;

        // Check hotbar slots (0-8)
        for i in 0..HOTBAR_SIZE {
            let slot_x = start_x + (i as f32 * (SLOT_SIZE + SLOT_PADDING));
            let slot_y = hotbar_y;

            if mouse_x >= slot_x && mouse_x < slot_x + SLOT_SIZE &&
               mouse_y >= slot_y && mouse_y < slot_y + SLOT_SIZE {
                return Some(i);
            }
        }

        // Check main inventory slots (9-35)
        let main_inv_start_y = hotbar_y - INVENTORY_GAP - (3.0 * (SLOT_SIZE + SLOT_PADDING));

        for row in 0..3 {
            for col in 0..9 {
                let slot_index = 9 + row * 9 + col;
                let slot_x = start_x + (col as f32 * (SLOT_SIZE + SLOT_PADDING));
                let slot_y = main_inv_start_y + ((2 - row) as f32 * (SLOT_SIZE + SLOT_PADDING));

                if mouse_x >= slot_x && mouse_x < slot_x + SLOT_SIZE &&
                   mouse_y >= slot_y && mouse_y < slot_y + SLOT_SIZE {
                    return Some(slot_index);
                }
            }
        }

        None
    }

    /// Get slot index at mouse position for hover highlighting
    pub fn hovered_slot(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> Option<usize> {
        if !self.is_open {
            return None;
        }
        self.get_slot_at(mouse_x, mouse_y, screen_width, screen_height)
    }
}

impl Default for Inventory {
    fn default() -> Self {
        Self::new()
    }
}
