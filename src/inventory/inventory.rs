//! Player inventory management
//!
//! Features:
//! - 9-slot hotbar + 27-slot main inventory (36 total)
//! - Click-and-drag item movement
//! - Sorting with locked slot protection
//! - Quick Stack support (deposit matching items to nearby containers)

use super::{Item, ItemStack, ItemTag};
use crate::world::Block;

/// Number of hotbar slots
pub const HOTBAR_SIZE: usize = 9;
/// Number of main inventory slots
pub const MAIN_INVENTORY_SIZE: usize = 27;
/// Total inventory slots (hotbar + main)
pub const TOTAL_SLOTS: usize = HOTBAR_SIZE + MAIN_INVENTORY_SIZE;

/// UI layout constants
pub mod layout {
    /// Slot size in pixels
    pub const SLOT_SIZE: f32 = 48.0;
    /// Padding between slots
    pub const SLOT_PADDING: f32 = 4.0;
    /// Margin from bottom of screen
    pub const MARGIN_BOTTOM: f32 = 20.0;
    /// Gap between hotbar and main inventory
    pub const INVENTORY_GAP: f32 = 12.0;
    /// Border thickness for slots
    pub const BORDER_WIDTH: f32 = 2.0;
    /// Selection highlight thickness
    pub const SELECTION_WIDTH: f32 = 3.0;
}

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
    /// Current mouse position (for held item rendering)
    mouse_pos: (f32, f32),
    /// Hovered slot index
    hovered_slot: Option<usize>,
}

impl Inventory {
    /// Create a new empty inventory
    pub fn new_empty() -> Self {
        Self {
            slots: [None; TOTAL_SLOTS],
            selected_slot: 0,
            is_open: false,
            held_item: None,
            mouse_pos: (0.0, 0.0),
            hovered_slot: None,
        }
    }

    /// Create a new inventory with starter items
    pub fn new() -> Self {
        let mut slots: [Option<ItemStack>; TOTAL_SLOTS] = [None; TOTAL_SLOTS];

        // Initialize hotbar with common blocks for building
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
            mouse_pos: (0.0, 0.0),
            hovered_slot: None,
        }
    }

    // ========== SELECTION ==========

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

    // ========== ITEM MANIPULATION ==========

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
    /// Tries to stack with existing items first, then uses empty slots
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

    /// Try to add a block to inventory, returns true if successful
    pub fn add_block(&mut self, block: Block) -> bool {
        if let Some(item) = Item::from_block(block) {
            self.add_item(item, 1) == 0
        } else {
            false // Block doesn't drop items
        }
    }

    /// Check if inventory has space for an item
    pub fn has_space_for(&self, item: Item, count: u8) -> bool {
        let mut space = 0u16;

        for slot in &self.slots {
            match slot {
                Some(stack) if stack.item == item => {
                    space += stack.space_remaining() as u16;
                }
                None => {
                    space += item.max_stack_size() as u16;
                }
                _ => {}
            }
            if space >= count as u16 {
                return true;
            }
        }

        false
    }

    /// Check if inventory is full (no empty slots and no stackable space)
    pub fn is_full(&self) -> bool {
        self.slots.iter().all(|slot| {
            match slot {
                Some(stack) => stack.is_full(),
                None => false,
            }
        })
    }

    // ========== SLOT ACCESS ==========

    /// Get hotbar slots (0-8) for rendering
    pub fn hotbar_slots(&self) -> &[Option<ItemStack>] {
        &self.slots[..HOTBAR_SIZE]
    }

    /// Get all slots for inventory grid rendering
    pub fn all_slots(&self) -> &[Option<ItemStack>] {
        &self.slots
    }

    /// Get a slot by index
    pub fn get_slot(&self, index: usize) -> Option<&ItemStack> {
        self.slots.get(index).and_then(|s| s.as_ref())
    }

    /// Get a mutable reference to a slot
    pub fn slot_mut(&mut self, index: usize) -> Option<&mut Option<ItemStack>> {
        self.slots.get_mut(index)
    }

    // ========== DRAG AND DROP ==========

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

    /// Try to merge held item into a slot (for stacking)
    /// Returns true if any items were transferred
    pub fn try_merge_into_slot(&mut self, slot_index: usize) -> bool {
        if slot_index >= TOTAL_SLOTS {
            return false;
        }

        let held = match &mut self.held_item {
            Some(h) => h,
            None => return false,
        };

        if let Some(slot) = &mut self.slots[slot_index] {
            if slot.item == held.item {
                let overflow = slot.try_add(held.count);
                if overflow < held.count {
                    held.count = overflow;
                    if held.count == 0 {
                        self.held_item = None;
                    }
                    return true;
                }
            }
        }
        false
    }

    /// Place a single item from held stack into slot
    /// Used for right-click placement
    pub fn place_one_in_slot(&mut self, slot_index: usize) -> bool {
        if slot_index >= TOTAL_SLOTS {
            return false;
        }

        let held = match &mut self.held_item {
            Some(h) if h.count > 0 => h,
            _ => return false,
        };

        match &mut self.slots[slot_index] {
            None => {
                // Empty slot: place one item
                self.slots[slot_index] = Some(ItemStack::new(held.item, 1));
                held.count -= 1;
                if held.count == 0 {
                    self.held_item = None;
                }
                true
            }
            Some(slot) if slot.item == held.item && !slot.is_full() => {
                // Same item, has space: add one
                slot.try_add(1);
                held.count -= 1;
                if held.count == 0 {
                    self.held_item = None;
                }
                true
            }
            _ => false,
        }
    }

    // ========== UI STATE ==========

    /// Update mouse position
    pub fn update_mouse(&mut self, x: f32, y: f32) {
        self.mouse_pos = (x, y);
    }

    /// Get mouse position
    pub fn mouse_pos(&self) -> (f32, f32) {
        self.mouse_pos
    }

    /// Toggle inventory open/closed
    pub fn toggle(&mut self) {
        self.is_open = !self.is_open;
        // Drop held item back into inventory when closing
        if !self.is_open {
            if let Some(held) = self.held_item.take() {
                self.add_item(held.item, held.count);
            }
            self.hovered_slot = None;
        }
    }

    /// Handle a left click at screen position when inventory is open
    /// Returns true if a slot was clicked
    pub fn handle_click(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.is_open {
            return false;
        }

        if let Some(slot_index) = self.get_slot_at(mouse_x, mouse_y, screen_width, screen_height) {
            // Try to merge first (for stacking), otherwise swap
            if !self.try_merge_into_slot(slot_index) {
                self.swap_with_slot(slot_index);
            }
            true
        } else {
            false
        }
    }

    /// Handle a right click at screen position when inventory is open
    /// Places one item from held stack
    pub fn handle_right_click(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.is_open {
            return false;
        }

        if let Some(slot_index) = self.get_slot_at(mouse_x, mouse_y, screen_width, screen_height) {
            if self.held_item.is_some() {
                self.place_one_in_slot(slot_index)
            } else {
                // Pick up half the stack
                if let Some(stack) = &mut self.slots[slot_index] {
                    let half = (stack.count + 1) / 2;
                    let taken = stack.try_remove(half);
                    if taken > 0 {
                        self.held_item = Some(ItemStack::new(stack.item, taken));
                        if stack.count == 0 {
                            self.slots[slot_index] = None;
                        }
                        return true;
                    }
                }
                false
            }
        } else {
            false
        }
    }

    /// Get the slot index at a screen position, if any
    fn get_slot_at(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> Option<usize> {
        use layout::*;

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

        // Check main inventory slots (9-35) - only when inventory is open
        if self.is_open {
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
        }

        None
    }

    /// Update hovered slot based on mouse position
    pub fn update_hover(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) {
        self.hovered_slot = self.get_slot_at(mouse_x, mouse_y, screen_width, screen_height);
    }

    /// Get slot index at mouse position for hover highlighting
    pub fn hovered_slot(&self) -> Option<usize> {
        self.hovered_slot
    }

    // ========== SORTING ==========

    /// Sort inventory (respects locked slots)
    /// Groups by category, then rarity, then stacks same items
    pub fn sort(&mut self) {
        // Collect all non-locked items
        let mut items: Vec<ItemStack> = Vec::new();
        let mut locked_slots: Vec<(usize, ItemStack)> = Vec::new();

        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(stack) = slot.take() {
                if stack.locked {
                    locked_slots.push((i, stack));
                } else {
                    items.push(stack);
                }
            }
        }

        // Merge stackable items
        let mut merged: Vec<ItemStack> = Vec::new();
        for item in items {
            let mut remaining = item.count;
            let item_type = item.item;

            // Try to add to existing stacks
            for existing in &mut merged {
                if remaining == 0 {
                    break;
                }
                if existing.item == item_type {
                    remaining = existing.try_add(remaining);
                }
            }

            // Create new stacks for remainder
            while remaining > 0 {
                let to_add = remaining.min(item_type.max_stack_size());
                merged.push(ItemStack::new(item_type, to_add));
                remaining -= to_add;
            }
        }

        // Sort by priority
        merged.sort_by_key(|stack| stack.item.sort_priority());

        // Place sorted items back
        let mut item_iter = merged.into_iter();
        for slot in &mut self.slots {
            if slot.is_none() {
                *slot = item_iter.next();
            }
        }

        // Restore locked slots
        for (i, stack) in locked_slots {
            // If slot is now occupied, swap
            if let Some(existing) = self.slots[i].take() {
                // Find an empty slot for displaced item
                for slot in &mut self.slots {
                    if slot.is_none() {
                        *slot = Some(existing);
                        break;
                    }
                }
            }
            self.slots[i] = Some(stack);
        }
    }

    /// Toggle lock on a slot
    pub fn toggle_slot_lock(&mut self, slot_index: usize) {
        if let Some(Some(stack)) = self.slots.get_mut(slot_index) {
            stack.toggle_lock();
        }
    }

    // ========== QUICK STACK (for future container integration) ==========

    /// Get items that could be quick-stacked (items that exist in target container)
    /// Returns list of (item, count) that would be deposited
    pub fn get_quick_stack_items(&self, container_has: impl Fn(&Item) -> bool) -> Vec<(Item, u8)> {
        let mut result = Vec::new();

        for slot in &self.slots {
            if let Some(stack) = slot {
                // Skip locked slots
                if stack.locked {
                    continue;
                }
                // Only stack items that already exist in container
                if container_has(&stack.item) {
                    result.push((stack.item, stack.count));
                }
            }
        }

        result
    }

    /// Remove items for quick stack (call after container confirms it can receive)
    pub fn remove_for_quick_stack(&mut self, items: &[(Item, u8)]) {
        for (item, count) in items {
            let mut to_remove = *count;

            for slot in &mut self.slots {
                if to_remove == 0 {
                    break;
                }
                if let Some(stack) = slot {
                    if !stack.locked && stack.item == *item {
                        let removed = stack.try_remove(to_remove);
                        to_remove -= removed;
                        if stack.is_empty() {
                            *slot = None;
                        }
                    }
                }
            }
        }
    }

    /// Get items matching a tag filter
    pub fn get_items_with_tag(&self, tag: ItemTag) -> Vec<(usize, &ItemStack)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref().filter(|s| s.item.has_tag(tag)).map(|s| (i, s))
            })
            .collect()
    }

    /// Count total items of a specific type
    pub fn count_item(&self, item: Item) -> u32 {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.item == item)
            .map(|s| s.count as u32)
            .sum()
    }
}

impl Default for Inventory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_creation() {
        let inv = Inventory::new();
        assert_eq!(inv.selected_index(), 0);
        assert!(!inv.is_open);

        // Should have starter items in hotbar
        assert!(inv.hotbar_slots()[0].is_some());
        assert!(inv.hotbar_slots()[7].is_some()); // Torch
    }

    #[test]
    fn test_inventory_slot_selection() {
        let mut inv = Inventory::new();
        assert_eq!(inv.selected_index(), 0);

        inv.select_slot(5);
        assert_eq!(inv.selected_index(), 5);

        // Invalid slot should be ignored
        inv.select_slot(20);
        assert_eq!(inv.selected_index(), 5);
    }

    #[test]
    fn test_inventory_slot_cycling() {
        let mut inv = Inventory::new();

        // Cycle forward
        inv.next_slot();
        assert_eq!(inv.selected_index(), 1);

        // Cycle backward
        inv.prev_slot();
        assert_eq!(inv.selected_index(), 0);

        // Wrap around backward
        inv.prev_slot();
        assert_eq!(inv.selected_index(), HOTBAR_SIZE - 1);

        // Wrap around forward
        inv.select_slot(HOTBAR_SIZE - 1);
        inv.next_slot();
        assert_eq!(inv.selected_index(), 0);
    }

    #[test]
    fn test_inventory_selected_item() {
        let inv = Inventory::new();
        let selected = inv.selected_item();
        assert!(selected.is_some());

        let block = inv.selected_block();
        assert!(block.is_some());
    }

    #[test]
    fn test_inventory_consume_selected() {
        let mut inv = Inventory::new();

        // Get initial count
        let initial_count = inv.selected_item().unwrap().count;

        // Consume one
        assert!(inv.consume_selected());
        assert_eq!(inv.selected_item().unwrap().count, initial_count - 1);
    }

    #[test]
    fn test_inventory_consume_clears_empty_slot() {
        let mut inv = Inventory::new();

        // Set slot to single item
        inv.slots[0] = Some(ItemStack::new(Item::Block(Block::Stone), 1));

        // Consume should clear the slot
        inv.consume_selected();
        assert!(inv.slots[0].is_none());
    }

    #[test]
    fn test_inventory_add_item_stacks() {
        let mut inv = Inventory::new();

        // Clear inventory first
        for slot in &mut inv.slots {
            *slot = None;
        }

        // Add stone
        let overflow = inv.add_item(Item::Block(Block::Stone), 32);
        assert_eq!(overflow, 0);
        assert!(inv.slots[0].is_some());
        assert_eq!(inv.slots[0].as_ref().unwrap().count, 32);

        // Add more stone - should stack
        let overflow = inv.add_item(Item::Block(Block::Stone), 20);
        assert_eq!(overflow, 0);
        assert_eq!(inv.slots[0].as_ref().unwrap().count, 52);
    }

    #[test]
    fn test_inventory_add_item_overflow_to_new_slot() {
        let mut inv = Inventory::new();

        // Clear inventory
        for slot in &mut inv.slots {
            *slot = None;
        }

        // Add more than one stack
        let overflow = inv.add_item(Item::Block(Block::Stone), 100);
        assert_eq!(overflow, 0);

        // Should have 64 in first slot, 36 in second
        assert_eq!(inv.slots[0].as_ref().unwrap().count, 64);
        assert_eq!(inv.slots[1].as_ref().unwrap().count, 36);
    }

    #[test]
    fn test_inventory_add_item_returns_overflow() {
        let mut inv = Inventory::new();

        // Fill all slots with different items
        for (i, slot) in inv.slots.iter_mut().enumerate() {
            *slot = Some(ItemStack::new(
                Item::Block(if i % 2 == 0 { Block::Stone } else { Block::Dirt }),
                64,
            ));
        }

        // Try to add grass - should overflow
        let overflow = inv.add_item(Item::Block(Block::Grass), 10);
        assert_eq!(overflow, 10);
    }

    #[test]
    fn test_inventory_swap_with_slot() {
        let mut inv = Inventory::new();

        // Clear and set up
        for slot in &mut inv.slots {
            *slot = None;
        }
        inv.slots[0] = Some(ItemStack::new(Item::Block(Block::Stone), 10));

        // Pick up item (swap empty held with slot 0)
        inv.swap_with_slot(0);
        assert!(inv.slots[0].is_none());
        assert!(inv.held_item().is_some());
        assert_eq!(inv.held_item().unwrap().count, 10);

        // Put down in different slot
        inv.swap_with_slot(5);
        assert!(inv.slots[5].is_some());
        assert!(inv.held_item().is_none());
    }

    #[test]
    fn test_inventory_toggle() {
        let mut inv = Inventory::new();

        assert!(!inv.is_open);
        inv.toggle();
        assert!(inv.is_open);
        inv.toggle();
        assert!(!inv.is_open);
    }

    #[test]
    fn test_inventory_toggle_returns_held_item() {
        let mut inv = Inventory::new();

        // Open inventory
        inv.toggle();

        // Pick up item
        inv.set_held_item(Some(ItemStack::new(Item::Block(Block::Stone), 5)));

        // Close inventory - held item should be returned
        inv.toggle();
        assert!(inv.held_item().is_none());
        // Item should be back in inventory (stacked or new slot)
    }

    #[test]
    fn test_inventory_hotbar_slots() {
        let inv = Inventory::new();
        let hotbar = inv.hotbar_slots();
        assert_eq!(hotbar.len(), HOTBAR_SIZE);
    }

    #[test]
    fn test_inventory_all_slots() {
        let inv = Inventory::new();
        let all = inv.all_slots();
        assert_eq!(all.len(), TOTAL_SLOTS);
    }

    #[test]
    fn test_inventory_constants() {
        assert_eq!(HOTBAR_SIZE, 9);
        assert_eq!(MAIN_INVENTORY_SIZE, 27);
        assert_eq!(TOTAL_SLOTS, 36);
    }

    #[test]
    fn test_inventory_add_block() {
        let mut inv = Inventory::new_empty();

        // Add a block
        assert!(inv.add_block(Block::Stone));
        assert_eq!(inv.count_item(Item::Block(Block::Stone)), 1);

        // Non-droppable blocks return false
        assert!(!inv.add_block(Block::Air));
        assert!(!inv.add_block(Block::Water));
    }

    #[test]
    fn test_inventory_has_space() {
        let mut inv = Inventory::new_empty();

        // Empty inventory has space
        assert!(inv.has_space_for(Item::Block(Block::Stone), 64));

        // Fill with stone
        for slot in &mut inv.slots {
            *slot = Some(ItemStack::new(Item::Block(Block::Stone), 64));
        }

        // Full inventory - no space for different item
        assert!(!inv.has_space_for(Item::Block(Block::Dirt), 1));

        // But still has "space" for stone (though stacks are full)
        assert!(!inv.has_space_for(Item::Block(Block::Stone), 1));
    }

    #[test]
    fn test_inventory_sorting() {
        let mut inv = Inventory::new_empty();

        // Add items in random order
        inv.slots[5] = Some(ItemStack::new(Item::Block(Block::Stone), 32));
        inv.slots[2] = Some(ItemStack::new(Item::Block(Block::DiamondOre), 5));
        inv.slots[8] = Some(ItemStack::new(Item::Block(Block::Stone), 20));
        inv.slots[0] = Some(ItemStack::new(Item::Block(Block::Dirt), 10));

        // Sort
        inv.sort();

        // Diamond ore should be first (ore + legendary)
        // Stones should be merged (52 total)
        let mut found_diamond = false;
        let mut total_stone = 0u32;

        for slot in &inv.slots {
            if let Some(stack) = slot {
                if stack.item == Item::Block(Block::DiamondOre) && !found_diamond {
                    found_diamond = true;
                }
                if stack.item == Item::Block(Block::Stone) {
                    total_stone += stack.count as u32;
                }
            }
        }

        assert!(found_diamond);
        assert_eq!(total_stone, 52); // 32 + 20 merged
    }

    #[test]
    fn test_inventory_locked_slots_survive_sort() {
        let mut inv = Inventory::new_empty();

        // Add a locked item at slot 3
        inv.slots[3] = Some(ItemStack::new_locked(Item::Block(Block::Torch), 16));
        // Add some other items
        inv.slots[0] = Some(ItemStack::new(Item::Block(Block::Stone), 32));
        inv.slots[7] = Some(ItemStack::new(Item::Block(Block::Dirt), 20));

        // Sort
        inv.sort();

        // Torch should still be at slot 3 and still locked
        let torch_slot = inv.slots[3].as_ref().unwrap();
        assert_eq!(torch_slot.item, Item::Block(Block::Torch));
        assert!(torch_slot.locked);
    }
}
