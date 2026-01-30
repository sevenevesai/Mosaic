//! Item types and item stacks

use crate::world::Block;

/// Items that can exist in inventory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Item {
    /// A placeable block
    Block(Block),
}

impl Item {
    /// Maximum stack size for this item type
    pub fn max_stack_size(&self) -> u8 {
        match self {
            Item::Block(_) => 64,
        }
    }

    /// Get texture index for rendering in UI
    /// Uses the front face texture for block items
    pub fn texture_index(&self) -> u32 {
        match self {
            Item::Block(block) => block.texture_indices()[2], // Front face
        }
    }

    /// Get the block this item places (if applicable)
    pub fn as_block(&self) -> Option<Block> {
        match self {
            Item::Block(block) => Some(*block),
        }
    }

    /// Create an item from a block (for drops)
    /// Returns None for blocks that shouldn't drop items
    pub fn from_block(block: Block) -> Option<Self> {
        match block {
            Block::Air | Block::Water | Block::Bedrock => None,
            _ => Some(Item::Block(block)),
        }
    }
}

/// A stack of items with count
#[derive(Debug, Clone, Copy)]
pub struct ItemStack {
    pub item: Item,
    pub count: u8,
}

impl ItemStack {
    /// Create a new item stack
    pub fn new(item: Item, count: u8) -> Self {
        Self {
            item,
            count: count.min(item.max_stack_size()),
        }
    }

    /// Try to add items to this stack, returns overflow count
    pub fn try_add(&mut self, amount: u8) -> u8 {
        let max = self.item.max_stack_size();
        let space = max.saturating_sub(self.count);
        let to_add = amount.min(space);
        self.count += to_add;
        amount - to_add
    }

    /// Try to remove items from this stack, returns actual removed
    pub fn try_remove(&mut self, amount: u8) -> u8 {
        let to_remove = amount.min(self.count);
        self.count -= to_remove;
        to_remove
    }

    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_max_stack_size() {
        let stone = Item::Block(Block::Stone);
        assert_eq!(stone.max_stack_size(), 64);

        let torch = Item::Block(Block::Torch);
        assert_eq!(torch.max_stack_size(), 64);
    }

    #[test]
    fn test_item_as_block() {
        let stone = Item::Block(Block::Stone);
        assert_eq!(stone.as_block(), Some(Block::Stone));

        let torch = Item::Block(Block::Torch);
        assert_eq!(torch.as_block(), Some(Block::Torch));
    }

    #[test]
    fn test_item_from_block() {
        // Normal blocks create items
        assert!(Item::from_block(Block::Stone).is_some());
        assert!(Item::from_block(Block::Dirt).is_some());
        assert!(Item::from_block(Block::Torch).is_some());

        // Special blocks don't create items
        assert!(Item::from_block(Block::Air).is_none());
        assert!(Item::from_block(Block::Water).is_none());
        assert!(Item::from_block(Block::Bedrock).is_none());
    }

    #[test]
    fn test_item_texture_index() {
        let stone = Item::Block(Block::Stone);
        // Should return front face texture (index 2 in texture_indices)
        let idx = stone.texture_index();
        assert!(idx < 100); // Reasonable texture index
    }

    #[test]
    fn test_item_stack_creation() {
        let stack = ItemStack::new(Item::Block(Block::Stone), 32);
        assert_eq!(stack.count, 32);
        assert!(!stack.is_empty());
    }

    #[test]
    fn test_item_stack_clamps_to_max() {
        let stack = ItemStack::new(Item::Block(Block::Stone), 100);
        assert_eq!(stack.count, 64); // Clamped to max stack size
    }

    #[test]
    fn test_item_stack_try_add() {
        let mut stack = ItemStack::new(Item::Block(Block::Stone), 60);

        // Add within limit
        let overflow = stack.try_add(2);
        assert_eq!(overflow, 0);
        assert_eq!(stack.count, 62);

        // Add with overflow
        let overflow = stack.try_add(10);
        assert_eq!(overflow, 8); // 64 - 62 = 2 can fit, 10 - 2 = 8 overflow
        assert_eq!(stack.count, 64);
    }

    #[test]
    fn test_item_stack_try_remove() {
        let mut stack = ItemStack::new(Item::Block(Block::Stone), 10);

        // Remove within count
        let removed = stack.try_remove(3);
        assert_eq!(removed, 3);
        assert_eq!(stack.count, 7);

        // Remove more than available
        let removed = stack.try_remove(20);
        assert_eq!(removed, 7); // Only 7 available
        assert_eq!(stack.count, 0);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_item_stack_is_empty() {
        let mut stack = ItemStack::new(Item::Block(Block::Stone), 1);
        assert!(!stack.is_empty());

        stack.try_remove(1);
        assert!(stack.is_empty());
    }
}
