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
