//! Item types, stacks, and categorization for inventory system
//!
//! Designed to support:
//! - Quick Stack (Terraria-style proximity sorting)
//! - Slot locking (prevent sorting of specific items)
//! - Item tags for filtering and categorization
//! - Future RPG/city-sim mechanics (tools, weapons, resources)

use crate::world::Block;
use std::collections::HashSet;

/// Item category tags for sorting, filtering, and Quick Stack
/// Items can have multiple tags for flexible categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemTag {
    // === Material Categories ===
    /// Raw materials that can be processed
    RawMaterial,
    /// Processed/refined materials
    ProcessedMaterial,
    /// Building blocks for construction
    BuildingBlock,

    // === Resource Types ===
    /// Ore blocks that can be smelted
    Ore,
    /// Wood and wood-derived items
    Wood,
    /// Stone and stone-derived items
    Stone,
    /// Soil types (dirt, sand, gravel)
    Soil,
    /// Plant/organic materials
    Organic,

    // === Usage Categories ===
    /// Can be used as fuel
    Combustible,
    /// Tools (pickaxes, axes, shovels)
    Tool,
    /// Weapons (swords, bows)
    Weapon,
    /// Armor and wearables
    Armor,
    /// Food and consumables
    Consumable,

    // === Special Properties ===
    /// Emits light when placed
    LightSource,
    /// Decorative items
    Decoration,
    /// Rare/valuable items
    Valuable,
    /// Quest-related items
    Quest,

    // === City-Sim Categories (future) ===
    /// Materials for roads/infrastructure
    Infrastructure,
    /// Items for NPC/villager use
    VillagerResource,
    /// Currency/trade items
    Currency,
}

/// Rarity tiers for items (affects sorting, display, loot tables)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ItemRarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

impl ItemRarity {
    /// Get color for UI display (RGBA as u32)
    pub fn color(&self) -> [f32; 4] {
        match self {
            ItemRarity::Common => [1.0, 1.0, 1.0, 1.0],      // White
            ItemRarity::Uncommon => [0.3, 0.9, 0.3, 1.0],   // Green
            ItemRarity::Rare => [0.3, 0.5, 1.0, 1.0],       // Blue
            ItemRarity::Epic => [0.7, 0.3, 0.9, 1.0],       // Purple
            ItemRarity::Legendary => [1.0, 0.7, 0.2, 1.0],  // Orange/Gold
        }
    }
}

/// Items that can exist in inventory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Item {
    /// A placeable block
    Block(Block),
    // Future item types:
    // Tool { tool_type: ToolType, material: Material, durability: u16 },
    // Weapon { weapon_type: WeaponType, damage: u16, durability: u16 },
    // Consumable { effect: ConsumableEffect, quantity: u8 },
}

impl Item {
    /// Maximum stack size for this item type
    pub fn max_stack_size(&self) -> u8 {
        match self {
            Item::Block(block) => {
                // Tools and special items don't stack as high
                if block.is_tool_block() {
                    1
                } else {
                    64
                }
            }
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

    /// Get tags for this item (used for sorting/filtering)
    pub fn tags(&self) -> HashSet<ItemTag> {
        let mut tags = HashSet::new();

        match self {
            Item::Block(block) => {
                // Add appropriate tags based on block type
                match block {
                    // Ores
                    Block::CoalOre => {
                        tags.insert(ItemTag::Ore);
                        tags.insert(ItemTag::Combustible);
                        tags.insert(ItemTag::RawMaterial);
                    }
                    Block::IronOre | Block::CopperOre | Block::TinOre |
                    Block::NickelOre | Block::LeadOre => {
                        tags.insert(ItemTag::Ore);
                        tags.insert(ItemTag::RawMaterial);
                    }
                    Block::GoldOre | Block::SilverOre => {
                        tags.insert(ItemTag::Ore);
                        tags.insert(ItemTag::RawMaterial);
                        tags.insert(ItemTag::Valuable);
                    }
                    Block::DiamondOre | Block::EmeraldOre | Block::SapphireOre |
                    Block::AmethystOre | Block::TitaniumOre | Block::UraniumOre => {
                        tags.insert(ItemTag::Ore);
                        tags.insert(ItemTag::RawMaterial);
                        tags.insert(ItemTag::Valuable);
                    }
                    Block::QuartzOre => {
                        tags.insert(ItemTag::Ore);
                        tags.insert(ItemTag::RawMaterial);
                    }
                    Block::RedstoneOre => {
                        tags.insert(ItemTag::Ore);
                        tags.insert(ItemTag::RawMaterial);
                    }

                    // Wood types
                    Block::Wood | Block::OakLog | Block::BirchLog |
                    Block::PineLog | Block::JungleLog => {
                        tags.insert(ItemTag::Wood);
                        tags.insert(ItemTag::Combustible);
                        tags.insert(ItemTag::BuildingBlock);
                    }

                    // Stone types
                    Block::Stone | Block::StoneDirt | Block::StoneRubble2 |
                    Block::MossyStone | Block::CrackedStone | Block::Gravel => {
                        tags.insert(ItemTag::Stone);
                        tags.insert(ItemTag::BuildingBlock);
                    }
                    Block::Sandstone | Block::RedSandstone | Block::ClayStone => {
                        tags.insert(ItemTag::Stone);
                        tags.insert(ItemTag::BuildingBlock);
                    }

                    // Soil types
                    Block::Dirt | Block::Grass | Block::SnowGrass |
                    Block::JungleGrass | Block::MudBlock | Block::Peat => {
                        tags.insert(ItemTag::Soil);
                        tags.insert(ItemTag::BuildingBlock);
                    }
                    Block::Sand => {
                        tags.insert(ItemTag::Soil);
                        tags.insert(ItemTag::BuildingBlock);
                    }

                    // Light sources
                    Block::Torch => {
                        tags.insert(ItemTag::LightSource);
                        tags.insert(ItemTag::Combustible);
                    }
                    Block::CrystalAmethyst | Block::CrystalQuartz => {
                        tags.insert(ItemTag::LightSource);
                        tags.insert(ItemTag::Decoration);
                        tags.insert(ItemTag::Valuable);
                    }

                    // Organic/plants
                    Block::Leaves | Block::JungleLeaves | Block::SpruceLeaves |
                    Block::Vines | Block::DeadBush | Block::Cactus => {
                        tags.insert(ItemTag::Organic);
                    }

                    // Building materials
                    Block::Brick => {
                        tags.insert(ItemTag::BuildingBlock);
                        tags.insert(ItemTag::ProcessedMaterial);
                        tags.insert(ItemTag::Infrastructure);
                    }

                    // Ice/Snow
                    Block::Snow | Block::Ice | Block::PackedIce => {
                        tags.insert(ItemTag::BuildingBlock);
                    }

                    // Cave decorations
                    Block::Stalactite | Block::Stalagmite => {
                        tags.insert(ItemTag::Decoration);
                        tags.insert(ItemTag::Stone);
                    }

                    // Default: building block
                    _ => {
                        tags.insert(ItemTag::BuildingBlock);
                    }
                }
            }
        }

        tags
    }

    /// Check if item has a specific tag
    pub fn has_tag(&self, tag: ItemTag) -> bool {
        self.tags().contains(&tag)
    }

    /// Get item rarity
    pub fn rarity(&self) -> ItemRarity {
        match self {
            Item::Block(block) => {
                match block {
                    // Legendary
                    Block::DiamondOre | Block::TitaniumOre | Block::UraniumOre => {
                        ItemRarity::Legendary
                    }
                    // Epic
                    Block::EmeraldOre | Block::SapphireOre => {
                        ItemRarity::Epic
                    }
                    // Rare
                    Block::GoldOre | Block::SilverOre | Block::AmethystOre |
                    Block::CrystalAmethyst | Block::CrystalQuartz => {
                        ItemRarity::Rare
                    }
                    // Uncommon
                    Block::IronOre | Block::CopperOre | Block::RedstoneOre |
                    Block::QuartzOre | Block::NickelOre | Block::LeadOre => {
                        ItemRarity::Uncommon
                    }
                    // Common
                    _ => ItemRarity::Common,
                }
            }
        }
    }

    /// Get display name for this item
    pub fn display_name(&self) -> &'static str {
        match self {
            Item::Block(block) => block.display_name(),
        }
    }

    /// Get sort priority (lower = sorted first)
    /// Groups items by category, then rarity, then alphabetically
    pub fn sort_priority(&self) -> u32 {
        let category_base = if self.has_tag(ItemTag::Tool) {
            0
        } else if self.has_tag(ItemTag::Weapon) {
            1000
        } else if self.has_tag(ItemTag::Armor) {
            2000
        } else if self.has_tag(ItemTag::Ore) {
            3000
        } else if self.has_tag(ItemTag::RawMaterial) {
            4000
        } else if self.has_tag(ItemTag::ProcessedMaterial) {
            5000
        } else if self.has_tag(ItemTag::BuildingBlock) {
            6000
        } else if self.has_tag(ItemTag::Consumable) {
            7000
        } else {
            8000
        };

        // Add rarity offset (higher rarity = lower number = sorted first)
        let rarity_offset = match self.rarity() {
            ItemRarity::Legendary => 0,
            ItemRarity::Epic => 100,
            ItemRarity::Rare => 200,
            ItemRarity::Uncommon => 300,
            ItemRarity::Common => 400,
        };

        category_base + rarity_offset
    }
}

/// A stack of items with count
#[derive(Debug, Clone, Copy)]
pub struct ItemStack {
    pub item: Item,
    pub count: u8,
    /// Whether this slot is locked (won't be affected by sorting)
    pub locked: bool,
}

impl ItemStack {
    /// Create a new item stack
    pub fn new(item: Item, count: u8) -> Self {
        Self {
            item,
            count: count.min(item.max_stack_size()),
            locked: false,
        }
    }

    /// Create a locked item stack
    pub fn new_locked(item: Item, count: u8) -> Self {
        Self {
            item,
            count: count.min(item.max_stack_size()),
            locked: true,
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

    /// Check if stack is full
    pub fn is_full(&self) -> bool {
        self.count >= self.item.max_stack_size()
    }

    /// Get remaining space in stack
    pub fn space_remaining(&self) -> u8 {
        self.item.max_stack_size().saturating_sub(self.count)
    }

    /// Toggle lock state
    pub fn toggle_lock(&mut self) {
        self.locked = !self.locked;
    }

    /// Check if items can merge (same item type, not full)
    pub fn can_merge_with(&self, other: &ItemStack) -> bool {
        self.item == other.item && !self.is_full()
    }
}

/// Extension trait for Block to support item system
pub trait BlockItemExt {
    fn is_tool_block(&self) -> bool;
    fn display_name(&self) -> &'static str;
}

impl BlockItemExt for Block {
    fn is_tool_block(&self) -> bool {
        // Currently no tool blocks, but ready for future
        false
    }

    fn display_name(&self) -> &'static str {
        match self {
            Block::Air => "Air",
            Block::Dirt => "Dirt",
            Block::Grass => "Grass Block",
            Block::Stone => "Stone",
            Block::StoneDirt => "Dirty Stone",
            Block::Brick => "Brick",
            Block::Water => "Water",
            Block::Wood => "Wood Planks",
            Block::Leaves => "Leaves",
            Block::StoneRubble2 => "Stone Rubble",
            Block::IronOre => "Iron Ore",
            Block::GoldOre => "Gold Ore",
            Block::DiamondOre => "Diamond Ore",
            Block::RedstoneOre => "Redstone Ore",
            Block::Bedrock => "Bedrock",
            Block::Torch => "Torch",
            Block::Sand => "Sand",
            Block::CoalOre => "Coal Ore",
            Block::CopperOre => "Copper Ore",
            Block::EmeraldOre => "Emerald Ore",
            Block::SapphireOre => "Sapphire Ore",
            Block::QuartzOre => "Quartz Ore",
            Block::AmethystOre => "Amethyst Ore",
            Block::SilverOre => "Silver Ore",
            Block::TinOre => "Tin Ore",
            Block::NickelOre => "Nickel Ore",
            Block::LeadOre => "Lead Ore",
            Block::TitaniumOre => "Titanium Ore",
            Block::UraniumOre => "Uranium Ore",
            Block::OakLog => "Oak Log",
            Block::BirchLog => "Birch Log",
            Block::Sandstone => "Sandstone",
            Block::RedSandstone => "Red Sandstone",
            Block::SnowGrass => "Snowy Grass",
            Block::Snow => "Snow",
            Block::Ice => "Ice",
            Block::PackedIce => "Packed Ice",
            Block::MudBlock => "Mud",
            Block::Peat => "Peat",
            Block::JungleGrass => "Jungle Grass",
            Block::JungleLeaves => "Jungle Leaves",
            Block::Vines => "Vines",
            Block::Gravel => "Gravel",
            Block::MossyStone => "Mossy Stone",
            Block::CrackedStone => "Cracked Stone",
            Block::ClayStone => "Claystone",
            Block::Stalactite => "Stalactite",
            Block::Stalagmite => "Stalagmite",
            Block::CrystalAmethyst => "Amethyst Crystal",
            Block::CrystalQuartz => "Quartz Crystal",
            Block::Cactus => "Cactus",
            Block::DeadBush => "Dead Bush",
            Block::PineLog => "Pine Log",
            Block::JungleLog => "Jungle Log",
            Block::SpruceLeaves => "Spruce Leaves",
        }
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
        assert!(!stack.locked);
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

    #[test]
    fn test_item_tags() {
        let coal = Item::Block(Block::CoalOre);
        assert!(coal.has_tag(ItemTag::Ore));
        assert!(coal.has_tag(ItemTag::Combustible));
        assert!(coal.has_tag(ItemTag::RawMaterial));

        let torch = Item::Block(Block::Torch);
        assert!(torch.has_tag(ItemTag::LightSource));
        assert!(torch.has_tag(ItemTag::Combustible));
    }

    #[test]
    fn test_item_rarity() {
        assert_eq!(Item::Block(Block::Stone).rarity(), ItemRarity::Common);
        assert_eq!(Item::Block(Block::IronOre).rarity(), ItemRarity::Uncommon);
        assert_eq!(Item::Block(Block::GoldOre).rarity(), ItemRarity::Rare);
        assert_eq!(Item::Block(Block::EmeraldOre).rarity(), ItemRarity::Epic);
        assert_eq!(Item::Block(Block::DiamondOre).rarity(), ItemRarity::Legendary);
    }

    #[test]
    fn test_item_stack_locking() {
        let mut stack = ItemStack::new(Item::Block(Block::Stone), 32);
        assert!(!stack.locked);

        stack.toggle_lock();
        assert!(stack.locked);

        let locked_stack = ItemStack::new_locked(Item::Block(Block::Torch), 16);
        assert!(locked_stack.locked);
    }

    #[test]
    fn test_item_stack_merge() {
        let stack1 = ItemStack::new(Item::Block(Block::Stone), 32);
        let stack2 = ItemStack::new(Item::Block(Block::Stone), 16);
        let stack3 = ItemStack::new(Item::Block(Block::Dirt), 16);
        let full_stack = ItemStack::new(Item::Block(Block::Stone), 64);

        assert!(stack1.can_merge_with(&stack2));
        assert!(!stack1.can_merge_with(&stack3)); // Different items
        assert!(!full_stack.can_merge_with(&stack2)); // Full stack
    }
}
