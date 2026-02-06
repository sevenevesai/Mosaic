//! Inventory system for player item management
//!
//! Provides:
//! - Hotbar (9 slots) and main inventory (27 slots) with item stacking
//! - Item tags for categorization and Quick Stack support
//! - Sorting with locked slot protection
//! - Drag-and-drop item movement

pub mod item;
pub mod inventory;

pub use item::{Item, ItemStack, ItemTag, ItemRarity, BlockItemExt, ToolKind, ToolMaterial};
pub use inventory::{Inventory, HOTBAR_SIZE, MAIN_INVENTORY_SIZE, TOTAL_SLOTS, layout};
