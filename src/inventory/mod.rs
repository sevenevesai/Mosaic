//! Inventory system for player item management
//!
//! Provides hotbar (9 slots) and main inventory (27 slots) with item stacking.

pub mod item;
pub mod inventory;

pub use item::{Item, ItemStack};
pub use inventory::{Inventory, HOTBAR_SIZE, TOTAL_SLOTS};
