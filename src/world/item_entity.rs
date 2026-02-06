//! Item entities - physical items that exist in the world
//!
//! Handles dropped items, physics, pickup, and merging.

use glam::Vec3;
use std::collections::HashMap;

use crate::inventory::Item;
use crate::world::Block;

/// Physics constants for item entities
pub mod physics {
    /// Gravity for items (slightly less than player for floatier feel)
    pub const ITEM_GRAVITY: f32 = 20.0;
    /// Terminal velocity
    pub const ITEM_TERMINAL_VELOCITY: f32 = 40.0;
    /// Bounce restitution (energy retained on bounce) - low for minimal bounce
    pub const ITEM_BOUNCE: f32 = 0.15;
    /// Ground friction (velocity multiplier per second) - high friction to stop quickly
    pub const ITEM_FRICTION: f32 = 0.02;
    /// Air drag
    pub const ITEM_AIR_DRAG: f32 = 0.95;
    /// Water buoyancy (upward acceleration in water)
    pub const ITEM_BUOYANCY: f32 = 12.0;
    /// Water drag
    pub const ITEM_WATER_DRAG: f32 = 0.9;
}

/// Pickup constants
pub mod pickup {
    /// Radius at which items start flying toward player
    pub const MAGNET_RADIUS: f32 = 1.5;
    /// Radius at which items are collected (larger since player pos is eye level)
    pub const COLLECT_RADIUS: f32 = 1.2;
    /// Speed at which items fly toward player
    pub const MAGNET_SPEED: f32 = 8.0;
    /// Delay after spawn before pickup is allowed (seconds)
    pub const PICKUP_DELAY: f32 = 0.5;
    /// Delay after player drops item (seconds)
    pub const DROP_PICKUP_DELAY: f32 = 1.5;
}

/// Lifetime constants
pub mod lifetime {
    /// Default despawn time (seconds)
    pub const DESPAWN_TIME: f32 = 300.0; // 5 minutes
    /// Time before despawn when item starts flashing (seconds)
    pub const FLASH_WARNING_TIME: f32 = 30.0;
    /// Merge check radius
    pub const MERGE_RADIUS: f32 = 0.5;
    /// Time before items can merge (seconds)
    pub const MERGE_DELAY: f32 = 1.0;
}

/// A physical item in the world
#[derive(Debug, Clone)]
pub struct ItemEntity {
    /// Unique ID
    pub id: u64,
    /// World position (center of item)
    pub position: Vec3,
    /// Velocity
    pub velocity: Vec3,
    /// The item type
    pub item: Item,
    /// Stack count
    pub count: u8,
    /// Time since spawn (seconds)
    pub age: f32,
    /// Time until pickup allowed (seconds, counts down)
    pub pickup_delay: f32,
    /// Whether this item should never despawn
    pub no_despawn: bool,
    /// Whether item is on ground
    pub on_ground: bool,
    /// Whether item is in water
    pub in_water: bool,
    /// Whether item is being magnetized to player
    pub magnetized: bool,
}

impl ItemEntity {
    /// Create a new item entity
    pub fn new(id: u64, position: Vec3, velocity: Vec3, item: Item, count: u8) -> Self {
        Self {
            id,
            position,
            velocity,
            item,
            count,
            age: 0.0,
            pickup_delay: pickup::PICKUP_DELAY,
            no_despawn: false,
            on_ground: false,
            in_water: false,
            magnetized: false,
        }
    }

    /// Create with custom pickup delay (for player drops)
    pub fn with_pickup_delay(mut self, delay: f32) -> Self {
        self.pickup_delay = delay;
        self
    }

    /// Check if item can be picked up
    pub fn can_pickup(&self) -> bool {
        self.pickup_delay <= 0.0
    }

    /// Check if item should despawn
    pub fn should_despawn(&self) -> bool {
        !self.no_despawn && self.age >= lifetime::DESPAWN_TIME
    }

    /// Check if item is flashing (warning before despawn)
    pub fn is_flashing(&self) -> bool {
        !self.no_despawn && self.age >= (lifetime::DESPAWN_TIME - lifetime::FLASH_WARNING_TIME)
    }

    /// Check if item can merge with another
    pub fn can_merge_with(&self, other: &ItemEntity) -> bool {
        self.item == other.item
            && self.age >= lifetime::MERGE_DELAY
            && other.age >= lifetime::MERGE_DELAY
            && (self.count as u16 + other.count as u16) <= self.item.max_stack_size() as u16
    }

    /// Get the block type if this is a block item
    pub fn as_block(&self) -> Option<Block> {
        self.item.as_block()
    }
}

/// Manages all item entities in the world
pub struct ItemEntityManager {
    /// All active item entities
    entities: HashMap<u64, ItemEntity>,
    /// Next entity ID
    next_id: u64,
    /// Items to spawn next frame (deferred to avoid borrow issues)
    spawn_queue: Vec<(Vec3, Vec3, Item, u8, f32)>, // pos, vel, item, count, delay
    /// IDs of items collected this frame
    collected_ids: Vec<u64>,
}

impl ItemEntityManager {
    /// Create a new item entity manager
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            next_id: 1,
            spawn_queue: Vec::new(),
            collected_ids: Vec::new(),
        }
    }

    /// Spawn a new item entity
    pub fn spawn(&mut self, position: Vec3, velocity: Vec3, item: Item, count: u8) -> u64 {
        self.spawn_with_delay(position, velocity, item, count, pickup::PICKUP_DELAY)
    }

    /// Spawn with custom pickup delay
    pub fn spawn_with_delay(
        &mut self,
        position: Vec3,
        velocity: Vec3,
        item: Item,
        count: u8,
        pickup_delay: f32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let entity = ItemEntity::new(id, position, velocity, item, count)
            .with_pickup_delay(pickup_delay);

        self.entities.insert(id, entity);
        tracing::debug!("Spawned item entity {} ({:?} x{})", id, item, count);
        id
    }

    /// Queue a spawn for next frame (use when you can't borrow mutably)
    pub fn queue_spawn(&mut self, position: Vec3, velocity: Vec3, item: Item, count: u8, delay: f32) {
        self.spawn_queue.push((position, velocity, item, count, delay));
    }

    /// Process the spawn queue
    fn process_spawn_queue(&mut self) {
        // Take the queue to avoid borrow issues
        let queue: Vec<_> = std::mem::take(&mut self.spawn_queue);
        for (pos, vel, item, count, delay) in queue {
            self.spawn_with_delay(pos, vel, item, count, delay);
        }
    }

    /// Get an item entity by ID
    pub fn get(&self, id: u64) -> Option<&ItemEntity> {
        self.entities.get(&id)
    }

    /// Get mutable item entity by ID
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ItemEntity> {
        self.entities.get_mut(&id)
    }

    /// Remove an item entity
    pub fn remove(&mut self, id: u64) -> Option<ItemEntity> {
        self.entities.remove(&id)
    }

    /// Get all entities for rendering
    pub fn entities(&self) -> impl Iterator<Item = &ItemEntity> {
        self.entities.values()
    }

    /// Number of active entities
    pub fn count(&self) -> usize {
        self.entities.len()
    }

    /// Mark an item as collected (will be removed on next update)
    pub fn mark_collected(&mut self, id: u64) {
        self.collected_ids.push(id);
    }

    /// Update all item entities
    ///
    /// Returns list of (item, count) that were collected by the player
    pub fn update(
        &mut self,
        dt: f32,
        player_pos: Vec3,
        get_block: impl Fn(i32, i32, i32) -> Option<Block>,
        inventory_can_add: impl Fn(&Item, u8) -> bool,
    ) -> Vec<(Item, u8)> {
        // Process spawn queue first
        self.process_spawn_queue();

        let mut collected = Vec::new();
        let mut to_remove = Vec::new();
        let mut merges: Vec<(u64, u64)> = Vec::new(); // (keep, remove)

        // Collect already-marked collected IDs
        for id in self.collected_ids.drain(..) {
            if let Some(entity) = self.entities.remove(&id) {
                collected.push((entity.item, entity.count));
            }
        }

        // Get entity IDs for iteration (avoid borrow issues)
        let entity_ids: Vec<u64> = self.entities.keys().copied().collect();

        for id in entity_ids {
            let entity = match self.entities.get_mut(&id) {
                Some(e) => e,
                None => continue,
            };

            // Update age and pickup delay
            entity.age += dt;
            entity.pickup_delay = (entity.pickup_delay - dt).max(0.0);

            // Check despawn
            if entity.should_despawn() {
                to_remove.push(id);
                continue;
            }

            // Physics update
            Self::update_physics(entity, dt, &get_block);

            // Pickup check
            if entity.can_pickup() {
                let to_player = player_pos - entity.position;
                let dist = to_player.length();

                if dist < pickup::COLLECT_RADIUS {
                    // Try to collect
                    if inventory_can_add(&entity.item, entity.count) {
                        collected.push((entity.item, entity.count));
                        to_remove.push(id);
                        continue;
                    }
                } else if dist < pickup::MAGNET_RADIUS {
                    // Magnetize toward player
                    entity.magnetized = true;
                    let dir = to_player.normalize_or_zero();
                    entity.velocity = dir * pickup::MAGNET_SPEED;
                } else {
                    entity.magnetized = false;
                }
            }
        }

        // Check for merges (separate pass to avoid borrow issues)
        let positions: Vec<(u64, Vec3, Item, u8, f32)> = self.entities.iter()
            .map(|(&id, e)| (id, e.position, e.item, e.count, e.age))
            .collect();

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let (id1, pos1, item1, count1, age1) = positions[i];
                let (id2, pos2, item2, count2, age2) = positions[j];

                if item1 == item2
                    && age1 >= lifetime::MERGE_DELAY
                    && age2 >= lifetime::MERGE_DELAY
                    && (pos1 - pos2).length() < lifetime::MERGE_RADIUS
                    && (count1 as u16 + count2 as u16) <= item1.max_stack_size() as u16
                {
                    // Merge into the older entity
                    if age1 >= age2 {
                        merges.push((id1, id2));
                    } else {
                        merges.push((id2, id1));
                    }
                }
            }
        }

        // Process merges
        for (keep_id, remove_id) in merges {
            if to_remove.contains(&keep_id) || to_remove.contains(&remove_id) {
                continue;
            }
            if let Some(removed) = self.entities.remove(&remove_id) {
                if let Some(keeper) = self.entities.get_mut(&keep_id) {
                    keeper.count = keeper.count.saturating_add(removed.count)
                        .min(keeper.item.max_stack_size());
                    tracing::debug!("Merged item {} into {} (now {})", remove_id, keep_id, keeper.count);
                }
            }
        }

        // Remove despawned/collected items
        for id in to_remove {
            self.entities.remove(&id);
        }

        collected
    }

    /// Update physics for a single entity
    fn update_physics(
        entity: &mut ItemEntity,
        dt: f32,
        get_block: &impl Fn(i32, i32, i32) -> Option<Block>,
    ) {
        // Don't apply physics if magnetized
        if entity.magnetized {
            entity.position += entity.velocity * dt;
            return;
        }

        // Check if in water
        let block_x = entity.position.x.floor() as i32;
        let block_y = entity.position.y.floor() as i32;
        let block_z = entity.position.z.floor() as i32;

        entity.in_water = get_block(block_x, block_y, block_z)
            .map(|b| b == Block::Water)
            .unwrap_or(false);

        if entity.in_water {
            // Water physics
            entity.velocity.y += physics::ITEM_BUOYANCY * dt;
            entity.velocity *= physics::ITEM_WATER_DRAG.powf(dt);
        } else {
            // Air physics
            entity.velocity.y -= physics::ITEM_GRAVITY * dt;
            entity.velocity.y = entity.velocity.y.max(-physics::ITEM_TERMINAL_VELOCITY);
            entity.velocity *= physics::ITEM_AIR_DRAG.powf(dt);
        }

        // Ground friction
        if entity.on_ground {
            entity.velocity.x *= physics::ITEM_FRICTION.powf(dt);
            entity.velocity.z *= physics::ITEM_FRICTION.powf(dt);
        }

        // Move with collision
        let new_pos = entity.position + entity.velocity * dt;

        // Simple AABB collision (item is 0.25 x 0.25 x 0.25)
        let half_size = 0.125;

        // Check Y collision first (ground/ceiling)
        let check_y = new_pos.y - half_size;
        let ground_block = get_block(
            new_pos.x.floor() as i32,
            check_y.floor() as i32,
            new_pos.z.floor() as i32,
        );

        if ground_block.map(|b| b.is_solid()).unwrap_or(false) {
            // Hit ground
            entity.on_ground = true;
            entity.position.y = check_y.floor() + 1.0 + half_size;

            // Bounce
            if entity.velocity.y < -1.0 {
                entity.velocity.y = -entity.velocity.y * physics::ITEM_BOUNCE;
            } else {
                entity.velocity.y = 0.0;
            }
        } else {
            entity.on_ground = false;
            entity.position.y = new_pos.y;
        }

        // Check X collision
        let check_x = if entity.velocity.x > 0.0 {
            new_pos.x + half_size
        } else {
            new_pos.x - half_size
        };

        let x_block = get_block(
            check_x.floor() as i32,
            entity.position.y.floor() as i32,
            entity.position.z.floor() as i32,
        );

        if x_block.map(|b| b.is_solid()).unwrap_or(false) {
            entity.velocity.x = -entity.velocity.x * physics::ITEM_BOUNCE * 0.5;
        } else {
            entity.position.x = new_pos.x;
        }

        // Check Z collision
        let check_z = if entity.velocity.z > 0.0 {
            new_pos.z + half_size
        } else {
            new_pos.z - half_size
        };

        let z_block = get_block(
            entity.position.x.floor() as i32,
            entity.position.y.floor() as i32,
            check_z.floor() as i32,
        );

        if z_block.map(|b| b.is_solid()).unwrap_or(false) {
            entity.velocity.z = -entity.velocity.z * physics::ITEM_BOUNCE * 0.5;
        } else {
            entity.position.z = new_pos.z;
        }
    }

    /// Spawn item from broken block with random velocity
    pub fn spawn_from_block(&mut self, block_pos: [i32; 3], item: Item, count: u8) {
        let center = Vec3::new(
            block_pos[0] as f32 + 0.5,
            block_pos[1] as f32 + 0.5,
            block_pos[2] as f32 + 0.5,
        );

        // Random upward velocity with slight spread
        let angle = rand_angle();
        let speed = 2.0 + rand_float() * 1.0;
        let velocity = Vec3::new(
            angle.cos() * speed * 0.3,
            2.0 + rand_float() * 2.0,
            angle.sin() * speed * 0.3,
        );

        self.spawn(center, velocity, item, count);
    }

    /// Spawn item dropped by player
    pub fn spawn_from_player(&mut self, player_pos: Vec3, forward: Vec3, item: Item, count: u8) {
        let spawn_pos = player_pos + forward * 0.5;
        let velocity = forward * 4.0 + Vec3::Y * 2.0;

        self.spawn_with_delay(spawn_pos, velocity, item, count, pickup::DROP_PICKUP_DELAY);
    }

    /// Clear all entities (for world unload)
    pub fn clear(&mut self) {
        self.entities.clear();
    }

    /// Get data for saving
    pub fn save_data(&self) -> Vec<SavedItemEntity> {
        self.entities.values().map(|e| SavedItemEntity {
            position: e.position.into(),
            velocity: e.velocity.into(),
            item_id: match e.item {
                Item::Block(b) => b as u8,
                Item::Tool { kind, material, .. } => 128 + kind.to_id() * 6 + material.to_id(),
            },
            durability: match e.item {
                Item::Tool { durability, .. } => durability,
                _ => 0,
            },
            count: e.count,
            age: e.age,
            no_despawn: e.no_despawn,
        }).collect()
    }

    /// Load from saved data
    pub fn load_data(&mut self, data: Vec<SavedItemEntity>) {
        use crate::inventory::{ToolKind, ToolMaterial};

        self.clear();
        for saved in data {
            let item = if saved.item_id >= 128 {
                // Decode tool
                let tool_idx = saved.item_id - 128;
                let kind_id = tool_idx / 6;
                let mat_id = tool_idx % 6;
                match (ToolKind::from_id(kind_id), ToolMaterial::from_id(mat_id)) {
                    (Some(kind), Some(material)) => {
                        Item::Tool { kind, material, durability: saved.durability }
                    }
                    _ => continue, // Skip invalid tool data
                }
            } else {
                Item::Block(Block::from(saved.item_id))
            };
            let count = if matches!(item, Item::Tool { .. }) { 1 } else { saved.count };
            let mut entity = ItemEntity::new(
                self.next_id,
                Vec3::from(saved.position),
                Vec3::from(saved.velocity),
                item,
                count,
            );
            entity.age = saved.age;
            entity.no_despawn = saved.no_despawn;
            entity.pickup_delay = 0.0;

            self.entities.insert(self.next_id, entity);
            self.next_id += 1;
        }
        tracing::info!("Loaded {} item entities", self.entities.len());
    }
}

impl Default for ItemEntityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Saved item entity data
#[derive(Debug, Clone)]
pub struct SavedItemEntity {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub item_id: u8,
    pub count: u8,
    pub age: f32,
    pub no_despawn: bool,
    /// Tool durability (only used when item_id >= 128)
    pub durability: u16,
}

// Simple random helpers (avoid pulling in rand crate)
fn rand_float() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 10000) as f32 / 10000.0
}

fn rand_angle() -> f32 {
    rand_float() * std::f32::consts::TAU
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Block;

    #[test]
    fn test_item_entity_creation() {
        let entity = ItemEntity::new(
            1,
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::ZERO,
            Item::Block(Block::Stone),
            32,
        );

        assert_eq!(entity.id, 1);
        assert_eq!(entity.count, 32);
        assert!(!entity.can_pickup()); // Has pickup delay
    }

    #[test]
    fn test_item_entity_manager_spawn() {
        let mut manager = ItemEntityManager::new();

        let id = manager.spawn(
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::ZERO,
            Item::Block(Block::Stone),
            16,
        );

        assert_eq!(id, 1);
        assert_eq!(manager.count(), 1);

        let entity = manager.get(id).unwrap();
        assert_eq!(entity.count, 16);
    }

    #[test]
    fn test_item_merge_check() {
        let e1 = ItemEntity::new(1, Vec3::ZERO, Vec3::ZERO, Item::Block(Block::Stone), 32);
        let mut e2 = ItemEntity::new(2, Vec3::ZERO, Vec3::ZERO, Item::Block(Block::Stone), 16);

        // Can't merge when too young
        assert!(!e1.can_merge_with(&e2));

        // Can merge after delay
        e2.age = lifetime::MERGE_DELAY + 0.1;
        let mut e1_aged = e1.clone();
        e1_aged.age = lifetime::MERGE_DELAY + 0.1;
        assert!(e1_aged.can_merge_with(&e2));

        // Can't merge different items
        let e3 = ItemEntity::new(3, Vec3::ZERO, Vec3::ZERO, Item::Block(Block::Dirt), 16);
        assert!(!e1_aged.can_merge_with(&e3));
    }

    #[test]
    fn test_despawn_timing() {
        let mut entity = ItemEntity::new(1, Vec3::ZERO, Vec3::ZERO, Item::Block(Block::Stone), 1);

        assert!(!entity.should_despawn());
        assert!(!entity.is_flashing());

        entity.age = lifetime::DESPAWN_TIME - lifetime::FLASH_WARNING_TIME + 1.0;
        assert!(!entity.should_despawn());
        assert!(entity.is_flashing());

        entity.age = lifetime::DESPAWN_TIME + 1.0;
        assert!(entity.should_despawn());
    }

    #[test]
    fn test_no_despawn_flag() {
        let mut entity = ItemEntity::new(1, Vec3::ZERO, Vec3::ZERO, Item::Block(Block::Stone), 1);
        entity.no_despawn = true;
        entity.age = lifetime::DESPAWN_TIME + 1000.0;

        assert!(!entity.should_despawn());
        assert!(!entity.is_flashing());
    }
}
