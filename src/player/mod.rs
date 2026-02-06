//! Player module
//!
//! Handles player input, movement, camera control, and inventory management.

mod persistence;

pub use persistence::{PlayerData, PlayerDataError};

use std::time::Duration;

use glam::Vec3;
use winit::{
    event::MouseButton,
    keyboard::KeyCode,
};

use crate::constants::{
    PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_EYE_HEIGHT,
    GRAVITY, JUMP_VELOCITY, TERMINAL_VELOCITY, PLAYER_REACH,
};
use crate::inventory::{Inventory, Item};
use crate::renderer::Camera;
use crate::world::{Block, WorldManager, raycast, raycast_props};

/// Default third-person camera distance behind the player (blocks).
const THIRD_PERSON_DISTANCE_DEFAULT: f32 = 10.0;
/// Minimum camera distance when colliding with terrain.
const THIRD_PERSON_MIN_DISTANCE: f32 = 1.0;
/// Maximum third-person camera distance (scroll zoom limit).
const THIRD_PERSON_MAX_DISTANCE: f32 = 20.0;
/// Minimum third-person camera distance (scroll zoom limit).
const THIRD_PERSON_MIN_ZOOM: f32 = 2.0;
/// Height offset above player center for camera target point.
const THIRD_PERSON_HEIGHT_OFFSET: f32 = 0.5;
/// Duration of camera mode transition (seconds).
const CAMERA_TRANSITION_DURATION: f32 = 0.3;
/// First-person FOV (wider for immersion).
const FIRST_PERSON_FOV: f32 = 75.0;
/// Third-person FOV (standard).
const THIRD_PERSON_FOV: f32 = 70.0;

/// Player movement state for animation selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementState {
    Idle,
    Walking,
    Sprinting,
    Airborne,
    Swimming,
}

/// Result of block interaction processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockInteraction {
    /// No interaction occurred
    None,
    /// A block was broken at the given world position
    Break { x: i32, y: i32, z: i32, block: Block },
    /// A block was placed at the given world position
    Place { x: i32, y: i32, z: i32, block: Block },
    /// A prop was harvested at the given world position
    HarvestProp { x: f32, y: f32, z: f32, prop_id: u8 },
}

/// Axis-aligned bounding box
#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    /// Create AABB from player feet position
    fn from_feet_pos(feet: Vec3) -> Self {
        let half_width = PLAYER_WIDTH / 2.0;
        Self {
            min: Vec3::new(feet.x - half_width, feet.y, feet.z - half_width),
            max: Vec3::new(feet.x + half_width, feet.y + PLAYER_HEIGHT, feet.z + half_width),
        }
    }

    /// Check if this AABB intersects a block at given world coordinates
    fn intersects_block(&self, bx: i32, by: i32, bz: i32) -> bool {
        let block_min = Vec3::new(bx as f32, by as f32, bz as f32);
        let block_max = block_min + Vec3::ONE;

        self.min.x < block_max.x && self.max.x > block_min.x &&
        self.min.y < block_max.y && self.max.y > block_min.y &&
        self.min.z < block_max.z && self.max.z > block_min.z
    }
}

/// Player state
pub struct Player {
    pub camera: Camera,
    velocity: Vec3,

    /// Player feet position (camera is offset by PLAYER_EYE_HEIGHT)
    feet_pos: Vec3,

    // Input state
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    jump_pressed: bool,
    sneak: bool,
    sprint: bool,

    // Block interaction
    break_requested: bool,
    place_requested: bool,
    drop_requested: bool,

    /// Player inventory (holds items, manages selection)
    pub inventory: Inventory,

    // Physics
    on_ground: bool,
    in_water: bool,

    // Mode
    fly_mode: bool,

    // Camera mode
    /// Whether third-person camera is active (true) or first-person (false).
    third_person: bool,
    /// Character model facing direction (yaw in radians). Updated from movement.
    facing_yaw: f32,
    /// Third-person camera distance (adjustable via scroll wheel).
    third_person_distance: f32,
    /// Camera mode transition timer (counts down to 0). 0 = no transition.
    camera_transition_t: f32,
    /// Camera position at start of transition (for smooth lerp).
    camera_transition_from: Vec3,
    /// FOV at start of transition (for smooth lerp).
    camera_transition_fov_from: f32,
}

impl Player {
    pub fn new(position: Vec3) -> Self {
        // Position is camera position, calculate feet position
        let feet_pos = Vec3::new(position.x, position.y - PLAYER_EYE_HEIGHT, position.z);
        let mut camera = Camera::new(position, 16.0 / 9.0);

        // Initialize camera behind the player for third-person view so the first frame
        // doesn't render with the camera inside the character mesh.
        let center = Vec3::new(
            feet_pos.x,
            feet_pos.y + PLAYER_HEIGHT * 0.5 + THIRD_PERSON_HEIGHT_OFFSET,
            feet_pos.z,
        );
        let forward = camera.forward();
        camera.position = center - forward * THIRD_PERSON_DISTANCE_DEFAULT;

        Self {
            camera,
            velocity: Vec3::ZERO,
            feet_pos,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            jump_pressed: false,
            sneak: false,
            sprint: false,
            break_requested: false,
            place_requested: false,
            drop_requested: false,
            inventory: {
                let mut inv = Inventory::new_empty();
                // Give player a chest for testing
                inv.add_item(crate::inventory::Item::Block(Block::Chest), 5);
                // Give player torches for lighting
                inv.add_item(crate::inventory::Item::Block(Block::Torch), 64);
                inv
            },
            on_ground: false,
            in_water: false,
            fly_mode: false,
            third_person: true,
            facing_yaw: 0.0,
            third_person_distance: THIRD_PERSON_DISTANCE_DEFAULT,
            camera_transition_t: 0.0,
            camera_transition_from: Vec3::ZERO,
            camera_transition_fov_from: THIRD_PERSON_FOV.to_radians(),
        }
    }

    pub fn position(&self) -> Vec3 {
        self.camera.position
    }

    /// Get the currently selected block (from inventory, for placement and held light)
    pub fn selected_block(&self) -> Block {
        self.inventory.selected_block().unwrap_or(Block::Air)
    }

    /// Get horizontal movement speed (for footstep audio)
    pub fn horizontal_speed(&self) -> f32 {
        (self.velocity.x * self.velocity.x + self.velocity.z * self.velocity.z).sqrt()
    }

    /// Check if player is on the ground
    pub fn is_on_ground(&self) -> bool {
        self.on_ground
    }

    /// Check if player is sprinting
    pub fn is_sprinting(&self) -> bool {
        self.sprint
    }

    /// Check if player is in fly mode
    pub fn is_flying(&self) -> bool {
        self.fly_mode
    }

    pub fn is_in_water(&self) -> bool {
        self.in_water
    }

    /// Get player feet position (model origin).
    pub fn feet_position(&self) -> Vec3 {
        self.feet_pos
    }

    /// Get the character model facing yaw (radians).
    pub fn facing_yaw(&self) -> f32 {
        self.facing_yaw
    }

    /// Whether 3rd person camera is active.
    pub fn is_third_person(&self) -> bool {
        self.third_person
    }

    /// Toggle between first-person and third-person camera.
    pub fn toggle_camera_mode(&mut self) {
        // Save current camera state for smooth transition
        self.camera_transition_from = self.camera.position;
        self.camera_transition_fov_from = self.camera.fov;
        self.camera_transition_t = CAMERA_TRANSITION_DURATION;

        self.third_person = !self.third_person;
        tracing::info!("Camera mode: {}", if self.third_person { "THIRD PERSON" } else { "FIRST PERSON" });
    }

    /// Adjust third-person camera distance via scroll wheel.
    /// Positive delta = zoom in, negative delta = zoom out.
    pub fn adjust_camera_zoom(&mut self, delta: f32) {
        self.third_person_distance = (self.third_person_distance - delta * 1.5)
            .clamp(THIRD_PERSON_MIN_ZOOM, THIRD_PERSON_MAX_DISTANCE);
    }

    /// Current movement state for animation selection.
    pub fn movement_state(&self) -> MovementState {
        if self.in_water {
            return MovementState::Swimming;
        }
        if !self.on_ground && !self.fly_mode {
            return MovementState::Airborne;
        }
        let horiz_speed = self.horizontal_speed();
        if horiz_speed > 0.5 {
            if self.sprint {
                MovementState::Sprinting
            } else {
                MovementState::Walking
            }
        } else {
            MovementState::Idle
        }
    }

    /// Check if inventory is open (affects input handling)
    pub fn is_inventory_open(&self) -> bool {
        self.inventory.is_open
    }

    /// Get the block currently being targeted (for rendering outline)
    pub fn get_target_block(&self, world: &WorldManager) -> Option<[i32; 3]> {
        let origin = self.camera.position;
        let direction = self.camera.forward();
        raycast(origin, direction, PLAYER_REACH, world).map(|hit| hit.block_pos)
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        // E key toggles inventory regardless of other state
        if key == KeyCode::KeyE && pressed {
            self.inventory.toggle();
            return;
        }

        // When inventory is open, ignore most gameplay keys
        if self.inventory.is_open {
            return;
        }

        match key {
            KeyCode::KeyW => self.move_forward = pressed,
            KeyCode::KeyS => self.move_backward = pressed,
            KeyCode::KeyA => self.move_left = pressed,
            KeyCode::KeyD => self.move_right = pressed,
            KeyCode::Space => self.jump_pressed = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.sneak = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.sprint = pressed,
            KeyCode::KeyF => {
                if pressed {
                    self.fly_mode = !self.fly_mode;
                    tracing::info!("Fly mode: {}", if self.fly_mode { "ON" } else { "OFF" });
                }
            }
            // Hotbar selection with number keys
            KeyCode::Digit1 => if pressed { self.inventory.select_slot(0); }
            KeyCode::Digit2 => if pressed { self.inventory.select_slot(1); }
            KeyCode::Digit3 => if pressed { self.inventory.select_slot(2); }
            KeyCode::Digit4 => if pressed { self.inventory.select_slot(3); }
            KeyCode::Digit5 => if pressed { self.inventory.select_slot(4); }
            KeyCode::Digit6 => if pressed { self.inventory.select_slot(5); }
            KeyCode::Digit7 => if pressed { self.inventory.select_slot(6); }
            KeyCode::Digit8 => if pressed { self.inventory.select_slot(7); }
            KeyCode::Digit9 => if pressed { self.inventory.select_slot(8); }
            // Drop selected item
            KeyCode::KeyQ => if pressed { self.drop_requested = true; }
            _ => {}
        }
    }

    /// Handle mouse scroll for hotbar slot selection
    pub fn handle_scroll(&mut self, delta: f32) {
        if self.inventory.is_open {
            return; // Could be used for inventory scrolling later
        }

        if delta > 0.0 {
            self.inventory.prev_slot();
        } else if delta < 0.0 {
            self.inventory.next_slot();
        }
    }

    pub fn handle_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        // Don't rotate camera when inventory is open
        if self.inventory.is_open {
            return;
        }

        const SENSITIVITY: f32 = 0.002;
        self.camera.rotate(delta_x, delta_y, SENSITIVITY);
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if !pressed {
            return;
        }

        // If inventory is open, don't process world clicks here
        // (inventory clicks are handled separately)
        if self.inventory.is_open {
            return;
        }

        match button {
            MouseButton::Left => {
                self.break_requested = true;
            }
            MouseButton::Right => {
                self.place_requested = true;
            }
            _ => {}
        }
    }

    /// Process block interactions (call from main loop with mutable world access)
    ///
    /// Returns the type of interaction that occurred (Break, Place, HarvestProp, or None),
    /// including the world position of the affected block/prop for voxel GI updates.
    pub fn process_block_interactions(&mut self, world: &mut WorldManager) -> BlockInteraction {
        if self.break_requested {
            self.break_requested = false;

            let origin = self.camera.position;
            let direction = self.camera.forward();

            // Raycast against both props and blocks, take the closest hit
            let block_hit = raycast(origin, direction, PLAYER_REACH, world);
            let prop_hit = {
                let chunks = world.chunks_read();
                raycast_props(origin, direction, PLAYER_REACH, &chunks)
            };

            // Determine which is closer
            let block_dist = block_hit.as_ref().map(|h| h.distance).unwrap_or(f32::MAX);
            let prop_dist = prop_hit.as_ref().map(|h| h.distance).unwrap_or(f32::MAX);

            // Prop is closer - harvest it
            if prop_dist < block_dist {
                if let Some(hit) = prop_hit {
                    let [x, y, z] = hit.world_pos;
                    // Remove the prop from the world
                    if let Some(removed) = world.try_remove_prop(x, y, z, 0.5) {
                        tracing::debug!("Harvested prop {} at ({:.2}, {:.2}, {:.2})", removed.prop_id, x, y, z);
                        return BlockInteraction::HarvestProp { x, y, z, prop_id: removed.prop_id };
                    }
                }
            }
            // Block is closer - break it
            else if let Some(hit) = block_hit {
                let [x, y, z] = hit.block_pos;
                if let Some(block) = world.get_block(x, y, z) {
                    if block != Block::Bedrock {
                        world.set_block(x, y, z, Block::Air);
                        tracing::debug!("Broke {:?} at ({}, {}, {})", block, x, y, z);
                        return BlockInteraction::Break { x, y, z, block };
                    }
                }
            }
        }

        if self.place_requested {
            self.place_requested = false;
            if let Some((x, y, z, block)) = self.try_place_block(world) {
                return BlockInteraction::Place { x, y, z, block };
            }
        }

        BlockInteraction::None
    }

    /// Process drop request (Q key)
    /// Returns (item, count) if an item was dropped from selected slot
    pub fn process_drop_request(&mut self) -> Option<(Item, u8)> {
        if !self.drop_requested {
            return None;
        }
        self.drop_requested = false;

        // Get selected slot
        let slot_idx = self.inventory.selected_index();
        if let Some(stack) = self.inventory.get_slot(slot_idx) {
            let item = stack.item;
            // Drop one item (or could drop whole stack with Ctrl+Q)
            if self.inventory.consume_selected() {
                return Some((item, 1));
            }
        }
        None
    }

    /// Get forward direction for item throwing
    pub fn forward(&self) -> Vec3 {
        self.camera.forward()
    }

    fn try_place_block(&mut self, world: &mut WorldManager) -> Option<(i32, i32, i32, Block)> {
        // Check if we have a block to place
        let block_to_place = self.inventory.selected_block()?;

        let origin = self.camera.position;
        let direction = self.camera.forward();

        if let Some(hit) = raycast(origin, direction, PLAYER_REACH, world) {
            let [x, y, z] = hit.adjacent_pos;

            // Don't place if it would be inside the player
            let half_width = PLAYER_WIDTH / 2.0;
            let player_min = Vec3::new(
                self.feet_pos.x - half_width,
                self.feet_pos.y,
                self.feet_pos.z - half_width,
            );
            let player_max = Vec3::new(
                self.feet_pos.x + half_width,
                self.feet_pos.y + PLAYER_HEIGHT,
                self.feet_pos.z + half_width,
            );

            let block_min = Vec3::new(x as f32, y as f32, z as f32);
            let block_max = block_min + Vec3::ONE;

            // Check AABB intersection
            let intersects = player_min.x < block_max.x && player_max.x > block_min.x
                && player_min.y < block_max.y && player_max.y > block_min.y
                && player_min.z < block_max.z && player_max.z > block_min.z;

            if !intersects {
                // Place if target is air or water (can replace water with solid blocks)
                if let Some(target_block) = world.get_block(x, y, z) {
                    if target_block == Block::Air || target_block == Block::Water {
                        world.set_block(x, y, z, block_to_place);

                        // Consume the item from inventory
                        self.inventory.consume_selected();

                        tracing::debug!("Placed {:?} at ({}, {}, {})", block_to_place, x, y, z);
                        return Some((x, y, z, block_to_place));
                    }
                }
            }
        }
        None
    }

    pub fn update(&mut self, dt: Duration, world: Option<&WorldManager>) {
        let dt_secs = dt.as_secs_f32();

        if self.fly_mode {
            self.update_fly_mode(dt_secs);
        } else {
            self.update_normal_mode(dt_secs, world);
        }

        // Update facing direction based on camera mode.
        if self.third_person {
            // Third person: facing direction from WASD input relative to camera.
            // Smoothly interpolate to avoid jarring snaps when changing direction.
            let mut input_dir = Vec3::ZERO;
            if self.move_forward { input_dir += self.camera.forward(); }
            if self.move_backward { input_dir -= self.camera.forward(); }
            if self.move_left { input_dir -= self.camera.right(); }
            if self.move_right { input_dir += self.camera.right(); }
            input_dir.y = 0.0;
            if input_dir.length_squared() > 0.01 {
                let target_yaw = input_dir.x.atan2(input_dir.z);
                let mut diff = target_yaw - self.facing_yaw;
                if diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
                if diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
                let rotation_speed = 15.0;
                let max_step = rotation_speed * dt_secs;
                if diff.abs() < max_step {
                    self.facing_yaw = target_yaw;
                } else {
                    self.facing_yaw += diff.signum() * max_step;
                }
                if self.facing_yaw > std::f32::consts::PI { self.facing_yaw -= 2.0 * std::f32::consts::PI; }
                if self.facing_yaw < -std::f32::consts::PI { self.facing_yaw += 2.0 * std::f32::consts::PI; }
            }
        } else {
            // First person: character faces camera direction (horizontal component of yaw).
            self.facing_yaw = self.camera.yaw;
        }

        // Compute target camera position and FOV for current mode
        let target_fov;
        let target_pos;

        if self.third_person {
            target_fov = THIRD_PERSON_FOV.to_radians();

            // 3rd person: camera orbits behind player
            let center = Vec3::new(
                self.feet_pos.x,
                self.feet_pos.y + PLAYER_HEIGHT * 0.5 + THIRD_PERSON_HEIGHT_OFFSET,
                self.feet_pos.z,
            );
            let forward = self.camera.forward();
            let desired_pos = center - forward * self.third_person_distance;

            // Camera-terrain collision: raycast from center toward desired camera position
            let cam_dir = (desired_pos - center).normalize_or_zero();
            let mut cam_dist = self.third_person_distance;

            if let Some(world) = world {
                let steps = 10;
                for i in 1..=steps {
                    let t = (i as f32 / steps as f32) * self.third_person_distance;
                    let sample = center + cam_dir * t;
                    let bx = sample.x.floor() as i32;
                    let by = sample.y.floor() as i32;
                    let bz = sample.z.floor() as i32;
                    if let Some(block) = world.get_block(bx, by, bz) {
                        if block.is_solid() {
                            cam_dist = (t - 0.5).max(THIRD_PERSON_MIN_DISTANCE);
                            break;
                        }
                    }
                }
            }

            target_pos = center + cam_dir * cam_dist;
        } else {
            target_fov = FIRST_PERSON_FOV.to_radians();

            // 1st person: camera at eye height
            target_pos = Vec3::new(
                self.feet_pos.x,
                self.feet_pos.y + PLAYER_EYE_HEIGHT,
                self.feet_pos.z,
            );
        }

        // Smooth camera transition between modes
        if self.camera_transition_t > 0.0 {
            self.camera_transition_t = (self.camera_transition_t - dt_secs).max(0.0);
            let t = 1.0 - (self.camera_transition_t / CAMERA_TRANSITION_DURATION);
            // Smoothstep for eased transition
            let t = t * t * (3.0 - 2.0 * t);
            self.camera.position = self.camera_transition_from.lerp(target_pos, t);
            self.camera.fov = self.camera_transition_fov_from + (target_fov - self.camera_transition_fov_from) * t;
        } else {
            self.camera.position = target_pos;
            self.camera.fov = target_fov;
        }
    }

    fn update_fly_mode(&mut self, dt_secs: f32) {
        // Calculate movement direction
        let mut move_dir = Vec3::ZERO;

        if self.move_forward {
            move_dir += self.camera.forward();
        }
        if self.move_backward {
            move_dir -= self.camera.forward();
        }
        if self.move_left {
            move_dir -= self.camera.right();
        }
        if self.move_right {
            move_dir += self.camera.right();
        }

        // Full 3D movement in fly mode
        if self.jump_pressed {
            move_dir.y += 1.0;
        }
        if self.sneak {
            move_dir.y -= 1.0;
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        // Speed
        let base_speed = 15.0;
        let speed = if self.sprint { base_speed * 3.0 } else { base_speed };

        // Instant velocity in fly mode
        self.velocity = move_dir * speed;
        self.feet_pos += self.velocity * dt_secs;
    }

    fn update_normal_mode(&mut self, dt_secs: f32, world: Option<&WorldManager>) {
        // Check if player is in water
        if let Some(world) = world {
            self.in_water = self.check_in_water(world);
        }

        if self.in_water {
            self.update_swimming(dt_secs, world);
        } else {
            self.update_walking(dt_secs, world);
        }
    }

    fn update_walking(&mut self, dt_secs: f32, world: Option<&WorldManager>) {
        // Calculate horizontal movement direction
        let mut move_dir = Vec3::ZERO;

        if self.move_forward {
            move_dir += self.camera.forward();
        }
        if self.move_backward {
            move_dir -= self.camera.forward();
        }
        if self.move_left {
            move_dir -= self.camera.right();
        }
        if self.move_right {
            move_dir += self.camera.right();
        }

        // Horizontal only for WASD
        move_dir.y = 0.0;
        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        // Speed
        let base_speed = 4.3; // Minecraft walking speed
        let speed = if self.sprint { base_speed * 1.3 } else { base_speed };

        // Set horizontal velocity
        let target_horizontal = move_dir * speed;
        let accel = if self.on_ground { 20.0 } else { 5.0 }; // Less control in air
        self.velocity.x = lerp(self.velocity.x, target_horizontal.x, accel * dt_secs);
        self.velocity.z = lerp(self.velocity.z, target_horizontal.z, accel * dt_secs);

        // Apply gravity
        self.velocity.y -= GRAVITY * dt_secs;
        self.velocity.y = self.velocity.y.max(-TERMINAL_VELOCITY);

        // Jump
        if self.jump_pressed && self.on_ground {
            self.velocity.y = JUMP_VELOCITY;
            self.on_ground = false;
        }

        // Move with collision detection
        if let Some(world) = world {
            self.move_with_collision(dt_secs, world);
        } else {
            // No world, just move freely
            self.feet_pos += self.velocity * dt_secs;
        }
    }

    fn update_swimming(&mut self, dt_secs: f32, world: Option<&WorldManager>) {
        // Swimming: 3D movement like flying but slower with water drag
        let mut move_dir = Vec3::ZERO;

        if self.move_forward {
            move_dir += self.camera.forward();
        }
        if self.move_backward {
            move_dir -= self.camera.forward();
        }
        if self.move_left {
            move_dir -= self.camera.right();
        }
        if self.move_right {
            move_dir += self.camera.right();
        }

        // Swim up with jump, sink with sneak
        if self.jump_pressed {
            move_dir.y += 1.0;
        }
        if self.sneak {
            move_dir.y -= 1.0;
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        // Water speed is slower
        let base_speed = 2.5;
        let speed = if self.sprint { base_speed * 1.5 } else { base_speed };

        // Target velocity with water drag
        let target_vel = move_dir * speed;
        let water_accel = 8.0;
        self.velocity.x = lerp(self.velocity.x, target_vel.x, water_accel * dt_secs);
        self.velocity.y = lerp(self.velocity.y, target_vel.y, water_accel * dt_secs);
        self.velocity.z = lerp(self.velocity.z, target_vel.z, water_accel * dt_secs);

        // Slight buoyancy when not actively moving down
        if !self.sneak && move_dir.length_squared() < 0.1 {
            self.velocity.y += 1.5 * dt_secs;  // Gentle float up
        }

        // Apply weak gravity (sinking when not swimming)
        if !self.jump_pressed && !self.sneak && move_dir.length_squared() < 0.1 {
            self.velocity.y -= 2.0 * dt_secs;
        }

        // Clamp vertical velocity in water
        self.velocity.y = self.velocity.y.clamp(-3.0, 3.0);

        // Move with collision detection
        if let Some(world) = world {
            self.move_with_collision(dt_secs, world);
        } else {
            self.feet_pos += self.velocity * dt_secs;
        }

        // Not "on ground" while swimming
        self.on_ground = false;
    }

    /// Check if player's body is in water
    fn check_in_water(&self, world: &WorldManager) -> bool {
        // Check at feet level and mid-body
        let check_positions = [
            (self.feet_pos.x as i32, self.feet_pos.y as i32, self.feet_pos.z as i32),
            (self.feet_pos.x as i32, (self.feet_pos.y + 1.0) as i32, self.feet_pos.z as i32),
        ];

        for (bx, by, bz) in check_positions {
            if let Some(block) = world.get_block(bx, by, bz) {
                if block == Block::Water {
                    return true;
                }
            }
        }
        false
    }

    fn move_with_collision(&mut self, dt_secs: f32, world: &WorldManager) {
        let delta = self.velocity * dt_secs;

        // Move and collide on each axis separately
        // This prevents getting stuck on corners and allows sliding along walls

        // X axis
        self.feet_pos.x += delta.x;
        if self.check_collision(world) {
            self.feet_pos.x -= delta.x;
            self.velocity.x = 0.0;
        }

        // Y axis
        self.feet_pos.y += delta.y;
        if self.check_collision(world) {
            self.feet_pos.y -= delta.y;
            if self.velocity.y < 0.0 {
                self.on_ground = true;
            }
            self.velocity.y = 0.0;
        } else {
            self.on_ground = false;
        }

        // Z axis
        self.feet_pos.z += delta.z;
        if self.check_collision(world) {
            self.feet_pos.z -= delta.z;
            self.velocity.z = 0.0;
        }

        // Additional ground check (for standing still on ground)
        if !self.on_ground {
            let ground_check_pos = Vec3::new(self.feet_pos.x, self.feet_pos.y - 0.01, self.feet_pos.z);
            let ground_aabb = Aabb::from_feet_pos(ground_check_pos);
            if self.aabb_collides_world(&ground_aabb, world) {
                self.on_ground = true;
            }
        }
    }

    fn check_collision(&self, world: &WorldManager) -> bool {
        let aabb = Aabb::from_feet_pos(self.feet_pos);
        self.aabb_collides_world(&aabb, world)
    }

    fn aabb_collides_world(&self, aabb: &Aabb, world: &WorldManager) -> bool {
        // Check all blocks that could intersect the AABB
        let min_bx = aabb.min.x.floor() as i32;
        let max_bx = aabb.max.x.ceil() as i32;
        let min_by = aabb.min.y.floor() as i32;
        let max_by = aabb.max.y.ceil() as i32;
        let min_bz = aabb.min.z.floor() as i32;
        let max_bz = aabb.max.z.ceil() as i32;

        for by in min_by..max_by {
            for bz in min_bz..max_bz {
                for bx in min_bx..max_bx {
                    if let Some(block) = world.get_block(bx, by, bz) {
                        if block.is_solid() && aabb.intersects_block(bx, by, bz) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    // ========== PERSISTENCE ==========

    /// Create a PlayerData snapshot for saving
    pub fn to_save_data(&self) -> PlayerData {
        PlayerData::from_player(
            self.camera.position.into(),
            [self.camera.yaw, self.camera.pitch],
            self.inventory.selected_index() as u8,
            self.fly_mode,
            self.third_person,
            &self.inventory,
        )
    }

    /// Restore player state from saved data
    pub fn restore_from_save(&mut self, data: &PlayerData) {
        // Restore position
        self.camera.position = Vec3::from_array(data.position);
        self.feet_pos = Vec3::new(
            data.position[0],
            data.position[1] - PLAYER_EYE_HEIGHT,
            data.position[2],
        );

        // Restore rotation
        self.camera.yaw = data.rotation[0];
        self.camera.pitch = data.rotation[1];

        // Restore fly mode (third_person defaults to true, don't override from old saves)
        self.fly_mode = data.fly_mode;

        // Restore inventory
        data.apply_to_inventory(&mut self.inventory);

        tracing::info!(
            "Restored player: pos=({:.1}, {:.1}, {:.1}), fly={}",
            data.position[0], data.position[1], data.position[2], data.fly_mode
        );
    }

    /// Create a new player from saved data
    pub fn from_save_data(data: &PlayerData) -> Self {
        let position = Vec3::from_array(data.position);
        let mut player = Self::new(position);
        player.restore_from_save(data);
        player
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::HOTBAR_SIZE;

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 0.001);
        assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < 0.001);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 0.001);

        // Clamped to 1.0
        assert!((lerp(0.0, 10.0, 2.0) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_aabb_from_feet_pos() {
        let aabb = Aabb::from_feet_pos(Vec3::new(0.0, 10.0, 0.0));

        // Check dimensions match player constants
        let expected_half_width = PLAYER_WIDTH / 2.0;
        assert!((aabb.min.x - (-expected_half_width)).abs() < 0.001);
        assert!((aabb.max.x - expected_half_width).abs() < 0.001);
        assert!((aabb.min.y - 10.0).abs() < 0.001);
        assert!((aabb.max.y - (10.0 + PLAYER_HEIGHT)).abs() < 0.001);
    }

    #[test]
    fn test_aabb_intersects_block() {
        // Player standing at origin
        let aabb = Aabb::from_feet_pos(Vec3::new(0.5, 0.0, 0.5));

        // Block at feet level should intersect
        assert!(aabb.intersects_block(0, 0, 0));

        // Block at head level should intersect
        assert!(aabb.intersects_block(0, 1, 0));

        // Block way below should not intersect
        assert!(!aabb.intersects_block(0, -5, 0));

        // Block way above should not intersect
        assert!(!aabb.intersects_block(0, 10, 0));

        // Block far away should not intersect
        assert!(!aabb.intersects_block(100, 0, 100));
    }

    #[test]
    fn test_aabb_edge_cases() {
        // Player at block boundary
        let aabb = Aabb::from_feet_pos(Vec3::new(1.0, 0.0, 1.0));

        // Adjacent blocks should intersect due to player width
        assert!(aabb.intersects_block(0, 0, 0));
        assert!(aabb.intersects_block(1, 0, 1));
    }

    #[test]
    fn test_player_creation() {
        let player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Camera is positioned behind player for third-person view, so Y won't match exactly.
        // Verify feet position instead (derived from input camera position).
        assert!((player.feet_position().y - (100.0 - PLAYER_EYE_HEIGHT)).abs() < 0.001);
        assert!(!player.is_flying());
        assert!(!player.is_on_ground());
        assert!(!player.is_sprinting());
        assert!(!player.is_inventory_open());
    }

    #[test]
    fn test_player_horizontal_speed() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Zero velocity = zero speed
        assert!((player.horizontal_speed() - 0.0).abs() < 0.001);

        // Set horizontal velocity
        player.velocity = Vec3::new(3.0, 5.0, 4.0); // 3-4-5 triangle, horizontal = 5
        assert!((player.horizontal_speed() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_player_selected_block() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Empty inventory returns Air
        assert_eq!(player.selected_block(), Block::Air);

        // Add stone and check selection
        player.inventory.add_block(Block::Stone);
        assert_eq!(player.selected_block(), Block::Stone);
    }

    #[test]
    fn test_player_inventory_integration() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Add items to inventory
        player.inventory.add_block(Block::Stone);
        player.inventory.add_block(Block::Torch);

        // First slot has stone
        assert_eq!(player.selected_block(), Block::Stone);

        // Can add more items
        assert!(player.inventory.add_block(Block::DiamondOre));
    }

    #[test]
    fn test_player_key_handling() {
        let mut player = Player::new(Vec3::ZERO);

        // Movement keys
        assert!(!player.move_forward);
        player.handle_key(KeyCode::KeyW, true);
        assert!(player.move_forward);
        player.handle_key(KeyCode::KeyW, false);
        assert!(!player.move_forward);

        // Sprint
        assert!(!player.sprint);
        player.handle_key(KeyCode::ControlLeft, true);
        assert!(player.sprint);

        // Fly mode toggle
        assert!(!player.fly_mode);
        player.handle_key(KeyCode::KeyF, true);
        assert!(player.fly_mode);
        player.handle_key(KeyCode::KeyF, true);
        assert!(!player.fly_mode);
    }

    #[test]
    fn test_player_hotbar_selection_keys() {
        let mut player = Player::new(Vec3::ZERO);

        player.handle_key(KeyCode::Digit1, true);
        assert_eq!(player.inventory.selected_index(), 0);

        player.handle_key(KeyCode::Digit5, true);
        assert_eq!(player.inventory.selected_index(), 4);

        player.handle_key(KeyCode::Digit9, true);
        assert_eq!(player.inventory.selected_index(), 8);
    }

    #[test]
    fn test_player_inventory_toggle() {
        let mut player = Player::new(Vec3::ZERO);

        assert!(!player.is_inventory_open());
        player.handle_key(KeyCode::KeyE, true);
        assert!(player.is_inventory_open());
        player.handle_key(KeyCode::KeyE, true);
        assert!(!player.is_inventory_open());
    }

    #[test]
    fn test_player_scroll_changes_slot() {
        let mut player = Player::new(Vec3::ZERO);

        assert_eq!(player.inventory.selected_index(), 0);

        // Scroll down (next slot)
        player.handle_scroll(-1.0);
        assert_eq!(player.inventory.selected_index(), 1);

        // Scroll up (previous slot)
        player.handle_scroll(1.0);
        assert_eq!(player.inventory.selected_index(), 0);

        // Wrap around
        player.handle_scroll(1.0);
        assert_eq!(player.inventory.selected_index(), HOTBAR_SIZE - 1);
    }

    #[test]
    fn test_player_mouse_button_handling() {
        let mut player = Player::new(Vec3::ZERO);

        // Left click requests break
        assert!(!player.break_requested);
        player.handle_mouse_button(MouseButton::Left, true);
        assert!(player.break_requested);

        // Right click requests place
        assert!(!player.place_requested);
        player.handle_mouse_button(MouseButton::Right, true);
        assert!(player.place_requested);

        // Release doesn't trigger
        player.break_requested = false;
        player.handle_mouse_button(MouseButton::Left, false);
        assert!(!player.break_requested);
    }

    #[test]
    fn test_block_interaction_enum() {
        let none = BlockInteraction::None;
        let brk = BlockInteraction::Break { x: 1, y: 2, z: 3, block: Block::Stone };
        let place = BlockInteraction::Place { x: 4, y: 5, z: 6, block: Block::Dirt };

        assert_eq!(none, BlockInteraction::None);
        assert_ne!(brk, none);
        assert_ne!(place, none);

        if let BlockInteraction::Break { x, y, z, block } = brk {
            assert_eq!((x, y, z), (1, 2, 3));
            assert_eq!(block, Block::Stone);
        } else {
            panic!("Expected Break variant");
        }
    }

    #[test]
    fn test_player_update_fly_mode() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));
        player.fly_mode = true;
        player.move_forward = true;

        let initial_z = player.feet_pos.z;
        player.update(Duration::from_millis(100), None);

        // Should have moved (fly mode moves in camera direction)
        assert!(player.feet_pos.z != initial_z || player.feet_pos.x != 0.0);
    }

    #[test]
    fn test_player_update_without_world() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));
        player.velocity = Vec3::new(0.0, -5.0, 0.0);

        let initial_y = player.feet_pos.y;
        player.update(Duration::from_millis(100), None);

        // Should have fallen (no collision without world)
        assert!(player.feet_pos.y < initial_y);
    }

    #[test]
    fn test_camera_mode_toggle() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Defaults to third person
        assert!(player.is_third_person());

        // Toggle to first person
        player.toggle_camera_mode();
        assert!(!player.is_third_person());

        // Toggle back to third person
        player.toggle_camera_mode();
        assert!(player.is_third_person());
    }

    #[test]
    fn test_camera_mode_transition() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Toggle triggers a transition timer
        player.toggle_camera_mode();
        assert!(player.camera_transition_t > 0.0);

        // After enough time, transition completes
        player.update(Duration::from_millis(500), None);
        assert!((player.camera_transition_t - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_camera_zoom_adjustment() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        let initial_dist = player.third_person_distance;

        // Zoom in (positive delta)
        player.adjust_camera_zoom(1.0);
        assert!(player.third_person_distance < initial_dist);

        // Zoom out (negative delta)
        player.adjust_camera_zoom(-1.0);
        assert!((player.third_person_distance - initial_dist).abs() < 0.01);

        // Clamp to min
        for _ in 0..50 {
            player.adjust_camera_zoom(1.0);
        }
        assert!(player.third_person_distance >= THIRD_PERSON_MIN_ZOOM);

        // Clamp to max
        for _ in 0..50 {
            player.adjust_camera_zoom(-1.0);
        }
        assert!(player.third_person_distance <= THIRD_PERSON_MAX_DISTANCE);
    }

    #[test]
    fn test_first_person_camera_position() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Switch to first person
        player.third_person = false;
        player.camera_transition_t = 0.0; // Skip transition

        player.update(Duration::from_millis(16), None);

        // Camera should be at eye height above feet
        let expected_y = player.feet_pos.y + PLAYER_EYE_HEIGHT;
        assert!((player.camera.position.y - expected_y).abs() < 0.1);
    }

    #[test]
    fn test_first_person_facing_follows_camera() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Switch to first person
        player.third_person = false;
        player.camera_transition_t = 0.0;

        // Rotate camera
        player.camera.yaw = 1.5;
        player.update(Duration::from_millis(16), None);

        // Facing should match camera yaw
        assert!((player.facing_yaw - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_fov_differs_by_mode() {
        let mut player = Player::new(Vec3::new(0.0, 100.0, 0.0));

        // Third person FOV
        player.third_person = true;
        player.camera_transition_t = 0.0;
        player.update(Duration::from_millis(16), None);
        let tp_fov = player.camera.fov;

        // First person FOV (should be wider)
        player.third_person = false;
        player.camera_transition_t = 0.0;
        player.update(Duration::from_millis(16), None);
        let fp_fov = player.camera.fov;

        assert!(fp_fov > tp_fov);
    }
}
