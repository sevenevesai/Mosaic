//! Player module
//!
//! Handles player input, movement, and camera control.

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
use crate::renderer::Camera;
use crate::world::{Block, WorldManager, raycast};

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
    selected_block: Block,

    // Physics
    on_ground: bool,

    // Mode
    fly_mode: bool,
}

impl Player {
    pub fn new(position: Vec3) -> Self {
        // Position is camera position, calculate feet position
        let feet_pos = Vec3::new(position.x, position.y - PLAYER_EYE_HEIGHT, position.z);
        Self {
            camera: Camera::new(position, 16.0 / 9.0),
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
            selected_block: Block::Stone,
            on_ground: false,
            fly_mode: false,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.camera.position
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

    /// Get the block currently being targeted (for rendering outline)
    pub fn get_target_block(&self, world: &WorldManager) -> Option<[i32; 3]> {
        let origin = self.camera.position;
        let direction = self.camera.forward();
        raycast(origin, direction, PLAYER_REACH, world).map(|hit| hit.block_pos)
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
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
            // Block selection with number keys
            KeyCode::Digit1 => if pressed { self.selected_block = Block::Stone; }
            KeyCode::Digit2 => if pressed { self.selected_block = Block::Dirt; }
            KeyCode::Digit3 => if pressed { self.selected_block = Block::Grass; }
            KeyCode::Digit4 => if pressed { self.selected_block = Block::Cobblestone; }
            KeyCode::Digit5 => if pressed { self.selected_block = Block::Sand; }
            KeyCode::Digit6 => if pressed { self.selected_block = Block::Wood; }
            KeyCode::Digit7 => if pressed { self.selected_block = Block::Leaves; }
            KeyCode::Digit8 => if pressed { self.selected_block = Block::Torch; }
            _ => {}
        }
    }

    pub fn handle_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        const SENSITIVITY: f32 = 0.002;
        self.camera.rotate(delta_x, delta_y, SENSITIVITY);
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if !pressed {
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
    pub fn process_block_interactions(&mut self, world: &mut WorldManager) {
        if self.break_requested {
            self.break_requested = false;
            self.try_break_block(world);
        }

        if self.place_requested {
            self.place_requested = false;
            self.try_place_block(world);
        }
    }

    fn try_break_block(&self, world: &mut WorldManager) {
        let origin = self.camera.position;
        let direction = self.camera.forward();

        if let Some(hit) = raycast(origin, direction, PLAYER_REACH, world) {
            let [x, y, z] = hit.block_pos;
            // Don't break bedrock
            if let Some(block) = world.get_block(x, y, z) {
                if block != Block::Bedrock {
                    world.set_block(x, y, z, Block::Air);
                    tracing::debug!("Broke block at ({}, {}, {})", x, y, z);
                }
            }
        }
    }

    fn try_place_block(&self, world: &mut WorldManager) {
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
                // Only place if target is air
                if let Some(Block::Air) = world.get_block(x, y, z) {
                    world.set_block(x, y, z, self.selected_block);
                    tracing::debug!("Placed {:?} at ({}, {}, {})", self.selected_block, x, y, z);
                }
            }
        }
    }

    pub fn update(&mut self, dt: Duration, world: Option<&WorldManager>) {
        let dt_secs = dt.as_secs_f32();

        if self.fly_mode {
            self.update_fly_mode(dt_secs);
        } else {
            self.update_normal_mode(dt_secs, world);
        }

        // Sync camera position with feet position
        self.camera.position = Vec3::new(
            self.feet_pos.x,
            self.feet_pos.y + PLAYER_EYE_HEIGHT,
            self.feet_pos.z,
        );
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
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.min(1.0)
}
