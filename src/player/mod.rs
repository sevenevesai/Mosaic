//! Player module
//!
//! Handles player input, movement, and camera control.

use std::time::Duration;

use glam::Vec3;
use winit::{
    event::MouseButton,
    keyboard::KeyCode,
};

use crate::renderer::Camera;
use crate::world::WorldManager;

/// Player state
pub struct Player {
    pub camera: Camera,
    velocity: Vec3,

    // Input state
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
    sprint: bool,

    // Physics
    on_ground: bool,
}

impl Player {
    pub fn new(position: Vec3) -> Self {
        Self {
            camera: Camera::new(position, 16.0 / 9.0),
            velocity: Vec3::ZERO,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
            sprint: false,
            on_ground: false,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.camera.position
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.move_forward = pressed,
            KeyCode::KeyS => self.move_backward = pressed,
            KeyCode::KeyA => self.move_left = pressed,
            KeyCode::KeyD => self.move_right = pressed,
            KeyCode::Space => self.move_up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.move_down = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.sprint = pressed,
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
                // Break block (raycast to find target)
                // TODO: Implement block breaking
            }
            MouseButton::Right => {
                // Place block
                // TODO: Implement block placing
            }
            _ => {}
        }
    }

    pub fn update(&mut self, dt: Duration, world: Option<&WorldManager>) {
        let dt_secs = dt.as_secs_f32();

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

        // Horizontal movement only for WASD
        move_dir.y = 0.0;
        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        // Vertical movement (creative mode fly)
        if self.move_up {
            move_dir.y += 1.0;
        }
        if self.move_down {
            move_dir.y -= 1.0;
        }

        // Speed
        let base_speed = 10.0;
        let speed = if self.sprint { base_speed * 2.5 } else { base_speed };

        // Apply movement
        let target_velocity = move_dir * speed;
        let acceleration = 20.0;

        // Smooth acceleration
        self.velocity = self.velocity.lerp(target_velocity, (acceleration * dt_secs).min(1.0));

        // Move camera
        self.camera.translate(self.velocity * dt_secs);

        // Simple collision with ground (y = 0)
        // In production, do proper voxel collision
        if self.camera.position.y < 2.0 {
            self.camera.position.y = 2.0;
            self.velocity.y = 0.0;
            self.on_ground = true;
        } else {
            self.on_ground = false;
        }
    }
}
