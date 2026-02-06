//! Entity system using generational arenas (SlotMap).
//!
//! Manages characters, NPCs, and other world entities with component-style
//! storage via SecondaryMaps. Phase 1: static T-pose character rendering.

pub mod loader;

use glam::{Vec3, Quat, Mat4};
use slotmap::{SlotMap, SecondaryMap, new_key_type};
use std::sync::Arc;

new_key_type! {
    /// Generational handle for entities - ABA-safe, O(1) lookup.
    pub struct EntityKey;
}

/// Type of entity for dispatch and rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    Player,
    Npc,
}

/// Core data every entity has.
pub struct EntityBase {
    pub entity_type: EntityType,
    pub active: bool,
}

/// Spatial transform component.
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
        }
    }
}

/// Render component linking an entity to its GPU mesh data.
pub struct RenderComponent {
    /// Index into the CharacterRenderer's mesh slots.
    pub mesh_slot: usize,
    /// Number of indices in the mesh.
    pub index_count: u32,
    /// Offset into the bone matrix storage buffer (in number of matrices).
    pub bone_offset: u32,
    /// Number of bones for this entity's skeleton.
    pub bone_count: u32,
}

/// Skeleton data shared across entities using the same rig.
pub struct Skeleton {
    /// Bone names for debugging and animation binding.
    pub bone_names: Vec<String>,
    /// Parent index for each bone (-1 for root).
    pub parent_indices: Vec<i32>,
    /// Inverse bind matrices (geometry space -> bone local space).
    pub inverse_bind_matrices: Vec<glam::Mat4>,
    /// Rest pose local transforms.
    pub rest_pose_local: Vec<glam::Mat4>,
}

impl Skeleton {
    /// Get the world-space rest positions of all bones.
    /// Uses inverse of inverse_bind_matrices for exact bind pose positions.
    pub fn get_bone_world_positions(&self) -> Vec<Vec3> {
        // inverse_bind = inverse(bone_world_transform_at_bind_time)
        // So bone_world_transform = inverse(inverse_bind)
        // And bone_world_pos = translation component of that
        self.inverse_bind_matrices.iter().map(|inv_bind| {
            let world_transform = inv_bind.inverse();
            world_transform.w_axis.truncate()
        }).collect()
    }

    /// Compute world-space bone matrices in T-pose (identity skinning).
    /// Returns matrices suitable for GPU upload: global * inverse_bind.
    pub fn compute_tpose_matrices(&self) -> Vec<Mat4> {
        let count = self.bone_names.len();
        let mut global = vec![Mat4::IDENTITY; count];

        // Walk hierarchy: compute global transforms
        for i in 0..count {
            let parent = self.parent_indices[i];
            if parent >= 0 && (parent as usize) < count {
                global[i] = global[parent as usize] * self.rest_pose_local[i];
            } else {
                global[i] = self.rest_pose_local[i];
            }
        }

        // Final matrices: global * inverse_bind
        let mut result = vec![Mat4::IDENTITY; count];
        for i in 0..count {
            result[i] = global[i] * self.inverse_bind_matrices[i];
        }
        result
    }

    /// Compute bone matrices from animated local poses.
    /// Writes results directly into `output` (global * inverse_bind per bone).
    pub fn compute_animated_matrices(&self, local_poses: &[Mat4], output: &mut Vec<Mat4>) {
        let count = self.bone_names.len();
        output.resize(count, Mat4::IDENTITY);

        // Temporary global transform storage
        let mut global = vec![Mat4::IDENTITY; count];

        for i in 0..count {
            let local = if i < local_poses.len() {
                local_poses[i]
            } else {
                self.rest_pose_local[i]
            };

            let parent = self.parent_indices[i];
            if parent >= 0 && (parent as usize) < count {
                global[i] = global[parent as usize] * local;
            } else {
                global[i] = local;
            }

            output[i] = global[i] * self.inverse_bind_matrices[i];
        }
    }
}

/// A single baked animation clip with per-bone keyframes.
pub struct AnimationClip {
    pub name: String,
    pub duration: f64,
    /// Per-bone keyframes. Index matches skeleton bone index.
    /// None if this bone has no animation in this clip.
    pub bone_tracks: Vec<Option<BoneTrack>>,
}

/// Keyframe tracks for a single bone (translation, rotation, scale).
pub struct BoneTrack {
    pub translation_keys: Vec<BakedKey<Vec3>>,
    pub rotation_keys: Vec<BakedKey<Quat>>,
    pub scale_keys: Vec<BakedKey<Vec3>>,
}

/// A single keyframe with time and value.
#[derive(Clone)]
pub struct BakedKey<T: Clone> {
    pub time: f64,
    pub value: T,
}

impl AnimationClip {
    /// Sample bone local transforms at a given time (looping).
    /// Returns local TRS per bone. Bones without animation get rest pose from skeleton.
    pub fn sample(&self, time: f64, skeleton: &Skeleton) -> Vec<Mat4> {
        let t = if self.duration > 0.0 {
            time % self.duration
        } else {
            0.0
        };

        let bone_count = skeleton.bone_names.len();
        let mut local_matrices = Vec::with_capacity(bone_count);

        for i in 0..bone_count {
            let mat = if i < self.bone_tracks.len() {
                if let Some(ref track) = self.bone_tracks[i] {
                    let translation = sample_vec3(&track.translation_keys, t);
                    let rotation = sample_quat(&track.rotation_keys, t);
                    // Scale defaults to 1.0 if no scale keys (not 0.0)
                    let scale = if track.scale_keys.is_empty() {
                        Vec3::ONE
                    } else {
                        sample_vec3(&track.scale_keys, t)
                    };
                    Mat4::from_scale_rotation_translation(scale, rotation, translation)
                } else {
                    skeleton.rest_pose_local[i]
                }
            } else {
                skeleton.rest_pose_local[i]
            };
            local_matrices.push(mat);
        }
        local_matrices
    }
}

/// Linearly interpolate Vec3 keyframes at time t.
fn sample_vec3(keys: &[BakedKey<Vec3>], t: f64) -> Vec3 {
    if keys.is_empty() {
        return Vec3::ZERO;
    }
    if keys.len() == 1 || t <= keys[0].time {
        return keys[0].value;
    }
    if t >= keys.last().unwrap().time {
        return keys.last().unwrap().value;
    }
    // Binary search for the right interval
    let idx = keys.partition_point(|k| k.time <= t).saturating_sub(1);
    let k0 = &keys[idx];
    let k1 = &keys[(idx + 1).min(keys.len() - 1)];
    let dt = k1.time - k0.time;
    if dt < 1e-10 {
        return k0.value;
    }
    let factor = ((t - k0.time) / dt) as f32;
    k0.value.lerp(k1.value, factor)
}

/// Spherically interpolate Quat keyframes at time t.
fn sample_quat(keys: &[BakedKey<Quat>], t: f64) -> Quat {
    if keys.is_empty() {
        return Quat::IDENTITY;
    }
    if keys.len() == 1 || t <= keys[0].time {
        return keys[0].value;
    }
    if t >= keys.last().unwrap().time {
        return keys.last().unwrap().value;
    }
    let idx = keys.partition_point(|k| k.time <= t).saturating_sub(1);
    let k0 = &keys[idx];
    let k1 = &keys[(idx + 1).min(keys.len() - 1)];
    let dt = k1.time - k0.time;
    if dt < 1e-10 {
        return k0.value;
    }
    let factor = ((t - k0.time) / dt) as f32;
    k0.value.slerp(k1.value, factor)
}

/// Animation state for a single entity with crossfade blending.
pub struct AnimationComponent {
    /// Shared skeleton data.
    pub skeleton: Arc<Skeleton>,
    /// Currently playing animation clip (index into EntityManager::animations).
    pub current_clip: usize,
    /// Playback time in seconds.
    pub playback_time: f64,
    /// Playback speed multiplier.
    pub speed: f64,
    /// Computed bone matrices ready for GPU upload.
    pub bone_matrices: Vec<Mat4>,
    /// Previous clip index for crossfade blending.
    prev_clip: Option<usize>,
    /// Playback time of the previous clip (frozen at transition point).
    prev_playback_time: f64,
    /// Blend factor: 0.0 = fully previous clip, 1.0 = fully current clip.
    blend_factor: f64,
    /// Duration of the crossfade transition in seconds.
    blend_duration: f64,
    /// Whether the current clip should loop. If false, clamps at the last frame.
    pub looping: bool,
}

impl AnimationComponent {
    pub fn new(skeleton: Arc<Skeleton>, clip_index: usize) -> Self {
        let bone_count = skeleton.bone_names.len();
        Self {
            skeleton,
            current_clip: clip_index,
            playback_time: 0.0,
            speed: 1.0,
            bone_matrices: vec![Mat4::IDENTITY; bone_count],
            prev_clip: None,
            prev_playback_time: 0.0,
            blend_factor: 1.0,
            blend_duration: 0.2,
            looping: true,
        }
    }

    /// Transition to a new animation clip with crossfade blending.
    /// `duration` is the blend time in seconds (0.0 for instant switch).
    pub fn transition_to(&mut self, clip_index: usize, duration: f64) {
        if clip_index == self.current_clip {
            return;
        }
        // Save current state as the "previous" clip for blending
        self.prev_clip = Some(self.current_clip);
        self.prev_playback_time = self.playback_time;
        self.blend_factor = 0.0;
        self.blend_duration = duration;
        // Start new clip
        self.current_clip = clip_index;
        self.playback_time = 0.0;
    }

    /// Advance animation and compute bone matrices for GPU upload.
    /// Returns the root bone's animated local translation for root motion extraction.
    pub fn update(&mut self, dt: f64, clips: &[AnimationClip]) -> Option<(Vec3, Quat)> {
        self.playback_time += dt * self.speed;

        if self.current_clip >= clips.len() {
            return None;
        }

        let clip = &clips[self.current_clip];

        // Wrap or clamp playback time depending on looping mode
        if clip.duration > 0.0 {
            if self.looping {
                self.playback_time = self.playback_time % clip.duration;
            } else {
                // Clamp at last frame (hold final pose)
                self.playback_time = self.playback_time.min(clip.duration - 0.001);
            }
        }

        // Advance blend factor toward 1.0 (fully current clip)
        if self.blend_factor < 1.0 {
            if self.blend_duration > 0.0 {
                self.blend_factor = (self.blend_factor + dt / self.blend_duration).min(1.0);
            } else {
                self.blend_factor = 1.0;
            }
            // Also advance the previous clip playback so it doesn't freeze mid-stride
            if let Some(prev_idx) = self.prev_clip {
                if prev_idx < clips.len() {
                    let prev_clip = &clips[prev_idx];
                    self.prev_playback_time += dt * self.speed;
                    if prev_clip.duration > 0.0 {
                        self.prev_playback_time = self.prev_playback_time % prev_clip.duration;
                    }
                }
            }
        }

        // Sample current clip
        let current_poses = clip.sample(self.playback_time, &self.skeleton);

        // If blending, sample previous clip and interpolate
        let final_poses = if self.blend_factor < 1.0 {
            if let Some(prev_idx) = self.prev_clip {
                if prev_idx < clips.len() {
                    let prev_clip = &clips[prev_idx];
                    let prev_poses = prev_clip.sample(self.prev_playback_time, &self.skeleton);
                    let t = self.blend_factor as f32;
                    // Blend per-bone local transforms
                    blend_poses(&prev_poses, &current_poses, t)
                } else {
                    current_poses
                }
            } else {
                current_poses
            }
        } else {
            // Blend complete, clear previous clip
            self.prev_clip = None;
            current_poses
        };

        // Extract root bone (bone 0) animated TRS for root motion
        let root_motion = if !clip.bone_tracks.is_empty() {
            if let Some(ref track) = clip.bone_tracks[0] {
                let t = if clip.duration > 0.0 {
                    self.playback_time % clip.duration
                } else {
                    0.0
                };
                let translation = sample_vec3(&track.translation_keys, t);
                let rotation = sample_quat(&track.rotation_keys, t);
                Some((translation, rotation))
            } else {
                None
            }
        } else {
            None
        };

        // Walk hierarchy: compute global transforms then final matrices
        self.skeleton.compute_animated_matrices(&final_poses, &mut self.bone_matrices);

        root_motion
    }
}

/// Blend two sets of bone local transforms using matrix decomposition.
/// `t` = 0.0 returns `from`, `t` = 1.0 returns `to`.
fn blend_poses(from: &[Mat4], to: &[Mat4], t: f32) -> Vec<Mat4> {
    from.iter().zip(to.iter()).map(|(a, b)| {
        // Decompose both matrices into TRS
        let (a_scale, a_rot, a_trans) = decompose_mat4(a);
        let (b_scale, b_rot, b_trans) = decompose_mat4(b);
        // Interpolate each component
        let translation = a_trans.lerp(b_trans, t);
        let rotation = a_rot.slerp(b_rot, t);
        let scale = a_scale.lerp(b_scale, t);
        Mat4::from_scale_rotation_translation(scale, rotation, translation)
    }).collect()
}

/// Decompose a Mat4 into (scale, rotation, translation).
fn decompose_mat4(m: &Mat4) -> (Vec3, Quat, Vec3) {
    let translation = Vec3::new(m.w_axis.x, m.w_axis.y, m.w_axis.z);
    let sx = Vec3::new(m.x_axis.x, m.x_axis.y, m.x_axis.z).length();
    let sy = Vec3::new(m.y_axis.x, m.y_axis.y, m.y_axis.z).length();
    let sz = Vec3::new(m.z_axis.x, m.z_axis.y, m.z_axis.z).length();
    let scale = Vec3::new(sx, sy, sz);
    // Build rotation matrix by removing scale
    let rot_mat = Mat4::from_cols(
        (m.x_axis / sx.max(1e-6)).into(),
        (m.y_axis / sy.max(1e-6)).into(),
        (m.z_axis / sz.max(1e-6)).into(),
        glam::Vec4::W,
    );
    let rotation = Quat::from_mat4(&rot_mat);
    (scale, rotation, translation)
}

/// Central entity manager.
pub struct EntityManager {
    pub entities: SlotMap<EntityKey, EntityBase>,
    pub transforms: SecondaryMap<EntityKey, Transform>,
    pub render: SecondaryMap<EntityKey, RenderComponent>,
    pub animations: SecondaryMap<EntityKey, AnimationComponent>,
    pub skeletons: Vec<Arc<Skeleton>>,
    /// Shared animation clip library.
    pub animation_clips: Vec<AnimationClip>,
    /// The player's character entity (if spawned).
    pub player_entity: Option<EntityKey>,
}

impl EntityManager {
    pub fn new() -> Self {
        Self {
            entities: SlotMap::with_key(),
            transforms: SecondaryMap::new(),
            render: SecondaryMap::new(),
            animations: SecondaryMap::new(),
            skeletons: Vec::new(),
            animation_clips: Vec::new(),
            player_entity: None,
        }
    }

    /// Spawn a new entity, returns its key.
    pub fn spawn(
        &mut self,
        entity_type: EntityType,
        transform: Transform,
    ) -> EntityKey {
        let key = self.entities.insert(EntityBase {
            entity_type,
            active: true,
        });
        self.transforms.insert(key, transform);
        key
    }

    /// Remove an entity and all its components.
    pub fn despawn(&mut self, key: EntityKey) {
        self.entities.remove(key);
        self.transforms.remove(key);
        self.render.remove(key);
        self.animations.remove(key);
    }

    /// Iterate active entities with transforms and render components.
    pub fn renderable_entities(&self) -> impl Iterator<Item = (EntityKey, &Transform, &RenderComponent)> {
        self.render.iter().filter_map(move |(key, render)| {
            let base = self.entities.get(key)?;
            if !base.active {
                return None;
            }
            let transform = self.transforms.get(key)?;
            Some((key, transform, render))
        })
    }
}
