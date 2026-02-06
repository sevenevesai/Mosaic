//! FBX model loading via ufbx.
//!
//! Extracts mesh data (positions, normals, UVs, bone weights) and skeleton
//! hierarchy from FBX files for GPU rendering.

use std::path::Path;
use std::sync::Arc;
use glam::{Mat4, Vec3, Vec4};

use super::{Skeleton, AnimationClip, BoneTrack, BakedKey};

/// Maximum bone influences per vertex (standard for real-time rendering).
const MAX_WEIGHTS_PER_VERTEX: usize = 4;

/// GPU-ready vertex data extracted from an FBX mesh.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CharacterVertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Surface normal.
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub tex_coords: [f32; 2],
    /// Bone indices (up to 4 bones).
    pub bone_indices: [u8; 4],
    /// Bone weights (normalized, sum to 255).
    pub bone_weights: [u8; 4],
    /// Index into texture array for per-part textures.
    pub texture_layer: u32,
}

/// Result of loading an FBX model.
pub struct LoadedModel {
    pub vertices: Vec<CharacterVertex>,
    pub indices: Vec<u32>,
    pub skeleton: Option<Arc<Skeleton>>,
}

/// Load an FBX file and extract mesh + skeleton data.
pub fn load_fbx(path: &Path) -> Result<LoadedModel, String> {
    let path_str = path.to_str()
        .ok_or_else(|| format!("Invalid path: {}", path.display()))?;

    let opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(path_str, opts)
        .map_err(|e| format!("Failed to load FBX '{}': {:?}", path.display(), e))?;

    // Find the first mesh in the scene
    let mesh = scene.meshes.first()
        .ok_or_else(|| format!("No mesh found in FBX '{}'", path.display()))?;

    // Extract skeleton if skinned
    let skeleton = extract_skeleton(mesh);

    // Extract mesh data
    let (vertices, indices) = extract_mesh_data(mesh)?;

    Ok(LoadedModel {
        vertices,
        indices,
        skeleton: skeleton.map(Arc::new),
    })
}

/// Extract skeleton hierarchy from the mesh's skin deformers.
fn extract_skeleton(mesh: &ufbx::Mesh) -> Option<Skeleton> {
    let skin = mesh.skin_deformers.first()?;

    let cluster_count = skin.clusters.len();
    if cluster_count == 0 {
        return None;
    }

    let mut bone_names = Vec::with_capacity(cluster_count);
    let mut parent_indices = vec![-1i32; cluster_count];
    let mut inverse_bind_matrices = Vec::with_capacity(cluster_count);
    let mut rest_pose_local = Vec::with_capacity(cluster_count);

    // Build a mapping from node element_id to cluster index for parent lookup
    let mut node_to_cluster: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for (i, cluster) in skin.clusters.iter().enumerate() {
        if let Some(ref bone_node) = cluster.bone_node {
            // Store the node -> cluster mapping
            node_to_cluster.insert(bone_node.element.element_id, i);

            // Bone name
            bone_names.push(bone_node.element.name.to_string());

            // Inverse bind matrix (geometry_to_bone)
            inverse_bind_matrices.push(ufbx_matrix_to_glam(&cluster.geometry_to_bone));

            // Rest pose local transform from node
            let local = &bone_node.local_transform;
            let translation = Vec3::new(
                local.translation.x as f32,
                local.translation.y as f32,
                local.translation.z as f32,
            );
            let rotation = glam::Quat::from_xyzw(
                local.rotation.x as f32,
                local.rotation.y as f32,
                local.rotation.z as f32,
                local.rotation.w as f32,
            );
            let scale = Vec3::new(
                local.scale.x as f32,
                local.scale.y as f32,
                local.scale.z as f32,
            );
            rest_pose_local.push(
                Mat4::from_scale_rotation_translation(scale, rotation, translation)
            );
        } else {
            // No bone node - use identity
            bone_names.push(format!("bone_{}", i));
            inverse_bind_matrices.push(Mat4::IDENTITY);
            rest_pose_local.push(Mat4::IDENTITY);
        }
    }

    // Resolve parent indices (second pass after all nodes are mapped)
    for (i, cluster) in skin.clusters.iter().enumerate() {
        if let Some(ref bone_node) = cluster.bone_node {
            if let Some(ref parent_node) = bone_node.parent {
                if let Some(&parent_idx) = node_to_cluster.get(&parent_node.element.element_id) {
                    parent_indices[i] = parent_idx as i32;
                }
            }
        }
    }

    Some(Skeleton {
        bone_names,
        parent_indices,
        inverse_bind_matrices,
        rest_pose_local,
    })
}

/// Extract triangulated mesh data with skinning weights.
fn extract_mesh_data(
    mesh: &ufbx::Mesh,
) -> Result<(Vec<CharacterVertex>, Vec<u32>), String> {
    let skin = mesh.skin_deformers.first();

    let mut vertices: Vec<CharacterVertex> = Vec::with_capacity(mesh.num_triangles * 3);
    let mut tri_indices_buf: Vec<u32> = vec![0u32; mesh.max_face_triangles * 3];

    // Triangulate each face and extract vertex data
    for face in mesh.faces.iter() {
        let num_tris = mesh.triangulate_face(&mut tri_indices_buf, *face) as usize;

        for i in 0..(num_tris * 3) {
            let index = tri_indices_buf[i] as usize;

            // Position
            let pos = mesh.vertex_position[index];
            let position = [pos.x as f32, pos.y as f32, pos.z as f32];

            // Normal (may not exist - check values count)
            let normal = if mesh.vertex_normal.values.len() > 0 {
                let n = mesh.vertex_normal[index];
                [n.x as f32, n.y as f32, n.z as f32]
            } else {
                [0.0, 1.0, 0.0]
            };

            // UV coordinates
            let tex_coords = if mesh.vertex_uv.values.len() > 0 {
                let uv = mesh.vertex_uv[index];
                [uv.x as f32, uv.y as f32]
            } else {
                [0.0, 0.0]
            };

            // Bone weights
            let (bone_indices, bone_weights) = if let Some(ref skin) = skin {
                extract_skin_weights(mesh, skin, index)
            } else {
                ([0u8; 4], [255u8, 0, 0, 0])
            };

            vertices.push(CharacterVertex {
                position,
                normal,
                tex_coords,
                bone_indices,
                bone_weights,
                texture_layer: 0,
            });
        }
    }

    // Generate deduplicated index buffer using ufbx
    let num_index_values = vertices.len();
    let mut indices = vec![0u32; num_index_values];

    let mut streams = [ufbx::VertexStream::new(&mut vertices)];
    let allocator = ufbx::AllocatorOpts::default();
    let num_vertices = ufbx::generate_indices(&mut streams, &mut indices, allocator)
        .map_err(|e| format!("Failed to generate indices: {:?}", e))?;
    vertices.truncate(num_vertices);

    Ok((vertices, indices))
}

/// Extract bone indices and weights for a single vertex.
fn extract_skin_weights(
    mesh: &ufbx::Mesh,
    skin: &ufbx::SkinDeformer,
    index: usize,
) -> ([u8; 4], [u8; 4]) {
    let vertex_ix = mesh.vertex_indices[index] as usize;

    if vertex_ix >= skin.vertices.len() {
        return ([0; 4], [255, 0, 0, 0]);
    }

    let skin_vertex = &skin.vertices[vertex_ix];
    let num_weights = (skin_vertex.num_weights as usize).min(MAX_WEIGHTS_PER_VERTEX);

    let mut raw_indices = [0u32; 4];
    let mut raw_weights = [0.0f32; 4];
    let mut total_weight = 0.0f32;

    for i in 0..num_weights {
        let w = &skin.weights[skin_vertex.weight_begin as usize + i];
        raw_indices[i] = w.cluster_index;
        raw_weights[i] = w.weight as f32;
        total_weight += w.weight as f32;
    }

    // Renormalize
    if total_weight > 0.0 {
        for i in 0..num_weights {
            raw_weights[i] /= total_weight;
        }
    } else {
        raw_weights[0] = 1.0;
    }

    // Pack into u8 (0-255 range)
    let mut packed_indices = [0u8; 4];
    let mut packed_weights = [0u8; 4];
    let mut weight_sum = 0u16;

    for i in 0..MAX_WEIGHTS_PER_VERTEX {
        packed_indices[i] = raw_indices[i].min(255) as u8;
        let w = (raw_weights[i] * 255.0).round() as u8;
        packed_weights[i] = w;
        weight_sum += w as u16;
    }

    // Ensure weights sum to 255 (fix rounding)
    if weight_sum > 0 && weight_sum != 255 {
        let diff = 255i16 - weight_sum as i16;
        let max_idx = packed_weights.iter()
            .enumerate()
            .max_by_key(|(_, &w)| w)
            .map(|(i, _)| i)
            .unwrap_or(0);
        packed_weights[max_idx] = (packed_weights[max_idx] as i16 + diff).max(0) as u8;
    }

    (packed_indices, packed_weights)
}

/// Convert a ufbx Matrix to glam Mat4.
/// ufbx Matrix is a 3x4 affine matrix (3 rows, 4 columns):
/// m00 m01 m02 m03
/// m10 m11 m12 m13
/// m20 m21 m22 m23
/// Column-major for glam:
fn ufbx_matrix_to_glam(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0),
        Vec4::new(m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0),
        Vec4::new(m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0),
        Vec4::new(m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0),
    )
}

/// Load animation clips from an FBX file.
/// Returns all animation stacks found as AnimationClips, plus the skeleton if present.
/// The skeleton from the animation file is used to map bone names to indices.
pub fn load_animations(
    path: &Path,
    target_skeleton: &Skeleton,
) -> Result<Vec<AnimationClip>, String> {
    let path_str = path.to_str()
        .ok_or_else(|| format!("Invalid path: {}", path.display()))?;

    let opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(path_str, opts)
        .map_err(|e| format!("Failed to load animation FBX '{}': {:?}", path.display(), e))?;

    let mut clips = Vec::new();

    // Process each animation stack in the file
    for anim_stack in &scene.anim_stacks {
        let clip_name = anim_stack.element.name.to_string();

        // Bake the animation to sampled keyframes
        let bake_opts = ufbx::BakeOpts::default();
        let baked = ufbx::bake_anim(&scene, &anim_stack.anim, bake_opts)
            .map_err(|e| format!("Failed to bake animation '{}': {:?}", clip_name, e))?;

        let nominal_duration = baked.playback_duration;

        // Build a name -> target_bone_index map from the target skeleton
        let mut name_to_bone: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (i, name) in target_skeleton.bone_names.iter().enumerate() {
            name_to_bone.insert(name.as_str(), i);
        }

        // Initialize bone tracks (one per target skeleton bone)
        let bone_count = target_skeleton.bone_names.len();
        let mut bone_tracks: Vec<Option<BoneTrack>> = (0..bone_count).map(|_| None).collect();

        // Map baked nodes to skeleton bones by name
        for baked_node in &baked.nodes {
            let node_index = baked_node.typed_id as usize;
            if node_index >= scene.nodes.len() {
                continue;
            }
            let scene_node = &scene.nodes[node_index];
            let node_name = scene_node.element.name.to_string();

            // Find matching bone in target skeleton
            if let Some(&bone_idx) = name_to_bone.get(node_name.as_str()) {
                bone_tracks[bone_idx] = Some(BoneTrack {
                    translation_keys: baked_node.translation_keys.iter()
                        .map(|k| BakedKey {
                            time: k.time,
                            value: Vec3::new(k.value.x as f32, k.value.y as f32, k.value.z as f32),
                        })
                        .collect(),
                    rotation_keys: baked_node.rotation_keys.iter()
                        .map(|k| BakedKey {
                            time: k.time,
                            value: glam::Quat::from_xyzw(
                                k.value.x as f32,
                                k.value.y as f32,
                                k.value.z as f32,
                                k.value.w as f32,
                            ),
                        })
                        .collect(),
                    scale_keys: baked_node.scale_keys.iter()
                        .map(|k| BakedKey {
                            time: k.time,
                            value: Vec3::new(k.value.x as f32, k.value.y as f32, k.value.z as f32),
                        })
                        .collect(),
                });
            }
        }

        // Normalize keyframe times: subtract root bone's first rotation key time
        // since ufbx gives absolute FBX timeline times.
        let min_time = bone_tracks.get(0)
            .and_then(|t| t.as_ref())
            .and_then(|t| t.rotation_keys.first().map(|k| k.time))
            .unwrap_or(0.0);

        if min_time > 0.1 {
            tracing::info!("Animation '{}': normalizing times (subtracting {:.2}s)", clip_name, min_time);
            for track in bone_tracks.iter_mut().flatten() {
                for k in &mut track.translation_keys { k.time -= min_time; }
                for k in &mut track.rotation_keys { k.time -= min_time; }
                for k in &mut track.scale_keys { k.time -= min_time; }
            }
        }

        // Detect actual animation duration by finding where bones stop moving.
        // Many FBX pipelines embed the same timeline length for all clips,
        // causing animations to "freeze" on their last pose for the remainder.
        let clip_duration = detect_animation_end(&bone_tracks, nominal_duration);

        // Trim keyframes beyond detected duration
        for track in bone_tracks.iter_mut().flatten() {
            track.translation_keys.retain(|k| k.time <= clip_duration + 0.02);
            track.rotation_keys.retain(|k| k.time <= clip_duration + 0.02);
            track.scale_keys.retain(|k| k.time <= clip_duration + 0.02);
        }

        let matched = bone_tracks.iter().filter(|t| t.is_some()).count();
        tracing::info!(
            "Animation '{}': duration={:.3}s (nominal={:.2}s), {}/{} bones, {} nodes",
            clip_name, clip_duration, nominal_duration, matched, bone_count, baked.nodes.len()
        );

        clips.push(AnimationClip {
            name: clip_name,
            duration: clip_duration,
            bone_tracks,
        });
    }

    Ok(clips)
}

/// Detect where an animation actually ends by finding when bones stop moving.
/// Many FBX exporters pad animations to a fixed timeline length. This function
/// scans rotation tracks (most reliable for detecting motion) and finds the
/// latest time where any bone is still actively changing.
fn detect_animation_end(bone_tracks: &[Option<BoneTrack>], nominal_duration: f64) -> f64 {
    let mut last_active_time = 0.0f64;

    for track in bone_tracks.iter().flatten() {
        // Check rotation keys - most reliable indicator of animation activity
        let keys = &track.rotation_keys;
        if keys.len() < 2 {
            continue;
        }

        // Find the last keyframe that differs meaningfully from the one before it
        for i in (1..keys.len()).rev() {
            let q0 = keys[i - 1].value;
            let q1 = keys[i].value;
            // Quaternion dot product: 1.0 = identical, check if actually different
            let dot = q0.dot(q1).abs();
            if dot < 0.9999 {
                // This keyframe has actual motion
                last_active_time = last_active_time.max(keys[i].time);
                break;
            }
        }

        // Also check translation keys for root motion
        let tkeys = &track.translation_keys;
        if tkeys.len() >= 2 {
            for i in (1..tkeys.len()).rev() {
                let diff = (tkeys[i].value - tkeys[i - 1].value).length();
                if diff > 0.001 {
                    last_active_time = last_active_time.max(tkeys[i].time);
                    break;
                }
            }
        }
    }

    // Use detected end, but clamp to nominal duration as upper bound.
    // Add a small buffer (~1 frame at 30fps) for the final pose to settle.
    if last_active_time > 0.05 {
        let detected = (last_active_time + 0.04).min(nominal_duration);
        tracing::info!("  -> detected end at {:.3}s (nominal {:.2}s)", detected, nominal_duration);
        detected
    } else {
        // No motion detected, use nominal
        nominal_duration
    }
}

/// Diagnostic: print info about what an FBX file contains.
pub fn diagnose_fbx(path: &Path) -> Result<(), String> {
    let path_str = path.to_str()
        .ok_or_else(|| format!("Invalid path: {}", path.display()))?;

    let opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(path_str, opts)
        .map_err(|e| format!("Failed to load FBX '{}': {:?}", path.display(), e))?;

    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    tracing::info!("FBX: {} | {} meshes, {} nodes, {} anims",
        file_name, scene.meshes.len(), scene.nodes.len(), scene.anim_stacks.len());

    for mesh in scene.meshes.iter() {
        let name = &mesh.element.name;
        let skin_count = mesh.skin_deformers.len();
        let has_skin = if skin_count > 0 { "SKINNED" } else { "static" };
        let bone_count = if let Some(skin) = mesh.skin_deformers.first() {
            skin.clusters.len()
        } else {
            0
        };
        let vert_count = mesh.num_vertices;
        tracing::info!("  -> Mesh '{}' {} verts, {} bones, {}",
            name, vert_count, bone_count, has_skin);
    }

    for stack in scene.anim_stacks.iter() {
        // Get duration via bake
        if let Ok(baked) = ufbx::bake_anim(&scene, &stack.anim, ufbx::BakeOpts::default()) {
            tracing::info!("  -> Anim '{}' {:.2}s, {} nodes",
                stack.element.name, baked.playback_duration, baked.nodes.len());
        }
    }

    Ok(())
}

/// Load skeleton from an FBX file that contains only skeleton data (like EmptyBody.fbx).
/// Also works for any FBX with a skeleton in its scene nodes.
pub fn load_skeleton_from_fbx(path: &Path) -> Result<Skeleton, String> {
    let path_str = path.to_str()
        .ok_or_else(|| format!("Invalid path: {}", path.display()))?;

    let opts = ufbx::LoadOpts::default();
    let scene = ufbx::load_file(path_str, opts)
        .map_err(|e| format!("Failed to load FBX '{}': {:?}", path.display(), e))?;

    // Try to get skeleton from skin deformers on any mesh first
    for mesh in &scene.meshes {
        if let Some(skeleton) = extract_skeleton(mesh) {
            return Ok(skeleton);
        }
    }

    // No skinned mesh — build skeleton from scene node hierarchy
    // Find all nodes that look like bones (have children or specific naming)
    let mut bone_names = Vec::new();
    let mut parent_indices = Vec::new();
    let mut inverse_bind_matrices = Vec::new();
    let mut rest_pose_local = Vec::new();

    // Map node element_id to bone index
    let mut node_to_bone: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for (i, node) in scene.nodes.iter().enumerate() {
        // Skip the root node (index 0 is usually the scene root)
        if i == 0 && node.element.name.is_empty() {
            continue;
        }

        let bone_idx = bone_names.len();
        node_to_bone.insert(node.element.element_id, bone_idx);
        bone_names.push(node.element.name.to_string());

        // Local transform
        let local = &node.local_transform;
        let translation = Vec3::new(
            local.translation.x as f32,
            local.translation.y as f32,
            local.translation.z as f32,
        );
        let rotation = glam::Quat::from_xyzw(
            local.rotation.x as f32,
            local.rotation.y as f32,
            local.rotation.z as f32,
            local.rotation.w as f32,
        );
        let scale = Vec3::new(
            local.scale.x as f32,
            local.scale.y as f32,
            local.scale.z as f32,
        );
        rest_pose_local.push(Mat4::from_scale_rotation_translation(scale, rotation, translation));

        // Inverse bind matrix: compute from node_to_world inverse
        let world_mat = ufbx_matrix_to_glam(&node.node_to_world);
        inverse_bind_matrices.push(world_mat.inverse());

        // Parent index (resolved after all nodes processed)
        parent_indices.push(-1i32);
    }

    // Resolve parent indices
    for (i, node) in scene.nodes.iter().enumerate() {
        if i == 0 && node.element.name.is_empty() {
            continue;
        }
        if let Some(&bone_idx) = node_to_bone.get(&node.element.element_id) {
            if let Some(ref parent) = node.parent {
                if let Some(&parent_bone_idx) = node_to_bone.get(&parent.element.element_id) {
                    parent_indices[bone_idx] = parent_bone_idx as i32;
                }
            }
        }
    }

    if bone_names.is_empty() {
        return Err(format!("No skeleton found in '{}'", path.display()));
    }

    tracing::info!("Loaded skeleton from '{}': {} bones", path.display(), bone_names.len());

    Ok(Skeleton {
        bone_names,
        parent_indices,
        inverse_bind_matrices,
        rest_pose_local,
    })
}

/// Compute the scale factor to convert body part mesh coordinates into skeleton bind-pose space.
///
/// The body part FBX files and the skeleton FBX often use different unit scales.
/// We detect this by comparing the FBX unit_meters settings from both files.
/// Fallback: measure the vertex height range vs skeleton bone height range.
fn compute_mesh_to_skeleton_scale(
    models_dir: &Path,
    skeleton: &Skeleton,
    parts: &[BodyPart],
) -> f32 {
    // Strategy 1: Compare FBX scene unit_meters between body part and skeleton files.
    if let Some(part) = parts.first() {
        let mesh_path = models_dir.join(part.folder).join(part.file);
        if let Some(path_str) = mesh_path.to_str() {
            if let Ok(scene) = ufbx::load_file(path_str, ufbx::LoadOpts::default()) {
                let mesh_unit = scene.settings.unit_meters as f32;
                tracing::info!("Body part FBX unit_meters: {:.6}", mesh_unit);

                let skel_path = "assets/char/animations/v1/Avatar_for_Animation.fbx";
                if let Ok(skel_scene) = ufbx::load_file(skel_path, ufbx::LoadOpts::default()) {
                    let skel_unit = skel_scene.settings.unit_meters as f32;
                    tracing::info!("Skeleton FBX unit_meters: {:.6}", skel_unit);

                    if mesh_unit > 0.0 && skel_unit > 0.0 && (mesh_unit - skel_unit).abs() > 1e-6 {
                        let scale = mesh_unit / skel_unit;
                        tracing::info!("Unit scale ratio (mesh/skel): {:.4}", scale);
                        return scale;
                    }
                }

                // Strategy 2: Check root/mesh node transforms for baked scale.
                for node in scene.nodes.iter() {
                    if node.mesh.is_some() {
                        let geo = &node.geometry_transform;
                        let geo_scale = Vec3::new(
                            geo.scale.x as f32,
                            geo.scale.y as f32,
                            geo.scale.z as f32,
                        );
                        if (geo_scale - Vec3::ONE).length() > 0.01 {
                            tracing::info!("Mesh node geometry scale: [{:.3}, {:.3}, {:.3}]", geo_scale.x, geo_scale.y, geo_scale.z);
                            return geo_scale.y;
                        }
                        break;
                    }
                }
            }
        }
    }

    // Strategy 3: Measure empirically from vertex Y range vs skeleton bone Y range.
    let bone_positions = skeleton.get_bone_world_positions();
    let skel_min_y = bone_positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let skel_max_y = bone_positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let skel_height = skel_max_y - skel_min_y;

    if skel_height < 0.1 {
        tracing::warn!("Skeleton has near-zero height range, using scale 1.0");
        return 1.0;
    }

    // Deduplicate mesh files to avoid loading the same file twice (mirrored parts)
    let mut seen_files = std::collections::HashSet::new();
    let mut mesh_min_y = f32::MAX;
    let mut mesh_max_y = f32::MIN;
    for part in parts {
        let key = format!("{}/{}", part.folder, part.file);
        if !seen_files.insert(key) { continue; }
        let mesh_path = models_dir.join(part.folder).join(part.file);
        if !mesh_path.exists() { continue; }
        if let Ok(model) = load_fbx(&mesh_path) {
            for v in &model.vertices {
                mesh_min_y = mesh_min_y.min(v.position[1]);
                mesh_max_y = mesh_max_y.max(v.position[1]);
            }
        }
    }

    let mesh_height = mesh_max_y - mesh_min_y;
    if mesh_height < 0.1 {
        tracing::warn!("Mesh has near-zero height range, using scale 1.0");
        return 1.0;
    }

    let scale = skel_height / mesh_height;
    tracing::info!(
        "Empirical scale: skel Y [{:.1}, {:.1}] h={:.1}, mesh Y [{:.1}, {:.1}] h={:.1}, scale={:.4}",
        skel_min_y, skel_max_y, skel_height,
        mesh_min_y, mesh_max_y, mesh_height,
        scale,
    );
    scale
}

/// Result of merging multiple body part meshes.
pub struct MergedCharacterMesh {
    pub vertices: Vec<CharacterVertex>,
    pub indices: Vec<u32>,
    /// Unique texture paths (indices match texture_layer in vertices).
    pub texture_paths: Vec<std::path::PathBuf>,
}

/// Body part definition for mesh merging.
pub struct BodyPart {
    /// Folder name under assets/char/models/ (e.g., "Root", "Chest")
    pub folder: &'static str,
    /// FBX file name within the folder
    pub file: &'static str,
    /// Bone name(s) to bind to (first match wins, fallback to root)
    pub bone_names: &'static [&'static str],
    /// Optional texture file (relative to assets/char/textures/)
    pub texture: Option<&'static str>,
    /// Mirror this part on X axis (for left-side limbs)
    pub mirror: bool,
}

/// Per-body-part position/scale adjustment for the in-game editor.
#[derive(Clone, Debug)]
pub struct BodyPartAdjustment {
    /// Offset applied after scaling (in skeleton bind-pose space).
    pub offset: Vec3,
    /// Scale multiplier (applied to the part's shape around the bone).
    pub scale: f32,
}

impl Default for BodyPartAdjustment {
    fn default() -> Self {
        Self { offset: Vec3::ZERO, scale: 1.0 }
    }
}

/// Names for all 13 body parts (9 unique + 4 right-side), matching HUMANOID_BODY_PARTS order.
pub const BODY_PART_NAMES: &[&str] = &[
    "Root", "Spine", "Chest", "Head", "Hair",
    "Hip_L", "Knee_L", "Shoulder_L", "Elbow_L",
    "Hip_R", "Knee_R", "Shoulder_R", "Elbow_R",
];

/// Default body part adjustments baked from in-game editor tuning.
/// Right-side defaults are initialized as X-mirrored left-side values;
/// they can be independently fine-tuned since skeleton bones may not be perfectly symmetric.
pub fn default_adjustments() -> Vec<BodyPartAdjustment> {
    vec![
        // Center/spine parts
        BodyPartAdjustment { offset: Vec3::new( 0.0, -2.0,  0.0), scale: 1.0 }, // Root
        BodyPartAdjustment { offset: Vec3::new( 0.0, -1.5,  0.0), scale: 1.0 }, // Spine
        BodyPartAdjustment { offset: Vec3::new( 0.0,  0.0,  0.0), scale: 1.0 }, // Chest
        BodyPartAdjustment { offset: Vec3::new( 0.0,  1.0,  0.0), scale: 1.0 }, // Head
        BodyPartAdjustment { offset: Vec3::new( 0.0,  0.0,  0.0), scale: 1.0 }, // Hair
        // Left side
        BodyPartAdjustment { offset: Vec3::new(-2.5, -3.0, -1.0), scale: 1.0 }, // Hip_L
        BodyPartAdjustment { offset: Vec3::new(-2.5, -3.0, -1.0), scale: 1.0 }, // Knee_L
        BodyPartAdjustment { offset: Vec3::new( 0.0, -4.0,  0.5), scale: 1.0 }, // Shoulder_L
        BodyPartAdjustment { offset: Vec3::new( 2.0, -4.5,  1.0), scale: 1.0 }, // Elbow_L
        // Right side (independently adjustable)
        BodyPartAdjustment { offset: Vec3::new( 2.5, -3.0, -0.5), scale: 1.0 }, // Hip_R
        BodyPartAdjustment { offset: Vec3::new( 2.5, -3.0,  0.0), scale: 1.0 }, // Knee_R
        BodyPartAdjustment { offset: Vec3::new(-0.5, -4.0,  0.0), scale: 1.0 }, // Shoulder_R
        BodyPartAdjustment { offset: Vec3::new( 0.0, -4.0,  1.5), scale: 1.0 }, // Elbow_R
    ]
}

/// Default humanoid body parts for a complete character.
/// Uses the Avatar_for_Animation skeleton naming: Root_M, Hip_R/L, Spine1_M, Chest_M, etc.
/// Paired limbs (Hip, Shoulder, Elbow, Knee) have both right and left entries.
pub const HUMANOID_BODY_PARTS: &[BodyPart] = &[
    // Center/spine parts - body texture for torso parts, separate textures for head/hair
    BodyPart { folder: "Root", file: "Root020.fbx", bone_names: &["Root_M", "Root"], texture: Some("Characters/020.png"), mirror: false },
    BodyPart { folder: "Spine", file: "Spine020.fbx", bone_names: &["Spine1_M", "Spine"], texture: Some("Characters/020.png"), mirror: false },
    BodyPart { folder: "Chest", file: "Chest020.fbx", bone_names: &["Chest_M", "Chest"], texture: Some("Characters/020.png"), mirror: false },
    BodyPart { folder: "Head", file: "Head001.fbx", bone_names: &["Head_M", "Head"], texture: Some("Head/Head001.png"), mirror: false },
    BodyPart { folder: "Hair", file: "Hair001.fbx", bone_names: &["Head_M", "Head"], texture: Some("Hair/Hair001.png"), mirror: false },
    // Left side limbs (FBX meshes are at positive X, which matches left side)
    BodyPart { folder: "Hip", file: "Hip020.fbx", bone_names: &["Hip_L"], texture: Some("Characters/020.png"), mirror: false },
    BodyPart { folder: "Knee", file: "Knee020.fbx", bone_names: &["Knee_L"], texture: Some("Characters/020.png"), mirror: false },
    BodyPart { folder: "Shoulder", file: "Shoulder020.fbx", bone_names: &["Shoulder_L"], texture: Some("Characters/020.png"), mirror: false },
    BodyPart { folder: "Elbow", file: "Elbow020.fbx", bone_names: &["Elbow_L"], texture: Some("Characters/020.png"), mirror: false },
    // Right side limbs (mirrored from left side meshes to negative X)
    BodyPart { folder: "Hip", file: "Hip020.fbx", bone_names: &["Hip_R"], texture: Some("Characters/020.png"), mirror: true },
    BodyPart { folder: "Knee", file: "Knee020.fbx", bone_names: &["Knee_R"], texture: Some("Characters/020.png"), mirror: true },
    BodyPart { folder: "Shoulder", file: "Shoulder020.fbx", bone_names: &["Shoulder_R"], texture: Some("Characters/020.png"), mirror: true },
    BodyPart { folder: "Elbow", file: "Elbow020.fbx", bone_names: &["Elbow_R"], texture: Some("Characters/020.png"), mirror: true },
];

/// Find bone index by name. Tries exact match first (case-insensitive),
/// then falls back to substring match for flexibility.
fn find_bone_index(skeleton: &Skeleton, names: &[&str]) -> Option<u8> {
    // First pass: exact match (case-insensitive)
    for name in names {
        let needle = name.to_lowercase();
        for (i, bone_name) in skeleton.bone_names.iter().enumerate() {
            if bone_name.to_lowercase() == needle {
                return Some(i.min(255) as u8);
            }
        }
    }
    // Second pass: substring match as fallback
    for name in names {
        let needle = name.to_lowercase();
        for (i, bone_name) in skeleton.bone_names.iter().enumerate() {
            if bone_name.to_lowercase().contains(&needle) {
                return Some(i.min(255) as u8);
            }
        }
    }
    None
}

/// Load and merge multiple body part meshes into a single skinned mesh.
///
/// Each body part is assigned 100% weight to its designated bone(s).
/// Vertices are scaled from mesh space into skeleton bind-pose space so that
/// the standard skinning equation (global * inverse_bind * vertex) works correctly.
///
/// If `adjustments` is provided, per-part offset and scale tweaks are applied.
/// Adjustments index 0..8 map to the 9 unique parts; mirrored parts (9..12) auto-mirror.
pub fn merge_body_parts(
    models_dir: &Path,
    skeleton: &Skeleton,
    parts: &[BodyPart],
    adjustments: Option<&[BodyPartAdjustment]>,
) -> Result<MergedCharacterMesh, String> {
    let mut all_vertices: Vec<CharacterVertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut texture_paths: Vec<std::path::PathBuf> = Vec::new();
    // Map texture path to layer index for deduplication
    let mut texture_to_layer: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

    // Compute mesh-to-skeleton scale factor.
    //
    // Body part FBX meshes and the skeleton FBX use different unit scales.
    // We determine the correct scale by loading a reference mesh (Root) and
    // comparing its geometry_to_node transform, or by measuring the ratio
    // between known mesh positions and skeleton bind-pose positions.
    //
    // We use the Root mesh center vs Root_M bone as the primary reference,
    // combined with the Head mesh vs Head_M bone to get two data points
    // and derive a uniform scale + offset if needed.
    //
    // For now: load the first body part's FBX scene to read its root node scale.
    let mesh_to_skeleton_scale = compute_mesh_to_skeleton_scale(models_dir, skeleton, parts);
    tracing::info!("Mesh-to-skeleton scale factor: {:.4}", mesh_to_skeleton_scale);

    for (part_index, part) in parts.iter().enumerate() {
        // Each part has its own adjustment slot (13 total: 9 unique + 4 right-side).
        // Right-side parts (9..12) have independent adjustments for asymmetric tuning.
        let adj = adjustments
            .and_then(|a| a.get(part_index))
            .cloned()
            .unwrap_or_default();

        let mesh_path = models_dir.join(part.folder).join(part.file);
        if !mesh_path.exists() {
            tracing::warn!("Body part mesh not found: {}", mesh_path.display());
            continue;
        }

        // Find the bone to bind this part to
        let bone_idx = find_bone_index(skeleton, part.bone_names).unwrap_or_else(|| {
            tracing::warn!("No bone match for {}, using bone 0", part.folder);
            0
        });

        // Load the mesh
        let model = match load_fbx(&mesh_path) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to load {}: {}", mesh_path.display(), e);
                continue;
            }
        };

        // Determine texture layer for this part (deduplicated)
        let texture_layer = if let Some(tex) = part.texture {
            let tex_key = tex.to_string();
            if let Some(&existing_layer) = texture_to_layer.get(&tex_key) {
                existing_layer
            } else {
                let new_layer = texture_paths.len() as u32;
                texture_to_layer.insert(tex_key, new_layer);
                texture_paths.push(std::path::PathBuf::from("assets/char/textures").join(tex));
                new_layer
            }
        } else {
            0 // Default to layer 0 if no texture specified
        };

        let base_vertex = all_vertices.len() as u32;
        let vert_count = model.vertices.len();

        // Compute mesh bounds and center
        let mut mesh_min = Vec3::splat(f32::MAX);
        let mut mesh_max = Vec3::splat(f32::MIN);
        for v in &model.vertices {
            let p = Vec3::from_array(v.position);
            mesh_min = mesh_min.min(p);
            mesh_max = mesh_max.max(p);
        }
        let mesh_center = (mesh_min + mesh_max) * 0.5;

        // Get bone's bind-pose world position
        let bone_bind_transform = skeleton.inverse_bind_matrices
            .get(bone_idx as usize)
            .map(|inv_bind| inv_bind.inverse())
            .unwrap_or(Mat4::IDENTITY);
        let bone_pos = bone_bind_transform.w_axis.truncate();

        let effective_mesh_center = if part.mirror {
            Vec3::new(-mesh_center.x, mesh_center.y, mesh_center.z)
        } else {
            mesh_center
        };

        tracing::info!(
            "  {} center=[{:.1},{:.1},{:.1}] bone=[{:.1},{:.1},{:.1}] scale={:.3}",
            part.file,
            effective_mesh_center.x, effective_mesh_center.y, effective_mesh_center.z,
            bone_pos.x, bone_pos.y, bone_pos.z,
            mesh_to_skeleton_scale,
        );

        // Per-bone centering with scale: each part is centered on its bone pivot,
        // with the part's shape scaled to fit the skeleton's coordinate proportions.
        // Formula: bind_pos = bone_pos + (vertex - mesh_center) * scale
        //
        // This ensures:
        // 1. Each part is centered on its bone pivot (animation rotates correctly)
        // 2. Each part's shape is scaled to fit the skeleton's coordinate range
        // 3. In bind pose (bone_matrix = identity), vertices appear at these positions
        for mut v in model.vertices {
            let mut pos = Vec3::from_array(v.position);
            let mut normal = Vec3::from_array(v.normal);

            // Mirror on X axis for right-side limbs (mirrored parts)
            if part.mirror {
                pos.x = -pos.x;
                normal.x = -normal.x;
            }

            // Center on bone, scale part shape to skeleton proportions.
            // Per-part adjustment: offset shifts the part, scale multiplies the shape.
            // Right-side parts have independent adjustments (no auto X-negation).
            let part_offset = adj.offset;
            let bind_pos = bone_pos + part_offset
                + (pos - effective_mesh_center) * mesh_to_skeleton_scale * adj.scale;

            v.position = bind_pos.to_array();
            // Normal direction is unchanged by uniform scale

            // FBX UV origin is bottom-left, wgpu expects top-left: flip V coordinate
            v.tex_coords[1] = 1.0 - v.tex_coords[1];

            // Override bone weights: 100% to the assigned bone
            v.bone_indices = [bone_idx, 0, 0, 0];
            v.bone_weights = [255, 0, 0, 0];
            v.texture_layer = texture_layer;
            all_vertices.push(v);
        }

        // Add indices (offset by base vertex)
        // For mirrored parts, reverse winding order (swap v1 and v2 of each triangle)
        if part.mirror {
            for chunk in model.indices.chunks(3) {
                if chunk.len() == 3 {
                    all_indices.push(base_vertex + chunk[0]);
                    all_indices.push(base_vertex + chunk[2]);
                    all_indices.push(base_vertex + chunk[1]);
                }
            }
        } else {
            for idx in model.indices {
                all_indices.push(base_vertex + idx);
            }
        }

        let mirror_str = if part.mirror { " [mirrored]" } else { "" };
        tracing::info!(
            "Merged {}/{}: {} verts -> bone {} '{}'{} (bone pos: [{:.1}, {:.1}, {:.1}])",
            part.folder, part.file,
            vert_count,
            bone_idx,
            skeleton.bone_names.get(bone_idx as usize).unwrap_or(&"?".to_string()),
            mirror_str,
            bone_pos.x, bone_pos.y, bone_pos.z,
        );
    }

    if all_vertices.is_empty() {
        return Err("No body parts were successfully loaded".to_string());
    }

    tracing::info!(
        "Merged character: {} total verts, {} indices, {} parts",
        all_vertices.len(), all_indices.len(), texture_paths.len()
    );

    Ok(MergedCharacterMesh {
        vertices: all_vertices,
        indices: all_indices,
        texture_paths,
    })
}
