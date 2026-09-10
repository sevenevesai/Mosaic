# Lighting design

## Mechanism (from .claude/rules/lighting-system.md)

### Layer 1: chunk vertex light

`LightEngine` in `src/world/lighting.rs` propagates sky and block light by BFS flood fill and
removal, into each chunk's packed `light` array (11 tests). `get_face_light()` in
`world/meshing.rs` samples it per vertex and packs the level into `ao_tile_packed.w` as 0-255.

Cross-chunk propagation needs neighbour chunk access and triggers a remesh of the neighbours,
which is why a single torch placement can queue several chunks.

### Layer 2: DDA torch point lights

This replaced an earlier flood-fill and jump-flood approach that wrote into the volume's alpha
channel. Torch positions are collected on the CPU and uploaded to a GPU storage buffer; the
fragment shader loops over them.

```
CPU: volume fill collects emissive block positions
  -> upload_torch_lights() writes torch_buffer (max 256)
  -> torch_count packed into VoxelVolumeUniform.sun_color.w

GPU (chunk and character fragment shaders):
  -> loop torch_lights[0..torch_count]
  -> distance cull -> falloff -> normal weighting -> DDA shadow -> flicker -> accumulate
```

Two collection paths exist: the CPU fill (`fill_voxel_cache`) returns a `light_sources` vector
that the caller uploads, while the GPU fill (`gpu_fill_volume`) has no CPU-side voxel data, so
`collect_torch_positions_from_chunks()` scans chunks separately. A change to emissive
collection has to be made in both or torches vanish under one fill path only.

### Per-torch calculation

1. Cull when `dist >= emission * 18.0`.
2. Quadratic falloff `(1 - dist / max_radius)^2`.
3. Normal weighting `max(dot(normal, light_dir), 0.05)` — the 0.05 floor gives faces turned
   away a subtle wraparound instead of a hard terminator.
4. DDA shadow trace when `torch_shadows` is on and `dist > 1.5`.
5. Hash-based per-torch flicker, three sine waves with a phase offset per position.
6. Accumulate `torch_color * falloff * ndotl * emission * flicker * shadow`.

### `dda_trace_point_shadow`

Amanatides & Woo, the same as the sun shadow trace: `textureLoad` with integer coordinates,
starting voxel skipped, returns 1.0 on reaching an emissive voxel, 0.0 on a solid one, and
0.35 for semi-transparent blocks such as leaves and water. Solid blocks within 1 voxel of the
light source are skipped so the torch's mount does not shadow it.

Held light needs no buffer lookup, since the player position is already known; it uses the
same shadow trace with a quartic falloff.

### Files

| File | Role |
|---|---|
| `src/world/lighting.rs` | `LightEngine`, BFS propagation and removal |
| `src/renderer/voxel_gi.rs` | `GpuTorchLight`, `torch_buffer`, upload and collection |
| `src/renderer/pipeline.rs` | `dda_trace_point_shadow()`, `sample_torch_light()`, `calculate_held_light()` |
| `src/renderer/character_renderer.rs` | `char_` copies of the same three functions |
| `src/settings.rs`, `src/settings_menu.rs` | `torch_shadows` flag and TORCH SHADOWS toggle |
