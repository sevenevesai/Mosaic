# Lighting System

Two layers: BFS flood fill for chunk vertex light (sky/block propagation) + DDA point-light torch system (GPU buffer, per-fragment shadows).

## Layer 1: Chunk Vertex Light (BFS Flood Fill)

Each chunk: `light: Box<[u8; 4096]>` - upper 4 bits = sky light (0-15), lower 4 bits = block light (0-15). Packed into vertex `ao_tile_packed.w` as 0-255. Used for basic light level data during meshing.

### Files

- `world/lighting.rs` - `LightEngine` with BFS propagation/removal (12 tests)
- `world/chunk.rs` - `Block::light_emission()`, `Block::is_light_transparent()`
- `world/meshing.rs` - `get_face_light()` samples light for vertices

### Adding Light Sources

1. Add block variant in `Block` enum
2. Set emission in `Block::light_emission()` (torch=14, max=15)
3. Add to `Block::is_light_transparent()` so light passes through it

## Layer 2: DDA Torch Point-Light System (GPU)

Replaced the old flood-fill/jump-flood A-channel approach. Torch positions are collected on the CPU and uploaded to a GPU storage buffer. The fragment shader loops over lights, computes point-light falloff, and DDA-traces a shadow ray to each torch.

### Architecture

```
CPU: Volume fill collects emissive block positions
  → Upload to torch_buffer (storage buffer, max 256 lights)
  → torch_count packed into VoxelVolumeUniform.sun_color.w

GPU Fragment Shader (chunk + character):
  → Loop over torch_lights[0..torch_count]
  → Per torch: distance check → falloff → normal weighting → DDA shadow → flicker
  → Accumulate contributions
```

### Files

| File | Role |
|------|------|
| `renderer/voxel_gi.rs` | `GpuTorchLight` struct, `torch_buffer`, `upload_torch_lights()`, `collect_torch_positions_from_chunks()` |
| `renderer/pipeline.rs` | `dda_trace_point_shadow()`, `sample_torch_light()`, `calculate_held_light()` (chunk shader) |
| `renderer/character_renderer.rs` | `char_dda_torch_shadow()`, `char_sample_torch_light()`, `char_held_light()` |
| `renderer/mod.rs` | GPU fill path torch collection call |
| `settings.rs` | `torch_shadows: bool` (default OFF) |
| `settings_menu.rs` | TORCH SHADOWS toggle in Graphics tab |

### Torch Buffer Format

`@group(3) @binding(4) var<storage, read> torch_lights: array<vec4<f32>>`

Each entry: `[world_x, world_y, world_z, emission_intensity]` where intensity is 0-1 (mapped from `light_emission() * 255 / 15`).

### DDA Shadow Trace (`dda_trace_point_shadow`)

Same Amanatides & Woo algorithm as sun shadows:
- `textureLoad` with integer coords (no interpolation)
- Skips starting voxel (prevents self-shadowing)
- Stops at emissive voxel (returns 1.0 = lit)
- Stops at solid voxel (returns 0.0 = shadow)
- **Mount-block skip**: solid blocks within 1 voxel of the light source are ignored, so a torch's support block doesn't shadow that torch's light
- Semi-transparent blocks (leaves, water) return 0.35

### Point-Light Calculation

Per torch per fragment:
1. Distance check: `dist >= emission * 18.0` → skip
2. Quadratic falloff: `(1 - dist/max_radius)^2`
3. Normal weighting: `max(dot(normal, light_dir), 0.05)` (0.05 min = subtle wraparound)
4. DDA shadow trace (if torch_shadows enabled and dist > 1.5)
5. Per-torch flicker: hash-based phase offset, 3 sine waves
6. Accumulate: `torch_color * falloff * ndotl * emission * flicker * shadow`

### Collection Paths

- **CPU fill** (`fill_voxel_cache`): Returns `light_sources` Vec; caller uploads via `upload_torch_lights()`
- **GPU fill** (`gpu_fill_volume`): `collect_torch_positions_from_chunks()` scans chunks on CPU, uploads to buffer

### Character Integration

Character renderer (`character_renderer.rs`) has its own copies of torch functions (prefixed `char_`). Uses 1.2 block start offset to escape character's own voxels. Same bind group (group 3 binding 4).

## Gotchas

- **torch_count in sun_color.w** — Not a separate uniform field. Read as `i32(voxel_volume.sun_color.w)` in shader
- **Mount-block skip range** is 1 voxel on each axis (Manhattan). Don't increase or torches behind walls will leak
- **MAX_TORCH_LIGHTS = 256** — If more emissive blocks exist in the 256^3 volume, excess are silently dropped
- **Character torch offset** — Start ray 1.2 blocks along light_dir to avoid character self-shadowing (vs 0.02 normal offset for terrain)
- **Held light** uses known player position (no buffer lookup needed). Same DDA shadow, quartic falloff
- **BFS flood fill still exists** for chunk vertex light data, but the torch visual lighting in the fragment shader is entirely from the GPU buffer system
- **torch_shadows toggle** (bit 0 of `volume_origin.w`) controls whether DDA shadow rays are traced. OFF = unshadowed point lights (cheaper)
