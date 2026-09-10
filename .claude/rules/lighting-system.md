---
paths:
  - "src/world/lighting.rs"
  - "src/renderer/voxel_gi.rs"
  - "src/renderer/pipeline.rs"
  - "src/renderer/character_renderer.rs"
---

# Lighting

Two independent layers. `world/lighting.rs` runs a BFS flood fill that fills each chunk's
`light: Box<[u8; 4096]>` — sky level in the upper nibble, block level in the lower, both
0-15 — and meshing bakes it into the vertex. Visible torch lighting does not come from that
array: it comes from a GPU storage buffer of point lights that the fragment shaders sample
directly. Mechanism: `docs/lighting-design.md`.

## Adding a light source

1. Add the variant to the `Block` enum in `world/chunk.rs`.
2. Give it an emission in `Block::light_emission()` — Torch is 14, CrystalQuartz 12,
   CrystalAmethyst 10, UraniumOre 8, maximum 15.
3. Add it to `Block::is_light_transparent()` so light passes through it rather than stopping
   at it.

## Torch buffer

`@group(3) @binding(4) var<storage, read> torch_lights: array<vec4<f32>>`, each entry
`[world_x, world_y, world_z, emission]` with emission normalized to 0-1. Both the chunk and
character shaders read the same binding.

## Gotchas

- `torch_count` is packed into `sun_color.w` of `VoxelVolumeUniform`, not carried as its own
  field. Shaders read it as `i32(voxel_volume.sun_color.w)`.
- `MAX_TORCH_LIGHTS` is 256. Emissive blocks beyond that inside the volume are dropped
  silently, so a torch farm loses lights with no error.
- The mount-block skip ignores solid blocks within 1 voxel (Manhattan) of a light, so a
  torch's own support block does not shadow it. Widening that range leaks torchlight through
  walls.
- Character torch rays start 1.2 blocks along the light direction to clear the character's own
  voxels, against 0.02 along the normal for terrain. The character copies of the functions are
  prefixed `char_` and drift from the terrain versions if only one is edited.
- `torch_shadows` (bit 0 of `volume_origin.w`, `settings.rs` default false) decides whether
  shadow rays are traced at all. Off means unshadowed point lights, which is the cheap path.
- The BFS array is still the source for vertex light data and for gameplay light queries.
  Deleting it because torches no longer read it breaks both.
