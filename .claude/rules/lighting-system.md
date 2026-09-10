---
paths:
  - "src/world/lighting.rs"
  - "src/renderer/voxel_gi.rs"
  - "src/renderer/pipeline.rs"
  - "src/renderer/character_renderer.rs"
---

# Lighting

Two layers. `world/lighting.rs` BFS-floods sky and block light into each chunk's
`light: Box<[u8; 4096]>` — sky in the upper nibble, block in the lower, both 0-15 — and
meshing bakes it into the vertex. Visible torch light does not come from that array but from a
GPU storage buffer of point lights the fragment shaders sample directly. Mechanism and the
torch buffer layout: `docs/lighting-design.md`.

## Adding a light source

1. Add the variant to the `Block` enum in `world/chunk.rs`.
2. Give it an emission in `Block::light_emission()` — Torch is 14, CrystalQuartz 12,
   CrystalAmethyst 10, UraniumOre 8, maximum 15.
3. Add it to `Block::is_light_transparent()` so light passes through it rather than stopping
   at it.

## Sun shadows

`shadow_quality` (bits 1-3 of `volume_origin.w`) sets the DDA step budget: OFF traces nothing,
LOW 48, MEDIUM 96, HIGH and ULTRA 128, and ULTRA adds a 2-tap PCF that offsets the ray origin
perpendicular to the sun so the taps straddle voxel boundaries. Rays run only where
`sun_intensity > 0.02` and `ndotl > 0`. The player's own shadow is the analytic
`player_shadow_box` uniform (feet position, half width 0.3) that the chunk, character, grass
and volumetric shaders read; the older voxel-injection path is gone, so do not cast a shadow
by writing into the volume.

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
  torch shadow rays are traced at all. Off means unshadowed point lights, the cheap path.
- The BFS array is still the source for vertex light data and for gameplay light queries.
  Deleting it because torches no longer read it breaks both.
