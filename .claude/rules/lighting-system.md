# Lighting System

Minecraft-style BFS flood fill light propagation. 12 tests including cross-chunk propagation.

## Data Storage

- Each chunk: `light: Box<[u8; 4096]>` - 1 byte per block (16³ = 4096)
- Upper 4 bits = sky light (0-15)
- Lower 4 bits = block light (0-15)
- Vertex packing: Light scaled from 0-15 to 0-255 in `ao_tile_packed.w`

## Key Files

| File | Purpose |
|------|---------|
| `world/lighting.rs` | `LightEngine` with BFS propagation/removal |
| `world/chunk.rs` | `Block::light_emission()`, `Block::is_light_transparent()` |
| `world/meshing.rs` | `get_face_light()` samples light for vertices |
| `renderer/pipeline.rs` | Shader applies warm torch color with gamma curve |

## Light Propagation Algorithm

**Placement:**
1. Set source block light level (e.g., torch = 14)
2. BFS queue starting from source
3. For each neighbor: if transparent and neighbor_light < current_light - 1, set neighbor_light = current_light - 1, enqueue
4. Repeat until queue empty

**Removal:**
1. BFS removal from removed source, collecting affected positions
2. Re-propagate from all nearby light sources

**Cross-chunk:**
- Light propagation checks neighbor chunks when at chunk boundaries
- Requires neighbor chunks to be loaded
- Triggers neighbor chunk remesh when light crosses boundary

## Adding Light Sources

1. Add block variant in `world/chunk.rs`:
   ```rust
   pub enum Block {
       // ...
       Lantern,
   }
   ```

2. Set emission in `Block::light_emission()`:
   ```rust
   pub fn light_emission(&self) -> u8 {
       match self {
           Block::Torch => 14,
           Block::Lantern => 15,  // Brighter than torch
           _ => 0,
       }
   }
   ```

3. Set transparency in `Block::is_light_transparent()`:
   ```rust
   pub fn is_light_transparent(&self) -> bool {
       matches!(self, Block::Air | Block::Water | Block::Torch | Block::Lantern)
   }
   ```

## Shader Integration

From `pipeline.rs` chunk shader:

```wgsl
// Block light uses warm color with gamma curve
let block_light = f32(light_level) / 15.0;
let torch_color = vec3<f32>(1.0, 0.7, 0.4);  // Warm orange
let torch_contribution = pow(block_light, 2.2) * torch_color;
```

## Performance Notes

- BFS is O(n) where n = affected blocks
- Torch radius 14 = ~11K blocks worst case (sphere volume)
- Batched updates recommended for multiple light changes
- Cross-chunk propagation triggers 2 remeshes minimum
