# Water Rendering System

Novel heightfield-based water with smooth animated surface.

## Architecture Overview

Water is NOT stored as blocks. Instead:
- 2D heightfield per chunk (`ChunkWaterData`)
- Each column stores: surface_y, foam factor, flow angle
- Separate mesh generation (`water_meshing.rs`)
- Dedicated render pass after opaque geometry

## Data Structures

```rust
// Per-chunk water heightfield (only allocated if chunk has water)
pub struct ChunkWaterData {
    columns: Box<[[WaterColumn; 16]; 16]>,  // 16x16 grid
    has_any_water: bool,
    min_water_y: f32,
    max_water_y: f32,
}

pub struct WaterColumn {
    surface_y: f32,      // Height of water surface
    foam: f32,           // Foam intensity at this column
    flow_angle: f32,     // Direction of flow
}

// WaterVertex format (24 bytes, matches ChunkVertex for alignment)
pub struct WaterVertex {
    position: [f32; 3],     // 12 bytes
    normal_packed: u32,     // 4 bytes
    uv_data: u32,           // [foam, depth, flow_lo, flow_hi]
    light_data: u32,        // [block_light, sky_light, 0, 0]
}
```

## Key Constants (lib.rs)

```rust
WATER_SEA_LEVEL: 32          // Y level where water generates
WATER_FULL_HEIGHT: 0.875     // Surface height within block (leaves wave headroom)
WATER_WAVE_STRENGTH: 0.12    // Wave displacement amplitude
WATER_FRESNEL_POWER: 4.0     // Fresnel reflection exponent
```

## Shader Effects

**Gerstner Waves** (4 overlapping):
- Different directions and frequencies for natural look
- Unified flow direction prevents grid/X-pattern artifacts

**Fresnel Reflection**:
```wgsl
let fresnel = pow(1.0 - max(dot(view_dir, normal), 0.0), FRESNEL_POWER);
// More reflective at grazing angles
```

**Depth-Based Coloring**:
- Shallow: turquoise
- Medium: teal
- Deep: dark blue
- Uses Schlick approximation for color absorption

**Foam**:
- White foam at shorelines (depth < threshold)
- Wave-responsive (more foam at wave crests)
- Block edge detection for foam around objects

## Rendering Pipeline

1. Opaque chunks render first (write depth buffer)
2. Water pass:
   - Alpha blending enabled
   - Depth test: enabled
   - Depth write: disabled (prevents sorting issues)
3. Separate bind groups for camera, light, water uniforms

## Underwater Effect

When camera Y < water surface Y:
- Blue-green fog applied
- Reduced visibility distance
- Detected per-frame in renderer

## Key Files

| File | Purpose |
|------|---------|
| `world/chunk.rs` | `ChunkWaterData`, `WaterColumn` structs |
| `world/water_meshing.rs` | Mesh generation, `WaterVertex` |
| `world/generation.rs` | Water heightfield generation |
| `renderer/pipeline.rs` | Water shader (301 LOC) |
| `renderer/mod.rs` | Water render pass, underwater detection |

## Adding Water Features

**Variable Water Levels:**
Currently water is full-height. For partial levels:
1. Add `level: u8` (0-8) to `WaterColumn`
2. Modify `water_meshing.rs` to use level for surface_y
3. Update shader for stepped appearance

**Water Flow Simulation:**
Would require:
1. Cellular automata in `world/` module
2. Per-tick flow updates
3. Flow direction passed to shader for wave alignment
