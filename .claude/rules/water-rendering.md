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

## Mesh Generation

**Skirt Geometry** (prevents gaps during wave animation):
- Vertical walls extend 1.5 blocks below water surface at boundaries
- Only added where no adjacent water exists (edges of water bodies)
- Uses upward normals to render identically to surface
- Constant `WATER_SKIRT_DEPTH: f32 = 1.5` in `water_meshing.rs`

**Vertex Interpolation**:
- Heights and foam averaged at grid corners for seamless adjacent cells
- Flow angles use uniform value (0.0) to prevent interference patterns
- Per-vertex flow angles cause visible banding - avoid!

## Shader Effects

**Smooth Ripples** (replaces Gerstner waves):
- Multi-octave sine waves for calm surface
- Large gentle swells + medium ripples + small detail
- Subtle height displacement (~30% of wave_strength)
- Normal perturbation for reflection variation

**Fresnel Reflection** (Schlick approximation):
```wgsl
let fresnel = schlick_fresnel(n_dot_v, 0.02);  // Water F0 = 0.02
let reflection_strength = fresnel * 0.85 + depth_boost + 0.08;
```
Strong reflections at grazing angles, transparent looking down.

**Depth-Based Coloring**:
- Shallow: nearly clear (0.06, 0.10, 0.12)
- Medium: subtle teal (0.10, 0.18, 0.26)
- Deep: blue-gray (0.05, 0.10, 0.18)

**Foam**:
- Shoreline foam only (base_foam from mesh)
- No wave-peak foam (removed for calm look)

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
| `world/water_meshing.rs` | Mesh generation, `WaterVertex`, skirt geometry |
| `world/water_flow.rs` | Pipe-model flow simulation engine |
| `world/generation.rs` | Water heightfield generation |
| `renderer/pipeline.rs` | Water shader (ripples, Fresnel, depth color) |
| `renderer/mod.rs` | Water render pass, underwater detection |

## Adding Water Features

**Variable Water Levels:**
Currently water is full-height. For partial levels:
1. Add `level: u8` (0-8) to `WaterColumn`
2. Modify `water_meshing.rs` to use level for surface_y
3. Update shader for stepped appearance

**Water Flow Simulation** (`world/water_flow.rs`):
Pipe-model simulation exists but visual flow uses uniform direction:
- `WaterFlowEngine` manages active cells, flux calculations
- Per-cell flow angles cause visual artifacts when interpolated
- Flow data could be used for: foam intensity, particle effects, gameplay
- To enable visual flow: would need shared vertex grid or screen-space solution
