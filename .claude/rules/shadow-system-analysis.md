# Shadow System Analysis & Architecture Review

**Date:** 2026-01-26
**Status:** Active investigation - shadows toggling/disappearing

## Identified Issues

### Issue 1: Shadow Map is Player-Centric (ROOT CAUSE)
**Location:** `renderer/mod.rs:1314-1347`

The shadow map is centered on `camera.position`, not the view frustum:
```rust
let snapped_center = light_right * snapped_x + light_up * snapped_y + shadow_dir * cam_light_z;
```

**Symptoms:**
- Walking under a tree causes its shadow to disappear (tree moves outside shadow coverage)
- Shadows "toggle" when walking to certain positions
- Cave shadows appear as "tube around player FOV"

### Issue 2: Limited Shadow Coverage (150 units)
**Location:** `renderer/mod.rs:729-739, 1323`

```rust
let shadow_size = 150.0;  // Only ±150 blocks from camera
```

But render distance is 24 chunks = 384 blocks. Trees at screen edges (>150 blocks away) have NO shadows.

### Issue 3: Aggressive Edge Softening
**Location:** `renderer/pipeline.rs:430-435`

```wgsl
let edge_fade_start = 0.03;  // 3% from edge
let edge_fade_end = 0.12;    // 12% from edge
```

Combined with player-centric shadow map, this causes shadows to fade as player moves.

### Issue 4: Single Resolution Shadow Map
4096x4096 covering 300x300 blocks = 13.6 texels/block everywhere.
- Near shadows could be sharper with cascades
- Distant shadows waste resolution

## Current Architecture

```
Shadow Pass:
  1. Calculate orthographic projection centered on camera position
  2. Texel snap for stabilization (working correctly)
  3. Render chunks within ±150 unit box to depth buffer
  4. Main pass samples shadow map with PCF

Light Uniform (mod.rs:1311-1365):
  - Creates view_proj matrix from camera position + sun direction
  - shadow_distance = 120.0 (light position offset)
  - shadow_size = 150.0 (orthographic extent)
```

## Recommended Fixes

### Option A: Cascaded Shadow Maps (Best Quality)
Implement 3 cascades:
- Cascade 0: 0-50 blocks, high detail
- Cascade 1: 50-150 blocks, medium detail
- Cascade 2: 150-400 blocks, low detail

**Pros:** Industry standard, handles all distances
**Cons:** 3x shadow pass cost, significant refactoring

### Option B: View-Frustum Aligned Shadow Map (Quick Fix)
Fit shadow map to view frustum instead of camera position:
1. Calculate visible frustum corners
2. Transform to light space
3. Fit orthographic projection to enclosing AABB
4. Apply texel snapping for stabilization

**Pros:** Covers exactly what's visible, better resolution usage
**Cons:** Still single resolution, more complex stabilization

### Option C: Increased Coverage (Simplest)
1. Increase `shadow_size` from 150 to 300-350
2. Optionally increase `SHADOW_MAP_SIZE` to 8192

**Pros:** Minimal code changes
**Cons:** Lower shadow quality, 4x memory if 8192

### Recommended Path: Option B + Larger Coverage

1. **Immediate fix:** Increase shadow_size to 250 (matches ~16 chunks)
2. **Then:** Implement view-frustum alignment for better coverage
3. **Optional:** Add 2 cascades for quality improvement

## Implementation Steps for Option B

### Step 1: Calculate View Frustum in Light Space
```rust
fn calculate_shadow_frustum(camera: &Camera, sun_dir: Vec3, near: f32, far: f32) -> (Mat4, f32) {
    let view_proj = camera.view_projection_matrix();
    let inv_view_proj = view_proj.inverse();

    // Get frustum corners in world space
    let ndc_corners = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0), // near plane
        Vec4::new( 1.0, -1.0, 0.0, 1.0),
        Vec4::new( 1.0,  1.0, 0.0, 1.0),
        Vec4::new(-1.0,  1.0, 0.0, 1.0),
        Vec4::new(-1.0, -1.0, 1.0, 1.0), // far plane
        Vec4::new( 1.0, -1.0, 1.0, 1.0),
        Vec4::new( 1.0,  1.0, 1.0, 1.0),
        Vec4::new(-1.0,  1.0, 1.0, 1.0),
    ];

    let world_corners: Vec<Vec3> = ndc_corners.iter()
        .map(|&c| {
            let w = inv_view_proj * c;
            (w.xyz() / w.w)
        })
        .collect();

    // Create light view matrix
    let center = world_corners.iter().sum::<Vec3>() / 8.0;
    let light_view = Mat4::look_at_rh(center + sun_dir * 200.0, center, Vec3::Y);

    // Transform corners to light space, find AABB
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for corner in &world_corners {
        let light_pos = (light_view * corner.extend(1.0)).xyz();
        min = min.min(light_pos);
        max = max.max(light_pos);
    }

    // Texel snap
    let size = (max - min).max_element();
    // ... snap min/max to texel boundaries

    let proj = Mat4::orthographic_rh(min.x, max.x, min.y, max.y, -max.z - 100.0, -min.z + 100.0);
    (proj * light_view, size)
}
```

### Step 2: Update Shadow Culling
Change chunk culling from fixed box to frustum-derived bounds.

### Step 3: Reduce Edge Softening
Once shadow map covers visible area, reduce aggressive edge fading:
```wgsl
let edge_fade_start = 0.01;  // 1% from edge
let edge_fade_end = 0.03;    // 3% from edge (was 12%)
```

## Related Issues Found

### Block Light (Torches) - Working Correctly
The BFS flood fill lighting in `lighting.rs` is solid:
- Propagates correctly through transparent blocks
- Proper removal when source destroyed
- Cross-chunk propagation handled

### AO - Working Correctly
Ambient occlusion in meshing uses proper corner sampling:
- Hermite smoothstep for soft transitions
- 65% floor prevents over-darkening

### "Glowing" Faces Issue
Mentioned: "ground and trees generate front faces that don't accept any shadows"

**Likely cause:** These faces are outside the shadow map coverage area. When `calculate_shadow()` returns 1.0 (fully lit) for faces outside shadow UV bounds:
```wgsl
if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || ...) {
    return 1.0;  // No shadow = looks brighter
}
```

**Fix:** Increasing shadow coverage will resolve this. Alternatively, could return ambient-only lighting for out-of-bounds samples.

## Files to Modify

1. `renderer/mod.rs` - Shadow matrix calculation, coverage size
2. `renderer/pipeline.rs` - Edge softening parameters, out-of-bounds handling
3. (Optional) New cascade system would need additional bind groups

## Testing Checklist

- [ ] Walk under large tree - shadow should stay consistent
- [ ] Look at trees at screen edges - should have shadows
- [ ] Walk through cave - lighting should be stable
- [ ] Day/night transition - no jarring shadow changes
- [ ] F3 overlay shows shadow coverage metrics
