---
paths:
  - "src/renderer/mod.rs"
  - "src/renderer/chunk_pool.rs"
  - "src/renderer/gpu_culling.rs"
---

# Frame order and chunk uploads

`Renderer::render()` runs: GPU frustum culling (compute), the main pass (sky, chunks by
multi-draw indirect from the pool, impostors, items, characters, scenery, grass, props), a
copy of `scene_texture`, water, clouds, block outline, crack overlay, SSAO, volumetrics, DoF,
AA, menu blur, UI, and finally bloom compositing to the swapchain.

- Water samples the copy of `scene_texture` taken when the main pass ends, because a shader
  cannot sample the target it writes. Insert a pass that needs the lit scene after that copy
  or it reads a stale frame.
- Bloom always runs: `render_or_blit()` blits with tone mapping when bloom is disabled, so it
  is the only thing writing the swapchain. Skipping it leaves a black window.
- Block outline and crack overlay test depth without writing it and use constant depth bias -2
  and -3 to sit on the block face. Without the bias they z-fight; the crack needs the stronger
  value because it draws over the outline.
- The crack texture is one 256x64 RGBA image holding four 64x64 stages, generated
  procedurally at startup.

## Chunk uploads

`MAX_UPLOADS_PER_FRAME` is 16 — about 960 chunks a second at 60 fps — which keeps a frame from
stalling on buffer writes. Urgent uploads, the chunks a player edit touched, are taken first
and outside that budget, so a broken block appears in the same frame. Raising the throttle
instead of using the urgent path costs frame time on every frame.

## GPU paths with CPU fallbacks

GPU frustum culling needs `MULTI_DRAW_INDIRECT` and `MULTI_DRAW_INDIRECT_COUNT` together;
`toggle_gpu_culling()` (G key) will not enable it without the count feature and the CPU path
stays. GPU voxel fill (B key) toggles freely. Both toggles exist so a suspect compute path can
be compared against the CPU result at runtime.
