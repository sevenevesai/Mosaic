# Neolith Engine Roadmap

## Must Do (Critical / Blocking)

- [x] **Fix greedy meshing** - Current simple meshing works but uses 6x more triangles. Greedy meshing had winding order issues. Need to carefully implement with correct CCW vertex order for merged quads.

- [x] **Block placement/breaking** - Core gameplay. Raycast from camera, modify chunk data, trigger remesh.

- [x] **Collision detection** - Player currently flies through blocks. Need AABB vs voxel collision for walking/jumping.

- [x] **Chunk save/load** - Persist world to disk. Simple format: chunk position + block data.

---

## Should Do (Important)

- [x] **Re-enable LOD system** - Infrastructure exists in `world/mod.rs`. Currently disabled (`calculate_lod` returns Full). Fix greedy meshing first, then enable LOD.

- [x] **On-screen debug overlay** - F3 currently logs to console. Render text showing FPS, position, chunk count, triangle count.

- [x] **Shadow mapping** - Directional sun shadows with PCF filtering and frustum-culled shadow pass.

- [x] **Audio system** - Basic footstep sounds using rodio. Alternating sounds, speed-based timing.

- [ ] **GPU profiler fix** - Async buffer mapping in wgpu 0.19 is tricky. Either fix properly or upgrade to wgpu 0.20+ with better async support.

- [ ] **Ambient occlusion** - AO calculation exists but simplified. Implement proper corner-based AO for better visual depth.

- [ ] **Water rendering** - Water blocks exist but render as solid. Need transparency, separate render pass, wave animation.

- [x] **Day/night cycle** - Full sun/moon system with warm color palette, smooth shadow transitions, T key speed control.

- [ ] **Block-specific footsteps** - Different sounds for stone, sand, wood, etc. Currently only grass.

---

## Could Do (Nice to Have)

- [ ] **Biome system** - Temperature/humidity noise → distinct biomes with different terrain and vegetation.

- [ ] **Tree generation** - Structure placement during chunk gen. Oak, birch, etc.

- [ ] **Cave improvements** - Larger cave systems, stalactites, underground lakes.

- [ ] **Inventory system** - Block selection, hotbar UI.

- [ ] **Crafting** - Recipe system, crafting table interaction.

- [ ] **Mobs** - Entity system, basic AI, spawning.

- [ ] **Multiplayer** - Server/client architecture, network protocol.

- [ ] **Mod support** - Plugin API, scripting (Lua/WASM).

- [ ] **Infinite height** - Remove 256 block height limit, use sparse chunk storage.

- [ ] **Compute shader world gen** - GPU-accelerated noise for faster chunk generation.

- [ ] **Voxel GI** - Global illumination using voxel cone tracing or similar.

---

## Technical Debt

- [x] **Fix MDI rendering** - Multi-draw indirect now working. Chunks sync to pool, single draw call for pooled chunks.
- [ ] Clean up unused code warnings (`cargo fix --lib -p blocks`)
- [ ] Remove `_unused` variable prefixes where possible
- [ ] Document public APIs with doc comments
- [ ] Add unit tests for chunk operations
- [ ] Benchmark meshing performance

---

## Version Milestones

### v0.1 - Playable Demo ✓
- [x] Greedy meshing working
- [x] Block place/break
- [x] Basic collision
- [x] Save/load (region files + zstd)
- [x] Shadow mapping
- [x] Basic audio (footsteps)

### v0.2 - Visual Polish
- [ ] Proper AO (GTAO or corner-based)
- [ ] Water rendering
- [x] Day/night cycle
- [ ] Block-specific sounds

### v0.3 - Gameplay
- Inventory
- Crafting
- Trees/vegetation

### v0.4 - Scale
- [x] LOD system active
- [x] Performance optimizations (greedy mesh, packed vertices)
- [x] Multi-draw indirect (MDI working, single draw call for pooled chunks)
- [ ] Larger render distance (32+ chunks)

---

## Performance Research (January 2025)

Comprehensive research into cutting-edge voxel rendering techniques. Organized by implementation priority and expected gains.

---

### Tier 1: High-Impact, Lower Effort

#### Binary Greedy Meshing
**Expected Gain:** 5-10x faster mesh generation (50-200μs vs ~1ms per chunk)

Uses 64-bit integers where each bit represents a voxel, performing bitwise operations to cull 64 faces simultaneously. Creates merged quads using parallel bit manipulation.

**How it works:**
1. Create 64x64 array of 64-bit occupancy masks (0=air, 1=opaque)
2. Bitwise operations cull faces into visibility masks
3. Greedy quad merging using bitwise operations

**Limitations:** Scales poorly with many block types (requires separate passes per type). Max chunk size 64x64x64.

**References:**
- GitHub: https://github.com/cgerikj/binary-greedy-meshing
- Discussion: https://news.ycombinator.com/item?id=40087213

#### Fix MDI + Vertex Pooling
**Expected Gain:** 30-45% frame time reduction

Our `chunk_pool.rs` has the infrastructure but causes visual bugs. The correct approach:

**Architecture:**
- Persistently mapped VBO with interleaved vertex data in fixed-size buckets
- Each bucket associates with an entry in indirect draw buffer
- Single `MultiDrawIndirect` call renders all chunks

**Advanced: Face-Oriented Bucket Masking**
- Split each chunk mesh into 6 buckets by face direction
- Mask buckets based on camera orientation
- True back-face culling without per-triangle checks
- Additional 22% performance gain reported

**Benchmarks from research:**
- 16³ chunks: 6.1ms (pool) vs 11ms (naive) = 45% improvement
- 32³ chunks: 27ms (pool) vs 46ms (naive) = 41% improvement

**References:**
- Vertex Pooling Tutorial: https://nickmcd.me/2021/04/04/high-performance-voxel-engine/
- Vulkan MDI Sample: https://docs.vulkan.org/samples/latest/samples/performance/multi_draw_indirect/README.html

#### Vertex Compression
**Expected Gain:** 33-67% GPU memory reduction

**Current format (24 bytes):**
```
position: [f32; 3]     // 12 bytes
normal_packed: u32     // 4 bytes
uv: [u16; 2]           // 4 bytes
ao_tile_packed: u32    // 4 bytes
```

**Optimized format (16 bytes):**
```
position: u32          // 4 bytes (10-10-10-2 packed, chunk-relative)
normal_packed: u32     // 4 bytes
uv: [u16; 2]           // 4 bytes
ao_tile_packed: u32    // 4 bytes
```

**Ultra-compact format (8 bytes):**
```
pos_face: u32          // 4 bytes (10-10-10 pos + 2-bit face ID)
uv_tex: u32            // 4 bytes (8-bit tex_idx, 8-bit tile_u, 8-bit tile_v, 8-bit ao)
```

Position quantization works because 16³ chunks at 64 sub-voxel precision fits in 10 bits per axis.

**References:**
- Vertex Quantization: https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html
- meshoptimizer: https://meshoptimizer.org/
- AMD Meshlet Compression: https://gpuopen.com/learn/mesh_shaders/mesh_shaders-meshlet_compression/

#### 16-bit Index Buffers
**Expected Gain:** 50% index buffer memory reduction

Chunks with greedy meshing typically have 1,000-3,000 vertices (worst case ~12,000 for solid cube surface). 16-bit indices (max 65,535) are sufficient.

#### Palette Compression for Chunks
**Expected Gain:** 2-4x chunk memory reduction

Minecraft 1.13+ format using local palettes with variable bit-width indices.

**Bit Width Scaling:**
| Unique Blocks | Bits/Block | Storage for 16³ chunk |
|---------------|------------|------------------------|
| 1 (air only)  | 0          | 0 bytes                |
| 2-16          | 4          | 2048 bytes             |
| 17-256        | 8          | 4096 bytes             |

**Single-Entry Optimization:** Chunks containing only one block type require zero storage for block array.

**References:**
- Palette Compression: https://voxel.wiki/wiki/palette-compression/
- Minecraft Protocol: https://wiki.vg/Chunk_Format
- RLE for Voxels: https://eisenwave.github.io/voxel-compression-docs/rle/rle.html

---

### Tier 2: Medium Effort, Significant Gains

#### GPU Frustum + Hi-Z Occlusion Culling
**Expected Gain:** 2-4x culling speed improvement

Move culling from CPU to compute shader pipeline.

**Pipeline:**
1. Build Hi-Z pyramid from depth buffer (mipmap chain)
2. Compute shader tests chunk AABBs against frustum
3. Test surviving chunks against Hi-Z for occlusion
4. Output visible chunk indices to indirect draw buffer

**Aokana Framework** achieves this with 5-pass GPU pipeline:
1. Chunk Selection (frustum cull with precomputed AABBs)
2. Tile Selection (screen 8x8 tiles, ray projection)
3. Ray Marching (indirect dispatch compute for SVDAG intersection)
4. Build Hi-Z (hierarchical Z-buffer construction)
5. Color Resolve (binary search across color blocks)

**References:**
- Aokana Paper (I3D 2025): https://arxiv.org/abs/2505.02017
- Vulkan GPU-Driven Guide: https://vkguide.dev/docs/gpudriven/gpu_driven_engines/

#### GTAO (Ground Truth Ambient Occlusion)
**Expected Cost:** ~0.5-1ms @ 1080p
**Quality:** Excellent, no halos like SSAO

Most accurate screen-space AO approach. Intel XeGTAO benchmarks:
- ~0.56ms @ 1080p on RTX 2060
- ~1.4ms @ 4K on RTX 3070

Works exceptionally well with voxel geometry due to many corners and crevices.

**References:**
- Intel XeGTAO: https://github.com/GameTechDev/XeGTAO
- NVIDIA HBAO+: https://developer.nvidia.com/rendering-technologies/horizon-based-ambient-occlusion-plus

#### Visibility Buffer Rendering
**Expected Gain:** Variable, enables decoupled shading

Stores only 4 bytes per pixel (primitive ID + draw call ID) vs 24-32 byte G-buffer.

**Visibility Buffer Layout (from Aokana, 64-bit):**
- 24 bits: depth (inverted)
- 3 bits: axis-aligned normal
- 13 bits: chunk ID
- 24 bits: voxel coordinates (X, Y, Z)

**Benefits:**
- Completely decouples shading rate from rendering resolution
- Better for complex lighting effects
- Wicked Engine (2024) uses this approach

**References:**
- Visibility Buffer Rendering: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
- Wicked Engine 2024: https://wickedengine.net/2024/12/wicked-engines-graphics-in-2024/

#### Bindless Rendering
**Required for:** GPU-driven pipelines

Access descriptors via index into global descriptor heaps, reducing bind calls dramatically.

**References:**
- DX12 Bindless: https://rtarun9.github.io/blogs/bindless_rendering/
- Vulkan Descriptor Indexing: https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html

#### Async Copy Queue for Chunk Loading
**Architecture:**
- CPU meshing on worker threads (we have 8)
- Separate copy queue for GPU buffer transfers
- Direct queue uses indirect command buffer for GPU culling

**References:**
- Async Copy Queue: https://rtarun9.github.io/blogs/async_copy/
- D3D12 Voxel Engine: https://github.com/rtarun9/voxel-engine

---

### Tier 3: Advanced / Future Techniques

#### Voxel Cone Tracing (VCT) for Global Illumination
**Expected Cost:** 4-15ms for real-time GI
**Key Advantage:** Our world is already voxelized - voxelization step is FREE

**How it works:**
- Scene geometry stored in 3D texture or Sparse Voxel Octree
- Trace 4-6 wide "cones" per pixel instead of hundreds of thin rays
- Cones sample progressively larger mip levels as they travel
- Provides diffuse GI, specular reflections, soft shadows, AO in one framework

**Performance (NVIDIA GTC 2012):**
- 8ms @ 512x512
- 27ms @ 720p
- 62ms @ 1080p
- Memory: 128³ volumes with SVO = ~14% of naive grid

**References:**
- NVIDIA GTC 2012: https://developer.download.nvidia.com/GTC/PDF/GTC2012/PresentationPDF/SB134-Voxel-Cone-Tracing-Octree-Real-Time-Illumination.pdf
- Implementation: https://github.com/Friduric/voxel-cone-tracing
- Dynamic VCT (2024): https://onlinelibrary.wiley.com/doi/10.1111/cgf.15262

#### DDGI (Dynamic Diffuse Global Illumination)
**Expected Cost:** 1-2ms per frame, 5MB GPU RAM per cascade
**Requires:** RT hardware for best results

3D grid of light probes updated via hardware ray tracing.

**Benefits:**
- Smooth, stable diffuse GI with no temporal noise
- No denoising required
- Good leak prevention with visibility data

**Limitations:**
- Cannot handle glossy/specular reflections
- Low-frequency - cannot capture fine detail shadows

**References:**
- NVIDIA DDGI: https://developer.nvidia.com/blog/an-engineers-guide-to-integrating-ddgi/
- DDGI Paper: https://www.jcgt.org/published/0008/02/01/paper-lowres.pdf
- RTXGI-DDGI: https://github.com/NVIDIAGameWorks/RTXGI-DDGI

#### Mesh Shaders (wgpu v28+ experimental)
**Expected Gain:** 20-50% geometry throughput improvement

wgpu v28 now supports mesh shaders experimentally via `EXPERIMENTAL_MESH_SHADER` feature.

**Optimal Configuration:**
- 64 vertices, 124 triangles, 64 threads per workgroup
- Per-meshlet frustum culling: 2.75ms → 2.43ms
- Combined frustum + occlusion: 44% total improvement

**Nvidium Minecraft Mod** demonstrates this:
- Uses mesh shaders to replace Sodium's rendering backend
- 32 invocations for 16 quads (1 invocation per triangle)
- Only NVIDIA GPUs with mesh shader support (16xx+)

**References:**
- wgpu Mesh Shaders: https://github.com/gfx-rs/wgpu/blob/v28/docs/api-specs/mesh_shading.md
- Nvidium: https://github.com/MCRcortex/nvidium
- Mesh Shader Guide (2025): https://interplayoflight.wordpress.com/2025/05/05/meshlets-and-mesh-shaders/
- NVIDIA Mesh Shaders: https://developer.nvidia.com/blog/advanced-api-performance-mesh-shaders/

#### Sparse 64-Trees for Distant LOD
**Expected Gain:** ~3x memory savings over SVO, 2x+ traversal speed

Alternative to traditional SVOs using 64-branching (4³ children per node).

**Data Structure (12 bytes per node):**
```
struct SvtNode64 {
    uint32_t IsLeaf : 1;
    uint32_t ChildPtr : 31;
    uint64_t ChildMask;  // 64-bit population mask
}
```

**Memory Efficiency:**
- ~0.19 bytes/voxel vs ~0.57 for traditional SVO
- Bistro scene (358M voxels): 224MB vs 368MB for SVO (39% savings)

**Performance Optimizations:**
- Ancestor memoization: 2x faster (16903 → 8896 cycles/ray)
- 2³ cell coalescing: 21% faster
- Ray-octant mirroring: 10% faster

**References:**
- Sparse 64-Trees Guide: https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/

#### SVDAG (Sparse Voxel Directed Acyclic Graph)
**Expected Gain:** 5-10x memory compression over SVO

Merges identical subtrees to eliminate redundancy.

**Compression Results:**
- 128K³ scene (19 billion voxels): 945MB SVDAG vs 5.1GB SVO
- Node count reduced by 1-3 orders of magnitude

**Variants:**
- **SSVDAG (Symmetry-aware):** Additional 2x compression via axis mirroring
- **HashDAG:** Enables dynamic editing via hash table embedding
- **LSVDAG (Lossy):** 10-50% additional compression with 1-5% voxel error

**References:**
- SVDAG Paper (SIGGRAPH 2013): https://dl.acm.org/doi/10.1145/2461912.2462024
- SSVDAG: https://dl.acm.org/doi/10.1145/2856400.2856420
- HashDAG: https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14757
- SVO Documentation: https://eisenwave.github.io/voxel-compression-docs/svo/svo.html

#### DirectX 12 Work Graphs
**Status:** Available since March 2024, not yet in wgpu

Paradigm shift where GPU dynamically schedules new workloads at runtime.

**Benefits:**
- Eliminates CPU-GPU roundtrips for work generation
- Draw calls as integral part of work graphs
- Higher performance alternative to ExecuteIndirect

**Adoption:** Epic Games (Nanite/Lumen), Remedy (Alan Wake 2)

**References:**
- D3D12 Work Graphs: https://devblogs.microsoft.com/directx/d3d12-work-graphs/
- NVIDIA Tutorial: https://developer.nvidia.com/blog/advancing-gpu-driven-rendering-with-work-graphs-in-direct3d-12
- Work Graph Playground: https://github.com/GPUOpen-LibrariesAndSDKs/WorkGraphPlayground

#### Temporal Upscaling (DLSS/FSR)
**FSR 4 "Redstone" (2025):** AMD's first fully AI-based upscaling
- Suite: FSR Upscaling, Frame Generation, Ray Regeneration, Radiance Caching
- Vendor-agnostic and open source

**Voxel Considerations:**
- Sharp edges and high-frequency patterns may need specialized jittering
- Block-based aesthetics present unique challenges for temporal accumulation

**References:**
- AMD FSR: https://www.amd.com/en/products/graphics/technologies/fidelityfx/super-resolution.html
- FSR Redstone: https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/
- OptiScaler (bridges upscalers): https://github.com/optiscaler/OptiScaler

#### Neural Radiance Caching
**AMD FSR Radiance Caching:**
- Predicts light propagation from sparse path-traced samples
- Neural network learns scene radiance distribution in real-time
- Available in FSR "Redstone" SDK for RDNA 4

**References:**
- AMD FSR Redstone: https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/
- NVIDIA Neural Radiance Caching: https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing
- GSCache (3D Gaussian Splatting): https://arxiv.org/abs/2507.19718

---

### Lighting Techniques Summary

| Technique | Cost (1080p) | Quality | Voxel Fit | Hardware |
|-----------|--------------|---------|-----------|----------|
| Flood-Fill (current) | ~0.5ms CPU | Medium | Excellent | Any |
| GTAO | ~0.5-1ms | High | High | Any |
| SSGI | ~1-3ms | Medium | Low-Med | Any |
| VCT | ~4-15ms | High | Excellent | Any |
| LPV | ~2-4ms | Medium | Medium | Any |
| DDGI | ~1-2ms | High | High | RT HW |
| RTAO | ~2-4ms | Very High | High | RT HW |

---

### Data Structure Comparison

| Approach | Memory | Ray Perf | Edit Speed | Complexity |
|----------|--------|----------|------------|------------|
| Flat Array | Poor | Good | Excellent | Very Low |
| SVO | Good | Good | Moderate | Medium |
| ESVO | Better | Very Good | Moderate | High |
| SVDAG | Excellent | Very Good | Poor | High |
| HashDAG | Good | Very Good | Good | Very High |
| Sparse 64-Tree | Very Good | Excellent | Good | Medium-High |
| Hybrid Chunks | Good | Good | Very Good | Medium |
| Brickmap | Good | Very Good | Good | High |

---

### Key Academic Papers & Industry Talks

**I3D 2025:**
- Aokana: GPU-Driven Voxel Rendering Framework - https://arxiv.org/abs/2505.02017
- Transform-Aware SVDAGs - https://dl.acm.org/doi/10.1145/3728301

**SIGGRAPH 2024-2025:**
- Adaptive Voxel-Based OIT (Call of Duty) - https://advances.realtimerendering.com/s2025/
- Real-Time Path Guiding Using Bounding Voxel Sampling - https://www.realtimerendering.com/kesen/sig2024.html

**GDC 2024:**
- AMD Work Graphs and Mesh Nodes - https://gpuopen.com/learn/gdc-2024-workgraphs-drawcalls/
- AMD FidelityFX Brixelizer (sparse distance fields) - https://gpuopen.com/events/amd-at-gdc-2024/

**NVIDIA Research:**
- Efficient Sparse Voxel Octrees (2010) - https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees
- GigaVoxels - https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf
- GVDB Voxels Library - https://developer.nvidia.com/gvdb

**Other Notable:**
- Teardown Frame Analysis - https://acko.net/blog/teardown-frame-teardown/
- Voxel Engine Architecture - https://voxely.net/blog/the-perfect-voxel-engine/
- Transvoxel Algorithm (LOD stitching) - https://transvoxel.org/

---

### Implementation Priority Matrix

```
                    EFFORT
              Low    Med    High
         ┌─────────────────────────┐
    High │ Binary   │ GPU    │ VCT  │
         │ Meshing  │ Cull   │      │
   GAIN  │ MDI Fix  │ Hi-Z   │ Mesh │
         │ Vertex   │ GTAO   │ Shader│
         │ Compress │        │      │
         ├─────────────────────────┤
    Med  │ 16-bit   │ Vis    │SVDAG │
         │ Indices  │ Buffer │      │
         │ Palette  │ Async  │64-Tree│
         │          │ Copy   │      │
         └─────────────────────────┘
```

**Recommended Implementation Order:**
1. ~~Binary Greedy Meshing~~ ✓ (pure bitwise operations)
2. Vertex compression to 16 bytes
3. 16-bit indices
4. ~~Debug & enable MDI~~ ✓ (working, single draw call)
5. GPU frustum culling (compute shader)
6. Hi-Z occlusion culling
7. GTAO post-process
8. Visibility buffer
9. Palette compression for chunks
10. VCT for global illumination
11. Mesh shaders (when wgpu ready)
12. SVDAG for ultra-distant LOD
