# Voxelot Engine Comparison & Upgrade Opportunities

Comprehensive technical comparison between Mosaic and [Voxelot](https://github.com/Roenbaeck/voxelot), a sparse voxel engine using hierarchical chunks with Roaring Bitmaps.

**Reference**: Local copy at `archive/voxelot/`

---

## Engine Overview

| Aspect | Mosaic | Voxelot |
|--------|--------|---------|
| **Focus** | Playable game engine | Massive world viewer |
| **Language** | Rust + wgpu | Rust + wgpu |
| **LOC** | ~32K | ~26K |
| **World Model** | Flat chunk grid (16³) | Hierarchical octree (16^n) |
| **Sparse Storage** | Region files + zstd | Roaring Bitmaps |

---

## Architecture Comparison

### World Structure

**Mosaic - Flat Grid:**
```
World
  └─ ChunkPos (x, y, z) → Chunk [16³ blocks]
      ├─ blocks: [4096]
      ├─ light: packed (sky/block 4-bit each)
      ├─ water: Optional ChunkWaterData
      └─ dirty flag
```

**Voxelot - Hierarchical:**
```
World (root Chunk)
  └─ Chunk L3 (4096³ units)
      ├─ Chunk L2 (256³ units)
      │   ├─ Chunk L1 (16³ units)
      │   │   └─ Voxel::Solid(type) or leaves
      │   └─ LOD metadata (avg_color, bounding_box)
```

**Tradeoffs:**
- Mosaic: Simpler coordinate math, O(1) chunk lookup, easier gameplay integration
- Voxelot: Automatic LOD via hierarchy, sparse storage, natural distance culling

### Rendering Pipeline

**Mosaic (Forward + Post):**
1. Sky pass (fullscreen triangle)
2. Opaque geometry (MDI, GPU-culled)
3. Water pass (heightfield mesh)
4. Volumetric pass (god rays)
5. UI/Post-process (bloom)

**Voxelot (Deferred + Multi-Pass Post):**
1. G-buffer pass
2. Lighting pass
3. SSAO + blur
4. SSR + blur
5. Bloom (Dual Kawase, 5 passes)
6. DOF (CoC + combine)
7. Final composite

---

## Detailed Technical Comparison

### Rendering

| Metric | Mosaic | Voxelot | Winner |
|--------|--------|---------|--------|
| Draw Calls/Frame | 1 (MDI) | 1 (MDI) | Tie |
| Max Chunk Slots | 2,200 | Unlimited (hierarchy) | Voxelot |
| Vertex Buffer | 324MB fixed pool | Dynamic slab allocator | Voxelot (memory) |
| Vertex Format | 28 bytes | 48 bytes | **Mosaic** |
| Culling Location | GPU compute shader | CPU (rayon parallel) | **Mosaic** |
| Culling Technique | Frustum + GPU atomic | Frustum + Roaring + shell | Voxelot (stages) |
| Visibility Cache | **Yes (88% hit rate)** | Yes (camera delta) | **Tie** |

### Scalability

| Metric | Mosaic | Voxelot | Winner |
|--------|--------|---------|--------|
| Render Distance | 24 chunks (~384 blocks) | 800+ blocks configurable | **Voxelot** |
| Far Plane | ~400 blocks | 5,000 blocks default | **Voxelot** |
| World Size | Unlimited (flat) | 65,536³ (4-level octree) | Tie |
| LOD System | 3 manual levels | Automatic via tree depth | **Voxelot** |
| Impostor Rendering | No | Yes (1-pixel distant) | **Voxelot** |
| Memory/Empty Chunk | 8KB minimum | ~0 bytes (sparse) | **Voxelot** |

### Meshing

| Metric | Mosaic | Voxelot |
|--------|--------|---------|
| Algorithm | Binary greedy (bitwise) | Binary greedy (bitwise) |
| Vertex Format | 28 bytes packed | 48 bytes (color+emissive+reflectivity) |
| Threading | 8 workers + priority channel | Rayon auto-scaled |
| Priority System | Yes (player actions first) | No |
| Typical Quads/Chunk | 800-1,500 | Similar |

### Lighting & GI

| Metric | Mosaic | Voxelot |
|--------|--------|---------|
| GI Type | Voxel volume (256³) + VCT | Radiance cascade probes |
| GI Resolution | Per-block (1:1) | Per-probe (lower) |
| GI Range | 256 blocks from camera | Unlimited (probe-based) |
| Shadow Technique | DDA ray march through voxels | None (ambient only) |
| Computation | GPU compute shader | CPU |
| Update Strategy | Incremental (dirty blocks) | Incremental (dirty chunks) |

### Memory Efficiency

| Buffer | Mosaic | Voxelot |
|--------|--------|---------|
| Vertex Buffer | 324MB (fixed pool) | Dynamic (slab allocator) |
| Index Buffer | 72MB (fixed pool) | Dynamic |
| Chunk Storage | 8KB/chunk (blocks + light) | Variable (Roaring bitmap) |
| Voxel GI Volume | 64MB (256³ × 4 bytes) | N/A (probes) |
| Empty Chunk | ~8KB | ~200 bytes |

**For dense worlds**: Similar
**For sparse worlds**: Voxelot wins 40:1

### Threading Model

| Aspect | Mosaic | Voxelot |
|--------|--------|---------|
| Worker Threads | 8 (configurable 2-8) | Rayon (auto-scaled) |
| Task Channels | Priority + Regular (crossbeam) | Single queue |
| Player Action Latency | Instant (priority bypass) | Standard queue |
| Generation Counter | Yes (discard stale) | No |

### Post-Processing Effects

| Effect | Mosaic | Voxelot |
|--------|--------|---------|
| Bloom | Simple (single pass) | Dual Kawase (5 passes) |
| SSAO | No | Yes |
| SSR | No | Yes (31KB shader) |
| DOF | No | Yes (CoC + blur) |
| God Rays | Yes (volumetric) | No |
| Deferred G-Buffer | No (forward) | Yes |

---

## Summary Scorecard

| Category | Mosaic | Voxelot | Notes |
|----------|--------|---------|-------|
| GPU Throughput | ⭐⭐⭐⭐ | ⭐⭐⭐ | Mosaic's GPU culling + tighter vertex format |
| Scalability | ⭐⭐ | ⭐⭐⭐⭐⭐ | Voxelot's hierarchical LOD + impostors |
| Lighting Quality | ⭐⭐⭐⭐⭐ | ⭐⭐ | Mosaic's per-block GI + shadows |
| Memory Efficiency | ⭐⭐⭐ | ⭐⭐⭐⭐ | Voxelot's sparse storage |
| Visual Effects | ⭐⭐⭐ | ⭐⭐⭐⭐ | Voxelot's SSR/SSAO/DOF |
| Gameplay Responsiveness | ⭐⭐⭐⭐⭐ | N/A | Mosaic's priority channels |
| Code Complexity | ⭐⭐⭐⭐ | ⭐⭐⭐ | Voxelot is simpler (viewer-only) |

---

## Upgrade Opportunities

### Tier 1: Quick Wins (1-2 days each)

#### Visibility Cache ✅ IMPLEMENTED

**Status**: Completed Feb 2026

**What**: Cache frustum culling results between frames when camera is stationary.

**Implementation** (`src/renderer/gpu_culling.rs`):
- `VisibilityCache` struct tracks last camera position/forward
- Thresholds: 2.0 blocks movement, ~5 degrees rotation
- `cull_with_cache()` method skips compute dispatch on cache hit
- Auto-invalidates when chunks are added/removed

**Results**:
- **88% cache hit rate** in typical gameplay
- Skips GPU compute shader dispatch when camera stationary
- Stats logging every 300 frames for monitoring

**Files Modified**:
- `src/renderer/gpu_culling.rs` - Added VisibilityCache struct and cull_with_cache()
- `src/renderer/mod.rs` - Updated to use cull_with_cache()

---

#### Chunk Delta Upload ✅ IMPLEMENTED

**Status**: Completed Feb 2026

**What**: Track which chunks are already uploaded to GPU; only upload NEW chunks on camera movement.

**Implementation** (`src/renderer/voxel_gi.rs`):
- `uploaded_chunks: AHashMap<ChunkPos, u32>` - Maps chunk position → buffer slot
- `free_slots: Vec<u32>` - Recycled slots for reuse when chunks leave volume
- `upload_chunk_data_for_gpu_delta()` - Incremental upload function
- Slot allocator with defragmentation when buffer gets fragmented
- Auto-invalidation when blocks change via `invalidate_chunk()`

**Results**:
- **94-100% chunk reuse** during normal gameplay
- **Upload time: 13ms → 0.15ms** when stationary (86x faster)
- **Upload time: 13ms → 0.4ms** when moving (32x faster)
- **Data saved: 11MB+ per update** (was uploading ~8MB every 2 blocks)

**Files Modified**:
- `src/renderer/voxel_gi.rs` - Added delta tracking, slot allocator, incremental upload

---

#### Flow-Based Foam
**From**: `voxelot/shaders/water.wgsl`

**What**: Use existing `flow_angle` data to drive foam intensity at water edges.

**Implementation**:
- Modify `water_meshing.rs` to set foam based on flow velocity
- Add particle emitters at high-flow edges

**Impact**: Visual polish, more dynamic water
**Effort**: Small (1 day)
**Files**: `src/world/water_meshing.rs`, `src/renderer/pipeline.rs`

---

#### Config TOML
**From**: `voxelot/src/config.rs`

**What**: Externalize render settings to TOML file.

**Current Mosaic**: Constants in `src/lib.rs`
**Voxelot approach**: `config.toml` with sections for rendering, culling, etc.

**Impact**: Quality of life, easier tuning
**Effort**: Small (1 day)
**Files**: New `config.rs`, modify `src/lib.rs`

---

### Tier 2: Medium Effort (3-5 days each)

#### SSR for Water
**From**: `voxelot/shaders/ssr.wgsl` (31KB)

**What**: Screen-space reflections for water surfaces.

**Key Concepts from Voxelot**:
- Ray march in screen space along reflection vector
- Sample depth buffer to find intersection
- Blend with Fresnel-based fallback

**Implementation**:
1. Create new `ssr.wgsl` shader
2. Add SSR render pass after water
3. Bind water depth texture
4. Composite onto water surface

**Impact**: Major visual upgrade - water reflects environment
**Effort**: Medium (3-4 days)
**Files**: New `src/renderer/ssr.rs`, `shaders/ssr.wgsl`

---

#### Impostor Rendering
**From**: `voxelot/shaders/impostor.wgsl`

**What**: Render ultra-distant chunks as single colored quads.

**Key Concepts**:
- Store average color per chunk (already have for LOD)
- Below pixel threshold, render as 1-pixel quad
- Skip mesh entirely for distant chunks

**Implementation**:
1. Add `average_color: [u8; 4]` to chunk metadata
2. Compute during mesh generation
3. New impostor shader pass
4. Distance threshold (e.g., 32+ chunks)

**Impact**: Enable 50+ chunk render distance
**Effort**: Medium (3-4 days)
**Files**: `src/renderer/chunk_pool.rs`, new impostor shader

---

#### SSAO ✅ IMPLEMENTED

**Status**: Completed Feb 2026

**What**: Screen-space ambient occlusion for depth perception and contact shadows.

**Implementation** (`src/renderer/ssao.rs`):
- Hemisphere sampling with 12-64 samples (quality presets)
- Depth-based normal reconstruction (no G-buffer needed)
- Bilateral blur with depth-aware edge preservation
- Multiplicative blend onto scene texture
- Quality presets: Off, Low, Medium, High, Ultra
- Half-resolution option for performance (Low/Medium)
- Distance fade (60-100 blocks)

**Results**:
- Soft contact shadows at corners and object bases
- Settings menu toggle (ESC → SSAO)
- Persists to settings.cfg

**Files Modified**:
- `src/renderer/ssao.rs` - New SSAO module (~1,100 LOC)
- `src/renderer/mod.rs` - Integration into render pipeline
- `src/settings.rs` - Quality preset persistence
- `src/settings_menu.rs` - UI toggle

---

### Tier 3: Large Effort (1-2 weeks)

#### Adaptive Probe GI
**From**: `voxelot/src/gi.rs`, `shaders/radiance_cascades.wgsl`

**What**: Supplement voxel GI with probe-based fallback for distant areas.

**Current Mosaic**: 256³ voxel volume = 256 block radius
**With probes**: Unlimited range at lower resolution

**Implementation**:
1. Keep voxel GI for nearby (< 128 blocks)
2. Add probe grid for distant (128+ blocks)
3. Blend between systems at transition
4. Probes updated incrementally per frame

**Impact**: Unlimited render distance with GI
**Effort**: Large (5-7 days)
**Files**: `src/renderer/voxel_gi.rs`, new probe system

---

#### Variable-Size Chunks
**From**: Roaring bitmap concept

**What**: RLE compression for chunk storage.

**Current Mosaic**: 8KB fixed per chunk
**Proposed**: Store only occupied Y-levels, RLE for runs

**Implementation**:
- Modify `chunk_io.rs` serialization
- Add RLE encoding for homogeneous regions
- Keep fast path for dense chunks

**Impact**: 50-80% storage reduction for typical worlds
**Effort**: Large (5-7 days)
**Files**: `src/world/storage/chunk_io.rs`

---

#### DOF (Depth of Field)
**From**: `voxelot/shaders/dof_*.wgsl`

**What**: Camera focus blur effect.

**Key Concepts**:
- Generate Circle of Confusion (CoC) from depth
- Blur based on CoC value
- Combine sharp focus plane with blurred background

**Impact**: Cinematic look
**Effort**: Medium (3-4 days)
**Files**: New post-process passes

---

### Tier 4: Major Refactor (4-6 weeks)

#### Hierarchical Chunks
**From**: `voxelot/src/lib_hierarchical.rs`

**What**: Replace flat chunk grid with octree structure.

**Voxelot's Approach**:
- "Chunks all the way down" - world is recursive chunk structure
- Each level is 16³ subdivisions
- 4 levels = 65,536³ world
- Automatic LOD via traversal depth

**Impact**: 10-20x larger worlds, automatic LOD
**Effort**: XL (4-6 weeks) - requires world/persistence refactor
**Risk**: High - touches every system

**Recommendation**: Not recommended for Mosaic. The flat grid is better suited for gameplay with block interactions. Consider impostor rendering instead for extended view distance.

---

## Key Files to Reference in Voxelot

| File | Size | What to Learn |
|------|------|---------------|
| `shaders/ssr.wgsl` | 31KB | Screen-space reflection algorithm |
| `shaders/ssao_blur.wgsl` | - | SSAO implementation |
| `shaders/impostor.wgsl` | - | 1-pixel LOD rendering |
| `shaders/dual_kawase_down.wgsl` | - | Better bloom blur |
| `shaders/radiance_cascades.wgsl` | - | Probe-based GI |
| `src/culling.rs` | 1.3K | Visibility caching pattern |
| `src/gi.rs` | 660 | Radiance cascade probes |
| `src/buffer_allocator.rs` | 377 | Slab allocator for dynamic buffers |
| `src/lib_hierarchical.rs` | - | Octree world structure |

---

## Priority Roadmap

### Phase 1: Quick Performance Wins
1. ~~**Visibility Cache**~~ ✅ DONE - 88% hit rate achieved
2. ~~**Chunk Delta Upload**~~ ✅ DONE - 94-100% reuse, 32x faster uploads
3. **Flow-Based Foam** - Visual polish

### Phase 2: Extended View Distance
3. **Impostor Rendering** - 2x render distance
4. **Adaptive Probe GI** - GI at any distance

### Phase 3: Visual Quality
5. **SSR for Water** - Environment reflections
6. ~~**SSAO**~~ ✅ DONE - Contact shadows, 5 quality presets
7. **Dual Kawase Bloom** - Better glow

### Phase 4: Optional Polish
8. **DOF** - Cinematic focus
9. **Variable-Size Chunks** - Storage optimization

---

## What Mosaic Already Does Better

These are areas where Mosaic's approach is superior - no upgrade needed:

| Feature | Why Mosaic Wins |
|---------|-----------------|
| **Gameplay Loop** | Full inventory, mining, placing - Voxelot has none |
| **Voxel GI Accuracy** | Per-block resolution vs Voxelot's probes |
| **Real Shadows** | DDA ray marching - Voxelot has ambient only |
| **Volumetric Lighting** | God rays - Voxelot doesn't have this |
| **World Generation** | 8+ biomes, caves, ores, trees |
| **Water Physics** | Pipe-model flow simulation |
| **Player Persistence** | Position, inventory, world saving |
| **Priority Threading** | Player actions bypass generation queue |

---

## Conclusion

**For a game engine**, Mosaic's architecture is correct:
- Responsive block interactions need priority channels
- Per-block GI accuracy matters for gameplay (lighting puzzles, etc.)
- Flat chunks are easier to reason about for gameplay mechanics

**Highest-value upgrades** (impact vs effort):
1. ~~Visibility Cache~~ ✅ DONE (88% hit rate, ~88% culling overhead reduction)
2. Impostor Rendering (medium, 2x render distance)
3. SSR for Water (medium, major visual upgrade)

**Skip these** (not worth the effort for a game engine):
- Hierarchical chunks (breaks gameplay assumptions)
- Roaring bitmaps (complexity not justified for dense game worlds)
- Full deferred pipeline (forward + post is simpler and fast enough)
