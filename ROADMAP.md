# Mosaic Engine Roadmap

**Philosophy:** Foundation before features. Tests and profiling enable confident iteration.

---

## Current State (v0.1 - Playable Demo)

### Completed
- [x] Binary greedy meshing with packed vertices (24 bytes)
- [x] LOD system (Full/Half/Quarter based on distance)
- [x] BFS flood fill lighting (block + sky light)
- [x] Shadow mapping with PCF filtering
- [x] Multi-draw indirect (MDI) rendering
- [x] Day/night cycle with warm color palette
- [x] World persistence (region files + zstd)
- [x] Water rendering (Gerstner waves, fresnel, foam)
- [x] Swimming physics (buoyancy, drag)
- [x] AABB collision detection
- [x] Block placement/breaking with raycast
- [x] Audio system (footsteps)
- [x] F3 debug overlay

### Known Issues
- [ ] GPU profiler shows 0.00ms (wgpu 0.19 async buffer mapping)
- [ ] No automated tests (11K LOC untested)
- [ ] No API documentation

### Performance Baseline (RTX 2080 Ti, render distance 24)
- 300-450 FPS with shadows + greedy meshing + LOD
- ~265K triangles, ~2.4K chunks rendered

---

## Tier 1: Foundation (Critical)

**Rationale:** 11K LOC with zero tests is high technical risk. Changes to meshing, lighting, or serialization could introduce subtle bugs. Fix profiling to enable data-driven optimization.

### Testing Infrastructure
- [ ] **Unit tests for meshing** - Verify greedy mesh output for known block configurations
- [ ] **Unit tests for lighting** - BFS propagation correctness, cross-chunk boundaries
- [ ] **Unit tests for chunk serialization** - Round-trip encode/decode, compression integrity
- [ ] **Integration tests for chunk lifecycle** - Load → mesh → render → unload cycle

### Profiling & Measurement
- [ ] **Fix GPU profiler** - Resolve wgpu 0.19 async buffer mapping or upgrade to 0.20+
- [ ] **Benchmark suite for meshing** - Track regression in mesh generation time
- [ ] **Benchmark suite for generation** - Track regression in terrain generation
- [ ] **Memory profiling** - Chunk allocation patterns, peak usage

---

## Tier 2: Developer Experience

**Rationale:** Documentation prevents knowledge loss between sessions. Better errors reduce debugging time.

- [ ] **Doc comments on public APIs** - WorldManager, Chunk, Renderer entry points
- [ ] **Error context in chunk loading** - Include chunk coords, region file path in errors
- [ ] **Structured logging** - Replace println! with tracing spans for async operations
- [ ] **Hot reload for constants** - Runtime config for RENDER_DISTANCE, LOD thresholds

---

## Tier 3: Visual Polish

**Rationale:** Low-effort, high-impact improvements to player experience.

### Audio
- [ ] **Block-specific footsteps** - Stone, sand, wood, water sounds
- [ ] **Ambient sounds** - Wind, water, cave echoes
- [ ] **Block interaction sounds** - Place/break feedback

### Water Enhancements
- [ ] **Screen-space refraction** - Distort scene behind water surface
- [ ] **Water mesh LOD** - Simplified mesh for distant water
- [ ] **Caustics projection** - Animated light patterns on underwater surfaces

### Rendering
- [ ] **Chunk-based water culling** - Skip fully submerged chunks
- [ ] **Particle system** - Block break particles, torch flames

---

## Tier 4: Gameplay Foundation

**Rationale:** Core gameplay systems that enable actual game mechanics.

### Inventory System
- [ ] **Hotbar UI** - Visual block selection (1-9 keys)
- [ ] **Inventory grid** - Storage beyond hotbar
- [ ] **Item stacking** - Stack limits, split/merge

### World Generation
- [ ] **Biome system** - Temperature/humidity noise → distinct biomes
- [ ] **Structure generation** - Villages, dungeons, ruins
- [ ] **More block types** - Slabs, stairs, fences, doors

### Interaction
- [ ] **Block metadata** - Rotation, state (open/closed doors)
- [ ] **Crafting system** - Recipe registry, crafting UI
- [ ] **Tool system** - Mining speed based on tool material

---

## Tier 5: Scale

**Rationale:** Infrastructure for larger worlds and multiplayer. Defer until gameplay is solid.

### Performance at Scale
- [ ] **Render distance 32+** - Requires aggressive LOD, occlusion culling
- [ ] **Compute shader world gen** - GPU-accelerated noise
- [ ] **Infinite height** - Remove 256 block limit, sparse chunk storage

### Multiplayer
- [ ] **Server/client architecture** - Authoritative server, client prediction
- [ ] **Network protocol** - Chunk streaming, player sync, block updates
- [ ] **Authentication** - Player accounts, permissions

### Extensibility
- [ ] **Mod support** - Plugin API for blocks, generation, rendering
- [ ] **Scripting** - Lua or WASM for mod logic

---

## Version Milestones

### v0.2 - Solid Foundation
- [ ] Unit tests for core algorithms (meshing, lighting, serialization)
- [ ] GPU profiler working
- [ ] Benchmark suite
- [ ] Doc comments on public APIs

### v0.3 - Polished Experience
- [ ] Block-specific footsteps
- [ ] Water refraction
- [ ] Inventory hotbar
- [ ] Biome system

### v0.4 - Gameplay Loop
- [ ] Crafting system
- [ ] Tool progression
- [ ] Structure generation

### v0.5 - Scale
- [ ] Render distance 32+
- [ ] Multiplayer prototype

---

## Technical Debt

- [ ] Remove `_unused` variable prefixes
- [ ] Clean up dead code paths
- [ ] Consistent error handling (Result vs panic)
- [ ] Audit unsafe blocks (currently none, keep it that way)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01 | Foundation before features | 11K LOC with 0 tests is unsustainable risk |
| 2025-01 | Defer multiplayer to Tier 5 | Requires stable single-player first |
| 2025-01 | Tests over documentation | Tests catch bugs; docs just describe them |
