# Mosaic Engine Roadmap

**Last Updated**: February 2026

**Philosophy**: Foundation before features. Tests and profiling enable confident iteration.

---

## Current State (v0.2 - Feature Rich)

### Engine Statistics
- **Codebase**: 72K LOC Rust
- **Test Coverage**: 175 unit/integration tests
- **Performance**: 200-450 FPS @ render distance 24 (RTX 2080 Ti)
- **Triangles**: ~265K rendered per frame
- **Chunks**: ~2,400 active chunks

### Completed Systems

#### Rendering Pipeline
| Feature | Status | LOC | Notes |
|---------|--------|-----|-------|
| Binary Greedy Meshing | Done | 2,619 | Bitwise face culling, 70-85% quad reduction |
| Multi-Draw Indirect | Done | - | Single GPU call for all chunks |
| GPU Frustum Culling | Done | - | Compute shader + visibility cache (88% hit rate) |
| LOD System | Done | - | Full/Half/Quarter (needs tuning) |
| Voxel GI + VCT | Done | 2,689 | 256³ volume, DDA shadows, cone tracing |
| Chunk Delta Uploads | Done | - | 94-100% reuse, 32x faster |
| SSAO | Done | 1,100 | 5 quality presets, bilateral blur |
| Depth of Field | Done | 2,087 | CoC + Bokeh, 5 presets |
| Bloom | Done | 800 | Dual Kawase pyramid |
| Volumetric Lighting | Done | 740 | God rays |
| Water Rendering | Done | 688 | Heightfield, Fresnel, waves, skirts |
| Day/Night Cycle | Done | - | Warm palette, sun/moon |
| Shadow Mapping | Done | - | PCF filtering |

#### World & Gameplay
| Feature | Status | LOC | Notes |
|---------|--------|-----|-------|
| 8 Biomes | Done | ~800 | Grasslands, Highlands, Mountains, Desert, Tundra, Swamp, Jungle, Ocean |
| 56 Block Types | Done | - | Ores, building, decorative |
| Cave Systems | Done | 449 | Multi-octave noise, depth-aware |
| Caverns | Done | 515 | Large underground chambers |
| Ruins | Done | 549 | Procedural structures |
| Tree Generation | Done | 1,247 | Multiple types with branches |
| Inventory System | Done | 881 | 36 slots, drag-drop, stacking |
| Storage Chests | Done | ~600 | 27-slot containers |
| Item Entities | Done | ~650 | Physics, pickup, merging |
| Block Interaction | Done | - | Place, break, raycast |
| Player Physics | Done | 938 | AABB, gravity, swimming, fly mode |
| World Persistence | Done | - | Region files, zstd compression |
| Audio System | Done | - | Basic footsteps |
| F3 Debug Overlay | Done | 711 | FPS, chunks, triangles, light |

---

## Priority Tiers

### Tier S: Foundation (Critical)

| Priority | Feature | Effort | Doc |
|----------|---------|--------|-----|
| **1** | Water Refraction | 2-3 days | [water-refraction.md](docs/features/water-refraction.md) |
| **2** | Particle System | 3-4 days | [particle-system.md](docs/features/particle-system.md) |
| **3** | Unit Testing | Ongoing | [unit-testing.md](docs/features/unit-testing.md) |

**Rationale**: 175 tests provide solid foundation. Water refraction is the highest-impact visual improvement. Particles enable many future effects. Continue expanding test coverage with new features.

### Tier A: High Value

| Priority | Feature | Effort | Doc |
|----------|---------|--------|-----|
| **4** | Block-Specific Audio | 1-2 days | [block-audio.md](docs/features/block-audio.md) |
| **5** | Screen-Space Reflections | 3-4 days | [ssr.md](docs/features/ssr.md) |
| **6** | Crafting System | 4-5 days | [crafting.md](docs/features/crafting.md) |
| **7** | Tool System | 2-3 days | [tool-system.md](docs/features/tool-system.md) |

**Rationale**: Audio provides visceral feedback. SSR makes water stunning. Crafting + Tools create the core gameplay loop.

### Tier B: Solid Improvements

| Priority | Feature | Effort | Doc |
|----------|---------|--------|-----|
| **8** | API Documentation | 2-3 days | [future-features.md](docs/features/future-features.md) |
| **9** | Block Variants | 4-5 days | [future-features.md](docs/features/future-features.md) |
| **10** | Caustics | 2-3 days | [future-features.md](docs/features/future-features.md) |
| **11** | Structure Generation | 5-7 days | [future-features.md](docs/features/future-features.md) |

### Tier C: Future

| Priority | Feature | Effort | Doc |
|----------|---------|--------|-----|
| **12** | Impostor Rendering | 3-4 days | [future-features.md](docs/features/future-features.md) |
| **13** | Extended LOD | 2-3 days | [future-features.md](docs/features/future-features.md) |
| **14** | Multiplayer | 4-6 weeks | [future-features.md](docs/features/future-features.md) |

---

## Version Milestones

### v0.3 - Visual Polish
- [x] Unit tests for meshing, lighting, serialization (175 tests)
- [ ] Water refraction effect
- [ ] Particle system (block break, torch flames)
- [ ] Block-specific footstep sounds

### v0.4 - Gameplay Loop
- [ ] Crafting system (2x2 inventory, 3x3 table)
- [ ] Tool system (tiers, durability, speed)
- [ ] SSR for water surfaces
- [ ] Block variants (slabs, stairs)

### v0.5 - Content Expansion
- [ ] Structure generation (villages, dungeons)
- [ ] More block types (doors, fences, glass)
- [ ] Caustics underwater
- [ ] Audio polish (ambient, break/place sounds)

### v0.6 - Scale
- [ ] Extended LOD tuning
- [ ] Impostor rendering for distant chunks
- [ ] Render distance 32+
- [ ] Multiplayer prototype

---

## Known Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| GPU profiler shows 0.00ms | Low | wgpu 0.19 async limitation |
| LOD thresholds all equal | Low | Effectively disabled, needs tuning |
| 2 unused methods in voxel_gi.rs | Low | `invalidate_cache` vestigial |

---

## Quick Implementation Guide

### For Claude Code Sessions

**Starting a new feature**:
1. Check `docs/features/` for detailed spec
2. Read relevant `.claude/rules/` docs for system context
3. Run `cargo check` to verify build state
4. Create feature branch
5. Implement incrementally, test frequently

**Key files by system**:
| System | Primary File | Related |
|--------|-------------|---------|
| Rendering | `renderer/mod.rs` | `pipeline.rs`, `voxel_gi.rs` |
| World | `world/mod.rs` | `meshing.rs`, `generation/` |
| Player | `player/mod.rs` | `inventory/` |
| Persistence | `world/storage/` | `chunk_io.rs`, `region.rs` |

**Verification checklist**:
```bash
cargo check              # Compilation
cargo test               # Unit tests (when added)
cargo build --release    # Full build
./target/release/mosaic.exe  # Manual test
```

---

## Design Principles

1. **Foundation first** - Tests before features
2. **Voxel-aware** - All effects handle hard edges and depth discontinuities
3. **Performance-conscious** - Profile before optimizing, budget GPU time
4. **Incremental delivery** - Each feature works standalone
5. **Persistence always** - Save everything to prevent test frustration

---

## Reference Documentation

| Doc | Purpose |
|-----|---------|
| `CLAUDE.md` | Quick-start for AI sessions |
| `docs/features/` | Detailed feature specs |
| `.claude/rules/` | System deep-dives |
| `.claude/rules/voxelot-comparison.md` | Performance upgrade ideas |
| `.claude/rules/meta-prompt-generator.md` | Template for new feature prompts |
