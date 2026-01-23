# Blocks Engine Roadmap

## Must Do (Critical / Blocking)

- [ ] **Fix greedy meshing** - Current simple meshing works but uses 6x more triangles. Greedy meshing had winding order issues. Need to carefully implement with correct CCW vertex order for merged quads.

- [ ] **Block placement/breaking** - Core gameplay. Raycast from camera, modify chunk data, trigger remesh.

- [ ] **Collision detection** - Player currently flies through blocks. Need AABB vs voxel collision for walking/jumping.

- [ ] **Chunk save/load** - Persist world to disk. Simple format: chunk position + block data.

---

## Should Do (Important)

- [ ] **Re-enable LOD system** - Infrastructure exists in `world/mod.rs`. Currently disabled (`calculate_lod` returns Full). Fix greedy meshing first, then enable LOD.

- [ ] **On-screen debug overlay** - F3 currently logs to console. Render text showing FPS, position, chunk count, triangle count.

- [ ] **GPU profiler fix** - Async buffer mapping in wgpu 0.19 is tricky. Either fix properly or upgrade to wgpu 0.20+ with better async support.

- [ ] **Ambient occlusion** - AO calculation exists but simplified. Implement proper corner-based AO for better visual depth.

- [ ] **Water rendering** - Water blocks exist but render as solid. Need transparency, separate render pass, wave animation.

- [ ] **Day/night cycle** - Animate sun direction in shader, adjust sky color and fog.

- [ ] **Custom texture loading** - Replace procedural atlas with actual PNG texture atlas from `assets/textures/`.

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

- [ ] Clean up unused code warnings (`cargo fix --lib -p blocks`)
- [ ] Remove `_unused` variable prefixes where possible
- [ ] Document public APIs with doc comments
- [ ] Add unit tests for chunk operations
- [ ] Benchmark meshing performance

---

## Version Milestones

### v0.1 - Playable Demo
- Greedy meshing working
- Block place/break
- Basic collision
- Save/load

### v0.2 - Visual Polish
- Custom textures
- Proper AO
- Water rendering
- Day/night

### v0.3 - Gameplay
- Inventory
- Crafting
- Trees/vegetation

### v0.4 - Scale
- LOD system active
- Larger render distance
- Performance optimizations
