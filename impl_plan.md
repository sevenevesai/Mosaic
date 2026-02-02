Refined Task: Implement a "Frontier Sound System" that leverages the engine's voxel architecture to provide immersive spatial audio with occlusion. This involves extending the existing audio system to support 3D positioning and implementing a Digital Differential Analyzer (DDA) raycasting algorithm—similar to the lighting engine's shadow logic—to detect solid blocks between the sound source and the listener. The system should attenuate volume based on occlusion to create a realistic sense of depth and environment, specifically for short sound effects like footsteps and block interactions, while maintaining high performance by minimizing raycast overhead.

Domain: Gameplay/Engine

Research Findings:
- DDA: The engine already has world::raycast and renderer::voxel_gi which use DDA. world::raycast is CPU-based and suitable for audio checks.
- Rodio: The audio crate rodio supports SpatialSink for 3D positioning (attenuation/panning), but not occlusion. Occlusion must be applied manually by modifying volume.
- Audio Design: Common occlusion techniques involve Low-Pass Filters (LPF) and volume reduction. Since rodio's standard sinks don't easily expose LPF without custom filters, volume attenuation is the standard "MVP" approach.
- Performance: Raycasting every frame for every sound can be expensive. For "fire-and-forget" sounds (footsteps), checking occlusion once at spawn is a valid optimization ("Initial Occlusion").

Success Criteria:
- AudioManager accepts 3D positions for sounds (play_spatial).
- Sounds play with correct L/R panning and distance attenuation (handled by rodio).
- Sounds originating behind walls are audibly quieter (occlusion factor).
- The DDA algorithm correctly identifies obstacles without false positives (e.g., passing through transparent blocks like air/water/leaves appropriately).

Failure Modes:
- Raycast Spam: FPS drop during heavy action (Cause: Too many rays per frame). Prevention: Limit raycasts or use "initial occlusion" only.
- Wall Hacking: Hearing players clearly through walls (Cause: Occlusion check fails/buggy). Prevention: Verify DDA logic matches visual geometry.
- Popping: Audio clicks when volume changes (Cause: Instant volume changes). Prevention: Smooth volume transitions or set volume before playing.
- Silence: Sounds don't play (Cause: Raycast hits source immediately). Prevention: Offset ray origin slightly away from wall.

Approach:
Extend AudioManager to handle play_spatial requests. Implement a calculate_occlusion helper in src/audio/mod.rs (or a new src/audio/spatial.rs) that borrows the WorldManager. Use rodio's SpatialSink or Sink with manually calculated spatial volume if SpatialSink is too rigid (actually SpatialSink is good, but occlusion volume needs to be multiplied on top). For this iteration, we will calculate occlusion *once* when the sound starts (for short SFX) and apply a volume modifier. This avoids per-frame raycasting overhead for hundreds of particles/sounds.

Implementation Steps:
1. Define SoundMaterial: Add SoundMaterial enum to src/audio/mod.rs (or separate file) and implement Block::sound_material() in src/world/chunk.rs (as per the block-audio.md doc I read earlier, this is good groundwork).
2. Occlusion Logic: Implement calculate_occlusion(start, end, world) that traces a ray and returns an occlusion factor (0.0 = clear, 1.0 = blocked). It should treat Air, Water, Leaves as transparent (or partial occlusion).
3. Spatial Playback: Add play_spatial(&mut self, sound_id, position, world) to AudioManager. It calculates distance + occlusion, then plays the sound.
4. Integration: Update Player or App to call play_spatial for footsteps and block interactions, passing the WorldManager.
5. Verify: Test walking behind a wall and breaking a block; it should be quieter.

Assumptions & Questions:
- I assume rodio is already set up and working (code shows it is).
- I assume WorldManager is accessible where audio is triggered (usually App or Player).
- I will use "Initial Occlusion" (check once at spawn) for now. Is dynamic occlusion (updating moving sounds every frame) required? Assumption: No, simpler is better for now.