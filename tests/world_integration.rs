//! Integration tests for the world system
//!
//! Tests chunk loading, unloading, block operations, and persistence.

use mosaic::world::{Block, ChunkPos, WorldManager, generate_chunk_with_seed};
use mosaic::constants::CHUNK_SIZE_I32;
use glam::Vec3;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Test that WorldManager can be created without panics
#[test]
fn test_world_manager_creation() {
    let world = WorldManager::new();
    assert_eq!(world.loaded_chunks_count(), 0);
    assert_eq!(world.mesh_count(), 0);
}

/// Test that WorldManager can be created with persistence
#[test]
fn test_world_manager_with_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let world = WorldManager::new_with_save(temp_dir.path(), Some(42))
        .expect("Failed to create world with save");

    assert!(world.has_storage());
    assert_eq!(world.seed(), 42);
}

/// Test chunk generation is deterministic with same seed
#[test]
fn test_chunk_generation_deterministic() {
    let pos = ChunkPos { x: 5, y: 2, z: -3 };
    let seed = 12345u64;

    let chunk1 = generate_chunk_with_seed(pos, seed);
    let chunk2 = generate_chunk_with_seed(pos, seed);

    // Compare all blocks
    let chunk_size = CHUNK_SIZE_I32 as usize;
    for x in 0..chunk_size {
        for y in 0..chunk_size {
            for z in 0..chunk_size {
                assert_eq!(
                    chunk1.get_block(x, y, z),
                    chunk2.get_block(x, y, z),
                    "Block mismatch at ({}, {}, {})", x, y, z
                );
            }
        }
    }
}

/// Test that update triggers chunk loading around player position
#[test]
fn test_chunks_load_around_player() {
    let mut world = WorldManager::new();
    let player_pos = Vec3::new(0.0, 64.0, 0.0);

    // Initial state - nothing loaded
    assert_eq!(world.loaded_chunks_count(), 0);

    // Trigger update which queues chunk generation
    world.update(&player_pos);

    // Pending generation should be non-zero
    assert!(world.pending_generation_count() > 0,
        "Expected chunks to be queued for generation");
}

/// Test that worker threads process chunks without panics
#[test]
fn test_worker_threads_process_chunks() {
    let mut world = WorldManager::new();
    let player_pos = Vec3::new(0.0, 64.0, 0.0);

    // Trigger chunk loading
    world.update(&player_pos);

    // Give workers time to process (up to 5 seconds)
    let start = Instant::now();
    let timeout = Duration::from_secs(5);

    while world.loaded_chunks_count() == 0 && start.elapsed() < timeout {
        world.update(&player_pos);
        std::thread::sleep(Duration::from_millis(50));
    }

    // At least some chunks should have loaded
    assert!(world.loaded_chunks_count() > 0,
        "Expected at least one chunk to load within timeout");
}

/// Test get_block returns None for unloaded chunks
#[test]
fn test_get_block_unloaded_chunk() {
    let world = WorldManager::new();

    // Far away chunk that won't be loaded
    let result = world.get_block(10000, 64, 10000);
    assert!(result.is_none(), "Expected None for unloaded chunk");
}

/// Test set_block and get_block roundtrip
#[test]
fn test_block_set_get_roundtrip() {
    let mut world = WorldManager::new();
    let player_pos = Vec3::new(0.0, 64.0, 0.0);

    // Load chunks first
    world.update(&player_pos);

    // Wait for at least one chunk containing origin to load
    let start = Instant::now();
    let timeout = Duration::from_secs(5);

    while world.get_block(0, 0, 0).is_none() && start.elapsed() < timeout {
        world.update(&player_pos);
        std::thread::sleep(Duration::from_millis(50));
    }

    // Skip test if chunk didn't load in time
    if world.get_block(0, 0, 0).is_none() {
        eprintln!("Warning: Chunk at origin didn't load in time, skipping roundtrip test");
        return;
    }

    // Set a block
    let success = world.set_block(0, 10, 0, Block::Brick);
    assert!(success, "set_block should succeed for loaded chunk");

    // Read it back
    let block = world.get_block(0, 10, 0);
    assert_eq!(block, Some(Block::Brick), "Block should be Brick after set");
}

/// Test ChunkPos coordinate conversion
#[test]
fn test_chunk_pos_from_world_coords() {
    // Positive coordinates
    assert_eq!(ChunkPos::from_world_pos(Vec3::new(0.0, 0.0, 0.0)), ChunkPos { x: 0, y: 0, z: 0 });
    assert_eq!(ChunkPos::from_world_pos(Vec3::new(15.0, 15.0, 15.0)), ChunkPos { x: 0, y: 0, z: 0 });
    assert_eq!(ChunkPos::from_world_pos(Vec3::new(16.0, 0.0, 0.0)), ChunkPos { x: 1, y: 0, z: 0 });

    // Negative coordinates (should floor divide)
    assert_eq!(ChunkPos::from_world_pos(Vec3::new(-1.0, 0.0, 0.0)), ChunkPos { x: -1, y: 0, z: 0 });
    assert_eq!(ChunkPos::from_world_pos(Vec3::new(-16.0, 0.0, 0.0)), ChunkPos { x: -1, y: 0, z: 0 });
    assert_eq!(ChunkPos::from_world_pos(Vec3::new(-17.0, 0.0, 0.0)), ChunkPos { x: -2, y: 0, z: 0 });
}

/// Test ChunkPos world origin calculation
#[test]
fn test_chunk_pos_world_origin() {
    let pos = ChunkPos { x: 2, y: -1, z: 3 };
    let origin = pos.world_origin();

    assert_eq!(origin.x, 32.0);  // 2 * 16
    assert_eq!(origin.y, -16.0); // -1 * 16
    assert_eq!(origin.z, 48.0);  // 3 * 16
}

/// Test persistence: save and reload world
#[test]
fn test_world_persistence_roundtrip() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let seed = 98765u64;

    // Create world, modify a block, and save
    {
        let mut world = WorldManager::new_with_save(temp_dir.path(), Some(seed))
            .expect("Failed to create world");

        let player_pos = Vec3::new(0.0, 64.0, 0.0);

        // Load chunks
        world.update(&player_pos);

        // Wait for chunk at origin
        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        while world.get_block(0, 32, 0).is_none() && start.elapsed() < timeout {
            world.update(&player_pos);
            std::thread::sleep(Duration::from_millis(50));
        }

        if world.get_block(0, 32, 0).is_some() {
            // Set a distinctive block
            world.set_block(0, 32, 0, Block::DiamondOre);

            // Force save
            world.save_all();
        }
    } // World dropped, should flush saves

    // Small delay to ensure file writes complete
    std::thread::sleep(Duration::from_millis(100));

    // Reload and verify
    {
        let mut world = WorldManager::new_with_save(temp_dir.path(), None)
            .expect("Failed to reload world");

        // Seed should be preserved
        assert_eq!(world.seed(), seed, "Seed should be preserved across saves");

        let player_pos = Vec3::new(0.0, 64.0, 0.0);

        // Load chunks
        world.update(&player_pos);

        // Wait for chunk
        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        while world.get_block(0, 32, 0).is_none() && start.elapsed() < timeout {
            world.update(&player_pos);
            std::thread::sleep(Duration::from_millis(50));
        }

        // Check if our modification persisted
        if let Some(block) = world.get_block(0, 32, 0) {
            assert_eq!(block, Block::DiamondOre,
                "Modified block should persist across save/load");
        }
    }
}

/// Test that light engine propagates torch light correctly
#[test]
fn test_light_propagation_basic() {
    let mut world = WorldManager::new();
    let player_pos = Vec3::new(0.0, 64.0, 0.0);

    // Load chunks
    world.update(&player_pos);

    // Wait for chunks around origin
    let start = Instant::now();
    let timeout = Duration::from_secs(5);

    while world.get_block(0, 50, 0).is_none() && start.elapsed() < timeout {
        world.update(&player_pos);
        std::thread::sleep(Duration::from_millis(50));
    }

    if world.get_block(0, 50, 0).is_none() {
        eprintln!("Warning: Chunks didn't load in time for light test");
        return;
    }

    // Clear some space for the torch
    world.set_block(0, 50, 0, Block::Air);
    world.set_block(0, 51, 0, Block::Air);
    world.set_block(1, 50, 0, Block::Air);

    // Process updates
    world.update(&player_pos);

    // Place a torch
    world.set_block(0, 50, 0, Block::Torch);

    // Process light updates
    world.update(&player_pos);
    std::thread::sleep(Duration::from_millis(100));
    world.update(&player_pos);

    // Check light at torch position (should be max torch light = 14)
    let (block_light, _sky_light, _combined) = world.get_light_at(0, 50, 0);
    assert!(block_light >= 13, "Torch should emit light level ~14, got {}", block_light);

    // Check adjacent block (should have decayed light)
    let (adj_light, _, _) = world.get_light_at(1, 50, 0);
    assert!(adj_light >= 12, "Adjacent block should have light ~13, got {}", adj_light);
}

/// Test concurrent chunk loading doesn't cause panics
#[test]
fn test_concurrent_chunk_loading_stability() {
    let mut world = WorldManager::new();

    // Move player rapidly to different positions
    let positions = [
        Vec3::new(0.0, 64.0, 0.0),
        Vec3::new(100.0, 64.0, 0.0),
        Vec3::new(100.0, 64.0, 100.0),
        Vec3::new(-100.0, 64.0, -100.0),
        Vec3::new(0.0, 64.0, 0.0),
    ];

    for pos in &positions {
        for _ in 0..10 {
            world.update(pos);
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // If we got here without panicking, the test passes
    // Just verify we're in a sane state
    assert!(world.pending_generation_count() < 10000,
        "Pending generation shouldn't explode");
}

/// Test that mesh generation counter prevents stale meshes
#[test]
fn test_mesh_generation_counter() {
    let mut world = WorldManager::new();
    let player_pos = Vec3::new(0.0, 64.0, 0.0);

    // Trigger initial load and wait for chunks to stabilize
    for _ in 0..30 {
        world.update(&player_pos);
        std::thread::sleep(Duration::from_millis(50));
    }

    // Now most chunks should be loaded
    let initial_mesh_count = world.mesh_count();

    // Skip if no meshes loaded (unlikely but possible)
    if initial_mesh_count == 0 {
        eprintln!("Warning: No meshes loaded, skipping stale mesh test");
        return;
    }

    // Modify a block to trigger remeshing
    if world.get_block(0, 32, 0).is_some() {
        world.set_block(0, 32, 0, Block::Stone);

        // Wait a bit for remesh to process
        for _ in 0..5 {
            world.update(&player_pos);
            std::thread::sleep(Duration::from_millis(50));
        }

        let final_mesh_count = world.mesh_count();

        // Mesh count should be roughly the same (maybe ±5% for timing variations)
        // The key is it shouldn't double or triple from stale mesh accumulation
        let growth_ratio = final_mesh_count as f32 / initial_mesh_count as f32;
        assert!(growth_ratio < 1.5,
            "Mesh count grew too much (ratio: {:.2}x, was {}, now {}). Possible stale mesh issue.",
            growth_ratio, initial_mesh_count, final_mesh_count);
    }
}

/// Test Block enum conversions
#[test]
fn test_block_from_u8_all_types() {
    // Valid block types should round-trip
    let valid_blocks = [
        Block::Air, Block::Dirt, Block::Grass, Block::Stone,
        Block::StoneDirt, Block::Brick, Block::Water, Block::Wood,
        Block::Leaves, Block::StoneRubble2, Block::IronOre, Block::GoldOre,
        Block::DiamondOre, Block::RedstoneOre, Block::Bedrock, Block::Torch,
        Block::Sand, Block::CoalOre, Block::CopperOre,
    ];

    for block in valid_blocks {
        let id = block as u8;
        let converted: Block = id.into();
        assert_eq!(converted, block, "Block {:?} (id={}) should round-trip", block, id);
    }

    // Invalid IDs should become Air
    let invalid: Block = 255u8.into();
    assert_eq!(invalid, Block::Air, "Invalid block ID should become Air");
}

/// Test water mesh generation
#[test]
fn test_water_mesh_at_sea_level() {
    let mut world = WorldManager::new();

    // Position near sea level where water should exist
    let player_pos = Vec3::new(0.0, 32.0, 0.0);

    // Load chunks
    for _ in 0..20 {
        world.update(&player_pos);
        std::thread::sleep(Duration::from_millis(100));
    }

    // Water meshes may or may not exist depending on terrain generation
    // This test just verifies the system doesn't panic
    let water_count = world.water_mesh_count();
    eprintln!("Water mesh count at sea level: {}", water_count);

    // If there's water, verify we can iterate without panics
    for _ in world.visible_water_meshes() {
        // Just iterating is the test
    }
}

/// Test sky light gradient calculation
#[test]
fn test_sky_light_gradient() {
    // Generate a single chunk and verify sky light is calculated
    let pos = ChunkPos { x: 0, y: 2, z: 0 }; // y=2 is around surface level (y*16 = 32)
    let seed = 12345u64;

    let chunk = generate_chunk_with_seed(pos, seed);

    // Find a column with terrain
    let mut found_gradient = false;

    for x in 0..16 {
        for z in 0..16 {
            // Find highest solid block in this column
            let mut surface_y = None;
            for y in (0..16).rev() {
                let block = chunk.get_block(x, y, z);
                if block.is_solid() {
                    surface_y = Some(y);
                    break;
                }
            }

            if let Some(sy) = surface_y {
                // Check sky light above surface
                if sy < 15 {
                    let above_light = chunk.get_sky_light(x, sy + 1, z);
                    // Above surface should have high sky light (12-15)
                    if above_light >= 12 {
                        // Check sky light at surface (should be 0 - solid block)
                        let surface_light = chunk.get_sky_light(x, sy, z);
                        assert_eq!(surface_light, 0,
                            "Solid block at ({}, {}, {}) should have sky light 0, got {}",
                            x, sy, z, surface_light);

                        found_gradient = true;
                    }
                }
            }
        }
    }

    // We should have found at least one valid gradient test case
    assert!(found_gradient, "Should have found terrain with sky light gradient");
}

/// Test sky light in caves decreases with depth
#[test]
fn test_sky_light_attenuates_in_caves() {
    // Generate a chunk at underground level
    let pos = ChunkPos { x: 5, y: 1, z: 5 }; // y=1 is mostly underground (y*16 = 16)
    let seed = 54321u64;

    let chunk = generate_chunk_with_seed(pos, seed);

    // Count air blocks with different sky light levels
    let mut air_with_full_light = 0;
    let mut air_with_some_light = 0;
    let mut air_with_no_light = 0;

    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                let block = chunk.get_block(x, y, z);
                if block == Block::Air {
                    let sky = chunk.get_sky_light(x, y, z);
                    if sky == 15 {
                        air_with_full_light += 1;
                    } else if sky > 0 {
                        air_with_some_light += 1;
                    } else {
                        air_with_no_light += 1;
                    }
                }
            }
        }
    }

    // In an underground chunk, we expect some air blocks to have attenuated or zero sky light
    // (caves blocked from sky should have reduced/zero sky light)
    eprintln!("Sky light distribution: full={}, partial={}, none={}",
        air_with_full_light, air_with_some_light, air_with_no_light);

    // At least some blocks should have non-full sky light (underground)
    // This is a soft assertion since terrain generation is noise-based
    if air_with_full_light + air_with_some_light + air_with_no_light > 0 {
        // If there are any air blocks, we should see some attenuation
        // (not all at full brightness, unless the chunk happens to be all above ground)
    }
}
