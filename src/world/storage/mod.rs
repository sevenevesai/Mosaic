//! World persistence and storage system
//!
//! Handles saving/loading chunks to disk using a region-based file format
//! with zstd compression for optimal performance and file size.

mod chunk_io;
mod dirty_tracker;
mod region;
mod world_meta;

pub use chunk_io::{deserialize_chunk, serialize_chunk};
pub use dirty_tracker::DirtyChunkTracker;
pub use region::{RegionFile, RegionPos};
pub use world_meta::WorldMeta;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use super::chunk::{Chunk, ChunkPos};

/// Maximum number of region files to keep open
const MAX_OPEN_REGIONS: usize = 16;

/// Errors that can occur during storage operations
#[derive(Debug)]
pub enum StorageError {
    /// I/O error from filesystem
    Io(io::Error),
    /// Invalid file format (wrong magic/version)
    InvalidFormat(String),
    /// Decompression failed
    DecompressionError(String),
    /// Chunk data corrupted
    CorruptedChunk(ChunkPos),
    /// Region file corrupted
    CorruptedRegion(RegionPos),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(e) => write!(f, "I/O error: {}", e),
            StorageError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
            StorageError::DecompressionError(s) => write!(f, "Decompression error: {}", s),
            StorageError::CorruptedChunk(pos) => write!(f, "Corrupted chunk at {:?}", pos),
            StorageError::CorruptedRegion(pos) => write!(f, "Corrupted region at {:?}", pos),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<io::Error> for StorageError {
    fn from(e: io::Error) -> Self {
        StorageError::Io(e)
    }
}

/// Main world storage coordinator
///
/// Manages saving and loading chunks to/from disk using region files.
/// Thread-safe and designed for async I/O operations.
pub struct WorldStorage {
    /// Base directory for this world save
    world_dir: PathBuf,
    /// Cached open region files (LRU cache)
    region_cache: LruCache<RegionPos, RegionFile>,
    /// Tracks which chunks have been modified since last save
    dirty_tracker: DirtyChunkTracker,
    /// World metadata (seed, spawn, timestamps)
    meta: WorldMeta,
}

impl WorldStorage {
    /// Create or load a world at the given path
    ///
    /// If the world exists, loads metadata. Otherwise creates a new world.
    pub fn open(path: &Path, seed: Option<u64>) -> Result<Self, StorageError> {
        let world_dir = path.to_path_buf();
        let meta_path = world_dir.join("world.dat");
        let region_dir = world_dir.join("region");

        // Load existing or create new
        let meta = if meta_path.exists() {
            WorldMeta::load(&meta_path)?
        } else {
            // Create directories
            std::fs::create_dir_all(&region_dir)?;

            // Create new metadata with provided or default seed
            let new_meta = WorldMeta::new(seed.unwrap_or(12345));
            new_meta.save(&meta_path)?;
            new_meta
        };

        Ok(Self {
            world_dir,
            region_cache: LruCache::new(std::num::NonZeroUsize::new(MAX_OPEN_REGIONS).unwrap()),
            dirty_tracker: DirtyChunkTracker::new(),
            meta,
        })
    }

    /// Get the world seed
    pub fn seed(&self) -> u64 {
        self.meta.seed
    }

    /// Get the region directory path
    fn region_dir(&self) -> PathBuf {
        self.world_dir.join("region")
    }

    /// Get the path to a region file
    fn region_path(&self, pos: RegionPos) -> PathBuf {
        self.region_dir().join(format!("r.{}.{}.{}.region", pos.x, pos.y, pos.z))
    }

    /// Load a chunk from disk
    ///
    /// Returns None if the chunk has never been saved.
    pub fn load_chunk(&mut self, pos: ChunkPos) -> Result<Option<Chunk>, StorageError> {
        let region_pos = RegionPos::from_chunk(pos);
        let region_path = self.region_path(region_pos);

        // Check if region file exists
        if !region_path.exists() {
            return Ok(None);
        }

        // Get or open the region file
        let region = self.get_or_open_region(region_pos)?;

        // Try to load the chunk
        region.load_chunk(pos)
    }

    /// Save a chunk to disk
    pub fn save_chunk(&mut self, pos: ChunkPos, chunk: &Chunk) -> Result<(), StorageError> {
        let region_pos = RegionPos::from_chunk(pos);

        // Get or create the region file
        let region = self.get_or_create_region(region_pos)?;

        // Save the chunk
        region.save_chunk(pos, chunk)?;

        // Clear dirty flag
        self.dirty_tracker.clear_dirty(pos);

        Ok(())
    }

    /// Mark a chunk as dirty (needs saving)
    pub fn mark_dirty(&mut self, pos: ChunkPos) {
        self.dirty_tracker.mark_dirty(pos);
    }

    /// Check if a chunk is dirty
    pub fn is_dirty(&self, pos: ChunkPos) -> bool {
        self.dirty_tracker.is_dirty(pos)
    }

    /// Get all dirty chunk positions and clear the dirty set
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkPos> {
        self.dirty_tracker.take_dirty_chunks()
    }

    /// Get dirty chunk count
    pub fn dirty_count(&self) -> usize {
        self.dirty_tracker.count()
    }

    /// Save world metadata
    pub fn save_meta(&mut self) -> Result<(), StorageError> {
        self.meta.update_last_played();
        let meta_path = self.world_dir.join("world.dat");
        self.meta.save(&meta_path)?;
        Ok(())
    }

    /// Get player spawn position
    pub fn spawn_position(&self) -> [f32; 3] {
        self.meta.spawn_position()
    }

    /// Set player spawn position
    pub fn set_spawn(&mut self, x: f32, y: f32, z: f32) {
        self.meta.set_spawn(x, y, z);
    }

    /// Flush all open region files to disk
    pub fn flush(&mut self) -> Result<(), StorageError> {
        // Flush all cached regions
        for (_, region) in self.region_cache.iter_mut() {
            region.flush()?;
        }
        Ok(())
    }

    /// Get or open a region file from cache
    fn get_or_open_region(&mut self, pos: RegionPos) -> Result<&mut RegionFile, StorageError> {
        if !self.region_cache.contains(&pos) {
            let path = self.region_path(pos);
            let region = RegionFile::open(&path)?;
            self.region_cache.put(pos, region);
        }
        Ok(self.region_cache.get_mut(&pos).unwrap())
    }

    /// Get or create a region file
    fn get_or_create_region(&mut self, pos: RegionPos) -> Result<&mut RegionFile, StorageError> {
        if !self.region_cache.contains(&pos) {
            let path = self.region_path(pos);

            // Ensure region directory exists
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let region = if path.exists() {
                RegionFile::open(&path)?
            } else {
                RegionFile::create(&path)?
            };
            self.region_cache.put(pos, region);
        }
        Ok(self.region_cache.get_mut(&pos).unwrap())
    }
}

/// Thread-safe wrapper for WorldStorage
pub type SharedStorage = Arc<Mutex<WorldStorage>>;

/// Create a shared storage instance
pub fn create_shared_storage(path: &Path, seed: Option<u64>) -> Result<SharedStorage, StorageError> {
    let storage = WorldStorage::open(path, seed)?;
    Ok(Arc::new(Mutex::new(storage)))
}
