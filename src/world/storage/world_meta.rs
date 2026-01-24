//! World metadata storage
//!
//! Stores and loads world-level information like seed, spawn point, and timestamps.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::StorageError;

/// Magic bytes for world.dat file: "BLKW"
const WORLD_MAGIC: [u8; 4] = [0x42, 0x4C, 0x4B, 0x57];

/// Current format version
const WORLD_VERSION: u16 = 1;

/// Fixed header size in bytes
const HEADER_SIZE: usize = 224;

/// World metadata stored in world.dat
#[derive(Debug, Clone)]
pub struct WorldMeta {
    /// World generation seed
    pub seed: u64,
    /// Player spawn position
    pub spawn: [f32; 3],
    /// World creation timestamp (Unix millis)
    pub created: u64,
    /// Last played timestamp (Unix millis)
    pub last_played: u64,
    /// Total play time in seconds
    pub play_time_secs: u32,
    /// Game mode (0 = survival, 1 = creative)
    pub game_mode: u32,
}

impl WorldMeta {
    /// Create new world metadata with the given seed
    pub fn new(seed: u64) -> Self {
        let now = current_time_millis();
        Self {
            seed,
            spawn: [0.0, 80.0, 0.0], // Default spawn above terrain
            created: now,
            last_played: now,
            play_time_secs: 0,
            game_mode: 1, // Creative by default
        }
    }

    /// Update last played timestamp
    pub fn update_last_played(&mut self) {
        self.last_played = current_time_millis();
    }

    /// Set player spawn position
    pub fn set_spawn(&mut self, x: f32, y: f32, z: f32) {
        self.spawn = [x, y, z];
    }

    /// Get player spawn position
    pub fn spawn_position(&self) -> [f32; 3] {
        self.spawn
    }

    /// Load world metadata from file
    pub fn load(path: &Path) -> Result<Self, StorageError> {
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_SIZE];

        file.read_exact(&mut header)?;

        // Verify magic
        if header[0..4] != WORLD_MAGIC {
            return Err(StorageError::InvalidFormat(
                "Invalid world.dat magic bytes".to_string(),
            ));
        }

        // Check version
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version > WORLD_VERSION {
            return Err(StorageError::InvalidFormat(format!(
                "Unsupported world version {} (max supported: {})",
                version, WORLD_VERSION
            )));
        }

        // Parse fields
        let seed = u64::from_le_bytes(header[8..16].try_into().unwrap());

        let spawn = [
            f32::from_le_bytes(header[16..20].try_into().unwrap()),
            f32::from_le_bytes(header[20..24].try_into().unwrap()),
            f32::from_le_bytes(header[24..28].try_into().unwrap()),
        ];

        let created = u64::from_le_bytes(header[28..36].try_into().unwrap());
        let last_played = u64::from_le_bytes(header[36..44].try_into().unwrap());
        let play_time_secs = u32::from_le_bytes(header[44..48].try_into().unwrap());
        let game_mode = u32::from_le_bytes(header[48..52].try_into().unwrap());

        Ok(Self {
            seed,
            spawn,
            created,
            last_played,
            play_time_secs,
            game_mode,
        })
    }

    /// Save world metadata to file
    pub fn save(&self, path: &Path) -> Result<(), StorageError> {
        let mut header = [0u8; HEADER_SIZE];

        // Magic
        header[0..4].copy_from_slice(&WORLD_MAGIC);

        // Version
        header[4..6].copy_from_slice(&WORLD_VERSION.to_le_bytes());

        // Reserved/flags (bytes 6-7)
        header[6..8].copy_from_slice(&[0u8; 2]);

        // Seed
        header[8..16].copy_from_slice(&self.seed.to_le_bytes());

        // Spawn position
        header[16..20].copy_from_slice(&self.spawn[0].to_le_bytes());
        header[20..24].copy_from_slice(&self.spawn[1].to_le_bytes());
        header[24..28].copy_from_slice(&self.spawn[2].to_le_bytes());

        // Timestamps
        header[28..36].copy_from_slice(&self.created.to_le_bytes());
        header[36..44].copy_from_slice(&self.last_played.to_le_bytes());

        // Play time
        header[44..48].copy_from_slice(&self.play_time_secs.to_le_bytes());

        // Game mode
        header[48..52].copy_from_slice(&self.game_mode.to_le_bytes());

        // Rest is reserved (already zeroed)

        // Write to file
        let mut file = File::create(path)?;
        file.write_all(&header)?;
        file.flush()?;

        Ok(())
    }
}

/// Get current time in milliseconds since Unix epoch
fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_world_meta_roundtrip() {
        let meta = WorldMeta::new(12345);

        // Create temp file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save
        meta.save(path).unwrap();

        // Load
        let loaded = WorldMeta::load(path).unwrap();

        assert_eq!(meta.seed, loaded.seed);
        assert_eq!(meta.spawn, loaded.spawn);
        assert_eq!(meta.game_mode, loaded.game_mode);
    }
}
