//! Region file format
//!
//! Each region file stores a 32x32x32 grid of chunks.
//! Format inspired by Minecraft's region files but extended for 3D.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

use super::chunk_io::{serialize_chunk, deserialize_chunk};
use super::StorageError;
use crate::world::chunk::{Chunk, ChunkPos};

/// Magic bytes for region file: "BLKR"
const REGION_MAGIC: [u8; 4] = [0x42, 0x4C, 0x4B, 0x52];

/// Current region format version
const REGION_VERSION: u16 = 1;

/// Region size in chunks per axis (32x32x32 = 32768 chunks)
const REGION_SIZE: i32 = 32;

/// Number of chunks per region
const CHUNKS_PER_REGION: usize = (REGION_SIZE * REGION_SIZE * REGION_SIZE) as usize;

/// Sector size (4KB, matches typical filesystem block size)
const SECTOR_SIZE: usize = 4096;

/// Header size: magic(4) + version(2) + flags(2) = 8 bytes
const HEADER_SIZE: usize = 8;

/// Chunk table size: 32768 entries * 8 bytes = 262144 bytes
const CHUNK_TABLE_SIZE: usize = CHUNKS_PER_REGION * 8;

/// First sector for chunk data (after header + table, aligned to 4KB)
const FIRST_DATA_SECTOR: u32 = ((HEADER_SIZE + CHUNK_TABLE_SIZE + SECTOR_SIZE - 1) / SECTOR_SIZE) as u32;

/// Region position (chunk coordinates >> 5)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl RegionPos {
    /// Calculate region position from chunk position
    pub fn from_chunk(pos: ChunkPos) -> Self {
        Self {
            x: pos.x >> 5, // divide by 32
            y: pos.y >> 5,
            z: pos.z >> 5,
        }
    }
}

/// Chunk table entry (8 bytes)
#[derive(Debug, Clone, Copy, Default)]
struct ChunkTableEntry {
    /// Sector offset (0 = chunk not present)
    sector_offset: u32,
    /// Number of sectors used
    sector_count: u8,
    /// Timestamp (lower 24 bits of hours since epoch)
    timestamp: [u8; 3],
}

impl ChunkTableEntry {
    fn is_present(&self) -> bool {
        self.sector_offset != 0
    }

    fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&self.sector_offset.to_le_bytes());
        bytes[4] = self.sector_count;
        bytes[5..8].copy_from_slice(&self.timestamp);
        bytes
    }

    fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            sector_offset: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            sector_count: bytes[4],
            timestamp: [bytes[5], bytes[6], bytes[7]],
        }
    }
}

/// Region file handler
pub struct RegionFile {
    /// Underlying file
    file: File,
    /// Path to the file (kept for debugging/logging)
    #[allow(dead_code)]
    path: std::path::PathBuf,
    /// Chunk lookup table
    chunk_table: Vec<ChunkTableEntry>,
    /// Next available sector for writing
    next_sector: u32,
}

impl RegionFile {
    /// Create a new region file
    pub fn create(path: &Path) -> Result<Self, StorageError> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Write header
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&REGION_MAGIC);
        header[4..6].copy_from_slice(&REGION_VERSION.to_le_bytes());
        // bytes 6-7 are flags (reserved)
        file.write_all(&header)?;

        // Write empty chunk table
        let empty_table = vec![0u8; CHUNK_TABLE_SIZE];
        file.write_all(&empty_table)?;

        // Pad to first data sector
        let current_pos = HEADER_SIZE + CHUNK_TABLE_SIZE;
        let padding_needed = (FIRST_DATA_SECTOR as usize * SECTOR_SIZE) - current_pos;
        if padding_needed > 0 {
            let padding = vec![0u8; padding_needed];
            file.write_all(&padding)?;
        }

        file.flush()?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            chunk_table: vec![ChunkTableEntry::default(); CHUNKS_PER_REGION],
            next_sector: FIRST_DATA_SECTOR,
        })
    }

    /// Open an existing region file
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        // Read and verify header
        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)?;

        if header[0..4] != REGION_MAGIC {
            return Err(StorageError::InvalidFormat(
                "Invalid region file magic bytes".to_string(),
            ));
        }

        let version = u16::from_le_bytes([header[4], header[5]]);
        if version > REGION_VERSION {
            return Err(StorageError::InvalidFormat(format!(
                "Unsupported region version {} (max: {})",
                version, REGION_VERSION
            )));
        }

        // Read chunk table
        let mut table_bytes = vec![0u8; CHUNK_TABLE_SIZE];
        file.read_exact(&mut table_bytes)?;

        let mut chunk_table = Vec::with_capacity(CHUNKS_PER_REGION);
        let mut max_sector: u32 = FIRST_DATA_SECTOR;

        for i in 0..CHUNKS_PER_REGION {
            let offset = i * 8;
            let entry = ChunkTableEntry::from_bytes(
                table_bytes[offset..offset + 8].try_into().unwrap(),
            );

            if entry.is_present() {
                let end_sector = entry.sector_offset + entry.sector_count as u32;
                if end_sector > max_sector {
                    max_sector = end_sector;
                }
            }

            chunk_table.push(entry);
        }

        Ok(Self {
            file,
            path: path.to_path_buf(),
            chunk_table,
            next_sector: max_sector,
        })
    }

    /// Calculate chunk index in the table from chunk position
    fn chunk_index(pos: ChunkPos) -> usize {
        // Local position within region (0-31 for each axis)
        let lx = pos.x.rem_euclid(REGION_SIZE) as usize;
        let ly = pos.y.rem_euclid(REGION_SIZE) as usize;
        let lz = pos.z.rem_euclid(REGION_SIZE) as usize;

        lx + ly * REGION_SIZE as usize + lz * REGION_SIZE as usize * REGION_SIZE as usize
    }

    /// Load a chunk from the region file
    pub fn load_chunk(&mut self, pos: ChunkPos) -> Result<Option<Chunk>, StorageError> {
        let index = Self::chunk_index(pos);
        let entry = &self.chunk_table[index];

        if !entry.is_present() {
            return Ok(None);
        }

        // Seek to chunk data
        let offset = entry.sector_offset as u64 * SECTOR_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        // Read chunk data (sector_count * SECTOR_SIZE bytes)
        let data_size = entry.sector_count as usize * SECTOR_SIZE;
        let mut data = vec![0u8; data_size];
        self.file.read_exact(&mut data)?;

        // Deserialize
        let chunk = deserialize_chunk(&data)?;
        Ok(Some(chunk))
    }

    /// Save a chunk to the region file
    pub fn save_chunk(&mut self, pos: ChunkPos, chunk: &Chunk) -> Result<(), StorageError> {
        // Serialize chunk
        let data = serialize_chunk(chunk)?;

        // Calculate sectors needed
        let sectors_needed = ((data.len() + SECTOR_SIZE - 1) / SECTOR_SIZE) as u8;

        // Check if we can reuse the existing allocation
        let index = Self::chunk_index(pos);
        let old_entry = &self.chunk_table[index];

        let sector_offset = if old_entry.is_present() && old_entry.sector_count >= sectors_needed {
            // Reuse existing location - chunk fits in same or smaller space
            // This handles the common case of minor chunk modifications
            old_entry.sector_offset
        } else {
            // Need new space - append to end
            // Old sectors become orphaned (acceptable fragmentation)
            let offset = self.next_sector;
            self.next_sector += sectors_needed as u32;
            offset
        };

        // Pad data to sector boundary
        let padded_size = sectors_needed as usize * SECTOR_SIZE;
        let mut padded_data = data;
        padded_data.resize(padded_size, 0);

        // Write chunk data
        let offset = sector_offset as u64 * SECTOR_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&padded_data)?;

        // Update chunk table (index already computed above)
        self.chunk_table[index] = ChunkTableEntry {
            sector_offset,
            sector_count: sectors_needed,
            timestamp: current_timestamp_24(),
        };

        // Write table entry to file
        let table_offset = (HEADER_SIZE + index * 8) as u64;
        self.file.seek(SeekFrom::Start(table_offset))?;
        self.file.write_all(&self.chunk_table[index].to_bytes())?;

        Ok(())
    }

    /// Check if a chunk exists in this region
    pub fn has_chunk(&self, pos: ChunkPos) -> bool {
        let index = Self::chunk_index(pos);
        self.chunk_table[index].is_present()
    }

    /// Flush all pending writes to disk
    pub fn flush(&mut self) -> Result<(), StorageError> {
        self.file.flush()?;
        Ok(())
    }
}

/// Get current timestamp as 24-bit value (hours since epoch, wraps every ~1900 years)
fn current_timestamp_24() -> [u8; 3] {
    use std::time::{SystemTime, UNIX_EPOCH};

    let hours = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() / 3600;

    let bytes = (hours as u32).to_le_bytes();
    [bytes[0], bytes[1], bytes[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::Block;
    use tempfile::tempdir;

    #[test]
    fn test_region_file_create_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.region");

        // Create region file
        {
            let mut region = RegionFile::create(&path).unwrap();

            // Create a test chunk
            let mut chunk = Chunk::new();
            chunk.set_block(0, 0, 0, Block::Stone);
            chunk.set_block(15, 15, 15, Block::Dirt);

            // Save it
            let pos = ChunkPos { x: 0, y: 0, z: 0 };
            region.save_chunk(pos, &chunk).unwrap();
            region.flush().unwrap();
        }

        // Reopen and load
        {
            let mut region = RegionFile::open(&path).unwrap();

            let pos = ChunkPos { x: 0, y: 0, z: 0 };
            let chunk = region.load_chunk(pos).unwrap().unwrap();

            assert_eq!(chunk.get_block(0, 0, 0), Block::Stone);
            assert_eq!(chunk.get_block(15, 15, 15), Block::Dirt);
            assert_eq!(chunk.get_block(8, 8, 8), Block::Air);
        }
    }

    #[test]
    fn test_region_pos_from_chunk() {
        // Chunk 0,0,0 -> Region 0,0,0
        assert_eq!(
            RegionPos::from_chunk(ChunkPos { x: 0, y: 0, z: 0 }),
            RegionPos { x: 0, y: 0, z: 0 }
        );

        // Chunk 31,31,31 -> Region 0,0,0
        assert_eq!(
            RegionPos::from_chunk(ChunkPos { x: 31, y: 31, z: 31 }),
            RegionPos { x: 0, y: 0, z: 0 }
        );

        // Chunk 32,0,0 -> Region 1,0,0
        assert_eq!(
            RegionPos::from_chunk(ChunkPos { x: 32, y: 0, z: 0 }),
            RegionPos { x: 1, y: 0, z: 0 }
        );

        // Negative chunks
        assert_eq!(
            RegionPos::from_chunk(ChunkPos { x: -1, y: 0, z: 0 }),
            RegionPos { x: -1, y: 0, z: 0 }
        );

        assert_eq!(
            RegionPos::from_chunk(ChunkPos { x: -32, y: 0, z: 0 }),
            RegionPos { x: -1, y: 0, z: 0 }
        );

        assert_eq!(
            RegionPos::from_chunk(ChunkPos { x: -33, y: 0, z: 0 }),
            RegionPos { x: -2, y: 0, z: 0 }
        );
    }
}
