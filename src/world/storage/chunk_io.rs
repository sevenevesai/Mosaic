//! Chunk serialization and compression
//!
//! Handles converting chunks to/from compressed binary format.

use std::io;

use super::StorageError;
use crate::world::chunk::{Block, Chunk};
use crate::constants::CHUNK_SIZE;

/// Compression level for zstd (1-22, 3 is a good balance)
const ZSTD_LEVEL: i32 = 3;

/// Chunk payload flags
const FLAG_HAS_BLOCKS: u8 = 0x01;
const FLAG_HAS_LIGHT: u8 = 0x02;
#[allow(dead_code)]
const FLAG_HAS_EXTRA: u8 = 0x04;  // Reserved for future use

/// Compression type identifiers
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionType {
    None = 0,
    Zstd = 1,
    Lz4 = 2, // Reserved for future
}

impl From<u16> for CompressionType {
    fn from(v: u16) -> Self {
        match v {
            0 => CompressionType::None,
            1 => CompressionType::Zstd,
            2 => CompressionType::Lz4,
            _ => CompressionType::None,
        }
    }
}

/// Chunk data header (12 bytes)
#[derive(Debug, Clone, Copy)]
pub struct ChunkDataHeader {
    /// Uncompressed payload size
    pub uncompressed_size: u32,
    /// Compressed payload size
    pub compressed_size: u32,
    /// Compression type
    pub compression: CompressionType,
    /// Chunk data format version
    pub version: u16,
}

impl ChunkDataHeader {
    pub const SIZE: usize = 12;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.compressed_size.to_le_bytes());
        bytes[8..10].copy_from_slice(&(self.compression as u16).to_le_bytes());
        bytes[10..12].copy_from_slice(&self.version.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        Self {
            uncompressed_size: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            compressed_size: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            compression: CompressionType::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap())),
            version: u16::from_le_bytes(bytes[10..12].try_into().unwrap()),
        }
    }
}

/// Serialize a chunk to compressed bytes
///
/// Format:
/// - 1 byte: flags (has_blocks, has_light, has_extra)
/// - 4096 bytes: block data (if FLAG_HAS_BLOCKS)
/// - 4096 bytes: light data (if FLAG_HAS_LIGHT)
/// - variable: extra data (if FLAG_HAS_EXTRA) - future use
pub fn serialize_chunk(chunk: &Chunk) -> Result<Vec<u8>, StorageError> {
    const BLOCKS_COUNT: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

    // Build uncompressed payload
    // flags (1) + blocks (4096) + light (4096) = 8193 bytes
    let mut payload = Vec::with_capacity(1 + BLOCKS_COUNT * 2);

    // Flags: we always save blocks and light
    let flags = FLAG_HAS_BLOCKS | FLAG_HAS_LIGHT;
    payload.push(flags);

    // Block data
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                payload.push(chunk.get_block(x, y, z) as u8);
            }
        }
    }

    // Light data
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                payload.push(chunk.get_light(x, y, z));
            }
        }
    }

    // Compress with zstd
    let compressed = zstd::encode_all(&payload[..], ZSTD_LEVEL)
        .map_err(|e| StorageError::Io(io::Error::new(io::ErrorKind::Other, e)))?;

    // Build final output with header
    let header = ChunkDataHeader {
        uncompressed_size: payload.len() as u32,
        compressed_size: compressed.len() as u32,
        compression: CompressionType::Zstd,
        version: 1,
    };

    let mut result = Vec::with_capacity(ChunkDataHeader::SIZE + compressed.len());
    result.extend_from_slice(&header.to_bytes());
    result.extend_from_slice(&compressed);

    Ok(result)
}

/// Deserialize a chunk from compressed bytes
pub fn deserialize_chunk(data: &[u8]) -> Result<Chunk, StorageError> {
    const BLOCKS_COUNT: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

    if data.len() < ChunkDataHeader::SIZE {
        return Err(StorageError::InvalidFormat(
            "Chunk data too small for header".to_string(),
        ));
    }

    // Parse header
    let header = ChunkDataHeader::from_bytes(data[0..ChunkDataHeader::SIZE].try_into().unwrap());

    let compressed_data = &data[ChunkDataHeader::SIZE..];
    if compressed_data.len() < header.compressed_size as usize {
        return Err(StorageError::InvalidFormat(
            "Chunk data truncated".to_string(),
        ));
    }

    // Decompress
    let payload = match header.compression {
        CompressionType::None => compressed_data[..header.compressed_size as usize].to_vec(),
        CompressionType::Zstd => {
            zstd::decode_all(&compressed_data[..header.compressed_size as usize])
                .map_err(|e| StorageError::DecompressionError(e.to_string()))?
        }
        CompressionType::Lz4 => {
            return Err(StorageError::InvalidFormat(
                "LZ4 compression not yet supported".to_string(),
            ));
        }
    };

    if payload.len() < 1 {
        return Err(StorageError::InvalidFormat(
            "Empty chunk payload".to_string(),
        ));
    }

    // Parse flags
    let flags = payload[0];
    let has_blocks = flags & FLAG_HAS_BLOCKS != 0;
    let has_light = flags & FLAG_HAS_LIGHT != 0;

    let mut chunk = Chunk::new();
    let mut offset = 1;

    // Read blocks
    if has_blocks {
        if payload.len() < offset + BLOCKS_COUNT {
            return Err(StorageError::InvalidFormat(
                "Chunk block data truncated".to_string(),
            ));
        }

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let block_byte = payload[offset];
                    let block = Block::from(block_byte);
                    chunk.set_block(x, y, z, block);
                    offset += 1;
                }
            }
        }
    }

    // Read light
    if has_light {
        if payload.len() < offset + BLOCKS_COUNT {
            return Err(StorageError::InvalidFormat(
                "Chunk light data truncated".to_string(),
            ));
        }

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    chunk.set_light(x, y, z, payload[offset]);
                    offset += 1;
                }
            }
        }
    }

    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_roundtrip() {
        let mut chunk = Chunk::new();

        // Set some blocks
        chunk.set_block(0, 0, 0, Block::Stone);
        chunk.set_block(1, 1, 1, Block::Dirt);
        chunk.set_block(15, 15, 15, Block::Grass);

        // Set some light
        chunk.set_light(0, 0, 0, 0xF0); // Sky light 15
        chunk.set_light(1, 1, 1, 0x0E); // Block light 14

        // Serialize
        let data = serialize_chunk(&chunk).unwrap();

        // Should be compressed (smaller than raw 8KB)
        assert!(data.len() < 8192);

        // Deserialize
        let loaded = deserialize_chunk(&data).unwrap();

        // Verify blocks
        assert_eq!(loaded.get_block(0, 0, 0), Block::Stone);
        assert_eq!(loaded.get_block(1, 1, 1), Block::Dirt);
        assert_eq!(loaded.get_block(15, 15, 15), Block::Grass);
        assert_eq!(loaded.get_block(8, 8, 8), Block::Air);

        // Verify light
        assert_eq!(loaded.get_light(0, 0, 0), 0xF0);
        assert_eq!(loaded.get_light(1, 1, 1), 0x0E);
    }
}
