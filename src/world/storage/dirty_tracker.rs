//! Dirty chunk tracking
//!
//! Tracks which chunks have been modified and need saving.

use ahash::AHashSet;

use crate::world::chunk::ChunkPos;

/// Tracks chunks that have been modified and need saving
#[derive(Debug)]
pub struct DirtyChunkTracker {
    /// Set of dirty chunk positions
    dirty: AHashSet<ChunkPos>,
}

impl DirtyChunkTracker {
    /// Create a new empty tracker
    pub fn new() -> Self {
        Self {
            dirty: AHashSet::new(),
        }
    }

    /// Mark a chunk as dirty (needs saving)
    pub fn mark_dirty(&mut self, pos: ChunkPos) {
        self.dirty.insert(pos);
    }

    /// Clear the dirty flag for a chunk
    pub fn clear_dirty(&mut self, pos: ChunkPos) {
        self.dirty.remove(&pos);
    }

    /// Check if a chunk is dirty
    pub fn is_dirty(&self, pos: ChunkPos) -> bool {
        self.dirty.contains(&pos)
    }

    /// Get all dirty chunk positions and clear the set
    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkPos> {
        let chunks: Vec<_> = self.dirty.drain().collect();
        chunks
    }

    /// Get the number of dirty chunks
    pub fn count(&self) -> usize {
        self.dirty.len()
    }

    /// Clear all dirty flags
    pub fn clear_all(&mut self) {
        self.dirty.clear();
    }

    /// Get an iterator over dirty chunk positions
    pub fn iter(&self) -> impl Iterator<Item = &ChunkPos> {
        self.dirty.iter()
    }
}

impl Default for DirtyChunkTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirty_tracking() {
        let mut tracker = DirtyChunkTracker::new();

        let pos1 = ChunkPos { x: 0, y: 0, z: 0 };
        let pos2 = ChunkPos { x: 1, y: 0, z: 0 };

        // Initially not dirty
        assert!(!tracker.is_dirty(pos1));
        assert_eq!(tracker.count(), 0);

        // Mark dirty
        tracker.mark_dirty(pos1);
        assert!(tracker.is_dirty(pos1));
        assert!(!tracker.is_dirty(pos2));
        assert_eq!(tracker.count(), 1);

        // Mark same position again (no change)
        tracker.mark_dirty(pos1);
        assert_eq!(tracker.count(), 1);

        // Mark another position
        tracker.mark_dirty(pos2);
        assert_eq!(tracker.count(), 2);

        // Clear one
        tracker.clear_dirty(pos1);
        assert!(!tracker.is_dirty(pos1));
        assert!(tracker.is_dirty(pos2));
        assert_eq!(tracker.count(), 1);

        // Take all
        let dirty = tracker.take_dirty_chunks();
        assert_eq!(dirty.len(), 1);
        assert_eq!(tracker.count(), 0);
    }
}
