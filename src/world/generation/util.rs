//! Utility functions for terrain generation
//!
//! Provides fast hash functions and noise helpers shared across generation modules.

/// Fast integer hash for deterministic pseudo-randomness without noise overhead
#[inline]
pub fn fast_hash(x: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393);
    h = h.wrapping_add((z as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

/// Fast 3D hash for volumetric lookups
#[inline]
pub fn fast_hash_3d(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393);
    h = h.wrapping_add((y as u32).wrapping_mul(668265263));
    h = h.wrapping_add((z as u32).wrapping_mul(1440670291));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

/// Fast 3D hash with seed for seeded pseudo-randomness
#[inline]
pub fn fast_hash_3d_seeded(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393);
    h = h.wrapping_add((y as u32).wrapping_mul(668265263));
    h = h.wrapping_add((z as u32).wrapping_mul(1440670291));
    h = h.wrapping_add(seed.wrapping_mul(2166136261));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

/// Convert hash to float in range [0, 1)
#[inline]
pub fn hash_to_float(hash: u32) -> f32 {
    (hash & 0xFFFFFF) as f32 / 16777216.0
}

/// Smooth interpolation function (smoothstep)
#[inline]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Linear interpolation for f32
#[inline]
pub fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_hash_deterministic() {
        // Same inputs should produce same outputs
        assert_eq!(fast_hash(0, 0), fast_hash(0, 0));
        assert_eq!(fast_hash(100, -50), fast_hash(100, -50));
    }

    #[test]
    fn test_fast_hash_different_inputs() {
        // Different inputs should (usually) produce different outputs
        assert_ne!(fast_hash(0, 0), fast_hash(1, 0));
        assert_ne!(fast_hash(0, 0), fast_hash(0, 1));
    }

    #[test]
    fn test_fast_hash_3d_deterministic() {
        assert_eq!(fast_hash_3d(0, 0, 0), fast_hash_3d(0, 0, 0));
        assert_eq!(fast_hash_3d(10, 20, 30), fast_hash_3d(10, 20, 30));
    }

    #[test]
    fn test_hash_to_float_range() {
        for i in 0..1000 {
            let f = hash_to_float(fast_hash(i, i * 7));
            assert!(f >= 0.0 && f < 1.0, "hash_to_float should be in [0, 1)");
        }
    }

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 0.001);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 0.001);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 0.001);
        // Clamping
        assert!((smoothstep(0.0, 1.0, -0.5) - 0.0).abs() < 0.001);
        assert!((smoothstep(0.0, 1.0, 1.5) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < 0.001);
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 0.001);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 0.001);
    }
}
