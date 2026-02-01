//! Water Refraction Quality System
//!
//! Controls the visual quality of water refraction effects including:
//! - Wave-based color distortion strength
//! - Chromatic aberration (color separation)
//! - Normal perturbation intensity
//!
//! The refraction effect creates the illusion of light bending through water
//! by modulating colors based on wave normals and depth.

/// Water refraction quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefractionQuality {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl RefractionQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            RefractionQuality::Off => "OFF",
            RefractionQuality::Low => "LOW",
            RefractionQuality::Medium => "MEDIUM",
            RefractionQuality::High => "HIGH",
            RefractionQuality::Ultra => "ULTRA",
        }
    }

    pub fn next(&self) -> RefractionQuality {
        match self {
            RefractionQuality::Off => RefractionQuality::Low,
            RefractionQuality::Low => RefractionQuality::Medium,
            RefractionQuality::Medium => RefractionQuality::High,
            RefractionQuality::High => RefractionQuality::Ultra,
            RefractionQuality::Ultra => RefractionQuality::Off,
        }
    }

    /// Get refraction strength multiplier for this quality level (0.0 - 1.0)
    pub fn strength(&self) -> f32 {
        match self {
            RefractionQuality::Off => 0.0,
            RefractionQuality::Low => 0.5,
            RefractionQuality::Medium => 0.75,
            RefractionQuality::High => 0.9,
            RefractionQuality::Ultra => 1.0,
        }
    }

    /// Get chromatic aberration strength (0.0 = none, higher = more color separation)
    pub fn chromatic_strength(&self) -> f32 {
        match self {
            RefractionQuality::Off => 0.0,
            RefractionQuality::Low => 0.0,
            RefractionQuality::Medium => 0.0,
            RefractionQuality::High => 0.01,
            RefractionQuality::Ultra => 0.02,
        }
    }

    /// Get wave detail level (affects visual complexity)
    pub fn wave_octaves(&self) -> u32 {
        match self {
            RefractionQuality::Off => 0,
            RefractionQuality::Low => 1,
            RefractionQuality::Medium => 2,
            RefractionQuality::High => 3,
            RefractionQuality::Ultra => 4,
        }
    }

    /// Whether depth-based distortion falloff is enabled
    pub fn depth_falloff(&self) -> bool {
        !matches!(self, RefractionQuality::Off)
    }

    /// Get the base distortion scale (UV offset multiplier)
    /// Values increased significantly for visible effect
    pub fn distortion_scale(&self) -> f32 {
        match self {
            RefractionQuality::Off => 0.0,
            RefractionQuality::Low => 0.15,
            RefractionQuality::Medium => 0.25,
            RefractionQuality::High => 0.35,
            RefractionQuality::Ultra => 0.5,
        }
    }
}

/// Refraction parameters (runtime adjustable)
#[derive(Debug, Clone, Copy)]
pub struct RefractionParams {
    /// Overall refraction effect strength (0.0 - 1.0)
    pub strength: f32,
    /// Chromatic aberration intensity (0.0 - 0.01)
    pub chromatic: f32,
    /// Distortion scale factor (0.0 - 0.1)
    pub distortion_scale: f32,
    /// Number of wave octaves for detail (1-4)
    pub wave_octaves: u32,
    /// Depth at which refraction starts fading (blocks)
    pub depth_fade_start: f32,
    /// Depth at which refraction is fully faded (blocks)
    pub depth_fade_end: f32,
}

impl Default for RefractionParams {
    fn default() -> Self {
        Self {
            strength: 0.5,
            chromatic: 0.0,
            distortion_scale: 0.25,  // Increased for visibility
            wave_octaves: 2,
            depth_fade_start: 2.0,
            depth_fade_end: 12.0,
        }
    }
}

impl RefractionParams {
    /// Create params from a quality preset
    pub fn from_quality(quality: RefractionQuality) -> Self {
        Self {
            strength: quality.strength(),
            chromatic: quality.chromatic_strength(),
            distortion_scale: quality.distortion_scale(),
            wave_octaves: quality.wave_octaves(),
            depth_fade_start: 2.0,
            depth_fade_end: 12.0,
        }
    }
}
