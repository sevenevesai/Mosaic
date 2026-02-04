//! Settings persistence
//!
//! Saves and loads game settings to/from a simple text file.

use std::fs;
use std::path::Path;

use crate::renderer::DisplayMode;

/// Game settings that persist between sessions
#[derive(Debug, Clone)]
pub struct Settings {
    pub render_distance: i32,
    pub shadow_mode: f32,
    pub bloom_enabled: bool,
    pub volumetrics_enabled: bool,
    pub gpu_culling: bool,
    pub gpu_voxel_fill: bool,
    pub cycle_speed: f32,
    pub debug_overlay: bool,
    pub display_mode: DisplayMode,
    pub windowed_width: u32,
    pub windowed_height: u32,
    /// Shadow volume update step in blocks (2, 4, 8, 16, 32, 48)
    pub shadow_update_step: f32,
    /// Whether torches cast shadows (more immersive but higher GPU cost)
    pub torch_shadows: bool,
    /// SSAO quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub ssao_quality: String,
    /// DoF quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub dof_quality: String,
    /// Anti-aliasing quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub aa_quality: String,
    /// DoF focus distance in blocks
    pub dof_focus_distance: f32,
    /// DoF blur strength (0.0-1.0)
    pub dof_blur_strength: f32,
    /// DoF focus range in blocks
    pub dof_focus_range: f32,
    /// Water refraction quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub refraction_quality: String,
    /// Impostor quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub impostor_quality: String,

    // === CLOUD SETTINGS ===
    /// Cloud quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub cloud_quality: String,
    /// Cloud coverage (0.0 - 1.0)
    pub cloud_coverage: f32,
    /// Cloud animation speed (0.0 - 1.0)
    pub cloud_speed: f32,
    /// Whether cloud shadows are enabled
    pub cloud_shadows: bool,

    // === SCENERY SETTINGS ===
    /// Scenery quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub scenery_quality: String,

    // === GRASS SETTINGS ===
    /// Grass quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub grass_quality: String,
    /// Grass render distance: NEAR, MEDIUM, FAR, ULTRA
    pub grass_distance: String,
    /// Grass shadow mode: OFF, SIMPLE
    pub grass_shadow: String,

    // === AUDIO SETTINGS ===
    /// Master volume (0.0 - 1.0)
    pub audio_master: f32,
    /// Sound effects volume (0.0 - 1.0)
    pub audio_sfx: f32,
    /// Ambient sounds volume (0.0 - 1.0)
    pub audio_ambient: f32,

    // === LOD SETTINGS ===
    /// LOD distance: "OFF" or chunk distance where LOD starts (6, 8, 12, 16, 20, 24, 32)
    pub lod_distance: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            render_distance: crate::constants::RENDER_DISTANCE,
            shadow_mode: 0.0, // HARD
            bloom_enabled: true,
            volumetrics_enabled: true,
            gpu_culling: true,
            gpu_voxel_fill: true,
            cycle_speed: 1.0,
            debug_overlay: false,
            display_mode: DisplayMode::Windowed,
            windowed_width: 1280,
            windowed_height: 720,
            shadow_update_step: 2.0, // Frequent small updates (options: 2, 4, 8, 16, 32, 48)
            torch_shadows: false,    // Default OFF for performance
            ssao_quality: "MEDIUM".to_string(), // Default SSAO quality
            dof_quality: "LOW".to_string(),     // Default DoF quality
            aa_quality: "MEDIUM".to_string(),   // Default AA quality
            dof_focus_distance: 20.0,           // Default focus distance in blocks
            dof_blur_strength: 0.5,             // Default aperture (0.0-1.0)
            dof_focus_range: 10.0,              // Default focus range in blocks
            refraction_quality: "MEDIUM".to_string(), // Default water refraction quality
            impostor_quality: "MEDIUM".to_string(), // Default impostor quality
            // Cloud defaults
            cloud_quality: "MEDIUM".to_string(),   // Default cloud quality
            cloud_coverage: 0.5,                   // 50% coverage
            cloud_speed: 0.3,                      // 30% animation speed
            cloud_shadows: true,                   // Cloud shadows on by default
            // Scenery defaults
            scenery_quality: "MEDIUM".to_string(),  // Default scenery quality
            // Grass defaults
            grass_quality: "MEDIUM".to_string(),    // Default grass quality
            grass_distance: "MEDIUM".to_string(),   // Default grass distance
            grass_shadow: "OFF".to_string(),        // Default grass shadow off
            // Audio defaults
            audio_master: 0.8,   // 80% master volume
            audio_sfx: 1.0,      // 100% SFX
            audio_ambient: 0.6,  // 60% ambient
            // LOD defaults
            lod_distance: "12".to_string(),  // Default: LOD starts at 12 chunks
        }
    }
}

impl Settings {
    const FILE_PATH: &'static str = "settings.cfg";

    /// Load settings from file, or return defaults if file doesn't exist
    pub fn load() -> Self {
        let path = Path::new(Self::FILE_PATH);
        if !path.exists() {
            tracing::info!("No settings file found, using defaults");
            return Self::default();
        }

        match fs::read_to_string(path) {
            Ok(content) => Self::parse(&content),
            Err(e) => {
                tracing::warn!("Failed to read settings file: {}, using defaults", e);
                Self::default()
            }
        }
    }

    /// Parse settings from text content
    fn parse(content: &str) -> Self {
        let mut settings = Self::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "render_distance" => {
                        if let Ok(v) = value.parse() {
                            settings.render_distance = v;
                        }
                    }
                    "shadow_mode" => {
                        if let Ok(v) = value.parse() {
                            settings.shadow_mode = v;
                        }
                    }
                    "bloom_enabled" => {
                        settings.bloom_enabled = value == "true";
                    }
                    "volumetrics_enabled" => {
                        settings.volumetrics_enabled = value == "true";
                    }
                    "gpu_culling" => {
                        settings.gpu_culling = value == "true";
                    }
                    "gpu_voxel_fill" => {
                        settings.gpu_voxel_fill = value == "true";
                    }
                    "cycle_speed" => {
                        if let Ok(v) = value.parse() {
                            settings.cycle_speed = v;
                        }
                    }
                    "debug_overlay" => {
                        settings.debug_overlay = value == "true";
                    }
                    "display_mode" => {
                        settings.display_mode = match value {
                            "BORDERLESS" => DisplayMode::Borderless,
                            "FULLSCREEN" => DisplayMode::Fullscreen,
                            _ => DisplayMode::Windowed,
                        };
                    }
                    "windowed_width" => {
                        if let Ok(v) = value.parse() {
                            settings.windowed_width = v;
                        }
                    }
                    "windowed_height" => {
                        if let Ok(v) = value.parse() {
                            settings.windowed_height = v;
                        }
                    }
                    "shadow_update_step" => {
                        if let Ok(v) = value.parse() {
                            settings.shadow_update_step = v;
                        }
                    }
                    "torch_shadows" => {
                        settings.torch_shadows = value == "true";
                    }
                    "ssao_quality" => {
                        settings.ssao_quality = value.to_string();
                    }
                    "dof_quality" => {
                        settings.dof_quality = value.to_string();
                    }
                    "aa_quality" => {
                        settings.aa_quality = value.to_string();
                    }
                    "dof_focus_distance" => {
                        if let Ok(v) = value.parse() {
                            settings.dof_focus_distance = v;
                        }
                    }
                    "dof_blur_strength" => {
                        if let Ok(v) = value.parse() {
                            settings.dof_blur_strength = v;
                        }
                    }
                    "dof_focus_range" => {
                        if let Ok(v) = value.parse() {
                            settings.dof_focus_range = v;
                        }
                    }
                    "refraction_quality" => {
                        settings.refraction_quality = value.to_string();
                    }
                    "impostor_quality" => {
                        settings.impostor_quality = value.to_string();
                    }
                    // Cloud settings
                    "cloud_quality" => {
                        settings.cloud_quality = value.to_string();
                    }
                    "cloud_coverage" => {
                        if let Ok(v) = value.parse() {
                            settings.cloud_coverage = v;
                        }
                    }
                    "cloud_speed" => {
                        if let Ok(v) = value.parse() {
                            settings.cloud_speed = v;
                        }
                    }
                    "cloud_shadows" => {
                        settings.cloud_shadows = value == "true";
                    }
                    // Scenery settings
                    "scenery_quality" => {
                        settings.scenery_quality = value.to_string();
                    }
                    // Grass settings
                    "grass_quality" => {
                        settings.grass_quality = value.to_string();
                    }
                    "grass_distance" => {
                        settings.grass_distance = value.to_string();
                    }
                    "grass_shadow" => {
                        settings.grass_shadow = value.to_string();
                    }
                    // Audio settings
                    "audio_master" => {
                        if let Ok(v) = value.parse() {
                            settings.audio_master = v;
                        }
                    }
                    "audio_sfx" => {
                        if let Ok(v) = value.parse() {
                            settings.audio_sfx = v;
                        }
                    }
                    "audio_ambient" => {
                        if let Ok(v) = value.parse() {
                            settings.audio_ambient = v;
                        }
                    }
                    // LOD settings
                    "lod_distance" => {
                        settings.lod_distance = value.to_string();
                    }
                    _ => {}
                }
            }
        }

        tracing::info!("Loaded settings from {}", Self::FILE_PATH);
        settings
    }

    /// Save settings to file
    pub fn save(&self) {
        let content = format!(
            "# Mosaic Settings\n\
             render_distance={}\n\
             shadow_mode={}\n\
             bloom_enabled={}\n\
             volumetrics_enabled={}\n\
             gpu_culling={}\n\
             gpu_voxel_fill={}\n\
             cycle_speed={}\n\
             debug_overlay={}\n\
             display_mode={}\n\
             windowed_width={}\n\
             windowed_height={}\n\
             shadow_update_step={}\n\
             torch_shadows={}\n\
             ssao_quality={}\n\
             dof_quality={}\n\
             aa_quality={}\n\
             dof_focus_distance={}\n\
             dof_blur_strength={}\n\
             dof_focus_range={}\n\
             refraction_quality={}\n\
             impostor_quality={}\n\
             # Cloud Settings\n\
             cloud_quality={}\n\
             cloud_coverage={}\n\
             cloud_speed={}\n\
             cloud_shadows={}\n\
             # Scenery Settings\n\
             scenery_quality={}\n\
             # Grass Settings\n\
             grass_quality={}\n\
             grass_distance={}\n\
             grass_shadow={}\n\
             # Audio Settings\n\
             audio_master={}\n\
             audio_sfx={}\n\
             audio_ambient={}\n\
             # LOD Settings\n\
             lod_distance={}\n",
            self.render_distance,
            self.shadow_mode,
            self.bloom_enabled,
            self.volumetrics_enabled,
            self.gpu_culling,
            self.gpu_voxel_fill,
            self.cycle_speed,
            self.debug_overlay,
            self.display_mode.as_str(),
            self.windowed_width,
            self.windowed_height,
            self.shadow_update_step,
            self.torch_shadows,
            self.ssao_quality,
            self.dof_quality,
            self.aa_quality,
            self.dof_focus_distance,
            self.dof_blur_strength,
            self.dof_focus_range,
            self.refraction_quality,
            self.impostor_quality,
            self.cloud_quality,
            self.cloud_coverage,
            self.cloud_speed,
            self.cloud_shadows,
            self.scenery_quality,
            self.grass_quality,
            self.grass_distance,
            self.grass_shadow,
            self.audio_master,
            self.audio_sfx,
            self.audio_ambient,
            self.lod_distance,
        );

        if let Err(e) = fs::write(Self::FILE_PATH, content) {
            tracing::error!("Failed to save settings: {}", e);
        } else {
            tracing::info!("Settings saved to {}", Self::FILE_PATH);
        }
    }
}
