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
    /// Water refraction quality preset: OFF, LOW, MEDIUM, HIGH, ULTRA
    pub refraction_quality: String,
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
            refraction_quality: "MEDIUM".to_string(), // Default water refraction quality
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
                    "refraction_quality" => {
                        settings.refraction_quality = value.to_string();
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
             refraction_quality={}\n",
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
            self.refraction_quality,
        );

        if let Err(e) = fs::write(Self::FILE_PATH, content) {
            tracing::error!("Failed to save settings: {}", e);
        } else {
            tracing::info!("Settings saved to {}", Self::FILE_PATH);
        }
    }
}
