//! Frame profiling and performance metrics
//!
//! Comprehensive profiling system for identifying performance bottlenecks.
//! Tracks CPU section timing, frame percentiles, and quality settings context.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::renderer::RenderStats;

/// Snapshot of quality settings for logging context
#[derive(Clone, Default)]
pub struct SettingsSnapshot {
    pub render_distance: i32,
    pub ssao_quality: String,
    pub ao_quality: String,
    pub dof_quality: String,
    pub water_quality: String,
    pub impostor_quality: String,
    pub bloom_enabled: bool,
    pub volumetrics_enabled: bool,
    pub gpu_culling: bool,
    pub shadow_mode: String,
    pub torch_shadows: bool,
    pub resolution: (u32, u32),
}

impl SettingsSnapshot {
    /// Format settings as a compact string for log lines
    pub fn as_compact_string(&self) -> String {
        format!(
            "SSAO:{} AO:{} DOF:{} Water:{} Imp:{} Bloom:{} Vol:{} Cull:{} Shad:{}",
            self.ssao_quality,
            self.ao_quality,
            self.dof_quality,
            self.water_quality,
            self.impostor_quality,
            if self.bloom_enabled { "ON" } else { "OFF" },
            if self.volumetrics_enabled { "ON" } else { "OFF" },
            if self.gpu_culling { "ON" } else { "OFF" },
            self.shadow_mode,
        )
    }

    /// Format settings for session header
    pub fn as_header_lines(&self) -> Vec<String> {
        vec![
            format!("  Resolution: {}x{}", self.resolution.0, self.resolution.1),
            format!("  Render Distance: {} chunks", self.render_distance),
            format!("  SSAO: {}", self.ssao_quality),
            format!("  Voxel AO: {}", self.ao_quality),
            format!("  DOF: {}", self.dof_quality),
            format!("  Water: {}", self.water_quality),
            format!("  Impostors: {}", self.impostor_quality),
            format!("  Bloom: {}", if self.bloom_enabled { "ON" } else { "OFF" }),
            format!("  Volumetrics: {}", if self.volumetrics_enabled { "ON" } else { "OFF" }),
            format!("  GPU Culling: {}", if self.gpu_culling { "ON" } else { "OFF" }),
            format!("  Shadow Mode: {}", self.shadow_mode),
            format!("  Torch Shadows: {}", if self.torch_shadows { "ON" } else { "OFF" }),
        ]
    }
}

/// Tracks frame timing and performance metrics
pub struct FrameProfiler {
    frame_start: Instant,
    sections: HashMap<String, SectionTimer>,
    frame_times: RingBuffer<Duration>,
    /// Longer history for percentile calculations (5 seconds at 60fps = 300 frames)
    frame_times_extended: RingBuffer<Duration>,
    display_enabled: bool,
    verbose_enabled: bool,
    last_verbose_report: Instant,
    render_stats: RenderStats,
    /// Current settings snapshot for context
    settings: SettingsSnapshot,
    /// Session start time
    session_start: Instant,
    /// Whether session header has been logged
    session_header_logged: bool,
}

struct SectionTimer {
    start: Option<Instant>,
    accumulated: Duration,
    samples: RingBuffer<Duration>,
    min_ms: f64,
    max_ms: f64,
}

impl SectionTimer {
    fn new() -> Self {
        Self {
            start: None,
            accumulated: Duration::ZERO,
            samples: RingBuffer::new(120),
            min_ms: f64::MAX,
            max_ms: 0.0,
        }
    }

    fn update_min_max(&mut self) {
        let ms = self.accumulated.as_secs_f64() * 1000.0;
        if ms > 0.0 {
            self.min_ms = self.min_ms.min(ms);
            self.max_ms = self.max_ms.max(ms);
        }
    }

    fn reset_min_max(&mut self) {
        self.min_ms = f64::MAX;
        self.max_ms = 0.0;
    }
}

/// Fixed-size ring buffer for storing samples
pub struct RingBuffer<T> {
    data: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Copy + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            head: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.data.len();
        if self.len < self.data.len() {
            self.len += 1;
        }
    }

    pub fn average(&self) -> T
    where
        T: std::ops::Add<Output = T> + std::ops::Div<u32, Output = T>,
    {
        if self.len == 0 {
            return T::default();
        }
        let mut sum = T::default();
        for i in 0..self.len {
            sum = sum + self.data[i];
        }
        sum / (self.len as u32)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Get all values in order (oldest to newest)
    pub fn values(&self) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.len);
        let start = if self.len < self.data.len() {
            0
        } else {
            self.head
        };
        for i in 0..self.len {
            let idx = (start + i) % self.data.len();
            result.push(self.data[idx]);
        }
        result
    }
}

impl FrameProfiler {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            frame_start: now,
            sections: HashMap::new(),
            frame_times: RingBuffer::new(120),
            frame_times_extended: RingBuffer::new(300), // 5 seconds at 60fps
            display_enabled: false,
            verbose_enabled: false,
            last_verbose_report: now,
            render_stats: RenderStats::default(),
            settings: SettingsSnapshot::default(),
            session_start: now,
            session_header_logged: false,
        }
    }

    /// Update settings snapshot from renderer state
    pub fn update_settings(&mut self, settings: SettingsSnapshot) {
        self.settings = settings;
    }

    /// Get current settings snapshot
    pub fn settings(&self) -> &SettingsSnapshot {
        &self.settings
    }

    /// Update render stats from the renderer
    pub fn update_render_stats(&mut self, stats: RenderStats) {
        self.render_stats = stats;
    }

    pub fn is_display_enabled(&self) -> bool {
        self.display_enabled
    }

    pub fn toggle_display(&mut self) {
        self.display_enabled = !self.display_enabled;
        if self.display_enabled {
            tracing::info!("Performance overlay enabled (F3)");
        } else {
            tracing::info!("Performance overlay disabled");
        }
    }

    pub fn toggle_verbose(&mut self) {
        self.verbose_enabled = !self.verbose_enabled;
        if self.verbose_enabled {
            tracing::info!("╔════════════════════════════════════════════════════════════════════════╗");
            tracing::info!("║ VERBOSE PROFILING ENABLED - Detailed breakdown every 10 seconds       ║");
            tracing::info!("╚════════════════════════════════════════════════════════════════════════╝");
            // Log current settings context
            tracing::info!("Settings: {}", self.settings.as_compact_string());
            // Reset min/max when enabling
            for section in self.sections.values_mut() {
                section.reset_min_max();
            }
        } else {
            tracing::info!("Verbose profiling disabled");
        }
    }

    pub fn is_verbose_enabled(&self) -> bool {
        self.verbose_enabled
    }

    pub fn begin_frame(&mut self) {
        self.frame_start = Instant::now();
        // Reset section accumulators
        for section in self.sections.values_mut() {
            section.accumulated = Duration::ZERO;
        }

        // Log session header after stabilization (5 seconds)
        if !self.session_header_logged && self.session_start.elapsed() > Duration::from_secs(5) {
            self.log_session_header();
            self.session_header_logged = true;
        }
    }

    pub fn end_frame(&mut self) {
        let frame_time = self.frame_start.elapsed();
        self.frame_times.push(frame_time);
        self.frame_times_extended.push(frame_time);

        // Record section samples and update min/max
        for section in self.sections.values_mut() {
            section.update_min_max();
            section.samples.push(section.accumulated);
        }

        // Periodic logging removed from tracing output to reduce console spam.
        // F3 overlay still renders on-screen; use performance_log.txt for file-based metrics.

        // Verbose logging with min/max breakdown (every 10 seconds)
        if self.verbose_enabled && self.last_verbose_report.elapsed() > Duration::from_secs(10) {
            self.log_verbose_stats();
            self.last_verbose_report = Instant::now();
            // Reset min/max after reporting
            for section in self.sections.values_mut() {
                section.reset_min_max();
            }
        }
    }

    pub fn begin_section(&mut self, name: &str) {
        let section = self.sections
            .entry(name.to_string())
            .or_insert_with(SectionTimer::new);
        section.start = Some(Instant::now());
    }

    pub fn end_section(&mut self, name: &str) {
        if let Some(section) = self.sections.get_mut(name) {
            if let Some(start) = section.start.take() {
                section.accumulated += start.elapsed();
            }
        }
    }

    /// Calculate frame time percentiles (returns 1% low and 0.1% low in ms)
    pub fn frame_percentiles(&self) -> (f64, f64) {
        let mut times: Vec<f64> = self.frame_times_extended
            .values()
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        if times.is_empty() {
            return (0.0, 0.0);
        }

        // Sort descending (worst frames first)
        times.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let len = times.len();
        // 1% low = average of worst 1% of frames
        let one_pct_count = (len as f64 * 0.01).max(1.0) as usize;
        let one_pct_low: f64 = times[..one_pct_count].iter().sum::<f64>() / one_pct_count as f64;

        // 0.1% low = average of worst 0.1% of frames (or worst frame if < 1000 samples)
        let point_one_pct_count = (len as f64 * 0.001).max(1.0) as usize;
        let point_one_pct_low: f64 = times[..point_one_pct_count].iter().sum::<f64>() / point_one_pct_count as f64;

        (one_pct_low, point_one_pct_low)
    }

    /// Calculate FPS from percentile frame times
    pub fn fps_percentiles(&self) -> (f64, f64) {
        let (one_pct_ms, point_one_pct_ms) = self.frame_percentiles();
        let one_pct_fps = if one_pct_ms > 0.0 { 1000.0 / one_pct_ms } else { 0.0 };
        let point_one_pct_fps = if point_one_pct_ms > 0.0 { 1000.0 / point_one_pct_ms } else { 0.0 };
        (one_pct_fps, point_one_pct_fps)
    }

    fn log_session_header(&self) {
        tracing::info!("╔════════════════════════════════════════════════════════════════════════╗");
        tracing::info!("║                    MOSAIC ENGINE - SESSION START                       ║");
        tracing::info!("╠════════════════════════════════════════════════════════════════════════╣");
        tracing::info!("║ Quality Settings:                                                      ║");
        for line in self.settings.as_header_lines() {
            tracing::info!("║ {}{}║", line, " ".repeat(70 - line.len()));
        }
        tracing::info!("╚════════════════════════════════════════════════════════════════════════╝");
    }

    fn log_verbose_stats(&self) {
        let avg_frame = self.frame_times.average();
        let fps = if avg_frame.as_secs_f64() > 0.0 {
            1.0 / avg_frame.as_secs_f64()
        } else {
            0.0
        };

        let (one_pct_ms, point_one_pct_ms) = self.frame_percentiles();
        let (one_pct_fps, point_one_pct_fps) = self.fps_percentiles();

        tracing::info!("╔════════════════════════════════════════════════════════════════════════╗");
        tracing::info!("║ PERFORMANCE BREAKDOWN (last 2 seconds)                                 ║");
        tracing::info!("╠════════════════════════════════════════════════════════════════════════╣");
        tracing::info!("║ Settings: {:62}║", self.settings.as_compact_string());
        tracing::info!("╠════════════════════════════════════════════════════════════════════════╣");
        tracing::info!(
            "║ FPS: {:>5.1} avg │ {:>5.1} 1%low │ {:>5.1} 0.1%low                            ║",
            fps, one_pct_fps, point_one_pct_fps
        );
        tracing::info!(
            "║ Frame: {:>5.2}ms avg │ {:>5.2}ms 1%low │ {:>5.2}ms 0.1%low                       ║",
            avg_frame.as_secs_f64() * 1000.0, one_pct_ms, point_one_pct_ms
        );
        tracing::info!(
            "║ GPU: {:>5.2}ms │ Chunks: {:>4} rendered {:>4} culled │ Tris: {:>5}K            ║",
            self.render_stats.gpu_time_ms,
            self.render_stats.chunks_rendered,
            self.render_stats.chunks_culled,
            self.render_stats.triangles / 1000
        );

        // GPU pass breakdown if available
        if self.render_stats.pass_times.len() > 0 {
            tracing::info!("╠════════════════════════════════════════════════════════════════════════╣");
            tracing::info!("║ GPU Pass Breakdown:                                                    ║");
            for (name, time_ms) in &self.render_stats.pass_times {
                if *time_ms > 0.01 {
                    let pct = if self.render_stats.gpu_time_ms > 0.0 {
                        (time_ms / self.render_stats.gpu_time_ms) * 100.0
                    } else {
                        0.0
                    };
                    tracing::info!(
                        "║   {:20} {:>6.2}ms ({:>4.1}%)                                    ║",
                        name, time_ms, pct
                    );
                }
            }
        }

        tracing::info!("╠════════════════════════════════════════════════════════════════════════╣");
        tracing::info!("║ CPU Section          │   Avg   │   Min   │   Max   │ Spiky │  % Frame  ║");
        tracing::info!("╠══════════════════════╪═════════╪═════════╪═════════╪═══════╪═══════════╣");

        // Sort sections by average time (descending) for easier reading
        let mut sections: Vec<_> = self.sections.iter().collect();
        sections.sort_by(|a, b| {
            let avg_a = a.1.samples.average().as_secs_f64();
            let avg_b = b.1.samples.average().as_secs_f64();
            avg_b.partial_cmp(&avg_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let frame_ms = avg_frame.as_secs_f64() * 1000.0;
        for (name, section) in sections {
            let avg = section.samples.average().as_secs_f64() * 1000.0;
            if avg < 0.01 {
                continue; // Skip negligible sections
            }
            let min = if section.min_ms == f64::MAX { 0.0 } else { section.min_ms };
            let max = section.max_ms;
            // Mark as spiky if max > 3x avg (indicates frame time variance)
            let spiky = if avg > 0.01 && max > avg * 3.0 { "!!" } else { "  " };
            let pct = if frame_ms > 0.0 { (avg / frame_ms) * 100.0 } else { 0.0 };
            tracing::info!(
                "║ {:20} │ {:>5.2}ms │ {:>5.2}ms │ {:>5.2}ms │   {}  │   {:>5.1}%  ║",
                name, avg, min, max, spiky, pct
            );
        }
        tracing::info!("╚════════════════════════════════════════════════════════════════════════╝");
    }

    pub fn fps(&self) -> f64 {
        let avg = self.frame_times.average();
        if avg.as_secs_f64() > 0.0 {
            1.0 / avg.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn frame_time_ms(&self) -> f64 {
        self.frame_times.average().as_secs_f64() * 1000.0
    }

    pub fn section_time_ms(&self, name: &str) -> f64 {
        self.sections
            .get(name)
            .map(|s| s.samples.average().as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    pub fn render_stats(&self) -> &RenderStats {
        &self.render_stats
    }

    /// Get CPU section averages for session summary: Vec<(name, avg_ms)>
    pub fn cpu_section_averages(&self) -> Vec<(String, f64)> {
        let mut result: Vec<_> = self.sections
            .iter()
            .map(|(name, section)| {
                let avg_ms = section.samples.average().as_secs_f64() * 1000.0;
                (name.clone(), avg_ms)
            })
            .filter(|(_, ms)| *ms > 0.01)
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Get session elapsed time in seconds
    pub fn session_elapsed_secs(&self) -> f32 {
        self.session_start.elapsed().as_secs_f32()
    }
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self::new()
    }
}
