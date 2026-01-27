//! Frame profiling and performance metrics

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::renderer::RenderStats;

/// Tracks frame timing and performance metrics
pub struct FrameProfiler {
    frame_start: Instant,
    sections: HashMap<String, SectionTimer>,
    frame_times: RingBuffer<Duration>,
    display_enabled: bool,
    verbose_enabled: bool,
    last_report: Instant,
    last_verbose_report: Instant,
    render_stats: RenderStats,
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
struct RingBuffer<T> {
    data: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Copy + Default> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, value: T) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.data.len();
        if self.len < self.data.len() {
            self.len += 1;
        }
    }

    fn average(&self) -> T
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
}

impl FrameProfiler {
    pub fn new() -> Self {
        Self {
            frame_start: Instant::now(),
            sections: HashMap::new(),
            frame_times: RingBuffer::new(120),
            display_enabled: false,
            verbose_enabled: false,
            last_report: Instant::now(),
            last_verbose_report: Instant::now(),
            render_stats: RenderStats::default(),
        }
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
            tracing::info!("=== VERBOSE PROFILING ENABLED ===");
            tracing::info!("Detailed timing breakdown every 2 seconds");
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
    }

    pub fn end_frame(&mut self) {
        let frame_time = self.frame_start.elapsed();
        self.frame_times.push(frame_time);

        // Record section samples and update min/max
        for section in self.sections.values_mut() {
            section.update_min_max();
            section.samples.push(section.accumulated);
        }

        // Periodic logging
        if self.display_enabled && self.last_report.elapsed() > Duration::from_secs(1) {
            self.log_stats();
            self.last_report = Instant::now();
        }

        // Verbose logging with min/max breakdown
        if self.verbose_enabled && self.last_verbose_report.elapsed() > Duration::from_secs(2) {
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

    fn log_stats(&self) {
        let avg_frame = self.frame_times.average();
        let fps = if avg_frame.as_secs_f64() > 0.0 {
            1.0 / avg_frame.as_secs_f64()
        } else {
            0.0
        };

        tracing::info!(
            "FPS: {:.1} | Frame: {:.2}ms | GPU: {:.2}ms",
            fps,
            avg_frame.as_secs_f64() * 1000.0,
            self.render_stats.gpu_time_ms
        );

        tracing::info!(
            "  Chunks: {} rendered, {} culled | Triangles: {}",
            self.render_stats.chunks_rendered,
            self.render_stats.chunks_culled,
            self.render_stats.triangles
        );

        for (name, section) in &self.sections {
            let avg = section.samples.average();
            tracing::info!(
                "  {}: {:.2}ms",
                name,
                avg.as_secs_f64() * 1000.0
            );
        }
    }

    fn log_verbose_stats(&self) {
        let avg_frame = self.frame_times.average();
        let fps = if avg_frame.as_secs_f64() > 0.0 {
            1.0 / avg_frame.as_secs_f64()
        } else {
            0.0
        };

        tracing::info!("╔════════════════════════════════════════════════════════════╗");
        tracing::info!("║ PERFORMANCE BREAKDOWN (last 2 seconds)                     ║");
        tracing::info!("╠════════════════════════════════════════════════════════════╣");
        tracing::info!(
            "║ FPS: {:.1} | Frame: {:.2}ms avg | GPU: {:.2}ms",
            fps,
            avg_frame.as_secs_f64() * 1000.0,
            self.render_stats.gpu_time_ms
        );
        tracing::info!(
            "║ Chunks: {} rendered, {} culled | Tris: {}K",
            self.render_stats.chunks_rendered,
            self.render_stats.chunks_culled,
            self.render_stats.triangles / 1000
        );
        tracing::info!("╠════════════════════════════════════════════════════════════╣");
        tracing::info!("║ Section              │  Avg    │  Min    │  Max    │ Spiky ║");
        tracing::info!("╠══════════════════════╪═════════╪═════════╪═════════╪═══════╣");

        // Sort sections by average time (descending) for easier reading
        let mut sections: Vec<_> = self.sections.iter().collect();
        sections.sort_by(|a, b| {
            let avg_a = a.1.samples.average().as_secs_f64();
            let avg_b = b.1.samples.average().as_secs_f64();
            avg_b.partial_cmp(&avg_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (name, section) in sections {
            let avg = section.samples.average().as_secs_f64() * 1000.0;
            let min = if section.min_ms == f64::MAX { 0.0 } else { section.min_ms };
            let max = section.max_ms;
            // Mark as spiky if max > 3x avg (indicates frame time variance)
            let spiky = if avg > 0.01 && max > avg * 3.0 { "!!" } else { "  " };
            tracing::info!(
                "║ {:20} │ {:6.2}ms │ {:6.2}ms │ {:6.2}ms │   {}  ║",
                name, avg, min, max, spiky
            );
        }
        tracing::info!("╚════════════════════════════════════════════════════════════╝");
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
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self::new()
    }
}
