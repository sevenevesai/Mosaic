//! GPU profiling using wgpu timestamp queries

use std::collections::VecDeque;
use wgpu::{Device, Queue};

/// Number of frames to average for GPU timing
const HISTORY_SIZE: usize = 60;

/// GPU profiler using timestamp queries
pub struct GpuProfiler {
    /// Whether GPU profiling is supported
    supported: bool,
    /// Query set for timestamps
    query_set: Option<wgpu::QuerySet>,
    /// Buffer to read query results
    resolve_buffer: Option<wgpu::Buffer>,
    /// Staging buffer for CPU readback
    staging_buffer: Option<wgpu::Buffer>,
    /// History of GPU frame times (nanoseconds)
    frame_times_ns: VecDeque<u64>,
    /// Timestamp period (nanoseconds per tick)
    #[allow(dead_code)]
    timestamp_period: f32,
    /// Whether we have pending results to read
    pending_readback: bool,
}

impl GpuProfiler {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        // Check if timestamp queries are supported
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        if !supported {
            tracing::warn!("GPU timestamp queries not supported - GPU profiling disabled");
            return Self {
                supported: false,
                query_set: None,
                resolve_buffer: None,
                staging_buffer: None,
                frame_times_ns: VecDeque::with_capacity(HISTORY_SIZE),
                timestamp_period: 1.0,
                pending_readback: false,
            };
        }

        // Get timestamp period from queue
        let timestamp_period = queue.get_timestamp_period();
        tracing::info!("GPU profiling enabled (timestamp period: {} ns)", timestamp_period);

        // Create query set for 2 timestamps (start + end)
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Profiler Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        // Buffer to resolve queries into
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Profiler Resolve Buffer"),
            size: 16, // 2 x u64
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for CPU readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Profiler Staging Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            supported: true,
            query_set: Some(query_set),
            resolve_buffer: Some(resolve_buffer),
            staging_buffer: Some(staging_buffer),
            frame_times_ns: VecDeque::with_capacity(HISTORY_SIZE),
            timestamp_period,
            pending_readback: false,
        }
    }

    /// Get the query set for writing timestamps
    pub fn query_set(&self) -> Option<&wgpu::QuerySet> {
        self.query_set.as_ref()
    }

    /// Check if profiling is supported
    pub fn is_supported(&self) -> bool {
        self.supported
    }

    /// Resolve queries and copy to staging buffer
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.supported {
            return;
        }

        if let (Some(query_set), Some(resolve_buf), Some(staging_buf)) =
            (&self.query_set, &self.resolve_buffer, &self.staging_buffer)
        {
            encoder.resolve_query_set(query_set, 0..2, resolve_buf, 0);
            encoder.copy_buffer_to_buffer(resolve_buf, 0, staging_buf, 0, 16);
        }
    }

    /// Begin async readback of results
    pub fn begin_readback(&mut self) {
        // Disabled for now - wgpu 0.19 async buffer mapping is tricky
        // GPU timing still works via timestamp queries, just not readable yet
    }

    /// Poll for readback completion and update stats
    pub fn poll_results(&mut self, device: &Device) {
        if !self.pending_readback {
            return;
        }

        // Poll device to check if mapping is complete
        device.poll(wgpu::Maintain::Wait);

        // Give the mapping a frame to complete, then try to read
        // This is a simplified approach - production code would use proper callbacks
        self.pending_readback = false;

        // For now, skip reading to avoid panics - GPU profiling needs async handling
        // which is complex to implement correctly with wgpu 0.19
    }

    /// Get average GPU frame time in milliseconds
    pub fn avg_gpu_time_ms(&self) -> f64 {
        if self.frame_times_ns.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frame_times_ns.iter().sum();
        (sum as f64 / self.frame_times_ns.len() as f64) / 1_000_000.0
    }

    /// Get last GPU frame time in milliseconds
    pub fn last_gpu_time_ms(&self) -> f64 {
        self.frame_times_ns.back()
            .map(|&ns| ns as f64 / 1_000_000.0)
            .unwrap_or(0.0)
    }
}
