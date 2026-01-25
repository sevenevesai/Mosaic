//! GPU profiling using wgpu timestamp queries

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    timestamp_period: f32,
    /// Whether we have pending results to read
    pending_readback: bool,
    /// Signal that mapping is complete
    mapping_complete: Arc<AtomicBool>,
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
                mapping_complete: Arc::new(AtomicBool::new(false)),
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
            mapping_complete: Arc::new(AtomicBool::new(false)),
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
    /// Call this after resolve() has been called and the command buffer submitted
    pub fn begin_readback(&mut self) {
        if !self.supported || self.pending_readback {
            return;
        }

        if let Some(staging_buf) = &self.staging_buffer {
            // Reset the mapping complete flag
            self.mapping_complete.store(false, Ordering::SeqCst);

            // Request async mapping
            let mapping_complete = Arc::clone(&self.mapping_complete);
            staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    mapping_complete.store(true, Ordering::SeqCst);
                }
            });

            self.pending_readback = true;
        }
    }

    /// Poll for readback completion and update stats
    /// Call this at the start of the next frame to read previous frame's results
    pub fn poll_results(&mut self, device: &Device) {
        if !self.supported || !self.pending_readback {
            return;
        }

        // Poll device to process the mapping callback
        device.poll(wgpu::Maintain::Wait);

        // Check if mapping completed
        if !self.mapping_complete.load(Ordering::SeqCst) {
            return;
        }

        // Read the mapped data
        if let Some(staging_buf) = &self.staging_buffer {
            {
                let view = staging_buf.slice(..).get_mapped_range();
                if view.len() >= 16 {
                    // Parse the two u64 timestamps
                    let start = u64::from_le_bytes(view[0..8].try_into().unwrap());
                    let end = u64::from_le_bytes(view[8..16].try_into().unwrap());

                    // Calculate delta in GPU ticks, then convert to nanoseconds
                    let delta_ticks = end.saturating_sub(start);
                    let delta_ns = (delta_ticks as f64 * self.timestamp_period as f64) as u64;

                    // Update history
                    if self.frame_times_ns.len() >= HISTORY_SIZE {
                        self.frame_times_ns.pop_front();
                    }
                    self.frame_times_ns.push_back(delta_ns);
                }
            }

            // Unmap the buffer so it can be used again
            staging_buf.unmap();
        }

        self.pending_readback = false;
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
