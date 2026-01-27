//! GPU profiling using wgpu timestamp queries
//!
//! Uses double-buffered resolve buffers to avoid Vulkan synchronization issues
//! where resolve and copy operations in the same encoder can return zeros.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::{Device, Queue};

/// Number of frames to average for GPU timing
const HISTORY_SIZE: usize = 60;

/// Double-buffered timestamp resources to avoid sync issues
struct TimestampBuffer {
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    mapping_complete: Arc<AtomicBool>,
    pending_readback: bool,
}

/// GPU profiler using timestamp queries with double-buffering
///
/// The Vulkan backend has a synchronization issue where resolve_query_set
/// and copy_buffer_to_buffer in the same command encoder can return zeros.
/// This profiler uses double-buffered resources:
/// - Frame N: Resolve to buffer A, copy buffer B (previous frame) to staging
/// - Frame N+1: Resolve to buffer B, copy buffer A to staging
///
/// This ensures the resolve operation has completed before we copy.
pub struct GpuProfiler {
    /// Whether GPU profiling is supported
    supported: bool,
    /// Query set for timestamps
    query_set: Option<wgpu::QuerySet>,
    /// Double-buffered timestamp resources (index 0 and 1)
    buffers: Option<[TimestampBuffer; 2]>,
    /// Current buffer index (alternates 0, 1, 0, 1...)
    current_buffer: usize,
    /// History of GPU frame times (nanoseconds)
    frame_times_ns: VecDeque<u64>,
    /// Timestamp period (nanoseconds per tick)
    timestamp_period: f32,
    /// Frame count (used to skip first frame where no previous data exists)
    frame_count: u64,
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
                buffers: None,
                current_buffer: 0,
                frame_times_ns: VecDeque::with_capacity(HISTORY_SIZE),
                timestamp_period: 1.0,
                frame_count: 0,
            };
        }

        // Get timestamp period from queue
        let timestamp_period = queue.get_timestamp_period();
        tracing::info!(
            "GPU profiling enabled (timestamp period: {:.2} ns, double-buffered)",
            timestamp_period
        );

        // Create query set for 2 timestamps (start + end)
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Profiler Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        // Create double-buffered resources
        let create_buffer_pair = |index: usize| -> TimestampBuffer {
            let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GPU Profiler Resolve Buffer {}", index)),
                size: 16, // 2 x u64
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GPU Profiler Staging Buffer {}", index)),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            TimestampBuffer {
                resolve_buffer,
                staging_buffer,
                mapping_complete: Arc::new(AtomicBool::new(false)),
                pending_readback: false,
            }
        };

        let buffers = [create_buffer_pair(0), create_buffer_pair(1)];

        Self {
            supported: true,
            query_set: Some(query_set),
            buffers: Some(buffers),
            current_buffer: 0,
            frame_times_ns: VecDeque::with_capacity(HISTORY_SIZE),
            timestamp_period,
            frame_count: 0,
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

    /// Resolve current frame's queries and copy previous frame's results
    ///
    /// This uses double-buffering to ensure proper synchronization:
    /// - Resolves current frame's timestamps to buffer[current]
    /// - Copies previous frame's buffer[1-current] to its staging buffer
    ///
    /// The copy happens one frame after the resolve, guaranteeing the GPU
    /// has finished writing the timestamp data.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.supported {
            return;
        }

        let (query_set, buffers) = match (&self.query_set, &self.buffers) {
            (Some(qs), Some(bufs)) => (qs, bufs),
            _ => return,
        };

        // Resolve current frame's timestamps to current buffer
        encoder.resolve_query_set(
            query_set,
            0..2,
            &buffers[self.current_buffer].resolve_buffer,
            0,
        );

        // Copy PREVIOUS frame's resolve buffer to its staging buffer
        // (Skip on first frame - no previous data yet)
        if self.frame_count > 0 {
            let prev_buffer = 1 - self.current_buffer;
            encoder.copy_buffer_to_buffer(
                &buffers[prev_buffer].resolve_buffer,
                0,
                &buffers[prev_buffer].staging_buffer,
                0,
                16,
            );
        }
    }

    /// Begin async readback of the previous frame's results
    ///
    /// Call this after resolve() and queue.submit()
    pub fn begin_readback(&mut self) {
        if !self.supported || self.frame_count == 0 {
            self.frame_count += 1;
            // Alternate buffer for next frame
            self.current_buffer = 1 - self.current_buffer;
            return;
        }

        let buffers = match &mut self.buffers {
            Some(bufs) => bufs,
            None => return,
        };

        // Begin readback on the PREVIOUS frame's staging buffer
        let prev_buffer = 1 - self.current_buffer;
        let buf = &mut buffers[prev_buffer];

        if buf.pending_readback {
            // Previous readback still pending - skip this frame
            self.frame_count += 1;
            self.current_buffer = 1 - self.current_buffer;
            return;
        }

        // Reset the mapping complete flag
        buf.mapping_complete.store(false, Ordering::SeqCst);

        // Request async mapping
        let mapping_complete = Arc::clone(&buf.mapping_complete);
        buf.staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    mapping_complete.store(true, Ordering::SeqCst);
                }
            });

        buf.pending_readback = true;

        self.frame_count += 1;
        // Alternate buffer for next frame
        self.current_buffer = 1 - self.current_buffer;
    }

    /// Poll for readback completion and update stats
    ///
    /// Call this at the start of the frame to read results from 2 frames ago
    pub fn poll_results(&mut self, device: &Device) {
        if !self.supported {
            return;
        }

        let buffers = match &mut self.buffers {
            Some(bufs) => bufs,
            None => return,
        };

        // Check both buffers for completed readbacks
        for buf in buffers.iter_mut() {
            if !buf.pending_readback {
                continue;
            }

            // Poll device to process the mapping callback (non-blocking check first)
            device.poll(wgpu::Maintain::Poll);

            // Check if mapping completed
            if !buf.mapping_complete.load(Ordering::SeqCst) {
                continue;
            }

            // Read the mapped data
            {
                let view = buf.staging_buffer.slice(..).get_mapped_range();
                if view.len() >= 16 {
                    // Parse the two u64 timestamps
                    let start = u64::from_le_bytes(view[0..8].try_into().unwrap());
                    let end = u64::from_le_bytes(view[8..16].try_into().unwrap());

                    // Calculate delta in GPU ticks, then convert to nanoseconds
                    if start > 0 && end > 0 && end > start {
                        let delta_ticks = end - start;
                        let delta_ns =
                            (delta_ticks as f64 * self.timestamp_period as f64) as u64;

                        // Update history
                        if self.frame_times_ns.len() >= HISTORY_SIZE {
                            self.frame_times_ns.pop_front();
                        }
                        self.frame_times_ns.push_back(delta_ns);
                    }
                }
            }

            // Unmap the buffer so it can be used again
            buf.staging_buffer.unmap();
            buf.pending_readback = false;
        }
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
        self.frame_times_ns
            .back()
            .map(|&ns| ns as f64 / 1_000_000.0)
            .unwrap_or(0.0)
    }
}
