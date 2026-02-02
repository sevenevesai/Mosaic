//! GPU profiling using wgpu timestamp queries
//!
//! Uses double-buffered resolve buffers to avoid Vulkan synchronization issues
//! where resolve and copy operations in the same encoder can return zeros.
//!
//! Supports per-pass timing breakdown for detailed GPU profiling.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::{Device, Queue};

/// Number of frames to average for GPU timing
const HISTORY_SIZE: usize = 60;

/// Maximum number of GPU passes to track (each pass needs 2 timestamps: start + end)
const MAX_PASSES: usize = 16;

/// Double-buffered timestamp resources to avoid sync issues
struct TimestampBuffer {
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    mapping_complete: Arc<AtomicBool>,
    pending_readback: bool,
}

/// GPU pass timing info
#[derive(Clone)]
pub struct PassTiming {
    pub name: String,
    pub time_ms: f64,
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
    /// Query set for timestamps (2 per pass: start + end)
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
    /// Current pass index for this frame
    current_pass: usize,
    /// Pass names for this frame (indexed by pass number)
    pass_names: Vec<String>,
    /// Per-pass timing results (averaged over history)
    pass_timings: Vec<PassTiming>,
    /// Per-pass timing history (pass_name -> history of times in ns)
    pass_history: Vec<VecDeque<u64>>,
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
                current_pass: 0,
                pass_names: Vec::new(),
                pass_timings: Vec::new(),
                pass_history: Vec::new(),
            };
        }

        // Get timestamp period from queue
        let timestamp_period = queue.get_timestamp_period();
        tracing::info!(
            "GPU profiling enabled (timestamp period: {:.2} ns, {} passes max)",
            timestamp_period,
            MAX_PASSES
        );

        // Create query set for MAX_PASSES * 2 timestamps (start + end per pass)
        let query_count = (MAX_PASSES * 2) as u32;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Profiler Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });

        // Create double-buffered resources
        let buffer_size = (query_count as u64) * 8; // 8 bytes per u64 timestamp
        let create_buffer_pair = |index: usize| -> TimestampBuffer {
            let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GPU Profiler Resolve Buffer {}", index)),
                size: buffer_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GPU Profiler Staging Buffer {}", index)),
                size: buffer_size,
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
            current_pass: 0,
            pass_names: Vec::with_capacity(MAX_PASSES),
            pass_timings: Vec::new(),
            pass_history: Vec::new(),
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

    /// Begin a new frame - reset pass counter
    pub fn begin_frame(&mut self) {
        self.current_pass = 0;
        self.pass_names.clear();
    }

    /// Begin timing a render pass. Returns the query index for the start timestamp.
    /// Call this BEFORE the render pass begins.
    pub fn begin_pass(&mut self, name: &str) -> Option<u32> {
        if !self.supported || self.current_pass >= MAX_PASSES {
            return None;
        }

        self.pass_names.push(name.to_string());
        let query_idx = (self.current_pass * 2) as u32;
        Some(query_idx)
    }

    /// End timing a render pass. Returns the query index for the end timestamp.
    /// Call this AFTER the render pass ends.
    pub fn end_pass(&mut self) -> Option<u32> {
        if !self.supported || self.current_pass >= MAX_PASSES {
            return None;
        }

        let query_idx = (self.current_pass * 2 + 1) as u32;
        self.current_pass += 1;
        Some(query_idx)
    }

    /// Write a timestamp to the query set at the given index
    pub fn write_timestamp(&self, encoder: &mut wgpu::CommandEncoder, query_idx: u32) {
        if let Some(query_set) = &self.query_set {
            encoder.write_timestamp(query_set, query_idx);
        }
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
        if !self.supported || self.current_pass == 0 {
            return;
        }

        let (query_set, buffers) = match (&self.query_set, &self.buffers) {
            (Some(qs), Some(bufs)) => (qs, bufs),
            _ => return,
        };

        // Resolve current frame's timestamps to current buffer
        let query_count = (self.current_pass * 2) as u32;
        encoder.resolve_query_set(
            query_set,
            0..query_count,
            &buffers[self.current_buffer].resolve_buffer,
            0,
        );

        // Copy PREVIOUS frame's resolve buffer to its staging buffer
        // (Skip on first frame - no previous data yet)
        if self.frame_count > 0 {
            let prev_buffer = 1 - self.current_buffer;
            let copy_size = query_count as u64 * 8;
            encoder.copy_buffer_to_buffer(
                &buffers[prev_buffer].resolve_buffer,
                0,
                &buffers[prev_buffer].staging_buffer,
                0,
                copy_size,
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

        // Poll device to process the mapping callback (non-blocking check first)
        device.poll(wgpu::Maintain::Poll);

        // Collect data from completed readbacks (avoiding borrow conflicts)
        let mut data_to_process: Option<Vec<u8>> = None;
        let mut buffer_to_unmap: Option<usize> = None;

        for (i, buf) in buffers.iter_mut().enumerate() {
            if !buf.pending_readback {
                continue;
            }

            // Check if mapping completed
            if !buf.mapping_complete.load(Ordering::SeqCst) {
                continue;
            }

            // Read the mapped data into a Vec to avoid holding the borrow
            let view = buf.staging_buffer.slice(..).get_mapped_range();
            data_to_process = Some(view.to_vec());
            buffer_to_unmap = Some(i);
            break;
        }

        // Process the data outside the buffer borrow
        if let Some(data) = data_to_process {
            self.process_timestamps(&data);
        }

        // Unmap the buffer after processing
        if let Some(idx) = buffer_to_unmap {
            if let Some(bufs) = &mut self.buffers {
                bufs[idx].staging_buffer.unmap();
                bufs[idx].pending_readback = false;
            }
        }
    }

    /// Process timestamp data from the staging buffer
    fn process_timestamps(&mut self, data: &[u8]) {
        // Parse timestamps (pairs of u64: start, end for each pass)
        let mut pass_idx = 0;
        let mut total_time_ns: u64 = 0;

        // Ensure we have enough history slots
        while self.pass_history.len() < self.pass_names.len() {
            self.pass_history.push(VecDeque::with_capacity(HISTORY_SIZE));
        }

        for chunk in data.chunks_exact(16) {
            if pass_idx >= self.pass_names.len() {
                break;
            }

            let start = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let end = u64::from_le_bytes(chunk[8..16].try_into().unwrap());

            if start > 0 && end > 0 && end > start {
                let delta_ticks = end - start;
                let delta_ns = (delta_ticks as f64 * self.timestamp_period as f64) as u64;
                total_time_ns += delta_ns;

                // Update pass history
                let history = &mut self.pass_history[pass_idx];
                if history.len() >= HISTORY_SIZE {
                    history.pop_front();
                }
                history.push_back(delta_ns);
            }

            pass_idx += 1;
        }

        // Update total frame time history
        if total_time_ns > 0 {
            if self.frame_times_ns.len() >= HISTORY_SIZE {
                self.frame_times_ns.pop_front();
            }
            self.frame_times_ns.push_back(total_time_ns);
        }

        // Update pass timing averages
        self.update_pass_timings();
    }

    /// Update averaged pass timings from history
    fn update_pass_timings(&mut self) {
        self.pass_timings.clear();

        for (i, name) in self.pass_names.iter().enumerate() {
            if i < self.pass_history.len() {
                let history = &self.pass_history[i];
                if !history.is_empty() {
                    let sum: u64 = history.iter().sum();
                    let avg_ns = sum as f64 / history.len() as f64;
                    let avg_ms = avg_ns / 1_000_000.0;
                    self.pass_timings.push(PassTiming {
                        name: name.clone(),
                        time_ms: avg_ms,
                    });
                }
            }
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

    /// Get per-pass timing breakdown
    pub fn pass_timings(&self) -> &[PassTiming] {
        &self.pass_timings
    }

    /// Get pass timings as a vec of (name, time_ms) tuples for RenderStats
    pub fn pass_times_vec(&self) -> Vec<(String, f64)> {
        self.pass_timings
            .iter()
            .map(|p| (p.name.clone(), p.time_ms))
            .collect()
    }

    /// Force-unmap any pending mapped buffers and reset state
    /// Call this before resize or other operations that need unmapped buffers
    pub fn force_unmap(&mut self) {
        let Some(buffers) = &mut self.buffers else { return };

        for buf in buffers.iter_mut() {
            // Try to unmap if mapping completed
            if buf.pending_readback && buf.mapping_complete.load(Ordering::SeqCst) {
                buf.staging_buffer.unmap();
            }
            // Reset state regardless - this ensures next frame starts clean
            buf.pending_readback = false;
            buf.mapping_complete.store(false, Ordering::SeqCst);
        }

        // Reset frame count to skip copy-from-previous on next frame
        // This avoids copying to a buffer that might have been in a bad state
        self.frame_count = 0;
    }
}
