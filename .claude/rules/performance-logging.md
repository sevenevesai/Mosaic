---
paths:
  - "src/renderer/debug_overlay.rs"
  - "src/utils/profiler.rs"
  - "src/utils/gpu_profiler.rs"
---

# Performance logging

## Reading results

Never read the game's raw stderr from a background shell. Tracing emits thousands of lines and
fills the session's context with nothing useful. Redirect it
(`./target/release/mosaic.exe 2>session_stderr.log`) and read `performance_log.txt` in the
working directory instead.

`performance_log.txt` carries a session header with the quality settings, periodic snapshots,
and a `SESSION SUMMARY` block written on shutdown: min/avg/max FPS, percentiles, GPU pass
averages, CPU section averages, mesh stats, memory estimates. Read the summary first; the
`DATA|` lines are the time series behind it.

## Cadence

`DebugOverlay` waits `stabilization_delay_secs` 10.0 for the initial chunk load to settle,
then logs every `log_interval_secs` 60.0, capped at `max_logs_per_session` 5 so a long session
cannot grow the file without bound. The P key toggles verbose output (about 20 lines every
10 s) and LOD distribution prints once a minute.

A five-minute session produces roughly 15 lines of stderr and 20 lines of
`performance_log.txt`.
