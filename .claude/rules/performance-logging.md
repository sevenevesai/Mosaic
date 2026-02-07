# Performance Logging & Session Output

## CRITICAL: Never Read Raw Game Session Output

When the game (mosaic.exe) runs as a background shell task, its tracing output can produce thousands of lines on stderr. **NEVER** read this raw output directly - it will blow up context.

### Instead, read `performance_log.txt`

The engine writes a compact, structured performance log to `performance_log.txt` in the working directory. This file contains:
- Session header with all quality settings
- Periodic snapshots (every 60s, max 5 per session) with FPS, GPU time, CPU sections, pass breakdown
- **Session summary** on shutdown: min/avg/max FPS, percentiles, GPU pass averages, CPU section averages, mesh stats, memory estimates

### How to review performance after a session

1. **Read `performance_log.txt`** - this is the single source of truth
2. Look at the `SESSION SUMMARY` block at the end for aggregates
3. Look at periodic `DATA|` lines for time-series if needed
4. **Never** use `TaskOutput` on a game background shell expecting to parse tracing output

### Launching the game for performance testing

When running the game in background for testing, always redirect stderr:
```bash
./target/release/mosaic.exe 2>session_stderr.log
```
This prevents tracing output from accumulating in the shell task buffer. Read `performance_log.txt` for results.

## Log Volume Reference

With these optimizations, a typical 5-minute session produces:
- ~15 lines of tracing to stderr (session header + shutdown)
- ~20 lines in performance_log.txt (header + 5 snapshots + summary)
- Verbose mode (P key): ~20 lines every 10 seconds while active (user-toggled only)
- LOD distribution: 1 line per minute
