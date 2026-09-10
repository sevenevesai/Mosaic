---
paths:
  - "src/renderer/mod.rs"
  - "src/renderer/pipeline.rs"
---

# Day/night cycle

`DayNightCycle` in `renderer/mod.rs` drives time and colour; the sky and chunk shaders in
`renderer/pipeline.rs` consume them.

`time_of_day` runs 0.0 midnight, 0.25 dawn, 0.5 noon, 0.75 dusk over a
`base_cycle_duration` of 600 seconds. The T key steps `cycle_speed` 1x, 5x, 20x, 50x, 100x,
paused. `day_count` increments on wrap and feeds the moon phase.

Colours come from `DayNightCycle::calculate_colors()` and are deliberately warm throughout —
coral at dawn, golden blue at noon, amber and purple at dusk, warm indigo at night. Anything
tending to harsh blue or white is off-palette; tune inside that function rather than in the
shaders.

The sun uses multi-layer bloom and warms toward the horizon at 50-100% intensity. The moon has
a crisp edge, warm amber, and at most 0.25 intensity.

## Shadow transitions

Shadows have to fade rather than pop, and three things together achieve that. Breaking any one
of them brings back a hard flip at sunrise or sunset.

- The sun direction interpolates smoothly through the horizon and never flips sign.
- `shadow_fade = smoothstep(0.0, 0.3, sun_intensity)` takes shadows to zero before the sun is
  down.
- The moon casts no shadows at all; it contributes ambient only.
