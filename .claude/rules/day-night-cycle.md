# Day/Night Cycle System

Full sun/moon system with warm color palette. Located in `renderer/mod.rs` (`DayNightCycle`) and `renderer/pipeline.rs` (sky/chunk shaders).

## Time System

- `time_of_day: f32` - 0.0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk
- Base cycle: 600 seconds (10 minutes). T key cycles speed: 1x/5x/20x/50x/100x/paused.

## Color Palette

All warm undertones - no harsh blues or whites. Dawn=coral/pink, Noon=golden blue, Dusk=amber/purple, Night=warm indigo. Edit `DayNightCycle::calculate_colors()`.

## Celestial Bodies

- **Sun**: Multi-layer bloom, warmer at horizon. Intensity 50-100%.
- **Moon**: Crisp edge, warm amber, dim (max 0.25 intensity). **No shadows** - ambient only.

## Shadow Transitions (Critical Gotcha)

Shadows must fade smoothly - no jarring pops:
1. Sun direction smoothly interpolates through horizon (no direction flip)
2. Shadows fade to zero via `smoothstep(0.0, 0.3, sun_intensity)` before sun sets
3. Moon never casts shadows
