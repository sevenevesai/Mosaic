# Day/Night Cycle System

Full sun/moon system with warm color palette.

## Time System

```rust
time_of_day: f32    // 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
cycle_speed: f32    // Multiplier (T key cycles: 1x → 5x → 20x → 50x → 100x → paused)
```

Base cycle: 600 seconds (10 minutes) for full day.

## Key Components

| File | Struct | Purpose |
|------|--------|---------|
| `renderer/mod.rs` | `DayNightCycle` | Time tracking, sun/moon positions |
| `renderer/mod.rs` | `SkyUniform` | Sky gradient colors, celestial directions |
| `renderer/mod.rs` | `LightUniform` | Sun/moon colors, ambient light |
| `renderer/pipeline.rs` | Sky shader | Gradient + sun/moon rendering |
| `renderer/pipeline.rs` | Chunk shader | Lighting based on time |

## Color Palette

| Time | Sky | Sun/Moon |
|------|-----|----------|
| Dawn (0.25) | Coral → pink | Warm orange |
| Noon (0.5) | Golden blue | Bright white-yellow |
| Dusk (0.75) | Amber → purple | Deep orange |
| Night (0.0) | Warm indigo | Amber moon |

All colors have warm undertones - no harsh blues or whites.

## Sun Behavior

- Rises east (0.25), sets west (0.75)
- Maximum intensity at noon
- Casts directional shadows
- Shadow fade: `smoothstep(0.0, 0.3, sun_intensity)` at sunset

## Moon Behavior

- Rises when sun sets
- Crisp edge (not glowing like sun)
- Warm amber color
- Very dim: max intensity 0.25
- **No shadows** - provides ambient only

## Shader Integration

**Sky shader calculates:**
- Gradient between horizon and zenith colors
- Sun disk with glow falloff
- Moon disk with sharp edge
- Star field (night only, fades at dawn/dusk)

**Chunk shader receives:**
- `sun_direction`: vec3 for shadow mapping
- `sun_color`: vec3 for direct lighting
- `moon_color`: vec3 for ambient
- `ambient`: vec3 base ambient light

## Shadow Transitions

Critical for avoiding jarring visual changes:

1. **No direction flip** - Sun direction smoothly interpolates through horizon
2. **Intensity fade** - Shadows fade to zero before sun fully sets
3. **No moon shadows** - Moon only contributes ambient

```wgsl
let shadow_strength = smoothstep(0.0, 0.3, sun_intensity);
let shadow = mix(1.0, calculate_shadow(), shadow_strength);
```

## Controls

- **T key**: Cycle through speed multipliers
- Speeds: 1x (real-time) → 5x → 20x → 50x → 100x → 0x (paused)
- F3 overlay shows current time and speed

## Modifying Colors

Edit `DayNightCycle::calculate_colors()` in `renderer/mod.rs`:

```rust
fn calculate_colors(&self) -> (Vec3, Vec3, Vec3) {
    let t = self.time_of_day;
    // Return (sky_color, sun_color, ambient)
    // Use smooth interpolation between keyframes
}
```

## Performance Notes

- Colors calculated once per frame
- Uniform buffer updated each frame
- No per-vertex time calculations
- Shadow map reused (direction changes slowly)
