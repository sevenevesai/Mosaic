# Settings Menu System

Two-stage menu: main menu (buttons) + tabbed settings (~950 LOC in `settings_menu.rs`).

## Menu Flow

ESC (game) -> Main menu -> "Settings" -> Tabbed settings. ESC backs up one level. "Save & Quit" saves and exits.

## Tabs & Control Types

Tabs: Display, Graphics, Audio, Debug.

| Type | Format | Interaction |
|------|--------|-------------|
| Toggle | `[ ON  ]` | Click to flip |
| EnumCycle | `[<  VALUE   >]` | Click arrows to cycle |
| PercentSlider | `[============]` | Click bar for continuous 0.0-1.0 |

## Adding a New Setting

1. Add `MenuItem` to `create_menu_items()` with: `label`, `tab`, `control_type`, get/set callbacks
2. Add renderer helper methods if needed: `cycle_*_prev()`, `set_*_index()`

## Gotchas

- **Tab line width is 43 chars** (4 tabs x 10 + 3 gaps), NOT 36 - rendered at fixed Y=0.85 NDC
- **Hit detection must match render** - both use same coordinate calculations
- **Scroll resets on tab change**: `scroll_offset = 0`
- **`click()` returns bool** - true when Save & Quit pressed
- **MenuRenderData**: Renderer receives struct, not just lines
- **Blur needs COPY_DST**: scene_texture must have COPY_DST usage flag
- **Menu blur**: 9-tap Gaussian, 50% darkening, uses intermediate `menu_blur_texture`
