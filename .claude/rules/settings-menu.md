---
paths:
  - "src/settings_menu.rs"
  - "src/settings.rs"
---

# Settings menu

Two stages in `settings_menu.rs`: main menu buttons, then tabbed settings (Display, Graphics,
Audio, Debug). ESC backs up one level; "Save & Quit" writes `settings.cfg` and exits.

| Control | Rendered as | Interaction |
|---|---|---|
| Toggle | `[ ON  ]` | click to flip |
| EnumCycle | `[<  VALUE   >]` | click an arrow to cycle |
| PercentSlider | `[============]` | click the bar, continuous 0.0-1.0 |

## Adding a setting

1. Add a field to `Settings` in `settings.rs` so it persists.
2. Add a `MenuItem` in `create_menu_items()` with label, tab, control type and get/set
   callbacks.
3. Add the renderer helpers the callbacks need (`cycle_*_prev()`, `set_*_index()`).

## Gotchas

- Layout and hit detection are computed twice — once in `recalculate_settings_layout()`, once
  where the menu is drawn — from the same numbers: tabs at NDC Y 0.85, content from 0.70, each
  tab 10 characters wide with a 1-character gap, the strip centred and sized from
  `MenuTab::all()`. Change one side only and clicks land on the wrong tab.
- `scroll_offset` resets to 0 on a tab change; without that a short tab scrolls past its end.
- `click()` returns a `MenuAction` — the Save & Quit case is what shuts the game down, so a
  caller that ignores the return value makes the button dead.
- The renderer is handed a `MenuRenderData` struct, not a list of lines.
- The blur pass reads `scene_texture`, so that texture needs the `COPY_DST` usage flag, and
  draws through the intermediate `menu_blur_texture`: a 9-tap Gaussian with 50% darkening.
