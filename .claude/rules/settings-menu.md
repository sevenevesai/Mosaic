# Settings Menu System

Tabbed settings UI with mouse-based hit detection (~760 LOC).

## Architecture

| Component | Purpose |
|-----------|---------|
| `MenuTab` | Tab categories: Display, Graphics, Audio, Debug |
| `ControlType` | Toggle, EnumCycle, PercentSlider |
| `MenuItem` | Label + callbacks for get/set/cycle values |
| `ClickRegion` | Hit test results: Tab, LeftArrow, Center, RightArrow, SliderBar |

## Layout Constants

```rust
CHAR_WIDTH_PX: 12.0      // 6px font × 2.0 scale
CHAR_HEIGHT_PX: 20.0     // 10px font × 2.0 scale
LINE_HEIGHT_PX: 28.0     // 20 × 1.4 line spacing
LABEL_WIDTH: 18          // chars for label column
CONTROL_WIDTH: 16        // chars for control column
TOTAL_WIDTH: 36          // chars for menu items (not tabs)
```

## Tab System

- 4 tabs × 10 chars + 3 gaps = **43 chars** (wider than item lines)
- Tab line renders independently, NOT constrained to TOTAL_WIDTH
- Hit detection uses: `tabs_start_x = (screen_width - 43 * CHAR_WIDTH_PX) / 2`

## Control Types

| Type | Format | Width | Interaction |
|------|--------|-------|-------------|
| Toggle | `[ ON  ]` | 7 chars | Click anywhere to flip |
| EnumCycle | `[<  VALUE   >]` | 14 chars | Click arrows to cycle |
| PercentSlider | `[============]` | 14 chars | Click bar to set value |

## Hit Detection

Two-column layout with controls left-aligned in right column:
```
control_col_start = menu_left + (LABEL_WIDTH + COLUMN_GAP) * CHAR_WIDTH_PX
```

EnumCycle arrow regions:
- Left arrow: first 2 chars (`[<`)
- Center: middle 10 chars
- Right arrow: last 2 chars (`>]`)

## Adding Settings

1. Add `MenuItem` to `create_menu_items()` with:
   - `label`, `tab`, `control_type`
   - Callbacks: `get_value`, `get_index`, `get_float`
   - Callbacks: `cycle_next`, `cycle_prev`, `set_index`, `set_float`

2. Add renderer helper methods if needed:
   - `cycle_*_prev()` for reverse cycling
   - `set_*_index()` for direct setting

## Gotchas

- **Tab line width**: 43 chars, NOT 36 - don't truncate or tabs won't render fully
- **Hit detection must match render**: Both use same centering math
- **Menu items are 36 chars**: Tab line is wider and centered separately
- **Line counting**: Title(1) + space(1) + tabs(1) + space(1) + items(N) + space(1) + footer(1)
