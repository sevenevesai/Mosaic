# Settings Menu System

Two-stage menu system with main menu and tabbed settings (~950 LOC).

## Architecture

| Component | Purpose |
|-----------|---------|
| `MenuMode` | Main (buttons) vs Settings (tabs) |
| `MainMenuButton` | Settings, Save & Quit |
| `MenuTab` | Tab categories: Display, Graphics, Audio, Debug |
| `ControlType` | Toggle, EnumCycle, PercentSlider |
| `MenuItem` | Label + callbacks for get/set/cycle values |
| `ClickRegion` | Hit test results: Button, Tab, LeftArrow, Center, RightArrow, SliderBar |
| `MenuRenderData` | Render output: lines, tabs, scroll state |

## Menu Flow

1. **ESC pressed** (game) → Opens main menu
2. **Main menu** → "Settings" button → Settings tabs
3. **ESC pressed** (settings) → Back to main menu
4. **ESC pressed** (main menu) → Close menu, resume game
5. **Save & Quit** button → Save all data, exit to desktop

## Layout Constants

```rust
CHAR_WIDTH_PX: 12.0      // 6px font × 2.0 scale
CHAR_HEIGHT_PX: 20.0     // 10px font × 2.0 scale
LINE_HEIGHT_PX: 28.0     // 20 × 1.4 line spacing
LABEL_WIDTH: 18          // chars for label column
CONTROL_WIDTH: 16        // chars for control column
TOTAL_WIDTH: 36          // chars for menu items
```

## Main Menu

- Centered on screen
- Two buttons: "SETTINGS" and "SAVE & QUIT"
- Hover highlights with brackets: `[ SETTINGS ]`

## Settings Menu

- **Fixed tabs at top** (Y = 0.85 NDC)
- **Scrollable content area** below tabs
- **Dynamic scroll arrows** (^ and v) only shown when content exists
- Tab line is 43 chars (4 tabs × 10 chars + 3 gaps), centered independently

## Control Types

| Type | Format | Width | Interaction |
|------|--------|-------|-------------|
| Toggle | `[ ON  ]` | 7 chars | Click anywhere to flip |
| EnumCycle | `[<  VALUE   >]` | 14 chars | Click arrows to cycle |
| PercentSlider | `[============]` | 14 chars | Click bar to set value (continuous) |

## Scrolling

- `scroll_offset: usize` tracks first visible item
- `max_visible_items` calculated from screen height
- Mouse wheel scrolls content area
- Arrows appear only when more content exists in that direction

## Hit Detection

**Main menu:**
- Button rectangles calculated from centered layout

**Settings:**
- Tabs at fixed Y position
- Items offset by `scroll_offset` for both rendering and hit detection
- Control column: `control_col_start = menu_left + (LABEL_WIDTH + COLUMN_GAP) * CHAR_WIDTH_PX`

## Slider Accuracy

Sliders use **continuous values** (0.0-1.0) - no snapping:
- Click anywhere on the bar for precise control
- Visual shows filled portion with `=` and empty with `.`

## Adding Settings

1. Add `MenuItem` to `create_menu_items()` with:
   - `label`, `tab`, `control_type`
   - Callbacks: `get_value`, `get_index`, `get_float`
   - Callbacks: `cycle_next`, `cycle_prev`, `set_index`, `set_float`

2. Add renderer helper methods if needed:
   - `cycle_*_prev()` for reverse cycling
   - `set_*_index()` for direct setting

## Key Functions

| Function | Purpose |
|----------|---------|
| `on_open()` | Reset to main menu, recalculate layout |
| `go_back()` | Return from settings to main menu |
| `click()` | Handle click, returns true if quit requested |
| `scroll()` | Handle scroll wheel, returns true if consumed |
| `generate_lines()` | Returns `MenuRenderData` for rendering |
| `can_scroll_up/down()` | Check if scroll arrows should show |

## Menu Blur Effect

When menu opens, scene is blurred and darkened:
- **9-tap Gaussian blur** with radius 2 pixels
- **50% darkening** applied after blur
- Uses intermediate `menu_blur_texture` (can't read/write same texture)
- Blur pass runs before UI pass, copies result back to scene_texture

## Crosshair

- Hidden when menu is open (`menu_data.is_none()` check)
- Hidden when inventory is open
- Size: 0.015 clip space, thickness: 0.0015 (fine/thin style)

## Gotchas

- **MenuRenderData type**: Renderer receives struct, not just lines
- **Tab line width**: 43 chars, NOT 36 - rendered at fixed Y position
- **Hit detection must match render**: Both use same coordinate calculations
- **Scroll resets on tab change**: `scroll_offset = 0` when switching tabs
- **Quit returns bool**: `click()` returns true when Save & Quit pressed
- **Blur needs COPY_DST**: scene_texture must have COPY_DST usage flag
