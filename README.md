# Mosaic

**Website:** https://seveneves.ai/mosaic/

High-performance voxel engine built with Rust and wgpu.

## Quick Start

```bash
cargo build --release
./target/release/mosaic.exe
```

## Controls

| Input | Action |
|---|---|
| WASD | Move |
| Space | Jump, or fly up in fly mode |
| Shift | Fly down |
| Ctrl | Sprint |
| F | Toggle fly mode |
| Mouse left / right | Break / place block |
| E | Inventory |
| 1-9, scroll wheel | Hotbar slot |
| Esc | Menu |
| F5 | Toggle first and third person |

Debug and quality keys: F3 debug overlay, F4 body part editor, T day/night speed, Y shadow
mode, U shadow debug, V volumetrics quality, G GPU culling, B GPU voxel fill, M next music
track, P verbose performance log.

See [CLAUDE.md](CLAUDE.md) for architecture and development docs.
