<div align="center">

<img src="docs/media/icon.png" alt="Mosaic logo" width="96">

# Mosaic

**A voxel engine written from scratch in Rust and wgpu.**

Procedural worlds that generate and mesh on worker threads, lit by shadows, ambient<br>
occlusion and light shafts traced through a voxel volume on the GPU.

[Website](https://seveneves.ai/mosaic/) ·
[Demo video](https://seveneves.ai/mosaic/#video) ·
[Build and run](#build-and-run) ·
[Controls](#controls) ·
[How it works](#how-it-works)

</div>

---

Mosaic is a playable sandbox. You walk or fly through eight biomes, mine and place blocks, and
light caves with torches. It is a personal project with no releases, so you build it from source.
The textures, music, sounds and character models are not in this repository, and a clone runs
without them; see [Assets](#assets).

## Features

### Rendering

- **Traced lighting.** Sun shadows, light from up to 256 torches with shadows, ambient occlusion
  and volumetric light shafts are all traced through a 256³ voxel volume that follows the
  player, using Amanatides & Woo DDA. There are no shadow maps.
- **Binary greedy meshing.** Bitwise face culling hides buried faces and merges coplanar ones
  into large quads, meshing each 16³ chunk at full, half or quarter detail by distance.
- **GPU-driven drawing.** A compute pass frustum-culls chunks, and the survivors draw from one
  shared buffer by multi-draw indirect.
- **Water.** A fullscreen pass intersects each pixel's ray with the sea-level water plane, so
  there is no water mesh, and adds Fresnel reflection, refraction and underwater fog.
- **Sky and post-processing.** A day/night cycle, voxel clouds, GPU-built grass, SSAO, depth of
  field, bloom, and anti-aliasing that combines morphological edge detection with temporal
  accumulation.

### World

- **Eight biomes:** grasslands, cavernous highlands, desert, tundra, jungle, swamp, mountains and
  ocean, with caves, caverns, ores, trees, vegetation, rock formations and ruins.
- **Deterministic generation** from the world seed and chunk position, so chunks that were never
  saved regenerate identically. Each new world gets a random seed.
- **Flowing water** as a cellular automaton: sources spread and fall, and the level decays with
  distance.
- **Region saves** compressed with zstd, written on quit along with the player, inventory and
  dropped items.

### Play

- **First- and third-person camera** with an animated FBX character. You change its outfit, hair
  and head under CHARACTER in the menu.
- **Inventory** with a nine-slot hotbar, a block catalog, chests and dropped items.
- **Spatial audio.** Footstep, break and place sounds depend on the block's material, and blocks
  between you and a sound make it quieter. Ambient sound crossfades between surface, cave and
  underwater, and music plays on top.
- **Settings menu** with DISPLAY, GRAPHICS, AUDIO and DEBUG tabs, saved to `settings.cfg`.

## Build and run

You need a stable Rust toolchain from [rustup](https://rustup.rs/) and a GPU with Vulkan,
Direct3D 12 or Metal drivers. Mosaic is developed on Windows. It has no platform-specific code,
but Linux and macOS builds are untested; on Linux, the audio backend needs the ALSA development
package (`libasound2-dev` on Debian and Ubuntu).

```sh
git clone https://github.com/sevenevesai/Mosaic.git
cd Mosaic
cargo run --release
```

Run it from the repository root. The engine reads `assets/` from the working directory, and
writes `settings.cfg`, the world in `saves/world1/` and `performance_log.txt` there. To start a
new world, delete `saves/world1/`.

### Assets

`assets/` is excluded from git. Without it the engine still starts, but blocks render without
their textures, the moon is generated procedurally, and there is no music, sound or character.
The engine looks for:

| Path | Contents |
|---|---|
| `assets/blocks/` | 64×64 PNG block textures, looked up by name in `build_block_textures()` in [`src/world/block_registry.rs`](src/world/block_registry.rs) |
| `assets/sfx/` | WAV sound effects in `footstep/`, `break/` and `place/` (named `<material>-<n>.wav`), and `ambient/` |
| `assets/music/` | music tracks in FLAC, Ogg, MP3 or WAV |
| `assets/char/` | FBX character models and animations, with their textures |
| `assets/props/` | MagicaVoxel `.vox` props |
| `assets/moon-1.png` | the moon texture |

`cargo run --release --bin generate_sounds` writes placeholder sound effects into `assets/sfx/`.

## Controls

| Input | Action |
|---|---|
| W A S D, mouse | Move, look |
| Space | Jump; rise in fly mode or water |
| Shift | Descend in fly mode or water |
| Ctrl | Sprint |
| F | Toggle fly mode |
| Left mouse (hold) / right mouse | Mine / place block |
| 1-9, scroll wheel | Hotbar slot |
| Q | Drop item |
| E | Inventory |
| F5 | Toggle first and third person |
| Esc | Menu: settings, character, save and quit |

Debug and quality keys:

| Key | Action |
|---|---|
| F3 | Debug overlay: FPS, chunk, triangle and light counts |
| F4 | Body part editor |
| T | Day/night speed, including pause |
| Y | Hard or soft shadows |
| U | Shadow debug view |
| V | Volumetrics quality |
| G | GPU culling on or off |
| B | GPU voxel fill on or off |
| M | Next music track |
| P | Verbose performance log |

## How it works

Each lighting effect has a design doc covering its mechanism:
[lighting and torches](docs/lighting-design.md),
[ambient occlusion](docs/ambient-occlusion-design.md) and
[volumetrics](docs/volumetrics-design.md). Every other subsystem is described in a short rule
file in [`.claude/rules/`](.claude/rules/), named after the subsystem.

| Path | What it holds |
|---|---|
| [`src/renderer/`](src/renderer/) | wgpu passes: chunks, voxel volume, water, clouds, grass, SSAO, volumetrics, depth of field, anti-aliasing, bloom, UI |
| [`src/world/`](src/world/) | chunks and block IDs, block registry, meshing, light propagation, water flow |
| [`src/world/generation/`](src/world/generation/) | terrain, biomes, caves, trees, vegetation, ruins |
| [`src/world/storage/`](src/world/storage/) | region files and world metadata |
| [`src/player/`](src/player/) | movement, collision, block raycast, mining, camera |
| [`src/entity/`](src/entity/) | FBX skeleton loading and the skinned character |
| [`src/audio/`](src/audio/) | spatial sound, ambience and music |
| [`src/inventory/`](src/inventory/) | items, inventory and hotbar |
| [`src/bin/`](src/bin/) | `generate_sounds` (placeholder WAVs) and `analyze_props` (checks `.vox` props) |
| [`benches/`](benches/), [`tests/`](tests/) | Criterion benchmarks and the world integration test |

World constants (chunk size 16, render distance 32 chunks, sea level 32) are in
[`src/lib.rs`](src/lib.rs).

## Development

```sh
cargo check                         # type-check, the gate for any edit
cargo test                          # 209 tests: unit tests and tests/world_integration.rs
cargo bench --bench meshing_bench   # one of the Criterion benchmarks in benches/
```

The tests cover CPU-side code: meshing, lighting, generation, saves, player and inventory. They
do not exercise shaders or the renderer, so check those changes in the running engine with F3.
Delete `saves/world1/` before testing a change to generation or the save format.

Planned survival gameplay (crafting, health, hostile mobs, smelting and farming) is laid out in
[ROADMAP.md](ROADMAP.md).

### For AI agents

[CLAUDE.md](CLAUDE.md) is the brief for coding agents: the build and test gates, the invariants
that break rendering or saves when violated, and an index of the subsystem rules and design docs.
