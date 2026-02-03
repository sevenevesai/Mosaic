# .claude/rules/ Documentation

This directory contains detailed system documentation for the Mosaic engine. These files are automatically loaded by Claude Code when working in this project.

## Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `_init-template.md` | **Template for creating/updating CLAUDE.md in any project** | Starting new projects, updating stale docs |
| `voxelot-comparison.md` | **Voxelot comparison, upgrade roadmap, implementation status** | Performance optimization, future features |
| `testing.md` | **175 tests: coverage map, running tests, benchmarks** | Adding tests, debugging failures |
| `audio-system.md` | 3D spatial audio, material sounds, ambient system | Audio bugs, adding new sounds |
| `grass-system.md` | Procedural grass, bomb density, per-block placement | Grass rendering, density tuning |
| `lighting-system.md` | BFS flood fill lighting, adding light sources | Modifying lighting, adding glowing blocks |
| `water-rendering.md` | Heightfield water, shaders, effects | Water features, underwater effects |
| `world-persistence.md` | Region files, chunk serialization | Save/load issues, storage optimization |
| `day-night-cycle.md` | Time system, sun/moon, color palette | Modifying sky, lighting transitions |
| `meshing-system.md` | Binary greedy meshing, vertex format, LOD | Mesh bugs, performance optimization |
| `meta-prompt-generator.md` | Template for generating feature implementation prompts | Creating new feature specs |
| `settings-menu.md` | Tabbed settings UI, control types, hit detection | Settings menu changes, adding options |

## Usage

These files are **automatically loaded** when Claude Code starts a session in this project. You don't need to explicitly reference them.

For specific tasks, Claude will consult the relevant file. For example:
- "Add a lantern block" → `lighting-system.md`
- "Fix water rendering bug" → `water-rendering.md`
- "Chunk not saving" → `world-persistence.md`
- "Performance upgrade ideas" → `voxelot-comparison.md`

## For Other Projects

Copy `_init-template.md` to any new project's `.claude/rules/` directory. It contains a complete protocol for:
1. Analyzing a codebase
2. Identifying critical information
3. Writing an effective CLAUDE.md
4. Creating detailed documentation

This enables consistent, high-quality AI assistance across all your projects without needing to ask clarifying questions at session start.

## Maintenance

Update these files when:
- System architecture changes significantly
- New gotchas are discovered
- Common questions/confusions arise repeatedly

Keep each file focused on one system. Target ~100 lines per file.
