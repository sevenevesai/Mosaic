# Mosaic Gameplay Roadmap

Engine foundation is solid (~13K LOC): rendering, world gen, persistence, inventory, entity system, audio.
Current state is a creative sandbox with no gameplay tension. This roadmap transitions to survival gameplay.

## Build Order

Each phase builds on the previous. Crafting unlocks tool progression. Health makes mobs meaningful.
Mobs make night/shelter meaningful. Furnace extends the crafting chain. Farming closes the survival loop.

```
Phase 1: Crafting + Tools + Mining Speed
    |
Phase 2: Health + Damage + Respawn
    |
Phase 3: Hostile Mobs (Night Cycle)
    |
Phase 4: Furnace + Smelting + Tiered Progression
    |
Phase 5: Hunger + Farming
```

---

## Phase 1: Crafting + Tools + Mining Speed

**Status:** NOT STARTED
**Impact:** Transforms creative sandbox into a resource economy. Makes all 16 ore types meaningful.

### 1a. Item System Expansion
- Extend `Item` enum beyond `Block(Block)` to support `Tool { kind, material, durability }`
- Tool kinds: Pickaxe, Axe, Shovel, Sword (start with Pick + Axe)
- Material tiers: Wood, Stone, Copper, Iron, Gold, Diamond (maps to existing ores)
- Durability per tool (decrements on use, breaks at 0)
- Tools stack to 1 (infrastructure already exists via `max_stack_size`)

### 1b. Mining Speed System
- Replace instant-break with progressive block damage
- Block hardness table: dirt/sand ~0.5s by hand, stone ~4s by hand, ores vary
- Tool effectiveness multiplier: matching tool + higher tier = faster break
- Visual crack overlay (4 stages) on block being mined
- Audio feedback: mining tick sounds per hit

### 1c. Block Drop Tables
- Blocks don't always drop themselves: stone -> cobblestone, ores -> raw ore item
- Silk touch concept for later; for now, some blocks have transformed drops
- Requires new item variants for raw materials (Raw Iron, Raw Copper, Coal, etc.)

### 1d. Crafting System
- 3x3 crafting grid UI (accessible via C key or crafting table block)
- Shaped recipes: grid position matters (pick = 3 material + 2 sticks)
- Recipe registry loaded at startup
- Starter recipes: planks from logs, sticks from planks, wooden tools, torches
- Crafting table block: 2x2 hand-craft for basics, 3x3 at table for tools
- UI: recipe list sidebar showing available recipes based on inventory contents

### 1e. New Items Required
- **Materials:** Stick, Planks (oak/birch/pine/jungle), Cobblestone
- **Raw ores:** Raw Iron, Raw Copper, Raw Tin, Raw Gold, Raw Silver, Coal
- **Tools:** Wooden/Stone/Copper/Iron/Gold/Diamond Pickaxe, Axe, Shovel, Sword

### Existing Infrastructure
- `Item` enum has commented-out `Tool` variant ready to implement
- `ItemTag::Tool`, `ItemTag::Weapon` already defined
- `ItemRarity` tiers map directly to tool material tiers
- `is_tool_block()` trait method exists as a stub
- `max_stack_size` returns 1 for tool blocks already
- `sort_priority()` already sorts tools first

---

## Phase 2: Health + Damage + Respawn

**Status:** NOT STARTED
**Impact:** Introduces risk and makes building shelter/armor meaningful.

### 2a. Health System
- Player HP (20 hearts / 100 HP), displayed as heart bar above hotbar
- Natural regeneration when not recently damaged (1 HP/s after 3s)
- Death state: drop inventory items as item entities, show death screen

### 2b. Damage Sources
- Fall damage: > 3 blocks fallen = (distance - 3) * 5 HP
- Drowning: when underwater without air, 2 HP/s
- Cactus contact: 1 HP per touch
- Void damage: below Y=-64, instant kill
- Fire/lava: future phase

### 2c. Respawn
- Respawn at world spawn point on death
- Bed block (future): sets personal spawn point
- Dropped items persist for 5 minutes (system already exists)

### Existing Infrastructure
- Item entity drop/pickup system already complete with physics, despawn, magnet
- Player position/velocity tracking in player module
- Fall distance could be computed from velocity on landing

---

## Phase 3: Hostile Mobs

**Status:** NOT STARTED
**Impact:** Makes night dangerous, gives reason to build shelter, creates combat gameplay.

### 3a. Mob Spawning
- Spawn on solid blocks with sky_light < 7 at night (time_of_day 0.75-0.25)
- Spawn radius: 24-48 blocks from player
- Mob cap: ~20 hostile mobs loaded at once
- Despawn when > 64 blocks from player

### 3b. Basic AI
- Zombie: melee, walks toward player when in range (16 blocks), 2 HP/hit
- Skeleton: ranged placeholder (start as melee variant, add projectiles later)
- Pathfinding: simple A* or greedy walk-toward on navmesh of solid block tops
- Collision with blocks (reuse player AABB system)

### 3c. Combat
- Sword deals damage on LMB swing (cooldown-based)
- Knockback on hit (both directions)
- Mob HP bars (small bar above head)
- Mob drops: XP orbs (future), basic loot (bones, arrows, etc.)

### Existing Infrastructure
- Entity system with generational arena (SlotMap), NPC variant exists
- Skeletal animation with crossfade blending ready
- FBX loader for mob models
- Day/night cycle provides time_of_day for spawn gating
- Light data per chunk available for spawn checks

---

## Phase 4: Furnace + Smelting

**Status:** NOT STARTED
**Impact:** Extends crafting into multi-step progression, creates "just one more upgrade" loop.

### 4a. Furnace Block
- New block type: Furnace (placeable, interactive)
- Furnace UI: fuel slot + input slot + output slot + progress bar
- Fuel: coal (8 smelts), wood/planks (1.5 smelts), sticks (0.5 smelts)
- Tick-based smelting: ~10 seconds per item

### 4b. Smelting Recipes
- Raw Iron -> Iron Ingot
- Raw Copper -> Copper Ingot
- Raw Gold -> Gold Ingot
- Sand -> Glass (new block)
- Cobblestone -> Stone (smooth)
- Raw ore items from Phase 1 become meaningful

### 4c. Tier Gating
- Copper tools: slightly better than stone
- Iron tools: significant upgrade, can mine gold/diamond
- Diamond tools: best tier, fastest mining, highest durability
- Some blocks require minimum tool tier to mine (diamond ore needs iron+ pick)

---

## Phase 5: Hunger + Farming

**Status:** NOT STARTED
**Impact:** Closes the survival loop, gives reason to build a permanent base.

### 5a. Hunger System
- Hunger bar (20 drumsticks / 100 points) above HP bar
- Drains on sprint (0.5/s), mining (0.1/action), jumping (0.05/jump)
- At 0 hunger: no sprint, take 1 HP/s starvation damage
- At full hunger: natural HP regen active

### 5b. Farming Blocks
- Farmland: created by using hoe on dirt/grass
- Hydration: farmland near water (4 blocks) is hydrated (darker texture)
- Crop growth: tick-based stages (seed -> sprout -> mature, ~3-5 min)

### 5c. Crops + Food
- Wheat: found in grass (seed drop), grows in 4 stages -> bread
- Carrots/Potatoes: found in village chests or mob drops
- Cooking: furnace turns raw food into cooked (more hunger restored)
- Each food item restores different hunger amounts

---

## Deferred (Post-Core Survival)

These features are valuable but depend on the core loop being fun first:

- **Armor system** - defense stat reducing damage, wearable slots
- **Enchanting/upgrades** - special tool properties (efficiency, fortune, unbreaking)
- **Villages + NPCs** - passive mobs, trading, economy
- **Redstone/automation** - logic gates, pistons, hoppers
- **Biome-specific resources** - jungle cocoa, desert cactus farms, tundra ice tools
- **Boss mobs** - dungeon encounters with unique drops
- **Multiplayer** - engine-level networking rewrite
- **Quests/achievements** - milestone tracking, guided progression
