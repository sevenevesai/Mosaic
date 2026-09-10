---
paths:
  - "src/player/**"
  - "src/entity/**"
  - "src/app.rs"
---

# Player, camera and animation

`player/mod.rs` owns movement, AABB collision, the block raycast, mining and both camera
modes; `entity/` loads FBX skeletons and drives the skinned character.

## Camera

Third person is the default. The camera sits `THIRD_PERSON_DISTANCE_DEFAULT` 10.0 blocks
behind the aim point, scroll-zoomable between 2.0 and 20.0, with an over-the-shoulder offset
of 1.5 blocks right and 0.3 up that scales to zero as the camera closes in so it does not
clip the player. F5 switches modes over `CAMERA_TRANSITION_DURATION` 0.3 s and the FOV changes
with it: 75 degrees first person, 70 third.

Terrain collision samples 10 points along the ray from the aim point outward and pulls the
camera to `hit - 0.5`, never nearer than `THIRD_PERSON_MIN_DISTANCE` 1.0.

`facing_yaw` comes from the WASD direction relative to camera yaw and turns at a capped rate
per frame, so the character rotates toward its movement instead of snapping.

## Animation

`Player::movement_state()` classifies from physics: in water is Swimming and wins over
everything else; airborne outside fly mode is Jumping above `velocity.y` 0.5 and Falling
below; horizontal speed over 0.5 is Walking or Sprinting. `app.rs` maps each state to (clip index, speed, blend, looping). Jumping
and Falling are the only non-looping pair, and Falling plays at 0.3x so the pose holds in the
air rather than looping. The root motion `AnimationComponent::update` returns is discarded for
the player: physics owns position, and applying both makes the character drift.

## Ground contact

A downward move that collides is undone and then re-applied by binary search, eight halvings
against `check_collision()`, before `velocity.y` is zeroed. Without it a fast fall leaves the
player standing above the ground it landed on.

## Mining

`Block::hardness()` is negative for unbreakable and zero for instant break; otherwise progress
accumulates at `tool_speed / hardness` per second and the tool loses durability when the block
breaks. `Block::preferred_tool()` says which tool is fast on it. Tools stack to 1
(`Item::max_stack_size()`), as do tool blocks; everything else stacks to 64.
