# Minimal planar bipedal walker — robot + gz-sim RL environment

Date: 2026-07-31

## Goal

Define a bipedal robot and a matching gz-sim RL environment so it can be
trained to walk forward with SB3 PPO. The core question driving this design
is: what joint-state data does the observation space need, and how does it
get read out of Gazebo's ECM?

This follows the same architectural pattern already established in this
repo by the `cart_pole_gz_train` port (`gz_scorer.py` / `train_cart_pole.py`):
an in-process `TestFixture`, joint-based ECM access (real physics-engine
velocities, no finite-difference estimation), a Gymnasium wrapper, and SB3
PPO with `VecNormalize`.

## Scope

In scope:
- `biped.sdf` — a static, hand-authored world + robot model
- `biped_scorer.py` — `BipedScorer`, the Gazebo System that applies torques
  and reads joint state
- `train_biped.py` — `CustomBipedGzTrain` Gymnasium wrapper + PPO training

Out of scope (deferred to follow-on work once this core loop trains, same
evolution path `cart_pole_gz_train` took):
- GUI inference script (`run_inference.py` equivalent)
- Manual disturbance helper (`nudge.py` equivalent)
- Scratch verification scripts (`verify_*.py` equivalents)

## Why a static hand-authored SDF, not xacro

There is no `ros2_ws` in this repo, and no existing xacro to convert for this
new robot — unlike the `cart_pole_gz_train` port, which regenerates its SDF
from a real, pre-existing `robot_description` xacro shared with a real robot
launch stack. Hand-authoring the SDF directly (the same pattern the *root*
`cart_pole/` project used for its `cart_pole.sdf`) avoids standing up a
`ros2_ws`/colcon build purely to convert a xacro that doesn't exist yet, and
avoids reproducing the `world_builder.py` `REPO_ROOT` path bug already
flagged elsewhere in this repo's `CLAUDE.md`. If a real/ROS2-visible version
of this robot is wanted later, a xacro can be authored from this SDF's
measured parameters at that point — not before.

## Robot geometry

A minimal planar (sagittal-plane) biped: 9 links, 9 joints total — 7
non-fixed joints (3 passive + 4 actuated, the ones with real degrees of
freedom) plus 2 fixed joints (one per foot, rigidly attaching it to its
shank — no added DOF).

**Planar mount** (torso to world — passive, physics-driven, not actuated,
the same role `pole_joint` plays in the cart-pole precedent). Neither URDF
nor SDF has a single "2D translation + rotation" joint primitive, so this is
three joints chained through two near-massless helper links:

```
world --(prismatic, x)--> torso_slider_x --(prismatic, z)--> torso_slider_z --(revolute, pitch/y)--> torso
```

- `torso_slider_x`, `torso_slider_z`: near-massless helper links (~0.01 kg)
- x: unconstrained (or a generous range) — forward travel
- z: limited to a generous range (e.g. 0–1.5 m) so the model can't fly off
- pitch: limited generously (e.g. ±1.0 rad) — termination happens well
  before this joint limit is reached

**torso** — box ~0.3 × 0.2 × 0.5 m, ~20 kg. The pelvis/body the legs hang
from.

**Each leg** (left/right, offset ±0.06 m in y purely for visual/collision
separation — every rotation axis stays on y, so dynamics remain effectively
planar even though the two legs occupy parallel y-offset planes rather than
literally one plane):

- **hip joint** (revolute, pitch, **actuated**) → **thigh** (~0.35 m, ~5 kg)
- **knee joint** (revolute, pitch, **actuated**, limited to ~0–2.5 rad so it
  can't hyperextend backward past straight, like a real knee) → **shank**
  (~0.35 m, ~3 kg)
- **foot** (fixed joint, small flat box ~0.15 × 0.08 × 0.03 m, ~0.5 kg, toe
  offset slightly forward of the ankle point for push-off leverage)

All masses, joint limits, and the exact standing/rest height are starting
points. Like every physical constant in the cart-pole precedent (spawn
height, effort limits, termination bounds), the real values get measured
once the SDF is actually simulated, not guessed upfront — expect
`biped_scorer.py`'s constants to be tuned against direct measurement during
implementation, the same way `gz_scorer.py`'s `CART_POSITION_LIMIT` etc.
were.

## Observation space

`Box(13,)`, float32, all read live from the ECM's `Joint` components
(position/velocity — real physics-engine values, no finite-difference
estimate, matching `gz_scorer.py`'s approach):

| # | Field | Source |
|---|-------|--------|
| 0 | `torso_x_vel` | x-prismatic joint velocity |
| 1 | `torso_z_pos` | z-prismatic joint position (height) |
| 2 | `torso_z_vel` | z-prismatic joint velocity |
| 3 | `torso_pitch` | pitch-revolute joint position |
| 4 | `torso_pitch_vel` | pitch-revolute joint velocity |
| 5 | `hip_L_pos` | left hip joint position |
| 6 | `hip_L_vel` | left hip joint velocity |
| 7 | `knee_L_pos` | left knee joint position |
| 8 | `knee_L_vel` | left knee joint velocity |
| 9 | `hip_R_pos` | right hip joint position |
| 10 | `hip_R_vel` | right hip joint velocity |
| 11 | `knee_R_pos` | right knee joint position |
| 12 | `knee_R_vel` | right knee joint velocity |

Deliberately **excludes** absolute `torso_x_pos`: it's translation-invariant
and including it would only hurt generalization, not help the policy —
the same reason MuJoCo's Walker2d excludes it by default.

A foot ground-contact boolean per foot (2 more dims) is a plausible future
addition, but needs a Contact system/sensor per foot that isn't part of this
initial scorer — deferred, not required to get a first training run going.

## Action space

`Box(4,)`, one torque per actuated joint, order `[hip_L, knee_L, hip_R,
knee_R]`, each in `[-max_torque, max_torque]`.

`max_torque` is read live from each joint's own `effort_limits(ecm)` rather
than hardcoded — the same "read the live limit, don't hardcode" convention
`gz_scorer.py` and `verify_dynamics.py` already established for `cart_joint`.
Applied via each joint's `set_force(ecm, [torque])` in `on_pre_update`,
mirroring the cart-pole precedent exactly.

## Reward

```
reward = forward_velocity - control_cost - fall_penalty
```

- `control_cost = 0.001 * sum(action**2)` — a small penalty discouraging
  jittery/wasteful torque (matches the BipedalWalker-v3/Walker2d convention)
- `fall_penalty` — a one-time penalty (starting point: 5) applied on the
  step that triggers termination, so falling is actively penalized rather
  than merely forfeiting future reward

These constants are starting points, expected to be tuned once training
actually runs — they can't be picked correctly without measuring, same as
every other constant in this design.

## Termination

Episode ends when either:
- `torso_z_pos` drops below a minimum height (collapsed/crouched too low) —
  threshold to be measured once the SDF's actual standing rest height is
  known
- `abs(torso_pitch)` exceeds a maximum tilt (tipped over forward or
  backward)

## Open questions / explicitly deferred

- Exact numeric values for masses, joint limits, termination thresholds,
  and reward constants — all measured/tuned during implementation, not
  fixed here.
- Foot ground-contact sensing (contact booleans in the observation).
- GUI inference, manual disturbance testing, and scratch verification
  scripts — same follow-on scripts the cart-pole port eventually grew, not
  needed for the first trainable version.
- Whether this robot ever needs a xacro/`ros2_ws` presence (e.g. to drive a
  real robot) — not needed for RL training itself, per this repo's
  `CLAUDE.md`, and not decided here.
