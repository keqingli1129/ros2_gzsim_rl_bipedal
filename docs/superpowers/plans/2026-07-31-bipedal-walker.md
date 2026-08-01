# Minimal Planar Bipedal Walker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal planar bipedal robot (hand-authored SDF), a joint-based gz-sim scorer, and an SB3 PPO training script, so it can be trained to walk forward.

**Architecture:** Mirrors this repo's existing `cart_pole_gz_train` pattern (`gz_scorer.py` / `train_cart_pole.py`): an in-process `TestFixture`, joint-based ECM access for both actuation and observation (real physics-engine velocities, no finite-difference estimate), a Gymnasium wrapper, and SB3 PPO with `VecNormalize`. Unlike that precedent, the robot is a static hand-authored SDF (no xacro/`ros2_ws` — see the design spec's rationale), and actuation is continuous per-joint torque (`Box(4,)`) rather than discrete bang-bang.

**Tech Stack:** `gz.sim8` (TestFixture/ECM), Gymnasium, Stable-Baselines3 (PPO + VecNormalize), numpy.

**Spec:** `docs/superpowers/specs/2026-07-31-bipedal-walker-design.md`

## Global Constraints

- Every invocation touching `gz.sim8` needs `PYTHONPATH=/usr/lib/python3/dist-packages uv run python ...` and must call `ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)` before any `gz.*` import (see this repo's `CLAUDE.md`).
- No xacro, no `ros2_ws`, no colcon build — `biped.sdf` is static and hand-authored (spec: "Why a static hand-authored SDF, not xacro").
- `VecNormalize` is required, not optional, for training (matches this repo's established cart-pole precedent).
- Actuator effort limits are read live from the ECM (`joint.effort_limits(ecm)`) — never hardcode a torque cap that could silently desync from `biped.sdf`.
- Out of scope for this plan (per spec): GUI inference, manual disturbance testing (`nudge.py` equivalent). Not needed to get a first trainable pipeline running.

---

## Empirical baseline (measured, not guessed)

The exact geometry, effort limits, and termination thresholds below were measured by prototyping `biped.sdf` directly against `gz.sim8.TestFixture` before writing this plan (not derived on paper) — this repo's whole convention (see `CLAUDE.md`) is measuring physical behavior rather than assuming it. Key findings that shape Tasks 1–2:

- With all 7 joints at their reference (zero) configuration, the standing pose is grounded and stable (torso height/pitch drift `< 0.01`) for roughly the first second of simulated time.
- Unactuated, it then falls over within 2–3 more seconds (torso pitch saturates at its `torso_pitch_joint` limit, height drops ~0.72m). This is expected and correct: a two-legged, ~0.08m-wide-footed stance with no active balance control is a genuinely unstable equilibrium, unlike the cart-pole precedent's naturally-stable wide base. The RL policy's job is to prevent this, not for the passive system to resist it indefinitely.
- 60 N·m torque on a hip joint (the declared effort limit) from rest reaches ~1.38 rad in 200ms — a large, clearly-directional response, confirming the actuation chain works and the joint limits (`hip: ±1.5 rad`, `knee: -0.05 to 2.5 rad`) are respected by the physics engine.
- `joint.effort_limits(ecm)[0]` correctly reads back `biped.sdf`'s declared `<effort>` value (60.0) — the "read live, don't hardcode" convention works for this new robot exactly as it does for `cart_joint` in the existing precedent.

These measurements set `HEIGHT_DROP_LIMIT = 0.4` and `PITCH_LIMIT = 0.6` (Task 2) — both comfortably inside the measured ~1s stable window and the joints' own harder mechanical stops, so termination fires while the robot is still recoverably tipping, not already collapsed.

---

### Task 1: `biped.sdf` — the robot definition

**Files:**
- Create: `biped.sdf`
- Test: `verify_biped_dynamics.py`

**Interfaces:**
- Produces: a world named `biped` containing a model named `biped` with links `torso_slider_x`, `torso_slider_z`, `torso`, `thigh_L`, `shank_L`, `foot_L`, `thigh_R`, `shank_R`, `foot_R`, and joints `torso_slide_x_joint`, `torso_slide_z_joint`, `torso_pitch_joint`, `hip_L_joint`, `knee_L_joint`, `foot_L_joint`, `hip_R_joint`, `knee_R_joint`, `foot_R_joint`. All 4 actuated joints (`hip_L_joint`, `knee_L_joint`, `hip_R_joint`, `knee_R_joint`) declare `<effort>60</effort>`. Later tasks load this via `TestFixture(SDF_PATH)`.

- [ ] **Step 1: Write the failing verification script**

Create `verify_biped_dynamics.py`:

```python
"""Verify biped.sdf's standing pose is grounded and physically sensible,
and that leg-joint torque produces the expected motion.

Unlike cart_pole_train's verify_dynamics.py, this does NOT assert the
model stays motionless indefinitely under zero force: a two-legged,
narrow-footed standing pose with no active balance control is an
inherently unstable equilibrium (measured separately, see this repo's
2026-07-31-bipedal-walker implementation plan: it stands for ~1s, then
falls over within 2-3s). This only checks the first SETTLE_MS, well
inside that measured stable window, is enough to catch a genuinely
broken spawn (falling/sinking immediately, penetrating the ground, or
exploding) without asserting indefinite passive stability the physical
design was never meant to have.
"""
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

from gz.sim8 import TestFixture, World, world_entity, Model, Joint

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SDF_PATH = os.path.join(FILE_DIR, "biped.sdf")

SETTLE_MS = 500  # well inside the measured ~1s stable window
SETTLE_TOLERANCE = 0.05  # meters / radians
PUSH_MS = 200

state = {"hip_L_force": 0.0}


def ensure_init(ecm):
    if "hip_L_joint" in state:
        return
    world = World(world_entity(ecm))
    model = Model(world.model_by_name(ecm, "biped"))
    for name in ["torso_slide_x_joint", "torso_slide_z_joint",
                 "torso_pitch_joint", "hip_L_joint", "knee_L_joint"]:
        joint = Joint(model.joint_by_name(ecm, name))
        joint.enable_position_check(ecm, True)
        state[name] = joint
    state["max_hip_force"] = state["hip_L_joint"].effort_limits(ecm)[0]


def on_pre_update(info, ecm):
    if info.paused:
        return
    ensure_init(ecm)
    state["hip_L_joint"].set_force(ecm, [state["hip_L_force"]])


def on_post_update(info, ecm):
    if info.paused:
        return
    ensure_init(ecm)
    state["x"] = state["torso_slide_x_joint"].position(ecm)[0]
    state["z"] = state["torso_slide_z_joint"].position(ecm)[0]
    state["pitch"] = state["torso_pitch_joint"].position(ecm)[0]
    state["hip_L_pos"] = state["hip_L_joint"].position(ecm)[0]


fixture = TestFixture(SDF_PATH)
fixture.on_pre_update(on_pre_update)
fixture.on_post_update(on_post_update)
fixture.finalize()
server = fixture.server()

# --- 1. grounded at spawn, stays grounded for SETTLE_MS ---
server.run(True, 1, False)
spawn_x, spawn_z, spawn_pitch = state["x"], state["z"], state["pitch"]
server.run(True, SETTLE_MS - 1, False)
settled_z, settled_pitch = state["z"], state["pitch"]

assert abs(spawn_z) < 1e-3 and abs(spawn_pitch) < 1e-3, (
    f"model did not spawn at its reference standing pose (z={spawn_z:.4f}, "
    f"pitch={spawn_pitch:.4f}) - check biped.sdf's link poses"
)
assert abs(settled_z) < SETTLE_TOLERANCE, (
    f"torso height drifted by {settled_z:.4f}m within the first {SETTLE_MS}ms "
    f"(tolerance {SETTLE_TOLERANCE}) - model is falling/sinking immediately, "
    "not just slowly toppling over several seconds"
)
assert abs(settled_pitch) < SETTLE_TOLERANCE, (
    f"torso pitch drifted by {settled_pitch:.4f}rad within the first "
    f"{SETTLE_MS}ms (tolerance {SETTLE_TOLERANCE}) - model is toppling "
    "immediately, not just slowly over several seconds"
)

# --- 2. hip torque produces the expected motion ---
state["hip_L_force"] = state["max_hip_force"]
server.run(True, PUSH_MS, False)
pushed_hip = state["hip_L_pos"]

assert pushed_hip > 0.5, (
    f"hip_L only reached {pushed_hip:.4f}rad after {PUSH_MS}ms at max torque "
    f"({state['max_hip_force']}N.m) - expected a large positive swing "
    "(measured directly: max torque for 200ms from rest reaches ~1.38rad)"
)

print(
    f"PASS: spawned at reference pose, stayed within {SETTLE_TOLERANCE} of it "
    f"for {SETTLE_MS}ms (z={settled_z:+.4f}, pitch={settled_pitch:+.4f}); "
    f"{state['max_hip_force']}N.m hip torque for {PUSH_MS}ms then reached "
    f"hip_L={pushed_hip:.4f}rad"
)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python verify_biped_dynamics.py`
Expected: FAIL — `TestFixture` doesn't raise a Python exception for a missing SDF (verified directly); instead it logs `[Err] [SystemPaths.cc:...] File [...] but the path does not exist` / `[Err] [ServerPrivate.cc:...] Failed to find world [...]` to stderr, the world never loads, `on_pre_update`/`on_post_update` never fire, and the script then crashes with `KeyError: 'x'` at the `spawn_x, spawn_z, spawn_pitch = state["x"], ...` line — confirming `biped.sdf` doesn't exist yet.

- [ ] **Step 3: Write `biped.sdf`**

Create `biped.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="biped">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <surface>
            <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </visual>
      </link>
    </model>

    <model name="biped">
      <pose>0 0 0 0 0 0</pose>

      <link name="torso_slider_x">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <mass>0.01</mass>
          <inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
      </link>

      <link name="torso_slider_z">
        <pose>0 0 0.98 0 0 0</pose>
        <inertial>
          <mass>0.01</mass>
          <inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
      </link>

      <link name="torso">
        <pose>0 0 0.98 0 0 0</pose>
        <inertial>
          <mass>20</mass>
          <inertia><ixx>0.5</ixx><iyy>0.5</iyy><izz>0.3</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <geometry><box><size>0.3 0.2 0.5</size></box></geometry>
          <material><ambient>0.25 0.25 0.3 1</ambient><diffuse>0.25 0.25 0.3 1</diffuse></material>
        </visual>
        <collision name="collision">
          <geometry><box><size>0.3 0.2 0.5</size></box></geometry>
        </collision>
      </link>

      <link name="thigh_L">
        <pose>0 0.06 0.73 0 0 0</pose>
        <inertial>
          <mass>5</mass>
          <inertia><ixx>0.05</ixx><iyy>0.05</iyy><izz>0.005</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.06 0.06 0.35</size></box></geometry>
          <material><ambient>0.85 0.45 0.05 1</ambient><diffuse>0.85 0.45 0.05 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.06 0.06 0.35</size></box></geometry>
        </collision>
      </link>

      <link name="shank_L">
        <pose>0 0.06 0.38 0 0 0</pose>
        <inertial>
          <mass>3</mass>
          <inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.003</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.05 0.05 0.35</size></box></geometry>
          <material><ambient>0.75 0.1 0.1 1</ambient><diffuse>0.75 0.1 0.1 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.05 0.05 0.35</size></box></geometry>
        </collision>
      </link>

      <link name="foot_L">
        <pose>0 0.06 0.03 0 0 0</pose>
        <inertial>
          <mass>0.5</mass>
          <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <pose>0.02 0 -0.015 0 0 0</pose>
          <geometry><box><size>0.15 0.08 0.03</size></box></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>0.02 0 -0.015 0 0 0</pose>
          <geometry><box><size>0.15 0.08 0.03</size></box></geometry>
          <surface>
            <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
          </surface>
        </collision>
      </link>

      <link name="thigh_R">
        <pose>0 -0.06 0.73 0 0 0</pose>
        <inertial>
          <mass>5</mass>
          <inertia><ixx>0.05</ixx><iyy>0.05</iyy><izz>0.005</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.06 0.06 0.35</size></box></geometry>
          <material><ambient>0.85 0.45 0.05 1</ambient><diffuse>0.85 0.45 0.05 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.06 0.06 0.35</size></box></geometry>
        </collision>
      </link>

      <link name="shank_R">
        <pose>0 -0.06 0.38 0 0 0</pose>
        <inertial>
          <mass>3</mass>
          <inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.003</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.05 0.05 0.35</size></box></geometry>
          <material><ambient>0.75 0.1 0.1 1</ambient><diffuse>0.75 0.1 0.1 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>0 0 -0.175 0 0 0</pose>
          <geometry><box><size>0.05 0.05 0.35</size></box></geometry>
        </collision>
      </link>

      <link name="foot_R">
        <pose>0 -0.06 0.03 0 0 0</pose>
        <inertial>
          <mass>0.5</mass>
          <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="visual">
          <pose>0.02 0 -0.015 0 0 0</pose>
          <geometry><box><size>0.15 0.08 0.03</size></box></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>0.02 0 -0.015 0 0 0</pose>
          <geometry><box><size>0.15 0.08 0.03</size></box></geometry>
          <surface>
            <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
          </surface>
        </collision>
      </link>

      <joint name="torso_slide_x_joint" type="prismatic">
        <parent>world</parent>
        <child>torso_slider_x</child>
        <axis>
          <xyz>1 0 0</xyz>
          <limit><lower>-1.79769e+308</lower><upper>1.79769e+308</upper></limit>
        </axis>
      </joint>

      <joint name="torso_slide_z_joint" type="prismatic">
        <parent>torso_slider_x</parent>
        <child>torso_slider_z</child>
        <axis>
          <xyz>0 0 1</xyz>
          <limit><lower>-1.0</lower><upper>2.0</upper></limit>
        </axis>
      </joint>

      <joint name="torso_pitch_joint" type="revolute">
        <parent>torso_slider_z</parent>
        <child>torso</child>
        <axis>
          <xyz>0 1 0</xyz>
          <limit><lower>-1.0</lower><upper>1.0</upper></limit>
        </axis>
      </joint>

      <joint name="hip_L_joint" type="revolute">
        <parent>torso</parent>
        <child>thigh_L</child>
        <axis>
          <xyz>0 1 0</xyz>
          <limit><lower>-1.5</lower><upper>1.5</upper><effort>60</effort></limit>
        </axis>
      </joint>

      <joint name="knee_L_joint" type="revolute">
        <parent>thigh_L</parent>
        <child>shank_L</child>
        <axis>
          <xyz>0 1 0</xyz>
          <limit><lower>-0.05</lower><upper>2.5</upper><effort>60</effort></limit>
        </axis>
      </joint>

      <joint name="foot_L_joint" type="fixed">
        <parent>shank_L</parent>
        <child>foot_L</child>
      </joint>

      <joint name="hip_R_joint" type="revolute">
        <parent>torso</parent>
        <child>thigh_R</child>
        <axis>
          <xyz>0 1 0</xyz>
          <limit><lower>-1.5</lower><upper>1.5</upper><effort>60</effort></limit>
        </axis>
      </joint>

      <joint name="knee_R_joint" type="revolute">
        <parent>thigh_R</parent>
        <child>shank_R</child>
        <axis>
          <xyz>0 1 0</xyz>
          <limit><lower>-0.05</lower><upper>2.5</upper><effort>60</effort></limit>
        </axis>
      </joint>

      <joint name="foot_R_joint" type="fixed">
        <parent>shank_R</parent>
        <child>foot_R</child>
      </joint>

    </model>

  </world>
</sdf>
```

- [ ] **Step 4: Run the verification script to confirm it passes**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python verify_biped_dynamics.py`
Expected: `PASS: spawned at reference pose, stayed within 0.05 of it for 500ms (z=+0.0000, pitch=+0.0002); 60.0N.m hip torque for 200ms then reached hip_L=1.38xxrad` (exact decimals may vary slightly by solver iteration count, but must satisfy both assertions).

- [ ] **Step 5: Commit**

```bash
git add biped.sdf verify_biped_dynamics.py
git commit -m "$(cat <<'EOF'
Add hand-authored biped.sdf and its dynamics verification script

Minimal planar (sagittal-plane) bipedal robot: torso on a
prismatic-x/prismatic-z/revolute-pitch planar mount, two legs each
with an actuated hip and knee. Geometry and effort limits measured
directly against TestFixture before committing (see the implementation
plan's "Empirical baseline" section), not guessed.
EOF
)"
```

---

### Task 2: `biped_scorer.py` — joint-based ECM scoring

**Files:**
- Create: `biped_scorer.py`
- Test: `verify_biped_scorer.py`

**Interfaces:**
- Consumes: `biped.sdf` (Task 1) — joint names `torso_slide_x_joint`, `torso_slide_z_joint`, `torso_pitch_joint`, `hip_L_joint`, `knee_L_joint`, `hip_R_joint`, `knee_R_joint`; model name `biped`.
- Produces: `BipedScorer` class with `reset() -> (obs: np.ndarray[13], info: dict)`, `step(action: np.ndarray[4]) -> (obs: np.ndarray[13], reward: float, terminated: bool, truncated: bool, info: dict)`, `close() -> None`. Also exports `HEIGHT_DROP_LIMIT = 0.4` and `PITCH_LIMIT = 0.6` (both consumed by Task 3). Observation order: `[torso_x_vel, torso_z_pos, torso_z_vel, torso_pitch, torso_pitch_vel, hip_L_pos, hip_L_vel, knee_L_pos, knee_L_vel, hip_R_pos, hip_R_vel, knee_R_pos, knee_R_vel]`. Action order: `[hip_L_torque, knee_L_torque, hip_R_torque, knee_R_torque]`.

- [ ] **Step 1: Write the failing verification script**

Create `verify_biped_scorer.py`:

```python
"""Exercise BipedScorer's step/reset against the real biped.sdf world.

Mirrors verify_scorer.py's structure for the cart-pole precedent, adapted
for the biped's genuinely different stability profile: a short idle
window (not an indefinite one) is the right check here, since standing
unactuated is an inherently unstable equilibrium for this two-legged,
narrow-footed design (measured separately: falls over within 2-3s with
zero control) - unlike cart-pole's naturally-stable base.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from biped_scorer import BipedScorer

IDLE_STEPS = 100  # 500ms at 5ms/step - well inside the measured ~1s stable window
ZERO_ACTION = np.zeros(4, dtype=np.float32)
HIP_L_PUSH = np.array([60.0, 0.0, 0.0, 0.0], dtype=np.float32)

scorer = BipedScorer()
obs, _info = scorer.reset()
assert obs.shape == (13,)
assert all(abs(v) < 1e-3 for v in obs), \
    f"env should reset to a motionless, upright, standing state, got {obs}"

# --- idle: no torque at all, for the measured-stable window ---
for i in range(IDLE_STEPS):
    obs, _reward, terminated, _truncated, _info = scorer.step(ZERO_ACTION)
    assert not terminated, (
        f"episode terminated at idle step {i + 1} ({(i + 1) * 5}ms) with "
        f"obs={obs} - the biped should stay upright for at least "
        f"{IDLE_STEPS * 5}ms unactuated (measured stable window is ~1s); "
        "this means the standing pose itself regressed"
    )
idle_obs = obs
assert abs(idle_obs[1]) < 0.1 and abs(idle_obs[3]) < 0.1, (
    f"state drifted more than expected while idle for {IDLE_STEPS} steps: "
    f"torso_z_pos={idle_obs[1]:.4f}, torso_pitch={idle_obs[3]:.4f}"
)

# --- actuated: push left hip forward and confirm it actually moves ---
scorer.reset()
for _ in range(20):
    obs, reward, terminated, truncated, _info = scorer.step(HIP_L_PUSH)
    assert not terminated, "should not fall over in 20 steps of 5ms each"

assert obs[5] > 0.3, \
    f"hip_L should have swung forward briskly after 100ms at max torque, got {obs[5]}"
scorer.close()
print(
    f"PASS: idle for {IDLE_STEPS} steps stayed upright "
    f"(torso_z_pos={idle_obs[1]:+.4f}, torso_pitch={idle_obs[3]:+.4f}); "
    f"after 20 steps of hip_L torque, hip_L_pos={obs[5]:.3f}"
)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python verify_biped_scorer.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'biped_scorer'`.

- [ ] **Step 3: Write `biped_scorer.py`**

Create `biped_scorer.py`:

```python
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import numpy as np
from gz.sim8 import TestFixture, World, world_entity, Model, Joint

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SDF_PATH = os.path.join(FILE_DIR, "biped.sdf")

# Measured against biped.sdf's standing pose (see this repo's
# 2026-07-31-bipedal-walker implementation plan, "Empirical baseline"):
# stays within 1cm of standing height/pitch for ~1s unactuated, then falls
# over within 2-3s (an inherently unstable standing pose with no active
# balance control, unlike cart-pole's naturally-stable base). These
# thresholds sit well inside that measured fall trajectory, and well
# inside the joints' own harder mechanical stops (torso_pitch_joint's
# +/-1.0 rad), so termination fires while still recoverably tipping, not
# already jammed against a limit.
HEIGHT_DROP_LIMIT = 0.4  # meters, drop from standing height
PITCH_LIMIT = 0.6  # radians

CONTROL_COST_WEIGHT = 0.001
FALL_PENALTY = 5.0

_JOINT_NAMES = [
    "torso_slide_x_joint", "torso_slide_z_joint", "torso_pitch_joint",
    "hip_L_joint", "knee_L_joint", "hip_R_joint", "knee_R_joint",
]
_ACTUATED_JOINTS = ["hip_L_joint", "knee_L_joint", "hip_R_joint", "knee_R_joint"]


class BipedScorer:
    """Gazebo System that scores the biped via joint-based ECM access -
    reads all 7 joints (3 passive planar-mount + 4 actuated leg joints)
    directly via their Joint components, same approach as gz_scorer.py's
    GzCartPoleScorer (real physics-engine velocity, no finite-difference
    estimate)."""

    def __init__(self):
        self.command = np.zeros(4, dtype=np.float32)
        self._build_fixture()
        self.terminated = False
        self._initialized = False
        self.state = np.zeros(13, dtype=np.float32)
        self.reward = 0.0

    def _build_fixture(self):
        """Rebuild TestFixture/server from scratch on reset rather than
        calling server.reset_all() - same gz-sim8 bug documented in
        gz_scorer.py: reset_all() desyncs the physics engine from the ECM
        while leaving entity IDs unchanged, silently breaking force
        application and state reads."""
        self.server = None
        self.fixture = None
        self.fixture = TestFixture(SDF_PATH)
        self.fixture.on_pre_update(self.on_pre_update)
        self.fixture.on_post_update(self.on_post_update)
        self.fixture.finalize()
        self.server = self.fixture.server()

    def _ensure_initialized(self, ecm):
        if self._initialized:
            return
        world = World(world_entity(ecm))
        model = Model(world.model_by_name(ecm, "biped"))
        self.joints = {}
        for name in _JOINT_NAMES:
            joint = Joint(model.joint_by_name(ecm, name))
            joint.enable_position_check(ecm, True)
            joint.enable_velocity_check(ecm, True)
            self.joints[name] = joint
        # Effort limits are a hard actuator clamp enforced by the physics
        # engine - read live rather than hardcoded, so a future biped.sdf
        # edit changing a limit doesn't silently desync this (same
        # convention as gz_scorer.py's cart_joint.effort_limits).
        self.max_torque = {
            name: self.joints[name].effort_limits(ecm)[0]
            for name in _ACTUATED_JOINTS
        }
        self._initialized = True

    def on_pre_update(self, info, ecm):
        if info.paused:
            return
        self._ensure_initialized(ecm)
        for i, name in enumerate(_ACTUATED_JOINTS):
            torque = float(np.clip(
                self.command[i], -self.max_torque[name], self.max_torque[name]))
            self.joints[name].set_force(ecm, [torque])

    def on_post_update(self, info, ecm):
        if info.paused:
            return
        self._ensure_initialized(ecm)
        j = self.joints
        torso_x_vel = j["torso_slide_x_joint"].velocity(ecm)[0]
        torso_z_pos = j["torso_slide_z_joint"].position(ecm)[0]
        torso_z_vel = j["torso_slide_z_joint"].velocity(ecm)[0]
        torso_pitch = j["torso_pitch_joint"].position(ecm)[0]
        torso_pitch_vel = j["torso_pitch_joint"].velocity(ecm)[0]
        hip_L_pos = j["hip_L_joint"].position(ecm)[0]
        hip_L_vel = j["hip_L_joint"].velocity(ecm)[0]
        knee_L_pos = j["knee_L_joint"].position(ecm)[0]
        knee_L_vel = j["knee_L_joint"].velocity(ecm)[0]
        hip_R_pos = j["hip_R_joint"].position(ecm)[0]
        hip_R_vel = j["hip_R_joint"].velocity(ecm)[0]
        knee_R_pos = j["knee_R_joint"].position(ecm)[0]
        knee_R_vel = j["knee_R_joint"].velocity(ecm)[0]

        self.state = np.array([
            torso_x_vel, torso_z_pos, torso_z_vel, torso_pitch, torso_pitch_vel,
            hip_L_pos, hip_L_vel, knee_L_pos, knee_L_vel,
            hip_R_pos, hip_R_vel, knee_R_pos, knee_R_vel,
        ], dtype=np.float32)

        if not self.terminated:
            self.terminated = (
                torso_z_pos < -HEIGHT_DROP_LIMIT or abs(torso_pitch) > PITCH_LIMIT
            )

        control_cost = CONTROL_COST_WEIGHT * float(np.sum(np.square(self.command)))
        fall_penalty = FALL_PENALTY if self.terminated else 0.0
        self.reward = float(torso_x_vel - control_cost - fall_penalty)

    def step(self, action):
        self.command = np.asarray(action, dtype=np.float32)
        self.server.run(True, 5, False)
        return self.state, self.reward, self.terminated, False, {}

    def reset(self):
        self._build_fixture()
        self.command = np.zeros(4, dtype=np.float32)
        self.terminated = False
        self._initialized = False
        obs, _reward, _term, _trunc, _info = self.step(np.zeros(4, dtype=np.float32))
        return obs, {}

    def close(self):
        self.server = None
        self.fixture = None
```

- [ ] **Step 4: Run the verification script to confirm it passes**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python verify_biped_scorer.py`
Expected: `PASS: idle for 100 steps stayed upright (torso_z_pos=+0.00xx, torso_pitch=+0.00xx); after 20 steps of hip_L torque, hip_L_pos=x.xxx` (must satisfy both assertions — `hip_L_pos > 0.3`).

- [ ] **Step 5: Commit**

```bash
git add biped_scorer.py verify_biped_scorer.py
git commit -m "$(cat <<'EOF'
Add BipedScorer: joint-based ECM scoring for the biped

13-dim observation (torso x-vel/z-pos/z-vel/pitch/pitch-vel + 4 leg
joints' pos/vel), 4-dim continuous torque action, forward-velocity
reward with a control-cost and one-time fall penalty. Mirrors
gz_scorer.py's GzCartPoleScorer structure and its
rebuild-fixture-on-reset workaround for the reset_all() ECM-desync bug.
EOF
)"
```

---

### Task 3: `train_biped.py` — Gymnasium wrapper + PPO training

**Files:**
- Create: `train_biped.py`

**Interfaces:**
- Consumes: `BipedScorer`, `HEIGHT_DROP_LIMIT`, `PITCH_LIMIT` from `biped_scorer.py` (Task 2).
- Produces: `CustomBipedGzTrain(gym.Env)` with `observation_space = Box(13,)`, `action_space = Box(4,)`; `main()` trains PPO + `VecNormalize`, saving `biped_ppo.zip` and `biped_vecnormalize.pkl` next to this file.

- [ ] **Step 1: Write `train_biped.py`**

Create `train_biped.py`:

```python
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_scorer import BipedScorer, HEIGHT_DROP_LIMIT, PITCH_LIMIT

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MAX_TORQUE = 60.0  # matches biped.sdf's declared <effort> on every leg joint


class CustomBipedGzTrain(gym.Env):
    """Wraps BipedScorer for Gymnasium/SB3."""

    def __init__(self, env_config=None):
        self.env = BipedScorer()
        # Order: [hip_L, knee_L, hip_R, knee_R], each a torque in
        # [-MAX_TORQUE, MAX_TORQUE]. BipedScorer itself clamps against the
        # live ECM effort limit (see its _ensure_initialized) - MAX_TORQUE
        # here only needs to match biped.sdf's declared value closely
        # enough to size this Box sanely, not be the actual enforced cap.
        self.action_space = gym.spaces.Box(
            np.array([-MAX_TORQUE] * 4, dtype=np.float32),
            np.array([MAX_TORQUE] * 4, dtype=np.float32),
        )
        # 13-dim observation - see BipedScorer.on_post_update for the exact
        # assembly order. Bounds mirror train_cart_pole.py's convention:
        # only the two dimensions with a real termination threshold
        # (torso_z_pos, torso_pitch) get a finite bound; everything else is
        # unbounded (VecNormalize handles the actual scaling).
        low = np.array([-np.inf, -HEIGHT_DROP_LIMIT, -np.inf, -PITCH_LIMIT, -np.inf,
                         -np.inf, -np.inf, -np.inf, -np.inf,
                         -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, PITCH_LIMIT, np.inf,
                          np.inf, np.inf, np.inf, np.inf,
                          np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, (13,), np.float32)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def main():
    # VecNormalize is required, not optional - see this repo's CLAUDE.md:
    # the cart-pole precedent's velocity dimensions are 10-30x the scale
    # of its position dimensions, which stalled learning outright until
    # VecNormalize was added. This env has the same shape of problem
    # (torque-scale/velocity dimensions vs. small angle/position ones).
    venv = DummyVecEnv([lambda: Monitor(CustomBipedGzTrain())])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, verbose=1, device="auto")
    model.learn(total_timesteps=100_000)
    model_path = os.path.join(FILE_DIR, "biped_ppo")
    model.save(model_path)
    vecnorm_path = os.path.join(FILE_DIR, "biped_vecnormalize.pkl")
    venv.save(vecnorm_path)
    venv.close()
    print(f"Training complete. Saved model to {model_path}.zip")
    print(f"Saved VecNormalize stats to {vecnorm_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the full pipeline end-to-end (reduced timesteps)**

This does not run the full 100k-timestep training (that's a real, separate training run for you to kick off once this plan is done) - it only confirms the whole chain (env → `Monitor` → `DummyVecEnv` → `VecNormalize` → `PPO` → save) executes without error. Run this from the repo root:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
import sys
sys.path.insert(0, '.')
from train_biped import CustomBipedGzTrain
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

venv = DummyVecEnv([lambda: Monitor(CustomBipedGzTrain())])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
model = PPO('MlpPolicy', venv, verbose=0, device='auto')
model.learn(total_timesteps=2000)
model.save('/tmp/biped_smoke_test_ppo')
venv.save('/tmp/biped_smoke_test_vecnormalize.pkl')
venv.close()
print('SMOKE TEST PASS: 2000-timestep training completed and artifacts saved')
"
```

Expected: `SMOKE TEST PASS: 2000-timestep training completed and artifacts saved`, with no exceptions. PPO's own rollout logging (`ep_rew_mean`, `ep_len_mean`, etc.) will print along the way if you bump `verbose` — that's expected noise, not a failure signal.

- [ ] **Step 3: Clean up the smoke-test artifacts**

```bash
rm -f /tmp/biped_smoke_test_ppo.zip /tmp/biped_smoke_test_vecnormalize.pkl
```

- [ ] **Step 4: Commit**

```bash
git add train_biped.py
git commit -m "$(cat <<'EOF'
Add train_biped.py: PPO + VecNormalize training for the biped

CustomBipedGzTrain wraps BipedScorer for Gymnasium/SB3 with a
Box(13,) observation space and Box(4,) continuous-torque action
space. Full pipeline (env -> Monitor -> DummyVecEnv -> VecNormalize
-> PPO) smoke-tested at 2000 timesteps before this commit; a real
100k-timestep run is a separate, longer step.
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** robot geometry (Task 1), observation space (Task 2, `BipedScorer.on_post_update`), action space (Task 2 `_ACTUATED_JOINTS` + Task 3 `CustomBipedGzTrain.action_space`), reward (Task 2), termination (Task 2), file plan (Tasks 1-3 match the spec's three files exactly). Deferred items (GUI inference, `nudge.py` equivalent, `verify_*` QA scripts beyond the two written here, foot contact sensing) are spec-deferred, not silently dropped.
- **No placeholders:** every step above has complete, runnable code — no `# TODO`/`# implement later` markers.
- **Type/name consistency:** `BipedScorer.reset()`/`step()` signatures match what `CustomBipedGzTrain` calls; `HEIGHT_DROP_LIMIT`/`PITCH_LIMIT` are defined once in `biped_scorer.py` and imported (not redefined) in both `verify_biped_scorer.py`'s assertions and `train_biped.py`'s observation bounds; observation/action ordering is stated identically in each task's Interfaces block.
