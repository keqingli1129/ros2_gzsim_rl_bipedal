# Biped Inference Script (`infer.py`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `infer.py`, a standalone script that runs a trained biped policy (`biped_ppo.zip` + `biped_vecnormalize.pkl`) against `biped.sdf` with a visible Gazebo GUI, controlling the robot entirely over `gz.transport13` topics rather than reusing `BipedScorer`'s in-process ECM access.

**Architecture:** `infer.py` launches `gz sim -s -r biped.sdf` (server) and `gz sim -g` (GUI) as subprocesses, subscribes to a `JointStatePublisher`-published `joint_state` topic to read the robot's state, and publishes torque commands to 4 `ApplyJointForce`-provided `cmd_force` topics — one per actuated joint. This mirrors the removed cart-pole `run_inference.py` architecture exactly (see `git show 769a970~1:run_inference.py` in this repo's history).

**Tech Stack:** Python (`uv run`), `gz.sim8`/`gz.transport13`/`gz.msgs10` (system dist-packages via `PYTHONPATH`), `stable-baselines3` (`PPO`, `VecNormalize`, `DummyVecEnv`), `gymnasium`.

## Global Constraints

- All commands run as: `PYTHONPATH=/usr/lib/python3/dist-packages uv run <script>` (per this repo's `CLAUDE.md`, "Environment setup").
- `infer.py` must preload `libgz-sim8.so` via `ctypes.CDLL(..., ctypes.RTLD_GLOBAL)` before any `gz.*` import, exactly like `biped_scorer.py` does.
- The 13-dim observation must be assembled in exactly this order (matches `BipedScorer.on_post_update` / `train_biped.py`'s observation space): `torso_x_vel, torso_z_pos, torso_z_vel, torso_pitch, torso_pitch_vel, hip_L_pos, hip_L_vel, knee_L_pos, knee_L_vel, hip_R_pos, hip_R_vel, knee_R_pos, knee_R_vel`.
- `HEIGHT_DROP_LIMIT` and `PITCH_LIMIT` must be imported from `biped_scorer.py`, never re-hardcoded.
- Actuated-joint effort limits must be read live from `biped.sdf`, never hardcoded (same convention as `biped_scorer.py`'s `_ensure_initialized`).
- No pytest / no new automated test framework — this repo's convention is plain assert-based `verify_*.py` scripts and ad-hoc `python -c` checks, run directly via `uv run`. Follow that convention.
- Spec doc: `docs/superpowers/specs/2026-08-02-biped-inference-design.md`. All design rationale referenced below lives there — do not re-derive it independently if a task references it.

---

### Task 1: Add transport plugins to `biped.sdf`, verify no training regression

**Files:**
- Modify: `biped.sdf:244-251`
- Test: run existing `verify_biped_scorer.py` unchanged (regression check, no new file)

**Interfaces:**
- Produces: `biped.sdf`'s `<model name="biped">` now exposes `/world/biped/model/biped/joint_state` (via `JointStatePublisher`) and `/model/biped/joint/<hip_L_joint|knee_L_joint|hip_R_joint|knee_R_joint>/cmd_force` (via 4x `ApplyJointForce`) over `gz.transport13`, for later tasks to subscribe/publish to.

- [ ] **Step 1: Add the plugin declarations to `biped.sdf`**

Insert immediately after the `foot_R_joint` block (currently ending at line 249) and before the closing `</model>` (currently line 251):

```xml
      <joint name="foot_R_joint" type="fixed">
        <parent>shank_R</parent>
        <child>foot_R</child>
      </joint>

      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"></plugin>

      <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
        <joint_name>hip_L_joint</joint_name>
      </plugin>
      <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
        <joint_name>knee_L_joint</joint_name>
      </plugin>
      <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
        <joint_name>hip_R_joint</joint_name>
      </plugin>
      <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
        <joint_name>knee_R_joint</joint_name>
      </plugin>

    </model>
```

Use the Edit tool with `old_string` matching the `foot_R_joint` block through `    </model>` (lines 246-251 in the current file) and `new_string` as shown above.

- [ ] **Step 2: Sanity-check the SDF still parses**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
import ctypes
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libgz-sim8.so', ctypes.RTLD_GLOBAL)
from gz.sim8 import TestFixture
f = TestFixture('biped.sdf')
f.finalize()
print('SDF loaded OK')
"`

Expected: `SDF loaded OK` with no errors. If this fails, check the new `<plugin>` blocks are inside `<model name="biped">` (not accidentally outside it or inside a `<joint>`), and that `foot_R_joint`'s closing `</joint>` tag wasn't dropped.

- [ ] **Step 3: Run the regression check — this is the critical risk verification from the design doc**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python verify_biped_scorer.py`

Expected: `PASS: idle for 100 steps stayed upright ...` printed, exit code 0 — in particular this confirms the actuated-movement assertion (`obs[5] > 0.3` after sustained `hip_L` torque) still holds, meaning the new `ApplyJointForce` plugins (which default to 0 force, since nothing publishes to their topics during this in-process run) are not winning the `JointForceCmd` write race against `BipedScorer.on_pre_update`'s real torque.

**If this fails:** do not proceed with tasks 2-6 against this `biped.sdf`. Instead, copy `biped.sdf` to `biped_infer.sdf`, move the 5 new `<plugin>` blocks from `biped.sdf` into `biped_infer.sdf` only (reverting `biped.sdf` to its original state), and use `biped_infer.sdf` as the path in Task 2's `SDF_PATH` override instead of importing it from `biped_scorer.py`. Re-run this task's Step 2/3 checks against `biped_infer.sdf` (Step 3 becomes optional in that branch, since `biped_scorer.py` would no longer load the modified file at all).

- [ ] **Step 4: Commit**

```bash
git add biped.sdf
git commit -m "$(cat <<'EOF'
Add JointStatePublisher and ApplyJointForce plugins to biped.sdf

Exposes joint_state and per-actuated-joint cmd_force topics over
gz.transport13 so infer.py can control the robot as an external
subprocess, without changing anything BipedScorer's in-process training
path reads or writes (verified via verify_biped_scorer.py).
EOF
)"
```

---

### Task 2: `infer.py` skeleton — constants, `_ObsSpaceStub`, `_load_normalizer`

**Files:**
- Create: `infer.py`

**Interfaces:**
- Consumes: `biped_scorer.SDF_PATH`, `biped_scorer.HEIGHT_DROP_LIMIT`, `biped_scorer.PITCH_LIMIT` (all module-level constants already defined in `biped_scorer.py`).
- Produces: `_ObsSpaceStub` (a `gym.Env` subclass), `_load_normalizer(vecnorm_path) -> VecNormalize`, module constants `FILE_DIR`, `WORLD_NAME`, `MODEL_NAME`, `STEP_PERIOD`, `MAX_ITERATIONS`, `MAX_RESET_ATTEMPTS`, `ACTUATED_JOINTS`, `JOINT_NAMES` — all consumed by later tasks in this same file.

- [ ] **Step 1: Create `infer.py` with the header, constants, and both classes/functions**

```python
"""Run a trained biped policy against biped.sdf with a visible Gazebo GUI.

Mirrors the removed cart-pole run_inference.py's architecture: drives
Gazebo as an external subprocess (gz sim -s -r + gz sim -g) and controls
the robot entirely over gz.transport13 topics - deliberately not reusing
BipedScorer's in-process TestFixture, since that has no GUI attachment
path. See docs/superpowers/specs/2026-08-02-biped-inference-design.md.
"""
import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, FILE_DIR)

from biped_scorer import SDF_PATH, HEIGHT_DROP_LIMIT, PITCH_LIMIT

from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

WORLD_NAME = "biped"
MODEL_NAME = "biped"
# 5 x 1ms physics steps per action, matching both biped.sdf's max_step_size
# (0.001) and BipedScorer.step's server.run(True, 5, False) cadence.
STEP_PERIOD = 0.005
MAX_ITERATIONS = 50000  # ~250s at STEP_PERIOD, same value cart-pole used
MAX_RESET_ATTEMPTS = 5

# Order matches biped_scorer.py's _ACTUATED_JOINTS - keep both in sync.
ACTUATED_JOINTS = ["hip_L_joint", "knee_L_joint", "hip_R_joint", "knee_R_joint"]
# Order matches biped_scorer.py's _JOINT_NAMES and BipedScorer.on_post_update's
# state assembly order - keep all three in sync.
JOINT_NAMES = [
    "torso_slide_x_joint", "torso_slide_z_joint", "torso_pitch_joint",
    "hip_L_joint", "knee_L_joint", "hip_R_joint", "knee_R_joint",
]


class _ObsSpaceStub(gym.Env):
    """Carries only the observation/action space VecNormalize.load needs to
    shape-check against - mirrors train_biped.py's CustomBipedGzTrain spaces
    exactly (by literal duplication, same approach the old cart-pole
    inference script used for its own stub), without constructing a real
    BipedScorer, which would build its own TestFixture and collide with the
    live inference server this script launches separately."""

    def __init__(self):
        self.action_space = gym.spaces.Box(
            np.array([-1.0] * 4, dtype=np.float32),
            np.array([1.0] * 4, dtype=np.float32),
        )
        low = np.array([-np.inf, -HEIGHT_DROP_LIMIT, -np.inf, -PITCH_LIMIT, -np.inf,
                         -np.inf, -np.inf, -np.inf, -np.inf,
                         -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, PITCH_LIMIT, np.inf,
                          np.inf, np.inf, np.inf, np.inf,
                          np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, (13,), np.float32)

    def reset(self, seed=None, options=None):
        raise NotImplementedError("stub env is never actually reset/stepped")

    def step(self, action):
        raise NotImplementedError("stub env is never actually reset/stepped")


def _load_normalizer(vecnorm_path):
    if not os.path.exists(vecnorm_path):
        raise SystemExit(
            f"ERROR: VecNormalize stats not found at {vecnorm_path!r}.\n"
            "The trained policy expects observations normalized with the "
            "running statistics saved during training; running inference "
            "without them silently reproduces random-baseline performance. "
            "Re-run train_biped.py (which writes biped_vecnormalize.pkl next "
            "to the model) or pass --vecnorm explicitly."
        )
    venv = DummyVecEnv([lambda: _ObsSpaceStub()])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


if __name__ == "__main__":
    print("infer.py skeleton loaded OK")
```

- [ ] **Step 2: Verify the skeleton imports and the stub spaces match `train_biped.py`'s exactly**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
from infer import _ObsSpaceStub
import numpy as np
env = _ObsSpaceStub()
assert env.action_space.shape == (4,)
assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)
assert env.observation_space.shape == (13,)
assert abs(float(env.observation_space.low[1]) - (-0.4)) < 1e-6
assert env.observation_space.high[1] == float('inf')
assert abs(float(env.observation_space.low[3]) - (-0.6)) < 1e-6
assert abs(float(env.observation_space.high[3]) - 0.6) < 1e-6
print('stub spaces OK')
"`

Expected: `stub spaces OK`. (The `-0.4`/`0.6` values are `HEIGHT_DROP_LIMIT`/`PITCH_LIMIT` from `biped_scorer.py` — if these ever change there, this check's literals must be updated too.)

- [ ] **Step 3: Commit**

```bash
git add infer.py
git commit -m "Add infer.py skeleton: constants, obs-space stub, VecNormalize loader"
```

---

### Task 3: `_read_effort_limits` — live SDF effort-limit lookup

**Files:**
- Modify: `infer.py` (append function)

**Interfaces:**
- Consumes: `ACTUATED_JOINTS` (from Task 2), `ET` (already imported in Task 2).
- Produces: `_read_effort_limits(sdf_path) -> dict[str, float]`, consumed by `main()` in Task 6.

- [ ] **Step 1: Add `_read_effort_limits` to `infer.py`**

Insert after `_load_normalizer`'s definition (before the `if __name__ == "__main__":` block):

```python
def _read_effort_limits(sdf_path):
    root = ET.parse(sdf_path).getroot()
    limits = {}
    for name in ACTUATED_JOINTS:
        effort_el = root.find(f".//joint[@name='{name}']/axis/limit/effort")
        if effort_el is None:
            raise RuntimeError(
                f"could not find an effort limit for joint {name!r} in "
                f"{sdf_path} - did biped.sdf's joint structure change?"
            )
        limits[name] = float(effort_el.text)
    return limits
```

- [ ] **Step 2: Verify it reads the real values from `biped.sdf`**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
from infer import _read_effort_limits, SDF_PATH
limits = _read_effort_limits(SDF_PATH)
assert limits == {'hip_L_joint': 60.0, 'knee_L_joint': 60.0, 'hip_R_joint': 60.0, 'knee_R_joint': 60.0}, limits
print('effort limits OK:', limits)
"`

Expected: `effort limits OK: {'hip_L_joint': 60.0, 'knee_L_joint': 60.0, 'hip_R_joint': 60.0, 'knee_R_joint': 60.0}`.

- [ ] **Step 3: Commit**

```bash
git add infer.py
git commit -m "Add infer.py: live effort-limit lookup from biped.sdf"
```

---

### Task 4: Process management and world-reset helpers

**Files:**
- Modify: `infer.py` (append functions)

**Interfaces:**
- Consumes: `SDF_PATH`, `WORLD_NAME` (from Task 2), `subprocess`, `time` (already imported).
- Produces: `_kill_stale_gz_processes()`, `_pkill_and_escalate(pattern)`, `_reset_world(node) -> bool`, `_wait_for_obs(latest, timeout=2.0)` — all consumed by `run_inference()` in Task 6.

- [ ] **Step 1: Add the four helpers to `infer.py`**

Insert after `_read_effort_limits` (before `if __name__ == "__main__":`):

```python
def _kill_stale_gz_processes():
    """Terminate any gz sim server/GUI left over from a prior run - ported
    from cart-pole's run_inference.py, which found plain SIGTERM
    unreliable against gz sim -g in this environment and needed to escalate
    to SIGKILL. Scoped to this exact SDF path (not a bare "gz sim"
    substring) so this can't kill an unrelated gz sim session on the same
    machine."""
    _pkill_and_escalate(f"gz sim -s -r {SDF_PATH}")
    _pkill_and_escalate("gz sim -g")


def _pkill_and_escalate(pattern):
    subprocess.run(["pkill", "-f", pattern], check=False)
    time.sleep(1)
    still_alive = subprocess.run(
        ["pgrep", "-f", pattern], capture_output=True, check=False)
    if still_alive.returncode == 0:
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)
        time.sleep(1)


def _reset_world(node):
    """Reset via reset.all - the same RPC cart-pole's run_inference.py used
    after confirming reset.model_only was a no-op there. Does not raise on
    ok=False: that flag was observed to read False even when the reset
    physically took effect, so the caller checks the real postcondition
    (the next joint_state observation) instead."""
    request = WorldControl()
    request.reset.all = True
    ok, _resp = node.request(
        f"/world/{WORLD_NAME}/control", request, WorldControl, Boolean, 5000)
    return ok


def _wait_for_obs(latest, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if latest["obs"] is not None:
            return
        time.sleep(0.01)
    raise RuntimeError(
        "no joint_state message received - is JointStatePublisher declared "
        "in biped.sdf?"
    )
```

- [ ] **Step 2: Verify `_kill_stale_gz_processes` runs cleanly with nothing to kill**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
from infer import _kill_stale_gz_processes
_kill_stale_gz_processes()
print('kill-stale ran OK')
"`

Expected: `kill-stale ran OK`, no traceback (pkill/pgrep finding nothing is a normal, silent case here).

- [ ] **Step 3: Commit**

```bash
git add infer.py
git commit -m "Add infer.py: stale-process cleanup and world-reset helpers"
```

---

### Task 5: `_assemble_obs` — pure observation-assembly function

**Files:**
- Modify: `infer.py` (append function)

**Interfaces:**
- Consumes: `JOINT_NAMES` (from Task 2, for documentation/reference — the function itself indexes by literal joint-name keys).
- Produces: `_assemble_obs(positions: dict, velocities: dict) -> np.ndarray | None`, consumed by the `on_joint_state` callback inside `run_inference()` in Task 6.

- [ ] **Step 1: Add `_assemble_obs` to `infer.py`**

Insert after `_wait_for_obs` (before `if __name__ == "__main__":`):

```python
def _assemble_obs(positions, velocities):
    """Pure function: builds the 13-dim observation from position/velocity
    dicts keyed by joint name, in BipedScorer.on_post_update's exact order.
    Kept separate from the joint_state subscription callback so this
    ordering-critical logic is testable without a live simulation. Returns
    None if a joint is missing - a mid-reset snapshot can arrive with an
    incomplete joint set, and the caller should just skip that message
    (same as cart-pole's KeyError-skip in its own callback)."""
    try:
        return np.array([
            velocities["torso_slide_x_joint"],
            positions["torso_slide_z_joint"], velocities["torso_slide_z_joint"],
            positions["torso_pitch_joint"], velocities["torso_pitch_joint"],
            positions["hip_L_joint"], velocities["hip_L_joint"],
            positions["knee_L_joint"], velocities["knee_L_joint"],
            positions["hip_R_joint"], velocities["hip_R_joint"],
            positions["knee_R_joint"], velocities["knee_R_joint"],
        ], dtype=np.float32)
    except KeyError:
        return None
```

- [ ] **Step 2: Verify ordering and the missing-joint case**

Run: `PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
import numpy as np
from infer import _assemble_obs

positions = {
    'torso_slide_x_joint': 0.0, 'torso_slide_z_joint': 0.01, 'torso_pitch_joint': 0.02,
    'hip_L_joint': 0.1, 'knee_L_joint': 0.2, 'hip_R_joint': 0.3, 'knee_R_joint': 0.4,
}
velocities = {
    'torso_slide_x_joint': 1.0, 'torso_slide_z_joint': 1.1, 'torso_pitch_joint': 1.2,
    'hip_L_joint': 1.3, 'knee_L_joint': 1.4, 'hip_R_joint': 1.5, 'knee_R_joint': 1.6,
}
obs = _assemble_obs(positions, velocities)
expected = np.array([1.0, 0.01, 1.1, 0.02, 1.2, 0.1, 1.3, 0.2, 1.4, 0.3, 1.5, 0.4, 1.6], dtype=np.float32)
assert np.allclose(obs, expected), obs
assert obs.dtype == np.float32

incomplete = dict(positions)
del incomplete['knee_R_joint']
assert _assemble_obs(incomplete, velocities) is None

print('assemble_obs OK')
"`

Expected: `assemble_obs OK`.

- [ ] **Step 3: Commit**

```bash
git add infer.py
git commit -m "Add infer.py: pure observation-assembly function with ordering test"
```

---

### Task 6: `run_inference()` main loop, `main()`, end-to-end smoke test

**Files:**
- Modify: `infer.py` (append `run_inference` and `main`, replace the temporary `if __name__ == "__main__":` block)
- Modify: `CLAUDE.md` (file-reference list)

**Interfaces:**
- Consumes: everything from Tasks 2-5 (`_kill_stale_gz_processes`, `_reset_world`, `_wait_for_obs`, `_assemble_obs`, `_read_effort_limits`, `_load_normalizer`, `SDF_PATH`, `HEIGHT_DROP_LIMIT`, `PITCH_LIMIT`, `WORLD_NAME`, `MODEL_NAME`, `ACTUATED_JOINTS`, `STEP_PERIOD`, `MAX_ITERATIONS`, `MAX_RESET_ATTEMPTS`, `FILE_DIR`).
- Produces: `run_inference(model, normalizer, effort_limits)`, `main()` — the script's public entry point (`python infer.py [--model PATH] [--vecnorm PATH]`).

- [ ] **Step 1: Replace the temporary `if __name__ == "__main__":` block with `run_inference` and `main`**

Replace:

```python
if __name__ == "__main__":
    print("infer.py skeleton loaded OK")
```

with:

```python
def run_inference(model, normalizer, effort_limits):
    _kill_stale_gz_processes()

    gz_server = None
    gz_gui = None
    try:
        print("Launching Gazebo server...")
        gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", SDF_PATH])
        time.sleep(3)
        if gz_server.poll() is not None:
            raise RuntimeError(
                f"gz sim server exited immediately (code {gz_server.returncode}) "
                "- check for a stale process still holding the SDF/transport "
                "bus, or an SDF validation error"
            )

        print("Launching Gazebo GUI...")
        gz_gui = subprocess.Popen(["gz", "sim", "-g"])
        time.sleep(5)  # wait for GUI to connect

        node = Node()
        force_pubs = {
            name: node.advertise(f"/model/{MODEL_NAME}/joint/{name}/cmd_force", Double)
            for name in ACTUATED_JOINTS
        }

        latest = {"obs": None}

        def on_joint_state(msg):
            positions = {j.name: j.axis1.position for j in msg.joint}
            velocities = {j.name: j.axis1.velocity for j in msg.joint}
            obs = _assemble_obs(positions, velocities)
            if obs is not None:
                latest["obs"] = obs

        joint_state_topic = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
        node.subscribe(Model, joint_state_topic, on_joint_state)

        print("Waiting for first joint_state message...")
        _wait_for_obs(latest)

        def publish_zero_forces():
            # ApplyJointForce has no Reset() and simply holds the last
            # commanded force (same as cart-pole's single-joint case) - zero
            # every actuated joint before resetting so nothing is still
            # driving the robot post-reset.
            for name in ACTUATED_JOINTS:
                zero_msg = Double()
                zero_msg.data = 0.0
                force_pubs[name].publish(zero_msg)

        print("Running inference with GUI... Press Ctrl+C to stop.")
        episode_start = time.monotonic()
        for _ in range(MAX_ITERATIONS):
            loop_start = time.monotonic()

            obs = latest["obs"]
            normalized = normalizer.normalize_obs(obs.reshape(1, -1))
            action, _state = model.predict(normalized, deterministic=True)
            action = np.clip(action[0], -1.0, 1.0)

            for i, name in enumerate(ACTUATED_JOINTS):
                force_msg = Double()
                force_msg.data = float(action[i]) * effort_limits[name]
                force_pubs[name].publish(force_msg)

            torso_z_pos, torso_pitch = float(obs[1]), float(obs[3])
            if torso_z_pos < -HEIGHT_DROP_LIMIT or abs(torso_pitch) > PITCH_LIMIT:
                episode_len = time.monotonic() - episode_start
                print(f"Biped fell after {episode_len:.2f}s, resetting world...")
                publish_zero_forces()
                # Unsubscribe before requesting - cart-pole's run_inference.py
                # found node.request() reliably deadlocks gz.transport13's
                # Python binding while joint_state's own subscription
                # callback is firing at physics-step rate.
                reset_ok = None
                for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
                    node.unsubscribe(joint_state_topic)
                    reset_ok = _reset_world(node)
                    latest["obs"] = None
                    node.subscribe(Model, joint_state_topic, on_joint_state)
                    _wait_for_obs(latest)
                    torso_z_pos = float(latest["obs"][1])
                    torso_pitch = float(latest["obs"][3])
                    if torso_z_pos >= -HEIGHT_DROP_LIMIT and abs(torso_pitch) <= PITCH_LIMIT:
                        if attempt > 1:
                            print(f"  (reset took effect on attempt {attempt})")
                        break
                else:
                    raise RuntimeError(
                        f"world reset did not take effect after {MAX_RESET_ATTEMPTS} "
                        f"attempts (last request ok={reset_ok}, post-reset "
                        f"torso_z_pos={torso_z_pos:.4f} torso_pitch={torso_pitch:.4f})"
                    )
                episode_start = time.monotonic()

            elapsed = time.monotonic() - loop_start
            remaining = STEP_PERIOD - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print(f"Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without interruption, stopping.")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # A second Ctrl+C landing mid-cleanup previously escaped past a bare
        # proc.wait() in cart-pole's script, leaving the GUI process alive -
        # retry across repeated interrupts and escalate to SIGKILL if a
        # process doesn't respond to SIGTERM promptly.
        for proc in (gz_gui, gz_server):
            if proc is None:
                continue
            while True:
                try:
                    proc.terminate()
                    break
                except KeyboardInterrupt:
                    continue
        for proc in (gz_gui, gz_server):
            if proc is None:
                continue
            killed = False
            while True:
                try:
                    proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    if not killed:
                        proc.kill()
                        killed = True
                except KeyboardInterrupt:
                    continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(FILE_DIR, "biped_ppo"))
    parser.add_argument("--vecnorm", default=os.path.join(FILE_DIR, "biped_vecnormalize.pkl"))
    args = parser.parse_args()

    effort_limits = _read_effort_limits(SDF_PATH)
    print(f"Read actuated-joint effort limits from {SDF_PATH}: {effort_limits}")

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}.zip")
    normalizer = _load_normalizer(args.vecnorm)
    print(f"Loaded VecNormalize stats from {args.vecnorm}")

    run_inference(model, normalizer, effort_limits)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm a trained model exists before the smoke test**

Run: `ls biped_ppo.zip biped_vecnormalize.pkl`

Expected: both files listed. If either is missing, run `PYTHONPATH=/usr/lib/python3/dist-packages uv run python train_biped.py` first (this takes a while — it's a full 100k-timestep training run) — the smoke test in Step 3 cannot proceed without both artifacts, per this script's own `_load_normalizer` check.

- [ ] **Step 3: End-to-end smoke test**

Run (from the repo root, with a hard wall-clock cap so it can't hang the session): `PYTHONPATH=/usr/lib/python3/dist-packages timeout 40 uv run python infer.py 2>&1 | tee /tmp/infer_smoke_test.log`

Expected in `/tmp/infer_smoke_test.log`:
- `Read actuated-joint effort limits from .../biped.sdf: {...}`
- `Loaded model from .../biped_ppo.zip`
- `Loaded VecNormalize stats from .../biped_vecnormalize.pkl`
- `Launching Gazebo server...`
- `Launching Gazebo GUI...`
- `Waiting for first joint_state message...`
- `Running inference with GUI... Press Ctrl+C to stop.`
- No Python traceback anywhere in the log.

`timeout 40` will SIGTERM the whole process after 40s (the loop has no natural exit before `MAX_ITERATIONS`) — that's expected and fine for this check; the point is confirming it launches, connects, and runs without crashing, not a full episode.

- [ ] **Step 4: Confirm cleanup left no stale processes**

Run: `pgrep -f "gz sim" || echo "no gz sim processes running"`

Expected: `no gz sim processes running`. If `gz sim` processes are still listed, the `finally` block's teardown didn't run to completion — re-check Step 1's `finally` block was pasted in full, then run `PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "from infer import _kill_stale_gz_processes; _kill_stale_gz_processes()"` to clean up before retrying.

- [ ] **Step 5: Add `infer.py` to `CLAUDE.md`'s file-reference list**

In `CLAUDE.md`, find the "File reference (this repo)" section's bullet list (it currently ends with `verify_biped_scorer.py`'s entry). Add a new bullet after it:

```markdown
- **`infer.py`** — runs a trained policy (`biped_ppo.zip` +
  `biped_vecnormalize.pkl`) against `biped.sdf` with a visible Gazebo GUI,
  driving Gazebo as an external subprocess over `gz.transport13` topics
  (`JointStatePublisher`/`ApplyJointForce`, added to `biped.sdf` for this
  purpose) rather than reusing `BipedScorer`'s in-process `TestFixture`.
  Loops forever, auto-resetting the world on every fall, until Ctrl+C. Run:
  `uv run python infer.py`.
```

- [ ] **Step 6: Commit**

```bash
git add infer.py CLAUDE.md
git commit -m "$(cat <<'EOF'
Add infer.py's run_inference()/main() and document it in CLAUDE.md

Completes the cart-pole-style inference script: launches gz sim as an
external subprocess + GUI, drives it over gz.transport13 topics using
the trained biped_ppo.zip/biped_vecnormalize.pkl, and loops forever
auto-resetting the world on every fall until Ctrl+C.
EOF
)"
```

---

## Known limitations (out of scope for this plan)

- **No NaN/inf guard on the transport-read observation.** `BipedScorer.on_post_update` has one (an ill-conditioned mass matrix under repeated torque can produce non-finite state); `infer.py`'s `_assemble_obs`/`run_inference` do not replicate it. A non-finite `torso_z_pos`/`torso_pitch` would fail both comparisons in the fall check silently (Python: `NaN < x` and `abs(NaN) > x` are both `False`), so termination would never fire on a NaN state. This wasn't part of the approved design doc's scope; flagging here rather than silently adding unscoped code. If this turns out to matter in practice, it's a small, separable follow-up (port the same guard from `biped_scorer.py`).
- **Sign convention of "positive command" per joint is not asserted anywhere** — consistent with `verify_biped_scorer.py`'s own approach (empirical, not re-derived from the SDF).
