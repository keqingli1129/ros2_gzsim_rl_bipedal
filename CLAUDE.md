# RL training environment setup and the bipedal-walker gz-sim project

This repo trains a bipedal-walker RL policy against Gazebo Sim (gz-sim,
Harmonic) using Stable-Baselines3 PPO. This doc covers how the Python/gz
environment is set up, why the training utility is deliberately not a
ROS2 node, and how the bipedal environment itself is built.

An earlier cart-pole precedent (a standalone project plus a ROS2-robot
port) lived in this repo and established the architectural pattern the
bipedal work follows — in-process `gz.sim8.TestFixture`, joint-based ECM
access, `VecNormalize`-wrapped SB3 PPO — but its source files have been
removed now that the bipedal port supersedes it. The lessons that still
apply are folded into the sections below rather than kept as a separate
history.

## What this project is

A bipedal-walker reinforcement-learning pipeline built on Gazebo Sim and
Stable-Baselines3 PPO. It has two deliberately separate parts:

1. **A training/inference utility — not a ROS2 node, not a colcon
   package.** Trains a PPO policy headlessly, in-process, talking to the
   robot's physics directly through `gz.sim8` bindings (no `rclpy`, no
   ROS2 topics — this is a deliberate speed choice, not an oversight).
   Produces two artifacts that must travel together: the trained policy
   (`.zip`) and its `VecNormalize` running statistics (`.pkl`) — using
   one without the other silently collapses to random-baseline behavior,
   not a visible error.
2. **A future ROS2 control node — not yet built.** Would load those two
   artifacts and drive a real/simulated robot over actual ROS2
   topics/services, fed the artifact paths via a launch argument rather
   than baked in. This node would only need inference-time dependencies
   (`stable-baselines3`, `torch`) — not the training stack.

The dependency and process boundary between the two is deliberate, not
incidental — see "Why a standalone training utility, not a ROS2 node"
below for why.

## Environment setup

Two separate dependency worlds, kept deliberately independent (this
separation is *why* part 2 above isn't built into the training utility):

**1. RL/training Python environment** — a venv (this repo uses `uv`, any
venv manager works) holding:

- `stable-baselines3[extra]` — pulls in `gymnasium`, `torch`, and
  `opencv-python` transitively.
- An explicit override of `opencv-python` → `opencv-python-headless`.
  SB3's *base* install (not just `[extra]`) requires opencv; the
  GUI-capable build isn't needed for training/inference and cannot
  coexist in the same environment as another package that needs real
  `cv2.imshow()` windows (both provide the same `cv2` module name — one
  install silently overwrites the other). With `uv`:

  ```toml
  # pyproject.toml
  dependencies = [
      "opencv-python-headless>=5.0.0",
      "stable-baselines3[extra]>=2.9.0",
  ]
  [tool.uv]
  override-dependencies = ["opencv-python-headless>=5.0.0"]
  ```

  **This override alone is not sufficient** — verified directly on this
  machine (`uv 0.11.28`): `override-dependencies` only overrides a
  version constraint for a package name already in the dependency graph,
  it does not substitute one package name for another. With both
  `opencv-python` (pulled in by `stable-baselines3[extra]`) and
  `opencv-python-headless` (the direct override) present, `uv sync`
  installs *both* distributions, and whichever finishes writing to
  `site-packages/cv2/` last wins the import — non-deterministic. Always
  run `scripts/setup_env.sh` instead of a bare `uv sync`: it runs
  `uv sync` then force-reinstalls `opencv-python-headless` last
  (`uv pip install --reinstall-package opencv-python-headless ...`) so
  the headless build deterministically wins, and verifies this by
  confirming `cv2.imshow()` raises "not implemented" rather than
  succeeding (checking `hasattr(cv2, 'imshow')` is not a valid test —
  the headless build still exposes the attribute as a stub that raises
  when called).

**2. Gazebo Python bindings** — installed system-wide via apt (the
`gz-harmonic` package group, Ubuntu Noble / gz-sim 8), landing in
`/usr/lib/python3/dist-packages` (`gz.sim8`, `gz.common5`, `gz.math7`,
`gz.transport13`, `gz.msgs10`). **Not** installed into the venv — exposed
to it only at run time:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run <script.py>
```

Any script touching `gz.sim8` must also preload its shared library
before any `gz.*` import, or symbol resolution fails:

```python
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)
```

This is also why Pylance/VSCode may show `gz.*` imports as unresolved
even though they work at runtime: the editor's selected interpreter (the
venv) genuinely doesn't have `gz` installed in it. Fixed here via
`.vscode/settings.json`'s `python.analysis.extraPaths` pointing at
`/usr/lib/python3/dist-packages` — but that's a separate, independently
configured universe from whatever `PYTHONPATH` a given invocation
actually runs under; one resolving cleanly is not evidence the other
does.

**Do not** install `stable-baselines3`/`gymnasium`/training-time `opencv`
into whatever Python interpreter a future ROS2 control node's
`colcon build`/`ros2 run` resolves to — `colcon build` won't do this for
you anyway (see below), and it risks the `cv2` conflict above with
anything else in that environment. Only the future control node's
minimal runtime deps (`stable-baselines3` for `.load()`/`.predict()`,
`torch`) should ever need to land there, and only once that node
actually exists.

**No `ros2_ws`/xacro in this repo currently.** The bipedal robot's
description (`biped.sdf`) is static and hand-authored — there's no xacro
to convert and no colcon build needed for the RL pipeline itself. If a
future robot description needs to be shared with a real ROS2 launch
stack (making xacro the canonical source, with SDF derived from it),
that would need a `ros2_ws` built with the venv's `python3` *not*
shadowing the system one, since `ament`/`catkin_pkg` need the system
Python:

```bash
cd ros2_ws
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:) VIRTUAL_ENV= colcon build --symlink-install
source install/setup.bash
```

## Why a standalone training utility, not a ROS2 node

- Training is a one-shot, offline, batch process (run it, get a
  `.zip`/`.pkl`, done). It never needs ROS2-graph presence — no topics,
  no services, nothing that benefits from `ros2 node list` or a launch
  file — unlike a live robot-control node, which genuinely needs
  real-time pub/sub.
- `colcon build` does not install Python dependencies. Verified directly
  against this machine's `colcon_core`/`colcon_ros` source
  (`/usr/lib/python3/dist-packages/colcon_core/task/python/build.py`,
  `colcon_ros/task/ament_python/build.py`): the `ament_python` build task
  invokes `setup.py develop --no-deps` (symlink-install) or
  `setup.py install --single-version-externally-managed` (regular
  install) — both deliberately suppress setuptools' normal
  `install_requires` auto-install behavior. `rosdep` doesn't cover this
  gap either — `gymnasium`/`stable-baselines3` have no official rosdep
  keys. So making training "a ROS2 package" buys `ros2 run`
  discoverability, but every dependency (`gymnasium`, `stable-baselines3`,
  `torch`, `opencv-python-headless`, ...) still has to be manually
  installed onto whatever Python interpreter `colcon build` happens to
  bake into the generated entry-point script's shebang — no different
  from installing them for a plain script.
- Real conflict risk: `stable-baselines3`'s base install requires
  `opencv-python` (confirmed above). `opencv-python` and
  `opencv-python-headless` both install the same `cv2` module name, so
  they cannot coexist in one Python environment — installing one removes
  the other for every package sharing that interpreter, not just the RL
  node. If some other node in the same ROS2 environment needs
  `cv2.imshow()`-style GUI windows, you have a genuine, unresolvable
  conflict unless the two nodes run under separate Python environments.

Conclusion: keep training fully independent of the ROS2/colcon
dependency-resolution story. Only a trained artifact (a `.zip` +
`vecnormalize.pkl`) would need to cross into ROS2 territory — not the
full `gymnasium`/`stable-baselines3`/training-time-opencv stack.

## The bipedal walker environment

Design spec: `docs/superpowers/specs/2026-07-31-bipedal-walker-design.md`
Implementation plan: `docs/superpowers/plans/2026-07-31-bipedal-walker.md`

A minimal planar (sagittal-plane) biped: torso on a passive 3-joint
planar mount (prismatic-x, prismatic-z, revolute-pitch — free, not
actuated, the same role a pole plays in a cart-pole balance task), two
legs each with an actuated hip and knee joint plus a fixed foot.

- **Joint-based ECM access.** `BipedScorer` reads all 7 non-fixed
  joints' position/velocity directly via their `Joint` components — real
  physics-engine velocity, no finite-difference estimate. Actuation is
  continuous, normalized: each of the 4 actuated joints takes a torque
  command in `[-1, 1]`, clamped and scaled by that joint's own live
  `effort_limits(ecm)` value inside `on_pre_update` — never a hardcoded
  torque cap that could silently desync from `biped.sdf`.
- **`self.command` is clamped once, in `step()`**, so both the applied
  torque and the reward's control-cost term always agree — this matters
  because reward terms sized for normalized `[-1,1]` actions (e.g. a
  BipedalWalker-v3/Walker2d-style control-cost weight) silently blow up
  by orders of magnitude if computed against raw, unnormalized torques.
- **A NaN/inf guard on the observation** latches termination and
  sanitizes the state if the physics ever produces a non-finite value —
  without it, `NaN < threshold` and `abs(NaN) > threshold` are both
  `False` in Python, so termination would never fire and the NaN would
  permanently poison `VecNormalize`'s running statistics.
- **`VecNormalize` is a hard requirement, not an optimization** (same
  lesson as the earlier cart-pole precedent): this env's velocity
  dimensions are on a different scale than its position dimensions, and
  PPO doesn't normalize observations by default.
- **Episodes are time-limited** (`gymnasium.wrappers.TimeLimit`) — a
  policy that merely balances in place with no forward-progress
  incentive would otherwise never terminate, meaning `Monitor` never
  logs an episode and a training run would have no visible learning
  curve at all.
- **Reset applies a small random torque impulse** to break left/right
  symmetry — every episode would otherwise start from a bit-identical,
  perfectly symmetric state, removing state-distribution coverage.

### File reference (this repo)

All commands need `PYTHONPATH=/usr/lib/python3/dist-packages` and
`uv run` per Environment setup above.

- **`biped.sdf`** — static, hand-authored world + robot model (9 links,
  9 joints: 3 passive planar-mount + 4 actuated leg joints + 2 fixed
  foot joints). Also declares a `JointStatePublisher` plugin and 4x
  `ApplyJointForce` plugins (added for `infer.py`, below) that let an
  external process read/write joint state over `gz.transport13` topics.
  Those `ApplyJointForce` plugins write to the same per-joint
  `JointForceCmd` component `BipedScorer.on_pre_update` writes to during
  training — verified harmless (verified via `verify_biped_scorer.py`'s
  actuated-movement assertion), but re-run that check after any future
  edit to this file's plugin section.
- **`biped_scorer.py`** — `BipedScorer`: the in-process Gazebo System
  that applies torques and reads joint state each step. Exports
  `HEIGHT_DROP_LIMIT`/`PITCH_LIMIT` (termination thresholds).
- **`train_biped.py`** — `CustomBipedGzTrain` (Gymnasium wrapper) plus
  PPO + `VecNormalize` training. Saves `biped_ppo.zip` +
  `biped_vecnormalize.pkl` (both gitignored — build outputs, not source).
  Run: `uv run python train_biped.py`.
- **`verify_biped_dynamics.py`** — scratch check (not pytest): the robot
  is grounded and at rest at spawn, and max-effort torque produces the
  expected motion. Distinguishes "broken spawn" from "this is an
  inherently unstable standing pose without active balance control" —
  it does not assert indefinite passive stability, since the design
  never has that (measured: stable for ~1s unactuated, then falls over
  within 2-3s, which is expected).
- **`verify_biped_scorer.py`** — scratch check: exercises `BipedScorer`
  directly — idle stability, actuated movement, and an explicit
  termination/fall-penalty/latch scenario.
- **`infer.py`** — runs a trained policy (`biped_ppo.zip` +
  `biped_vecnormalize.pkl`) against `biped.sdf` with a visible Gazebo GUI,
  driving Gazebo as an external subprocess over `gz.transport13` topics
  (`JointStatePublisher`/`ApplyJointForce`, added to `biped.sdf` for this
  purpose) rather than reusing `BipedScorer`'s in-process `TestFixture`.
  Auto-resets the world on every fall; runs until Ctrl+C or a
  `MAX_ITERATIONS` safety cap (~250s). Run:
  `uv run python infer.py`.

## Open: how a trained model gets used in a real ROS2 node (future work)

Not yet implemented or designed in detail:

- **No separate "inference node."** A trained model should be plugged
  into an existing launch file (e.g. a launch argument pointing at the
  model/`vecnormalize.pkl` path, fed to whatever node actually runs
  inference) rather than a standalone script/node with its own CLI.
- **Only the control node itself needs ROS2-side Python deps**
  (`stable-baselines3`, `torch`, and whatever `vecnormalize.pkl`
  unpickling needs) — not the full training stack. Confirm at
  implementation time whether a bare `stable-baselines3` install (no
  `[extra]`) is sufficient for load+predict, to minimize the new
  dependency footprint on whatever Python environment the control node
  runs under.
- **Still open:** where the trained artifacts (`.zip`,
  `vecnormalize.pkl`) physically live so a colcon-built control node can
  find them (a path argument, package share-data, or something else),
  and which node actually does this — none of this is decided yet.
- The general pattern a control node would need: build a
  `DummyVecEnv`-wrapped stub matching the trained observation/action
  space, `VecNormalize.load(path, venv)` with `venv.training = False`,
  normalize each observation before calling `model.predict(...,
  deterministic=True)`.
