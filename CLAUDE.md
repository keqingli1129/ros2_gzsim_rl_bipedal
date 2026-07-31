# RL training/inference, xacro→SDF conversion, and future ROS2 model integration

Context: we considered converting `ros2_ws/src/cart_pole_gz_train/` (a
non-colcon SB3 PPO training utility) into a real colcon package with
`rclpy.Node`-wrapped entry points (`robot_rl_node`), so it would be
launchable via `ros2 run`/show up in the ROS2 graph. That idea was
scrapped. This doc captures why, how the existing training/inference
pipelines actually work (both the root `cart_pole/` project and its
`ros2_ws` port), how the xacro→SDF conversion works, and what's left
open for when a trained model needs to actually drive a robot through a
real ROS2 node.

## What this project is

A cart-pole reinforcement-learning pipeline built on Gazebo Sim (gz-sim,
Harmonic) and Stable-Baselines3 PPO, where the robot's canonical
description lives in a ROS2 package as xacro. It has two deliberately
separate parts:

1. **A training/inference utility — not a ROS2 node, not a colcon
   package.** Trains a PPO policy headlessly, in-process, talking to the
   robot's physics directly through `gz.sim8` bindings (no `rclpy`, no
   ROS2 topics — this is a deliberate speed choice, not an oversight).
   Can also run inference with a live GUI by spawning `gz sim` as
   subprocesses and driving the robot over `gz.transport13`. Produces two
   artifacts that must travel together: the trained policy (`.zip`) and
   its `VecNormalize` running statistics (`.pkl`) — using one without the
   other silently collapses to random-baseline behavior, not a visible
   error.
2. **A future ROS2 control node — not yet built.** Loads those two
   artifacts and drives the real/simulated robot over actual ROS2
   topics/services (`ros_gz_bridge`, `/joint_states`, a force/effort
   command topic, a reset service), fed the artifact paths via a launch
   argument rather than baked in. This node only needs inference-time
   dependencies (`stable-baselines3`, `torch`) — not the training stack.

The dependency and process boundary between the two is deliberate, not
incidental — see the next section for why.

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

  Install with `uv sync`.

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

**3. The ROS2 workspace** (only needed for the xacro→SDF conversion in
part 2 below, or eventually building the control node) — build with the
venv's `python3` *not* shadowing the system one, since `ament`/
`catkin_pkg` need the system Python, not the isolated venv:

```bash
cd ros2_ws
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:) VIRTUAL_ENV= colcon build --symlink-install
source install/setup.bash
```

The xacro→SDF conversion below shells out to `xacro`/`gz sdf -p`, which
needs this build to already exist and must run with the venv stripped
from `PATH` for the same reason.

**Do not** install `stable-baselines3`/`gymnasium`/training-time `opencv`
into whatever Python interpreter `colcon build`/`ros2 run` resolves to —
`colcon build` won't do this for you anyway (see below), and it risks
the `cv2` conflict above with anything else in that environment. Only
the future control node's minimal runtime deps (`stable-baselines3` for
`.load()`/`.predict()`, `torch`) should ever need to land there, and only
once that node actually exists.

## Why the ROS2-node conversion was scrapped

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
  `opencv-python` (confirmed indirectly — this repo's `pyproject.toml`
  needs a project-wide `[tool.uv] override-dependencies` to force
  `opencv-python-headless` instead, which wouldn't be necessary if the
  headless variant only mattered for `[extra]`). `opencv-python` and
  `opencv-python-headless` both install the same `cv2` module name, so
  they cannot coexist in one Python environment — installing one removes
  the other for every package sharing that interpreter, not just the RL
  node. If some other node in the same ROS2 environment needs
  `cv2.imshow()`-style GUI windows, you have a genuine, unresolvable
  conflict unless the two nodes run under separate Python environments.
- VSCode/Pylance resolving an import cleanly is not evidence it will
  resolve at `ros2 run` time — Pylance checks the selected interpreter
  (usually a venv) plus `python.analysis.extraPaths`
  (`.vscode/settings.json`), which is a different, independently
  configured universe from whatever interpreter `ros2 run` actually
  executes the node under.

Conclusion: keep training fully independent of the ROS2/colcon
dependency-resolution story. Only a trained artifact (a `.zip` +
`vecnormalize.pkl`) needs to cross into ROS2 territory — not the full
`gymnasium`/`stable-baselines3`/training-time-opencv stack.

## 1. How training and inference work today (root `cart_pole/` project)

`cart_pole/cart_pole_env.py` has two independent sim-interaction paths;
see the root `CLAUDE.md`'s Architecture section for the full detail.
Summary:

**Training (headless, in-process).** `GzRewardScorer` loads
`cart_pole.sdf` through `gz.sim8.TestFixture` and hooks
`on_pre_update`/`on_post_update`:
- `on_pre_update` applies ±2000N world force to the chassis based on the
  pending discrete action.
- `on_post_update` reads pole pitch / cart position from the ECM,
  computes reward and termination (`|pitch| > 0.48 rad` or
  `|cart x| > 4.8 m`), and estimates velocity by **finite difference**
  across the 5ms step (`(pose - prev_pose) / 0.005`) — deliberately
  matching the estimator inference actually sees, rather than reading
  the physics engine's true instantaneous velocity, to avoid a
  train/inference distribution mismatch (measured ~76% relative error
  between the two estimators).
- Each Gym `step()` runs the server for 5 blocking sim iterations
  (1ms physics step × 5 = 5ms sim time per env step).
- `reset()` rebuilds the `TestFixture`/server from scratch rather than
  calling `server.reset_all()` — the latter desyncs the physics engine
  from the ECM (force application/velocity reads silently stop working)
  while leaving entity IDs unchanged, so it doesn't even look like a
  stale-handle bug.
- `CustomCartPole` wraps this in the Gymnasium API: `Discrete(2)`
  actions, 4-dim `Box` observation (cart x, cart velocity, pole pitch,
  pole angular velocity).

**Inference (out-of-process, with GUI).** After training, the script
spawns `gz sim -s -r` and `gz sim -g` as **subprocesses** and talks over
Gazebo transport instead of the in-process ECM:
- Forces are published as `EntityWrench` messages to
  `/world/cart_pole/wrench/persistent` — not the plain `/wrench` topic,
  which only holds a force for one 1ms physics tick before dropping it
  (a fifth of training's per-action authority).
- State is read via synchronous request/response to the
  `/world/cart_pole/state` service, decoded by replicating gz-sim's
  internal FNV-1a component-type hash to identify `Name`/`Pose`
  components, composing model+link local poses into world frame.
  Velocities are finite-differenced against the response's own
  `sim_time`.
- `/wrench/persistent` entries can't be cleared in this gz-sim build
  (a known bug — `OnWrenchClear`'s entity match never succeeds) and
  survive a world reset untouched, so inference tracks the net force
  already applied and publishes only the delta needed to reach a new
  target (including zeroing before a reset).
- Requires `ctypes.CDLL(".../libgz-sim8.so", RTLD_GLOBAL)` to run
  **before** any `gz.*` import, or symbol resolution fails. Depends on
  the `ApplyLinkWrench` and `SceneBroadcaster` plugins declared in
  `cart_pole.sdf`.

`cart_pole.sdf` here is a **static, hand-authored** file — there is no
xacro anywhere in the root `cart_pole/` project, so no conversion step
is needed for this path at all.

## The `ros2_ws` port (`cart_pole_gz_train/`) — same idea, real robot

`ros2_ws/src/cart_pole_gz_train/` re-implements the same training/
inference split against the actual ROS2 workspace's robot
(`robot_description`'s `cart_pole` model, joints `cart_joint`/
`pole_joint`) instead of the root project's standalone `vehicle_green`
model. Key differences from the root project, since this is the more
relevant precedent for "a trained model driving a real robot":

- **Joint-based ECM access, not wrench-based.** `gz_scorer.py`'s
  `GzCartPoleScorer` reads `cart_joint`/`pole_joint` position/velocity
  directly via their `Joint` components — real physics-engine velocity,
  no finite-difference estimate needed (unlike the root project, which
  has no joint to read from, only a free-floating wrench-driven body).
  Actuation is `cart_joint.set_force(ecm, [±max_force])`, where
  `max_force` is read live from the joint's actual effort limit
  (verified: 1,000,000N produces the same realized acceleration as 30N —
  it's a hard actuator clamp), not hardcoded.
- **Inference drives the same joint over ROS2-adjacent gz-transport
  topics**: `/model/cart_pole/joint/cart_joint/cmd_force` (a plain
  `Double`, consumed by the `ApplyJointForce` gz-sim plugin declared in
  the xacro), reading state from
  `/world/cart_pole_train/model/cart_pole/joint_state` (`JointStatePublisher`).
- **Reset semantics were measured, not assumed**, and differ from the
  root project and from `commander`'s DQN:
  - This world's `reset.model_only` is a **complete no-op** (measured:
    position/velocity keep evolving along their pre-reset trajectory,
    zero discontinuity).
  - `reset.all` **does** work here and does *not* hit the
    `JointStatePublisher`-killing teardown bug `commander` hit on its
    own, differently-spawned world (`robomaster_rale`) — that bug is
    real but specific to how a model enters the world (runtime-spawned
    vs. declared whole in the SDF `gz sim -s -r` loads), so reset-type
    safety does not generalize between worlds without re-measuring.
  - `reset.all`'s own `ok` RPC flag is not trustworthy in either
    direction (~1-in-9 false `ok=False` on a reset that actually
    succeeded) — the caller checks the real postcondition (position/
    velocity against the termination thresholds) instead of trusting the
    flag, retrying up to `MAX_RESET_ATTEMPTS` times.
- **`VecNormalize` is a hard requirement, not an optimization.** This
  env's velocity dimensions run 10-30x larger than its position
  dimensions, which stalled learning outright until `VecNormalize`
  (running mean/std observation normalization) was added. Any consumer
  of the saved policy — training, evaluation, or a future control node —
  **must** load `vecnormalize.pkl` alongside the `.zip` and feed it
  normalized observations; feeding raw observations doesn't degrade
  gracefully, it silently collapses to random-baseline performance. See
  `evaluate_policy.py`/`run_inference.py`'s `_load_normalizer`, which
  hard-errors if the stats file is missing rather than proceeding.

This is the pattern a future control node would need to replicate:
build a `DummyVecEnv`-wrapped stub matching the trained observation/
action space, `VecNormalize.load(path, venv)` with `venv.training = False`,
normalize each observation before calling `model.predict(...,
deterministic=True)`.

## 2. Xacro → SDF conversion (`ros2_ws` side only)

The root `cart_pole/` project never needs this — its SDF is static and
hand-authored. `ros2_ws`'s robot is different: its canonical description
is `robot_description/robot/cart_pole.urdf.xacro`, shared with the real
`robot_launch` stack, so the training world has to be *derived* from it
rather than hand-authored, to avoid a second, silently-drifting source
of truth for the robot's physical parameters (mass, inertia, joint
limits). `ros2_ws/src/cart_pole_gz_train/world_builder.py` does this in
four steps, run fresh every process (not cached from a previous run's
leftover file — a stale leftover could silently reflect an older xacro
checkout):

1. **`run_xacro()`** — shells out to `xacro cart_pole.urdf.xacro`,
   producing URDF text. Requires `ros2_ws` to already be
   `colcon build`-ed, since `xacro` (via `ament_index_python`) needs to
   resolve the `robot_description` package — this is why
   `cart_pole_gz_train` must be run from the repo root with
   `ros2_ws/install/setup.bash` sourced (venv stripped from `PATH`; see
   `_run_in_ros_env`).
2. **`convert_urdf_to_sdf()`** — shells out to `gz sdf -p <urdf>`,
   converting URDF to SDF text.
3. **`postprocess_model_sdf()`** — strips mesh visuals/collisions down to
   primitive shapes (headless training never renders, so simplified
   collision is cheaper and mesh-based collision isn't needed), replacing
   each with a matching primitive `<visual>` too (`run_inference.py`'s
   GUI loads this same generated world, and a model with collision but
   zero visuals renders as an empty viewport even though it exists in the
   ECM). Also drops `tip_link`/`tip_joint` (mass 0.0001, physically
   negligible).
4. **`wrap_in_world()`** — injects a spawn pose (`z=0.3`, half of
   `base_footprint`'s 0.6m collision box height — measured to be the
   smallest lift that spawns already at rest; `z=0` leaves the base
   half-buried and jams the cart at ~1% authority, `z=2` costs a
   ~590ms free-fall every episode), then wraps the processed `<model>`
   in a full `<world>` with physics/plugin declarations and a `<gui>`
   block (camera pose, per-link visual materials — for
   `run_inference.py`'s benefit only; headless training never renders).
   A `<gui>` element in SDF *replaces* gz-sim's entire default GUI config
   rather than extending it — every plugin the GUI should show
   (`WorldControl`, `WorldStats`, `EntityTree`, not just the 3D view) has
   to be declared explicitly or it silently disappears.

Output is `cart_pole_train.sdf` — gitignored, an output not a source;
don't hand-edit it, edit the xacro or `world_builder.py`'s
postprocessing instead.

**Dead end to avoid**: `robot_description/robot/cart_pole.sdf` and
`cart_pole.urdf` are static, checked-in files from the very first commit
to this repo (June 2022), predating the switch to gz-sim/Harmonic.
Verified directly (diffed a live `xacro` run against the checked-in
file): the checked-in URDF declares
`<plugin filename="libgazebo_ros_control.so" name="gazebo_ros_control"/>`
— a **Gazebo Classic** (ROS1-era) plugin — instead of the
`ApplyJointForce`/`JointStatePublisher` gz-sim plugins this entire
pipeline actually depends on. Nothing in `ros2_ws` reads these two static
files; they're dead weight, not a valid shortcut around the
xacro→SDF regeneration.

## 3. Open: how a trained model gets used in a real ROS2 node (future work)

Not yet implemented or designed in detail. What's already settled from
prior discussion, to pick up from rather than re-derive:

- **No separate "inference node."** The trained model should be plugged
  into the existing launch file (`robot_launch`'s
  `launch_simulation.launch.py`) — e.g. a launch argument pointing at
  the model/`vecnormalize.pkl` path, fed to whatever node actually runs
  inference — rather than a standalone script/node with its own CLI.
- **Only the control node itself needs ROS2-side Python deps**
  (`stable-baselines3`, `torch`, and whatever `vecnormalize.pkl`
  unpickling needs) — not the full training stack (no `gymnasium` env
  authoring, no training-time `opencv` requirement beyond whatever
  `stable-baselines3`'s base install pulls in for `PPO.load()`/
  `.predict()` to work at all). Confirm at implementation time whether
  a bare `stable-baselines3` install (no `[extra]`) is sufficient for
  load+predict, to minimize the new dependency footprint on whatever
  Python environment the control node runs under.
- **Still open / not yet decided:**
  - Where the trained artifacts (`.zip`, `vecnormalize.pkl`) physically
    live so a colcon-built control node can find them — a path argument,
    package share-data, or something else.
  - Which node actually does this — extend `commander`, or add
    something new — and how it reconciles with `commander`'s existing
    `reset.model_only` convention for the `robomaster_rale` world (the
    `cart_pole_gz_train` port above uses `reset.all` instead, but that
    was measured against a *different*, SDF-declared world — do not
    assume it generalizes to `robomaster_rale` without re-measuring, per
    the reset-semantics note above).
  - Whether to read pole angle from `JointState.position` directly
    (available on the real `/joint_states` message) instead of
    `commander`'s current manual yaw-angle integration workaround, which
    was written for the DQN's own observation quirk rather than being a
    hard constraint on what `JointState` carries.
