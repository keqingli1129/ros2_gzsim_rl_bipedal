# Biped inference script (`infer.py`) — design

Date: 2026-08-02
Status: approved (pending spec review)

## Purpose

Run a trained biped policy (`biped_ppo.zip` + `biped_vecnormalize.pkl`) against
`biped.sdf` with a visible Gazebo GUI, so the walking behavior can actually be
watched rather than only measured through training logs. This mirrors the
role the now-removed cart-pole `run_inference.py` played for that project.

## Why not reuse `BipedScorer`/`TestFixture`

`BipedScorer` drives physics in-process via `gz.sim8.TestFixture` — headless,
no ROS2, no transport topics, by deliberate design (see this repo's
`CLAUDE.md`, "Why a standalone training utility, not a ROS2 node"). That
design is right for training (fast, no subprocess/IPC overhead) but doesn't
lend itself to attaching a live GUI for a human to watch.

The cart-pole precedent solved this by running Gazebo as an **external
subprocess** (`gz sim -s -r <sdf>` + `gz sim -g`) and controlling it entirely
over `gz.transport13` topics — decoupling "watch it run" from "how training
drives physics." `infer.py` follows that same architecture. This was a
deliberate choice (confirmed with the user) over the simpler alternative of
reusing `BipedScorer` in-process and merely attaching a GUI to it, because it
matches the existing cart-pole precedent exactly rather than introducing a
new pattern.

## SDF changes

`biped.sdf`'s `<model name="biped">` currently declares no topic interface at
all — `BipedScorer` reads/writes joints directly via ECM `Joint` components.
For `infer.py` to control the same robot over transport, `biped.sdf` gains:

- One `JointStatePublisher` plugin (no `<joint_name>` filter — confirmed from
  `/usr/share/gz/gz-sim8/worlds/lift_drag.sdf` that a bare tag publishes all
  joints by default), publishing `Model` messages on
  `/world/biped/model/biped/joint_state` with each joint's `axis1.position`/
  `axis1.velocity`.
- Four `ApplyJointForce` plugin instances, one per actuated joint
  (`hip_L_joint`, `knee_L_joint`, `hip_R_joint`, `knee_R_joint`), each
  configured with its own `<joint_name>` and subscribing to
  `/model/biped/joint/<name>/cmd_force` (confirmed against
  `/usr/share/gz/gz-sim8/worlds/apply_joint_force.sdf`).

**Risk, and how it's handled:** `biped_scorer.py` loads this exact same
`biped.sdf` via `TestFixture` during training. Both the new `ApplyJointForce`
systems and `BipedScorer.on_pre_update` would write to the same joint's
`JointForceCmd` component every `PreUpdate` — `ApplyJointForce` writing 0 (no
message is ever published to its topic during training) and `on_pre_update`
writing the real torque. Whichever system's callback runs later in that
step's `PreUpdate` order wins. This is expected to be harmless (the
`TestFixture` Python callback is expected to run after the SDF's own declared
plugins), but this is **not** independently verified from source/docs.

Mitigation: after adding the plugins, run `verify_biped_scorer.py` unchanged
against the modified `biped.sdf` and confirm it still passes — in particular
the actuated-movement check (`obs[5] > 0.3` after sustained `hip_L` torque),
which would fail if `ApplyJointForce`'s zero-default were winning the
component-write race. If it fails, fall back to a separate
`biped_infer.sdf` (a copy with the extra plugins) so the training path never
touches the new plugins at all, and update `infer.py` to point at that file
instead.

## Does the existing trained model need retraining?

No — `infer.py` doesn't change the observation space, action space, reward,
or termination logic. Those are unchanged in `biped_scorer.py`/
`train_biped.py`. The SDF changes above are additive (new plugins/topics)
and, per the mitigation above, are verified not to alter the physics the
existing `biped_ppo.zip`/`biped_vecnormalize.pkl` were trained against.

## Architecture

`infer.py` is a standalone script (not reusing `BipedScorer`):

1. Load `PPO.load(model_path)` and `VecNormalize.load(vecnorm_path, ...)`
   (via a `DummyVecEnv`-wrapped observation/action-space stub — no gz
   involved, so constructing it can't collide with the transport bus the
   real inference server will use).
2. Kill any stale `gz sim` processes scoped to `biped.sdf`'s path (same
   escalating-SIGKILL approach as cart-pole's `_kill_stale_gz_processes`).
3. Launch `gz sim -s -r biped.sdf` (server) and `gz sim -g` (GUI) as
   subprocesses.
4. Subscribe to the `joint_state` topic; on each message, assemble the
   13-dim observation and hold it in a shared `latest["obs"]` slot.
5. Loop, paced at 5ms/step (matches both `biped.sdf`'s
   `max_step_size = 0.001` and `BipedScorer.step`'s `server.run(True, 5,
   False)` cadence):
   - Normalize the latest observation via the loaded `VecNormalize` stats.
   - `model.predict(normalized, deterministic=True)` → 4-vector in
     `[-1, 1]`, one per actuated joint.
   - Clip defensively to `[-1, 1]` (SB3 already does this, but
     `BipedScorer.step` treats this as its own single source of truth — same
     principle applied here).
   - Scale each of the 4 values by that joint's live effort limit (read from
     `biped.sdf`, never hardcoded — same convention as
     `_ensure_initialized`'s `effort_limits(ecm)` read) and publish as a
     `Double` message to that joint's `cmd_force` topic.
   - Check termination using `HEIGHT_DROP_LIMIT`/`PITCH_LIMIT` imported
     directly from `biped_scorer.py` (never re-hardcoded), against the
     latest `torso_z_pos`/`torso_pitch` — the identical condition
     `BipedScorer.on_post_update` evaluates.
   - On termination: zero **all 4** `cmd_force` topics (each
     `ApplyJointForce` instance independently latches its last commanded
     force — same "has no `Reset()`" behavior documented for cart-pole's
     single joint), then run the reset sequence below.
   - Runs until a `MAX_ITERATIONS` safety cap (50000 steps at 5ms/step =
     ~250s, same value cart-pole used) or Ctrl+C.

## Data flow / ordering invariant

The 13-dim observation **must** be assembled in exactly
`BipedScorer.on_post_update`'s order:

```text
torso_x_vel, torso_z_pos, torso_z_vel, torso_pitch, torso_pitch_vel,
hip_L_pos, hip_L_vel, knee_L_pos, knee_L_vel,
hip_R_pos, hip_R_vel, knee_R_pos, knee_R_vel
```

`VecNormalize`'s saved running statistics were fit against this exact
ordering during training — if `infer.py` assembled the observation in a
different order, normalization would silently apply the wrong dimension's
mean/std to each value, and inference would perform far worse than training
suggested, without any error.

## Reset behavior

Loops forever, auto-resetting on every fall, until Ctrl+C — matching the
cart-pole precedent (confirmed with the user). Reset reuses cart-pole's
empirically-derived `reset.all` sequence unchanged in structure:

- Issue `WorldControl` `reset.all` (not `reset.model_only` — confirmed
  against the cart-pole world to be the one that actually resets joint
  state, not just keep publishing).
- Don't trust the RPC's own `ok` flag — it was observed to read `False`
  roughly 1-in-9 times even when the reset physically took effect. Check
  the real postcondition instead: whether the first fresh post-reset
  observation is back within `HEIGHT_DROP_LIMIT`/`PITCH_LIMIT`.
- Retry up to `MAX_RESET_ATTEMPTS` (5, same margin as cart-pole, which
  measured recovery within 2 attempts across repeated stress runs) if the
  first attempt doesn't visibly take effect.
- Unsubscribe from `joint_state` before issuing the reset request and
  resubscribe after — the cart-pole precedent found that issuing a
  `node.request()` while `joint_state`'s own subscription callback is
  firing at physics-step rate reliably deadlocks `gz.transport13`'s Python
  binding.

## Error handling / cleanup

- Stale-process kill on startup, scoped to `biped.sdf`'s path (not a bare
  `"gz sim"` substring, to avoid killing unrelated `gz sim` sessions on the
  same machine) — ported from cart-pole's `_kill_stale_gz_processes`.
- SIGINT-safe subprocess teardown: `terminate()` then escalate to `kill()`
  if a process doesn't exit within a timeout, tolerating a second Ctrl+C
  arriving mid-cleanup without abandoning the teardown — ported from
  cart-pole's `finally` block.

## CLI

```bash
uv run python infer.py [--model biped_ppo.zip] [--vecnorm biped_vecnormalize.pkl]
```

Defaults point at `train_biped.py`'s actual output filenames, resolved
relative to `infer.py`'s own directory. No `world_builder`-style generation
step is needed first — `biped.sdf` is already static.

## Testing

No automated test. Like the cart-pole precedent, this is a manually-watched
GUI script, not something a `verify_*.py` script asserts against —
`verify_biped_scorer.py` and `verify_biped_dynamics.py` remain the automated
checks, both exercising `BipedScorer` directly. The one required verification
step is the `biped.sdf`-plugin regression check described above (running
`verify_biped_scorer.py` after the SDF change, before considering that change
complete).

## Out of scope

- Anything ROS2-related — this script is still part of the standalone
  training/inference utility, not the future ROS2 control node described in
  this repo's `CLAUDE.md`.
- Changing `biped_scorer.py`, `train_biped.py`, or the reward/observation
  design in any way.
