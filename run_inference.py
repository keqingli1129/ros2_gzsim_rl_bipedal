import argparse
import os
import sys
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import subprocess
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, FILE_DIR)

from world_builder import generate_training_world
from gz_scorer import SDF_PATH, CART_POSITION_LIMIT, POLE_PITCH_LIMIT

from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

WORLD_NAME = "cart_pole_train"
MODEL_NAME = "cart_pole"
STEP_PERIOD = 0.005  # matches training's 5 x 1ms action cadence
MAX_ITERATIONS = 50000  # ~250s, mirrors cart_pole_env.py's run_inference bound
# reset.all was measured (via a dedicated scratch repro, not just here) to
# occasionally not take effect on the very first request - the post-reset
# observation reads back nearly identical to the pre-reset one, independent
# of the RPC's own ok flag (seen with both ok=True and ok=False). Across two
# separate 10-consecutive-reset stress runs, retrying always recovered
# within 2 attempts (6/10 and non-trivial 2nd-attempt successes, 0 needing a
# 3rd) - this cap gives a large margin over that.
MAX_RESET_ATTEMPTS = 5


class _ObsSpaceStub(gym.Env):
    """Carries only the observation/action space VecNormalize.load needs to
    shape-check against - never calls into gz, so constructing it can't
    double-register on the training world's transport name alongside the
    live inference server (unlike instantiating CustomCartPoleGzTrain,
    whose __init__ builds a real GzCartPoleScorer)."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            np.array([-CART_POSITION_LIMIT, -np.inf, -POLE_PITCH_LIMIT, -np.inf], dtype=np.float32),
            np.array([CART_POSITION_LIMIT, np.inf, POLE_PITCH_LIMIT, np.inf], dtype=np.float32),
            (4,), np.float32,
        )

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
            "Re-run train_cart_pole.py (which writes vecnormalize.pkl next "
            "to the model) or pass --vecnorm explicitly."
        )
    venv = DummyVecEnv([lambda: _ObsSpaceStub()])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def _read_effort_limit(sdf_path, joint_name):
    root = ET.parse(sdf_path).getroot()
    effort_el = root.find(f".//joint[@name='{joint_name}']/axis/limit/effort")
    if effort_el is None:
        raise RuntimeError(
            f"could not find an effort limit for joint {joint_name!r} in "
            f"{sdf_path} - did the xacro or gz sdf conversion change its "
            f"structure?"
        )
    return float(effort_el.text)


def _kill_stale_gz_processes():
    """Terminate any gz sim server/GUI left over from a prior run - a
    leftover server registers on the same transport bus under the same
    world name as the one about to be launched, and the new GUI can attach
    to that stale instance instead of ours.

    SIGTERM alone (plain `pkill -f`) is not reliable here: measured
    directly, `gz sim -g` can survive SIGTERM indefinitely in this
    environment (its libEGL/dri2 warnings suggest a stuck GL context
    teardown) while still showing up as "killed" in pkill's own output -
    it looks terminated but isn't. Escalate to SIGKILL for anything still
    alive after the graceful attempt, so a stale instance can never be
    left registered on the transport bus for the new launch to collide
    with.

    Matches are scoped as narrowly as each process type allows rather than
    a bare "gz sim" substring - a plain "gz sim" pkill has a wide blast
    radius and could kill an unrelated gz sim session on the same machine
    (e.g. ros2_ws's own robot_launch, which runs a completely different
    world). The server is matched on this exact SDF path, since it's the
    one process here that's uniquely identifiable. The GUI (`gz sim -g`)
    carries no world-identifying argument at all - it just attaches to
    whatever's on the transport bus - so it can only be scoped down to
    "launched in GUI mode", not to this specific world; that residual
    ambiguity is inherent to how `gz sim -g` works, not something this
    function can narrow further."""
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
    """Reset via reset.all, not reset.model_only.

    Task 6's verify_reset_preserves_joint_state.py validated
    reset.model_only against this same world, but only checked that
    joint_state kept publishing - it never checked that the reset
    actually changed cart_joint/pole_joint's position or velocity.
    Direct measurement while implementing this script showed
    reset.model_only is a complete no-op here: after a disturbed episode
    (cart drifting, pole falling) issuing reset.model_only and polling
    position/velocity immediately afterward shows both continuing to
    evolve exactly along their pre-reset trajectory, with no
    discontinuity at all - so a real out-of-bounds episode could never
    recover, and the main loop's out-of-bounds branch would spin forever
    re-triggering on the same still-diverging state.

    reset.all does not have this problem for this project's generated
    world: measured directly (polling position/velocity every 20-50ms
    after the reset request returns, the resolution actually used - not
    validated at physics-step granularity), position and velocity read at
    or near 0 on the first post-reset poll, and joint_state keeps
    publishing afterward at full rate across repeated resets (unlike the
    unrelated commander/robomaster_rale world documented in
    ros2_ws/src/CLAUDE.md, where reset.all's entity teardown permanently
    stops JointStatePublisher from advertising - that finding does not
    generalize to this world/SDF).

    This does not raise on ok=False. Direct measurement (both while
    implementing this script and again while reviewing it) showed
    node.request() can come back with ok=False - an RPC-level
    timeout/lost acknowledgment, roughly 1-in-9 in testing - even though
    the reset physically happened: position/velocity were confirmed at
    ~0 immediately afterward despite the False. So the RPC's own ok flag
    is not a trustworthy signal of whether the reset took effect in
    either direction, and raising on it would abort a run that actually
    succeeded. The caller checks the real postcondition instead (the
    post-reset observation against CART_POSITION_LIMIT/POLE_PITCH_LIMIT)
    once it has a fresh reading, and raises there if the reset truly
    didn't take."""
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
        "in the generated SDF?"
    )


def run_inference(model, normalizer, effort_limit):
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
        force_pub = node.advertise(
            f"/model/{MODEL_NAME}/joint/cart_joint/cmd_force", Double)

        latest = {"obs": None}

        def on_joint_state(msg):
            positions = {j.name: j.axis1.position for j in msg.joint}
            velocities = {j.name: j.axis1.velocity for j in msg.joint}
            try:
                obs = np.array([
                    positions["cart_joint"], velocities["cart_joint"],
                    positions["pole_joint"], velocities["pole_joint"],
                ], dtype=np.float32)
            except KeyError:
                return  # mid-reset snapshot missing a joint; skip it
            latest["obs"] = obs

        joint_state_topic = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
        node.subscribe(Model, joint_state_topic, on_joint_state)

        print("Waiting for first joint_state message...")
        _wait_for_obs(latest)

        print("Running inference with GUI... Press Ctrl+C to stop.")
        episode_start = time.monotonic()
        for _ in range(MAX_ITERATIONS):
            loop_start = time.monotonic()

            obs = latest["obs"]
            normalized = normalizer.normalize_obs(obs.reshape(1, -1))
            action, _state = model.predict(normalized, deterministic=True)
            action = int(action[0])

            force_msg = Double()
            force_msg.data = effort_limit if action == 1 else -effort_limit
            force_pub.publish(force_msg)

            cart_pos, _cart_vel, pole_pos, _pole_vel = obs
            if abs(cart_pos) > CART_POSITION_LIMIT or abs(pole_pos) > POLE_PITCH_LIMIT:
                episode_len = time.monotonic() - episode_start
                print(f"Cart-pole out of bounds after {episode_len:.2f}s, resetting world...")
                # ApplyJointForce has no Reset() and simply holds the last
                # commanded force (confirmed via nm -DC on its .so) - left
                # un-zeroed, the pre-fall +/-30N force stays latched through
                # reset.all and keeps driving the freshly-reset cart/pole,
                # which measurably caused runaway "0.00s, resetting
                # world..." storms (dozens of consecutive resets, request
                # ok=True every time) before this fix. Zero it before
                # resetting so nothing is still pushing post-reset.
                zero_msg = Double()
                zero_msg.data = 0.0
                force_pub.publish(zero_msg)
                # JointStatePublisher publishes at the physics-step rate
                # (~1kHz, no rate limit configured) - issuing node.request()
                # while this node's own subscription callback is firing at
                # that rate reliably deadlocks gz.transport13's Python
                # binding (confirmed in Task 6's
                # verify_reset_preserves_joint_state.py). Unsubscribe for
                # the duration of the request, resubscribe once it returns.
                #
                # reset.all was separately measured (dedicated scratch
                # repro, not just this loop) to occasionally not take
                # effect at all on its first attempt - the post-reset
                # observation reads back nearly identical to the pre-reset
                # one, independent of the RPC's own ok flag (reproduced
                # with both ok=True and ok=False). Retrying immediately
                # reliably recovers, always within 2 attempts across two
                # separate 10-consecutive-reset stress runs - so this loop
                # retries up to MAX_RESET_ATTEMPTS times, checking the
                # FIRST fresh post-reset reading each time (not one taken
                # after a deliberate delay - a previous version slept 0.5s
                # before reading, during which the still-latched pre-fix
                # force could drive the system back out of bounds before
                # the postcondition was ever evaluated, so a genuine
                # runaway-reset storm would read as a normal in-bounds
                # state instead of raising).
                reset_ok = None
                for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
                    node.unsubscribe(joint_state_topic)
                    reset_ok = _reset_world(node)
                    latest["obs"] = None
                    node.subscribe(Model, joint_state_topic, on_joint_state)
                    _wait_for_obs(latest)
                    cart_pos, _cart_vel, pole_pos, _pole_vel = latest["obs"]
                    if abs(cart_pos) <= CART_POSITION_LIMIT and abs(pole_pos) <= POLE_PITCH_LIMIT:
                        if attempt > 1:
                            print(f"  (reset took effect on attempt {attempt})")
                        break
                else:
                    raise RuntimeError(
                        f"world reset did not take effect after {MAX_RESET_ATTEMPTS} "
                        f"attempts (last request ok={reset_ok}, post-reset "
                        f"cart={cart_pos:.4f} pole={pole_pos:.4f})"
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
        # Ctrl+C sometimes arrives twice in quick succession here (observed
        # reliably when the process is launched under `timeout
        # --signal=INT ... uv run ...`, which forwards SIGINT through both
        # its own process-group signaling and uv's child supervision) - a
        # second KeyboardInterrupt landing mid-cleanup previously escaped
        # straight past a bare proc.wait(), leaving gz sim's GUI process
        # (which can take longer than expected to tear down its EGL/GL
        # context, per the libEGL warnings this environment logs) alive
        # and unmonitored - confirmed directly: it did not exit on its own
        # even minutes later, only dying to a subsequent manual SIGTERM.
        # Retry across repeated interrupts and escalate to SIGKILL if a
        # process doesn't respond to SIGTERM promptly, so cleanup can never
        # be skipped just because it got interrupted itself.
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
    parser.add_argument(
        "--model", default=os.path.join(FILE_DIR, "cart_pole_gz_train_ppo"))
    parser.add_argument(
        "--vecnorm", default=os.path.join(FILE_DIR, "vecnormalize.pkl"))
    args = parser.parse_args()

    generate_training_world(SDF_PATH)
    effort_limit = _read_effort_limit(SDF_PATH, "cart_joint")
    print(f"Read cart_joint effort limit from generated SDF: {effort_limit}N")

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}.zip")
    normalizer = _load_normalizer(args.vecnorm)
    print(f"Loaded VecNormalize stats from {args.vecnorm}")

    run_inference(model, normalizer, effort_limit)


if __name__ == "__main__":
    main()
