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
