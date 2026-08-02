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


if __name__ == "__main__":
    print("infer.py skeleton loaded OK")
