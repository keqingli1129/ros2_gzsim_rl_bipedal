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
