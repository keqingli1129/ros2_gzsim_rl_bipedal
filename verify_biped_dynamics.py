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
