"""Verify the generated world is *grounded and free-moving*, not merely
airborne or jammed.

The original version of this script only ran 100ms at max effort and
asserted the cart reached >0.5 m/s. That passed for the wrong reason: the
world back then spawned the model at z=2, so the whole 100ms window sat
inside a ~590ms free-fall, where cart_joint is trivially free to
accelerate because nothing is touching the ground yet. It would equally
have passed for a world whose robot never lands correctly.

This version checks the two things that actually matter:
  1. With zero applied force, the model is already at rest on the ground
     at t=0 and stays there (no fall, no sinking, no pole knocked over).
  2. From that grounded rest state, max effort accelerates the cart at
     close to effort_limit / cart_mass.
"""
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import SPAWN_Z, generate_training_world
import gz.math7  # noqa: F401  registers Pose3 so Link.world_pose() converts
from gz.sim8 import TestFixture, World, world_entity, Model, Joint, Link

scratch_dir = os.path.dirname(__file__)
sdf_path = os.path.join(scratch_dir, "cart_pole_train.sdf")
world_text = generate_training_world(sdf_path)

cart_mass = float(
    ET.fromstring(world_text)
    .find(".//model[@name='cart_pole']/link[@name='cart_link']/inertial/mass")
    .text
)

SETTLE_MS = 1000   # comfortably longer than the ~590ms free-fall the old
                   # z=2 spawn produced, so a regression to it can't hide here
PUSH_MS = 200

state = {"force": 0.0}


def ensure_init(ecm):
    if "cart_joint" in state:
        return
    world = World(world_entity(ecm))
    model = Model(world.model_by_name(ecm, "cart_pole"))
    cart_joint = Joint(model.joint_by_name(ecm, "cart_joint"))
    pole_joint = Joint(model.joint_by_name(ecm, "pole_joint"))
    for joint in (cart_joint, pole_joint):
        joint.enable_position_check(ecm, True)
        joint.enable_velocity_check(ecm, True)
    base = Link(model.link_by_name(ecm, "base_footprint"))
    base.enable_velocity_checks(ecm, True)
    state["cart_joint"] = cart_joint
    state["pole_joint"] = pole_joint
    state["base"] = base
    state["max_force"] = cart_joint.effort_limits(ecm)[0]


def on_pre_update(info, ecm):
    if info.paused:
        return
    ensure_init(ecm)
    state["cart_joint"].set_force(ecm, [state["force"]])


def on_post_update(info, ecm):
    if info.paused:
        return
    ensure_init(ecm)
    state["cart_pos"] = state["cart_joint"].position(ecm)[0]
    state["cart_vel"] = state["cart_joint"].velocity(ecm)[0]
    state["pole_pos"] = state["pole_joint"].position(ecm)[0]
    state["base_z"] = state["base"].world_pose(ecm).pos().z()


fixture = TestFixture(sdf_path)
fixture.on_pre_update(on_pre_update)
fixture.on_post_update(on_post_update)
fixture.finalize()
server = fixture.server()

# --- 1. grounded at spawn, and stays grounded ---
server.run(True, 1, False)
spawn_z = state["base_z"]
server.run(True, SETTLE_MS - 1, False)
rest_z = state["base_z"]
rest_pole_pos = state["pole_pos"]

assert abs(spawn_z - SPAWN_Z) < 1e-3, (
    f"base_footprint spawned at z={spawn_z:.4f}, expected {SPAWN_Z} "
    "(world_builder's spawn pose is wrong or was not injected)"
)
assert abs(rest_z - spawn_z) < 1e-3, (
    f"base_footprint moved from z={spawn_z:.4f} to z={rest_z:.4f} over "
    f"{SETTLE_MS}ms of zero-force settling - the model is either falling "
    "(spawn pose too high) or sinking into the ground plane (too low)"
)
assert abs(rest_pole_pos) < 1e-3, (
    f"pole_joint drifted to {rest_pole_pos:.4f} rad while just standing "
    "there - the pole's collision geometry is probably intersecting the "
    "ground plane (check the pole collision <pose> offset)"
)

# --- 2. max effort produces the expected acceleration from that rest state ---
v0 = state["cart_vel"]
state["force"] = state["max_force"]
server.run(True, PUSH_MS, False)
v1 = state["cart_vel"]
accel = (v1 - v0) / (PUSH_MS / 1000.0)

ideal_accel = state["max_force"] / cart_mass
assert 0.7 * ideal_accel < accel < 1.05 * ideal_accel, (
    f"cart accelerated at {accel:.3f} m/s^2 from a grounded rest state, but "
    f"effort_limit/cart_mass = {ideal_accel:.3f} m/s^2. Near-zero means the "
    "model is jammed against the ground plane; far above means it is not "
    "actually grounded."
)

print(
    f"PASS: model rests grounded at base_z={rest_z:.4f} with pole_pitch="
    f"{rest_pole_pos:+.5f} after {SETTLE_MS}ms of zero force; "
    f"{state['max_force']}N then gives accel={accel:.3f} m/s^2 "
    f"(effort_limit/cart_mass = {ideal_accel:.3f}), reaching "
    f"vel={v1:.3f} m/s (pos={state['cart_pos']:.3f})"
)
