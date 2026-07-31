import os
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import SPAWN_Z, generate_training_world

scratch_dir = os.path.dirname(__file__)
output_path = os.path.join(scratch_dir, "cart_pole_train.sdf")
world_text = generate_training_world(output_path)

assert "<mesh>" not in world_text, "primitive replacement left a mesh behind"
assert "tip_link" not in world_text, "tip_link should have been dropped"
assert 'cart_joint' in world_text and 'pole_joint' in world_text
assert '<world name="cart_pole_train">' in world_text

world = ET.fromstring(world_text)
model = next(m for m in world.find("world").findall("model")
             if m.get("name") == "cart_pole")

# Invariant 1: the model spawns with base_footprint's collision box resting
# flush on the ground plane. The box is centred on the model origin, so the
# spawn height must equal half its height - any less and the 88kg base
# starts buried and the contact solver never frees it (measured: cart_joint
# accelerates at 0.10 m/s^2 instead of 10.27); any more and every episode
# opens with a pointless free-fall.
#
# SPAWN_Z = box_height / 2 is only a valid resting height if
# base_footprint's collision is itself centred on the model origin (pose
# 0 0 0) - assert that precondition directly, so a future xacro change that
# gives the collision a nonzero pose can't silently invalidate the derivation
# while this invariant still reads green.
base_collision_pose_el = model.find(
    ".//link[@name='base_footprint']/collision/pose"
)
base_collision_pose = (
    [float(v) for v in base_collision_pose_el.text.split()]
    if base_collision_pose_el is not None
    else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
assert all(abs(v) < 1e-9 for v in base_collision_pose), (
    f"base_footprint's collision pose is {base_collision_pose}, not centred "
    "on the model origin - SPAWN_Z = box_height / 2 is only a valid resting "
    "height when the collision box is centred (pose all-zero); a nonzero "
    "pose here would silently invalidate that derivation"
)

spawn_pose = [float(v) for v in model.find("pose").text.split()]
base_box = model.find(".//link[@name='base_footprint']//box/size").text.split()
expected_z = float(base_box[2]) / 2.0
assert spawn_pose[:2] == [0.0, 0.0] and spawn_pose[3:] == [0.0, 0.0, 0.0], \
    f"spawn pose should only offset z, got {spawn_pose}"
assert abs(spawn_pose[2] - expected_z) < 1e-9, (
    f"model spawns at z={spawn_pose[2]} but base_footprint's collision box is "
    f"{base_box[2]}m tall and centred on the origin, so it must spawn at "
    f"z={expected_z} to rest flush on the ground"
)
assert abs(SPAWN_Z - expected_z) < 1e-9, "SPAWN_Z desynced from the base box size"

# Invariant 2: the pole's collision cylinder spans the pole's real physical
# extent. Compared directly against pole_link's own <inertial><pose>, which
# comes live from the xacro (not against the hardcoded cylinder-length
# constant this same module used to write the cylinder) - so this catches
# the actual failure mode from the original bug: the hand-maintained
# collision geometry desyncing from the real xacro pole if someone changes
# the pole's length there. A centred/mismatched cylinder buries the pole's
# lower half in the ground, which drags pole_joint to its +/-1.7rad limit on
# the very first contact (measured) - dynamics unlike the real robot.
pole_collision = model.find(".//link[@name='pole_link']/collision")
length = float(pole_collision.find(".//cylinder/length").text)
pole_pose = [float(v) for v in pole_collision.find("pose").text.split()]
assert pole_pose[:2] == [0.0, 0.0] and pole_pose[3:] == [0.0, 0.0, 0.0], \
    f"pole collision pose should only offset z, got {pole_pose}"

pole_inertial_pose = [
    float(v)
    for v in model.find(".//link[@name='pole_link']/inertial/pose").text.split()
]
assert abs(pole_pose[2] - pole_inertial_pose[2]) < 0.05, (
    f"pole collision cylinder (length {length}) is at z={pole_pose[2]}, but "
    f"pole_link's real inertial pose (sourced live from the xacro) is at "
    f"z={pole_inertial_pose[2]} - the hand-maintained collision cylinder has "
    "desynced from the real pole geometry (e.g. the xacro's pole length "
    "changed without this module's constants being updated to match)"
)

print(
    "PASS: generated world has no meshes, no tip_link, correct joints, "
    f"spawns flush at z={spawn_pose[2]:g}, and its pole collision cylinder "
    f"spans z in [{-length:g}, 0] in pole_link's frame"
)
