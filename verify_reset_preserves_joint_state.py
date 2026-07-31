# Amendment (post-implementation, from Task 7's implementation and the
# final whole-branch review): this script originally validated
# reset.model_only using message-count-only assertions (before/after
# joint_state counts on a system at rest). That reset type was later found
# to be a complete no-op on this world - reset.all is what run_inference.py
# actually ships with - and the message-count-only check would have passed
# identically either way, since it never exercised a system with nonzero
# position/velocity to reset. This script now disturbs the system first and
# asserts reset.all actually zeros position/velocity, closing that gap.
#
# Amendment 2 (found while re-verifying run_inference.py's own retry fix):
# a single reset.all request was separately measured to occasionally not
# take effect at all on its first attempt (post-reset reading nearly
# identical to pre-reset, independent of the ok flag) - roughly 1-in-3 in
# one measured run. run_inference.py retries up to MAX_RESET_ATTEMPTS times
# to absorb this; this script does the same, so it doesn't flake on the
# same underlying behavior it isn't trying to test here (that's covered by
# run_inference.py's own retry logic - this script's job is validating the
# unsubscribe/resubscribe transport pattern and that a reset CAN reach ~0,
# not measuring first-attempt reliability).

import os
import sys
import subprocess
import time

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, FILE_DIR)

from world_builder import generate_training_world

from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

sdf_path = os.path.join(FILE_DIR, "cart_pole_train.sdf")
generate_training_world(sdf_path)

WORLD_NAME = "cart_pole_train"
MODEL_NAME = "cart_pole"
TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
FORCE_TOPIC = f"/model/{MODEL_NAME}/joint/cart_joint/cmd_force"
# Generous margin around 0 for "the reset actually took effect" - not a
# precision check, just enough to distinguish a real reset from the
# no-op documented below.
POSITION_TOLERANCE = 0.05
# See Amendment 2 above - matches run_inference.py's own retry bound.
MAX_RESET_ATTEMPTS = 5

gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", sdf_path])
try:
    time.sleep(4)  # let the server come up and start publishing
    if gz_server.poll() is not None:
        raise RuntimeError(
            f"gz sim server exited immediately (code {gz_server.returncode})"
        )

    node = Node()
    counts = {"before": 0, "after": 0}
    phase = {"value": "before"}
    latest = {"obs": None}

    def on_joint_state(msg):
        counts[phase["value"]] += 1
        positions = {j.name: j.axis1.position for j in msg.joint}
        velocities = {j.name: j.axis1.velocity for j in msg.joint}
        if "cart_joint" in positions and "pole_joint" in positions:
            latest["obs"] = (
                positions["cart_joint"], velocities["cart_joint"],
                positions["pole_joint"], velocities["pole_joint"],
            )

    node.subscribe(Model, TOPIC, on_joint_state)

    time.sleep(2)
    assert counts["before"] > 0, (
        "no joint_state messages received before reset - is the server up "
        "and is JointStatePublisher declared in the generated SDF?"
    )

    # Disturb the system before resetting, so a no-op reset is
    # distinguishable from a working one. This script originally only
    # checked message counts with the model at rest both before and after
    # the reset - which is exactly the gap that let reset.model_only ship
    # as "verified" here despite being a complete no-op (see Amendment
    # below): a no-op reset and a working reset look identical on an
    # at-rest system.
    force_pub = node.advertise(FORCE_TOPIC, Double)
    force_msg = Double()
    force_msg.data = 30.0
    for _ in range(50):  # ~0.25s at max effort - enough to move cart/pole
        force_pub.publish(force_msg)
        time.sleep(0.005)
    force_msg.data = 0.0
    force_pub.publish(force_msg)
    time.sleep(0.2)

    assert latest["obs"] is not None, "no joint_state received after the disturbance burst"
    pre_reset_obs = latest["obs"]
    assert any(abs(v) > POSITION_TOLERANCE for v in pre_reset_obs), (
        f"disturbance burst didn't move the system enough to make the reset "
        f"postcondition meaningful (obs={pre_reset_obs}) - increase the "
        f"force/duration above"
    )

    # JointStatePublisher has no rate limit in the generated SDF, so it
    # publishes every physics step (~1kHz, confirmed empirically: ~2000
    # msgs over the 2s sleep above). Sending node.request() while this
    # node's own subscription callback is being invoked at that rate
    # reliably deadlocks/times out gz.transport13's Python binding here
    # (reproduced identically under both `uv run` and plain system
    # python3, with timeouts up to 15s and with the request issued from a
    # second Node instance - so it's a genuine contention/backpressure bug
    # in the binding, not a venv/protobuf mismatch or a too-short timeout).
    # Unsubscribing for the duration of the request sidesteps it, but note
    # this narrows what's actually tested: the "after" count comes from a
    # brand-new subscription created after the reset returns, not from the
    # original subscription surviving the reset uninterrupted, so this
    # confirms "a fresh subscription resumes receiving after reset" rather
    # than the plan's original, stronger claim that "an already-open
    # subscription keeps receiving through the reset." That narrower scope
    # is intentional and sufficient here: Task 7's run_inference.py drives
    # its own episode resets with this exact same
    # unsubscribe-before-reset/resubscribe-after-reset pattern (see its
    # main loop's call to _reset_world), so this script validates that
    # pattern.
    ok = None
    post_reset_obs = None
    for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
        node.unsubscribe(TOPIC)

        request = WorldControl()
        request.reset.all = True
        latest["obs"] = None
        ok, _resp = node.request(
            f"/world/{WORLD_NAME}/control", request, WorldControl, Boolean, 5000)
        # node.request()'s ok flag is not a trustworthy signal by itself
        # here - run_inference.py's _reset_world docstring documents
        # ok=False measured on a reset that physically succeeded
        # (~1-in-9). The position/velocity postcondition below is the real
        # check; ok is only reported for diagnostics.

        phase["value"] = "after"
        counts["after"] = 0
        node.subscribe(Model, TOPIC, on_joint_state)
        time.sleep(2)

        assert counts["after"] > 0, (
            f"JointStatePublisher stopped publishing after reset.all=True "
            f"(before={counts['before']} msgs, after={counts['after']} msgs, "
            f"request ok={ok}) - unlike the unrelated commander/robomaster_rale "
            f"world (ros2_ws/src/CLAUDE.md), this project's SDF-declared world "
            f"was measured to keep publishing across reset.all; if this "
            f"regresses, run_inference.py's reset strategy needs revisiting."
        )

        assert latest["obs"] is not None, "no joint_state received after the reset"
        post_reset_obs = latest["obs"]
        if all(abs(v) < POSITION_TOLERANCE for v in post_reset_obs):
            if attempt > 1:
                print(f"(reset took effect on attempt {attempt} - see Amendment 2)")
            break
    else:
        raise AssertionError(
            f"reset.all did not reset position/velocity to ~0 after "
            f"{MAX_RESET_ATTEMPTS} attempts (pre-reset obs={pre_reset_obs}, "
            f"last post-reset obs={post_reset_obs}, last request ok={ok}) - "
            f"reset.model_only was already found to be a no-op here (see "
            f"run_inference.py's _reset_world docstring); if reset.all is "
            f"also consistently failing, run_inference.py has no working "
            f"reset left and needs a new strategy before being used."
        )

    print(
        f"PASS: reset.all reset position/velocity to ~0 (pre-reset "
        f"obs={pre_reset_obs}, post-reset obs={post_reset_obs}) and "
        f"joint_state kept publishing afterward (before={counts['before']} "
        f"msgs, after={counts['after']} msgs)"
    )
finally:
    gz_server.terminate()
    try:
        gz_server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        gz_server.kill()
        gz_server.wait()
