"""Manual testing helper: publish a force burst to cart_joint's cmd_force
topic to disturb a live run_inference.py session and watch it recover.

A single one-off publish gets overwritten within ~5ms by run_inference.py's
own policy action (ApplyJointForce just holds the last commanded value), so
this holds one Node/publisher open and republishes continuously for the
requested duration - fast enough to actually override the policy before it
corrects. A raw `gz topic -p` one-liner is too slow per-call for this (each
invocation re-establishes a transport connection), which is why this exists
as a small script instead.

Run this in a separate terminal while run_inference.py is already running:

    PYTHONPATH=/usr/lib/python3/dist-packages python3 \
        ros2_ws/src/cart_pole_gz_train/nudge.py [duration] [force]
"""
import argparse
import time

from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double

FORCE_TOPIC = "/model/cart_pole/joint/cart_joint/cmd_force"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duration", nargs="?", type=float, default=0.3,
                         help="seconds to hold the force (default: 0.3)")
    parser.add_argument("force", nargs="?", type=float, default=30.0,
                         help="signed force in Newtons (default: 30.0)")
    args = parser.parse_args()

    node = Node()
    pub = node.advertise(FORCE_TOPIC, Double)
    time.sleep(0.5)  # let the publisher register before the first message

    msg = Double()
    msg.data = args.force
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        pub.publish(msg)
        time.sleep(0.002)

    msg.data = 0.0
    pub.publish(msg)
    print(f"burst ({args.duration}s at {args.force}N) delivered and zeroed")


if __name__ == "__main__":
    main()
