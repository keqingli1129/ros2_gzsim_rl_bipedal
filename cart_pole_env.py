import argparse
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np

from gz.common5 import set_verbosity
from gz.sim8 import TestFixture, World, world_entity, Model, Link
from gz.math7 import Vector3d

from stable_baselines3 import PPO
import math
import time
import subprocess

# --- Inference-time imports (Gazebo transport) ---
from gz.transport13 import Node
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.wrench_pb2 import Wrench
from gz.msgs10.vector3d_pb2 import Vector3d as Vector3dMsg
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.serialized_map_pb2 import SerializedStepMap

file_path = os.path.dirname(os.path.realpath(__file__))

def run_gui():
    """
    This function looks for your gz sim installation and looks for
    an instance of the gui client
    """
    subprocess.Popen(["gz", "sim", "-g"])

class GzRewardScorer:
    """
    This Gazebo System is used to introspect and score the world.
    """
    def __init__(self):
        """
        We initialize a TestFixture: This is a simple fixture that is used
        to load our gazebo world. We also inject the code to be executed
        on each run.
        """
        self.command = None # This variable is used as a bridge between Gymnasium and gazebo
        self._build_fixture()
        self.terminated = False
        self._initialized = False
        self.state = np.zeros(4, dtype=np.float32)
        self.reward = 0.0
        self.prev_cart_pose = None
        self.prev_pole_pose = None

    def _build_fixture(self):
        """
        Load a fresh TestFixture/Server pair.

        server.reset_all() leaves the physics engine desynced from the ECM:
        gravity-driven dynamics keep working, but Link.add_world_force() and
        velocity reads silently stop taking effect for the rest of the
        process (verified live - the chassis entity ID is unchanged before
        and after reset_all(), so it isn't a stale-handle issue; it's
        specific to that call). Tearing down and rebuilding the TestFixture
        avoids this entirely, since a freshly loaded fixture always has
        working force application.
        """
        self.server = None
        self.fixture = None
        self.fixture = TestFixture(os.path.join(file_path, 'cart_pole.sdf'))
        self.fixture.on_pre_update(self.on_pre_update)
        self.fixture.on_post_update(self.on_post_update)
        self.fixture.finalize()
        self.server = self.fixture.server()

    def _ensure_initialized(self, ecm):
        """Look up entities if not yet initialized (or after a reset)."""
        if not self._initialized:
            world = World(world_entity(ecm))
            self.model = Model(world.model_by_name(ecm, "vehicle_green"))
            self.pole_entity = self.model.link_by_name(ecm, "pole")
            self.chassis_entity = self.model.link_by_name(ecm, "chassis")
            self.pole = Link(self.pole_entity)
            self.chassis = Link(self.chassis_entity)
            self._initialized = True

    def on_pre_update(self, info, ecm):
        """
        on_pre_update is used to command the model vehicle.
        """
        if info.paused:
            return
        self._ensure_initialized(ecm)
        if self.command == 1:
            self.chassis.add_world_force(ecm, Vector3d(2000, 0, 0))
        elif self.command == 0:
            self.chassis.add_world_force(ecm, Vector3d(-2000, 0, 0))

    def on_post_update(self, info, ecm):
        """
        on_post_update is used to read the current state of the world. We write the
        state to a local field.
        """
        if info.paused:
            return
        self._ensure_initialized(ecm)
        pole_pose = self.pole.world_pose(ecm).rot().euler().y()
        cart_pose = self.chassis.world_pose(ecm).pos().x()
        # Estimate velocity via finite difference over the 5ms/5-tick step,
        # matching run_inference()'s estimator exactly (average velocity
        # over the step) rather than reading the physics engine's true
        # instantaneous velocity - the out-of-process inference server has
        # no way to get the latter over transport (enable_velocity_checks
        # only works from in-process code with direct ECM access), so a
        # policy trained on exact velocities sees a substantially different
        # signal at inference (~76% relative error between the two, verified
        # empirically) and its control quality degrades badly. Training on
        # the same estimator the deployed policy will actually receive
        # removes that train/inference distribution mismatch.
        step_dt = 0.005  # 5 physics ticks x 1ms, deterministic in training
        cart_vel = ((cart_pose - self.prev_cart_pose) / step_dt
                    if self.prev_cart_pose is not None else 0.0)
        pole_angular_vel = ((pole_pose - self.prev_pole_pose) / step_dt
                             if self.prev_pole_pose is not None else 0.0)
        # Write the state to the environment
        self.state = np.array([cart_pose, cart_vel, pole_pose, pole_angular_vel], dtype=np.float32)
        if not self.terminated:
            self.terminated = pole_pose > 0.48 or pole_pose < -0.48 or cart_pose > 4.8 or cart_pose < -4.8

        if self.terminated:
            self.reward = 0.0
        else:
            self.reward = 1.0

    def step(self, action, paused=False):
        """
        Execute the server.

        There is a bit of nuance in this instance,
        our environment has control over every 5 simulation steps.
        We block the server till those 5 steps are completed.
        """
        self.command = action
        self.server.run(True, 5, paused)
        obs = self.state
        reward = self.reward
        self.prev_cart_pose = obs[0]
        self.prev_pole_pose = obs[2]
        return obs, reward, self.terminated, False, {}

    def reset(self):
        """
        This function rebuilds the fixture/server rather than calling
        server.reset_all() - see _build_fixture()'s docstring for why.
        """
        self._build_fixture()
        self.command = None
        self.terminated = False
        self._initialized = False
        self.prev_cart_pose = None
        self.prev_pole_pose = None
        obs, reward_, term_, tunc_, other_= self.step(None, paused=False)
        return obs, {}

    def close(self):
        """
        Drop references to the in-process fixture/server so their pybind11
        destructors run and release the transport node. Needed before
        run_inference spawns a separate out-of-process server under the
        same world name - otherwise both stay registered at once and the
        GUI can attach to the wrong (stale) one.
        """
        self.server = None
        self.fixture = None



class CustomCartPole(gym.Env):
    """
    Wrapper around GzRewardScorer that adapts the reward scorer to work with
    gymnasium.
    """
    def __init__(self, env_config):
        self.env = GzRewardScorer()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            np.array([-10, float("-inf"), -0.418, -3.4028235e+38]),
            np.array([10, float("inf"), 0.418, 3.4028235e+38]),
            (4,), np.float32)

    def reset(self, seed=123):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return  obs, reward, done, truncated, info

    def close(self):
        self.env.close()

def _quat_mult(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def _quat_rotate(q, v):
    """Rotate vector v=(x,y,z) by quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y*vz - z*vy)
    ty = 2.0 * (z*vx - x*vz)
    tz = 2.0 * (x*vy - y*vx)
    return (
        vx + w*tx + (y*tz - z*ty),
        vy + w*ty + (z*tx - x*tz),
        vz + w*tz + (x*ty - y*tx),
    )

def _pitch_from_quat(q):
    """Extract Y-axis pitch from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    sinp = 2.0 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)

def _gz_component_hash(type_name):
    """Replicate gz::common::hash64() (FNV-1a, 64-bit), used by gz-sim's
    component Factory (components/Factory.hh) to assign each registered
    component type a runtime ComponentTypeId. SerializedComponent.type on
    the /world/<world>/state service carries this same value, so decoding
    that service's response requires reproducing the hash here rather than
    hardcoding IDs (they're derived from the type name string, not stable
    across gz-sim versions/builds otherwise).
    """
    prime = 0x100000001b3
    h = 0xcbf29ce484222325
    mask = (1 << 64) - 1
    for byte in type_name.encode("utf-8"):
        h ^= byte
        h = (h * prime) & mask
    if h >= (1 << 63):
        # SerializedComponent.type is a signed int64 field on the wire;
        # values with the high bit set decode as negative.
        h -= (1 << 64)
    return h

_NAME_COMPONENT_ID = _gz_component_hash("gz_sim_components.Name")
_POSE_COMPONENT_ID = _gz_component_hash("gz_sim_components.Pose")

def _query_world_state(node):
    """Synchronously fetch the full ECS snapshot from /world/cart_pole/state.

    Unlike the dynamic_pose/info topic (published on a fixed ~17ms timer by
    the SceneBroadcaster plugin, independent of when we actually need an
    observation), this is a request/response service we can call whenever we
    want a fresh reading - so the caller's own loop cadence becomes the
    observation rate, not the publisher's.
    """
    ok, resp = node.request(
        "/world/cart_pole/state", Empty(), Empty, SerializedStepMap, 2000)
    if not ok:
        return None, {}, {}

    sim_time = resp.stats.sim_time.sec + resp.stats.sim_time.nsec * 1e-9
    names_by_id = {}
    pose_text_by_id = {}
    for entity_id, entity_map in resp.state.entities.items():
        for comp_id, comp in entity_map.components.items():
            if comp_id == _NAME_COMPONENT_ID:
                names_by_id[entity_id] = comp.component.decode("utf-8")
            elif comp_id == _POSE_COMPONENT_ID:
                pose_text_by_id[entity_id] = comp.component.decode("utf-8")
    return sim_time, names_by_id, pose_text_by_id

def _euler_to_quat(roll, pitch, yaw):
    """Convert roll/pitch/yaw (radians) to a (w, x, y, z) quaternion.

    Mirrors gz::math::Quaternion<T>::SetFromEuler exactly (see
    gz/math7/gz/math/Quaternion.hh) so composing poses decoded from the ECS
    Pose component's Euler-angle text matches what Link.world_pose() would
    have produced from the same underlying quaternion during training.
    """
    phi, the, psi = roll / 2.0, pitch / 2.0, yaw / 2.0
    cphi, sphi = math.cos(phi), math.sin(phi)
    cthe, sthe = math.cos(the), math.sin(the)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    w = cphi * cthe * cpsi + sphi * sthe * spsi
    x = sphi * cthe * cpsi - cphi * sthe * spsi
    y = cphi * sthe * cpsi + sphi * cthe * spsi
    z = cphi * cthe * spsi - sphi * sthe * cpsi
    return (w, x, y, z)

def _resolve_target_entities(node):
    """Look up entity IDs for vehicle_green/chassis/pole by name, once.

    Entity IDs are assigned at world-load time from the SDF and stay fixed
    across a reset (confirmed live: same IDs before/after
    WorldControl.reset.all=True), so this only needs to run once at
    startup, not after every reset.
    """
    _, names_by_id, _ = _query_world_state(node)
    wanted = {"vehicle_green", "chassis", "pole"}
    ids = {name: eid for eid, name in names_by_id.items() if name in wanted}
    missing = wanted - ids.keys()
    if missing:
        raise RuntimeError(f"Could not resolve entities: {missing}")
    return ids

def _parse_pose_text(text):
    x, y, z, roll, pitch, yaw = (float(v) for v in text.split())
    return (x, y, z), _euler_to_quat(roll, pitch, yaw)


def _world_frame_pose(names_by_id, pose_text_by_id, entity_ids):
    """Compose (cart_pose, pole_pose) in world frame from one state query.

    Returns None if any of the three tracked entities' pose text is
    missing from this particular response (e.g. mid-reset).
    """
    try:
        model_pos, model_quat = _parse_pose_text(
            pose_text_by_id[entity_ids["vehicle_green"]])
        chassis_local_pos, _ = _parse_pose_text(
            pose_text_by_id[entity_ids["chassis"]])
        _, pole_local_quat = _parse_pose_text(
            pose_text_by_id[entity_ids["pole"]])
    except KeyError:
        return None

    cart_pose = model_pos[0] + _quat_rotate(model_quat, chassis_local_pos)[0]
    pole_pose = _pitch_from_quat(_quat_mult(model_quat, pole_local_quat))
    return cart_pose, pole_pose

def _kill_stale_gz_processes():
    """Terminate any gz sim server/GUI left over from a prior run.

    A leftover server registers on the same transport bus under the same
    world name as the one we're about to launch, and the new GUI can attach
    to that stale instance instead of ours, showing blank or wrong content.
    """
    subprocess.run(["pkill", "-f", "gz sim"], check=False)
    time.sleep(1)

def _reset_world(node):
    """Reset the running world via its control service.

    Without this, a fallen/out-of-bounds cart-pole keeps getting shoved by
    the policy's forces indefinitely: positions grow unbounded until they
    exceed what the physics engine's collision AABB math can represent,
    crashing the server (ODE assertion "aabbBound ... dMaxIntExact").
    """
    request = WorldControl()
    request.reset.all = True
    node.request(
        "/world/cart_pole/control", request, WorldControl, Boolean, 5000)


def run_inference(model):
    """
    Launch a Gazebo server + GUI and drive the trained model over Gazebo
    transport until Ctrl+C.
    """
    _kill_stale_gz_processes()

    sdf_path = os.path.join(file_path, "cart_pole.sdf")
    gz_server = None
    gz_gui = None
    try:
        print("Launching Gazebo server...")
        gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", sdf_path])
        time.sleep(3)

        print("Launching Gazebo GUI...")
        gz_gui = subprocess.Popen(["gz", "sim", "-g"])
        time.sleep(5)  # Wait for GUI to connect

        node = Node()

        # Advertise on the persistent wrench topic. The plain /wrench topic
        # applies a queued force for exactly one PreUpdate (one 1ms physics
        # step) then drops it (confirmed in gz-sim's ApplyLinkWrench source)
        # - at our 5ms action cadence that's 1 tick of force out of every 5,
        # roughly a fifth of training's authority (training's on_pre_update
        # re-applies the same force every physics tick for all 5 ticks per
        # action). /wrench/persistent keeps the force applied every tick,
        # matching that.
        #
        # There's no working way to REMOVE a persistent entry in this gz-sim
        # build: OnWrenchClear's entity match against the stored persistent
        # entries never succeeds (verified directly - a wrench published to
        # /wrench/clear, by bare or fully-scoped link name, leaves the prior
        # persistent force fully in effect). What DOES work as documented is
        # that persistent entries accumulate and their forces sum every tick
        # (ApplyLinkWrench has no ISystemReset either, so this state also
        # survives a world reset untouched). So instead of clearing, we track
        # the net force we've applied so far in net_force_x and only ever
        # publish the DELTA needed to move the running sum to the new target
        # - e.g. to flip from a net +2000N to -2000N we publish -4000N, which
        # added to the still-present +2000N entry nets to -2000N.
        wrench_pub = node.advertise("/world/cart_pole/wrench/persistent", EntityWrench)
        time.sleep(1)

        entity_ids = _resolve_target_entities(node)
        _query_world_state(node)  # warm-up call; first request has ~200ms
                                   # one-time connection setup cost that
                                   # would otherwise skew the first loop
                                   # iteration's timing measurement below.

        print("Running inference with GUI... Press Ctrl+C to stop.")
        obs = np.zeros(4, dtype=np.float32)
        prev_cart_pose = None
        prev_pole_pose = None
        prev_sim_time = None
        net_force_x = 0.0  # running sum of persistent force actually applied
        target_period = 0.005  # match training's 5ms (5 x 1ms) action cadence
        episode_start = time.monotonic()
        for _ in range(50000):
            loop_start = time.monotonic()
            action, _s = model.predict(obs, deterministic=True)

            # Apply force to the chassis link directly, matching training.
            # Entity name must be unscoped ("chassis", not
            # "vehicle_green::chassis") - this world's transport topics only
            # match against links' bare names, confirmed via the ECS state
            # query's Name components.
            force_x = 2000.0 if action == 1 else -2000.0
            if force_x != net_force_x:
                msg = EntityWrench()
                msg.entity.name = "chassis"
                msg.entity.type = 3  # LINK type
                msg.wrench.force.x = force_x - net_force_x  # delta, see note above
                msg.wrench.force.y = 0.0
                msg.wrench.force.z = 0.0
                wrench_pub.publish(msg)
                net_force_x = force_x

            sim_time, names_by_id, pose_text_by_id = _query_world_state(node)
            frame = _world_frame_pose(names_by_id, pose_text_by_id, entity_ids)

            if frame is not None and sim_time is not None:
                cart_pose, pole_pose = frame
                dt = (sim_time - prev_sim_time
                      if prev_sim_time is not None else None)
                if dt is not None and dt <= 0:
                    # Sim-time reset (world reset) or a duplicate reading.
                    dt = None

                cart_vel = ((cart_pose - prev_cart_pose) / dt
                            if dt and prev_cart_pose is not None
                            else obs[1])
                pole_angular_vel = ((pole_pose - prev_pole_pose) / dt
                                     if dt and prev_pole_pose is not None
                                     else obs[3])

                obs = np.array(
                    [cart_pose, cart_vel, pole_pose, pole_angular_vel],
                    dtype=np.float32)
                prev_cart_pose, prev_pole_pose, prev_sim_time = (
                    cart_pose, pole_pose, sim_time)

            # Same bounds training uses to end an episode. Inference has no
            # episode boundary of its own, so without this the policy keeps
            # applying force to an already-fallen/out-of-bounds cart forever.
            cart_pose, pole_pose = obs[0], obs[2]
            if pole_pose > 0.48 or pole_pose < -0.48 or cart_pose > 4.8 or cart_pose < -4.8:
                episode_len = time.monotonic() - episode_start
                print(f"Cart-pole out of bounds after {episode_len:.2f}s, resetting world...")
                # Persistent wrenches survive a world reset (ApplyLinkWrench
                # has no ISystemReset), so zero the net force here too via a
                # delta - otherwise the pre-reset force keeps being applied
                # on the fresh episode instead of starting unforced like
                # training's reset() (command=None) does.
                if net_force_x != 0.0:
                    zero_msg = EntityWrench()
                    zero_msg.entity.name = "chassis"
                    zero_msg.entity.type = 3  # LINK type
                    zero_msg.wrench.force.x = -net_force_x
                    wrench_pub.publish(zero_msg)
                    net_force_x = 0.0
                _reset_world(node)
                time.sleep(0.5)  # let the reset propagate before next query
                obs = np.zeros(4, dtype=np.float32)
                prev_cart_pose = None
                prev_pole_pose = None
                prev_sim_time = None
                episode_start = time.monotonic()

            elapsed = time.monotonic() - loop_start
            remaining = target_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for proc in (gz_gui, gz_server):
            if proc is not None:
                proc.terminate()
        for proc in (gz_gui, gz_server):
            if proc is not None:
                proc.wait()


def main():
    """
    Train PPO on the cart-pole (headless), then run inference with a GUI.
    Pass --infer-only to skip training and load the previously saved model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer-only", action="store_true",
        help="Skip training and run inference with the saved cart_pole_ppo.zip")
    args = parser.parse_args()

    model_path = os.path.join(file_path, "cart_pole_ppo")

    if args.infer_only:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}.zip")
    else:
        # --- Training (headless, in-process) ---
        env = CustomCartPole({})
        model = PPO("MlpPolicy", env, verbose=1, device="auto")
        model.learn(total_timesteps=100_000)
        model.save(model_path)
        print("Training complete. Saved model to cart_pole_ppo.zip")

        # Release the in-process training world before the inference server
        # (spawned below, out-of-process, under the same world name) starts.
        env.close()

    # --- Inference with GUI via Gazebo transport ---
    run_inference(model)


if __name__ == "__main__":
    main()