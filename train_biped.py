import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_scorer import BipedScorer, HEIGHT_DROP_LIMIT, PITCH_LIMIT

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class CustomBipedGzTrain(gym.Env):
    """Wraps BipedScorer for Gymnasium/SB3."""

    def __init__(self, env_config=None):
        self.env = BipedScorer()
        # Actions are normalized to [-1, 1] per joint (order: [hip_L,
        # knee_L, hip_R, knee_R]). BipedScorer scales each by the live ECM
        # effort limit internally (see its on_pre_update/_ensure_initialized),
        # so no torque constant needs to live in this file at all - this
        # also keeps CONTROL_COST_WEIGHT (copied from BipedalWalker-v3/
        # Walker2d's [-1,1]-action convention) in the reward-scale range it
        # was designed for, instead of a raw-N*m action blowing it up.
        self.action_space = gym.spaces.Box(
            np.array([-1.0] * 4, dtype=np.float32),
            np.array([1.0] * 4, dtype=np.float32),
        )
        # 13-dim observation - see BipedScorer.on_post_update for the exact
        # assembly order. Bounds mirror train_cart_pole.py's convention:
        # only the two dimensions with a real termination threshold
        # (torso_z_pos, torso_pitch) get a finite bound; everything else is
        # unbounded (VecNormalize handles the actual scaling).
        low = np.array([-np.inf, -HEIGHT_DROP_LIMIT, -np.inf, -PITCH_LIMIT, -np.inf,
                         -np.inf, -np.inf, -np.inf, -np.inf,
                         -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, PITCH_LIMIT, np.inf,
                          np.inf, np.inf, np.inf, np.inf,
                          np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, (13,), np.float32)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def main():
    # VecNormalize is required, not optional - see this repo's CLAUDE.md:
    # the cart-pole precedent's velocity dimensions are 10-30x the scale
    # of its position dimensions, which stalled learning outright until
    # VecNormalize was added. This env has the same shape of problem
    # (torque-scale/velocity dimensions vs. small angle/position ones).
    # A TimeLimit is required for learning-curve visibility, not just tidiness:
    # a balancing-only policy with no forward-progress incentive would
    # otherwise never trip HEIGHT_DROP_LIMIT/PITCH_LIMIT, meaning Monitor
    # never logs a completed episode and the training run has no visible
    # ep_rew_mean/ep_len_mean curve at all - the same trap this repo's
    # train_cart_pole.py precedent avoids via its own termination bounds,
    # but a policy that just stands still still needs *some* bound here.
    # 1000 steps * 5ms/step = 5 simulated seconds per episode.
    venv = DummyVecEnv([lambda: Monitor(TimeLimit(CustomBipedGzTrain(), max_episode_steps=1000))])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, verbose=1, device="auto")
    model.learn(total_timesteps=1_000_000)
    model_path = os.path.join(FILE_DIR, "biped_ppo")
    model.save(model_path)
    vecnorm_path = os.path.join(FILE_DIR, "biped_vecnormalize.pkl")
    venv.save(vecnorm_path)
    venv.close()
    print(f"Training complete. Saved model to {model_path}.zip")
    print(f"Saved VecNormalize stats to {vecnorm_path}")


if __name__ == "__main__":
    main()
