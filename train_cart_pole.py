import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gz_scorer import GzCartPoleScorer, CART_POSITION_LIMIT, POLE_PITCH_LIMIT

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class CustomCartPoleGzTrain(gym.Env):
    """Wraps GzCartPoleScorer for Gymnasium/SB3."""

    def __init__(self, env_config=None):
        self.env = GzCartPoleScorer()
        self.action_space = gym.spaces.Discrete(2)
        # Bounds are derived from gz_scorer's CART_POSITION_LIMIT/
        # POLE_PITCH_LIMIT (this robot's termination thresholds: cart_joint
        # +/-0.9m, pole_joint +/-0.48rad - not the root project's arbitrary
        # bounds tuned for its unrelated model, and not this joint's harder
        # mechanical stops either, +/-1m / +/-1.7rad) rather than hardcoded
        # literals, so the two can't silently drift apart if the termination
        # thresholds ever change.
        self.observation_space = gym.spaces.Box(
            np.array([-CART_POSITION_LIMIT, -np.inf, -POLE_PITCH_LIMIT, -np.inf], dtype=np.float32),
            np.array([CART_POSITION_LIMIT, np.inf, POLE_PITCH_LIMIT, np.inf], dtype=np.float32),
            (4,), np.float32,
        )

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def main():
    # SB3's PPO does not normalize observations by default. Measured raw
    # magnitudes for this env (see docs/plan.md Task 5 amendment / task-5
    # tuning report) show a large cross-dimension scale mismatch: cart_pos
    # and pole_pitch are O(0.1-0.5), but cart_vel and especially pole_vel
    # (the light ~0.86kg pole has low rotational inertia) reach O(1-10) -
    # roughly 10-30x larger. A freshly-initialized MlpPolicy expects
    # roughly unit-scale inputs, so the velocity dimensions dominate the
    # gradient and the position dimensions are effectively drowned out.
    # VecNormalize (running mean/std normalization of observations) fixes
    # this without touching the reward function or termination logic.
    #
    # The Monitor wrapper is applied explicitly: SB3 only auto-wraps Monitor
    # when the env handed to PPO() is *not* already a VecEnv, so passing a
    # pre-built DummyVecEnv (which VecNormalize requires) silently drops
    # episode bookkeeping and the rollout/ep_rew_mean and ep_len_mean rows
    # vanish from the training log entirely - i.e. no visible learning curve.
    venv = DummyVecEnv([lambda: Monitor(CustomCartPoleGzTrain())])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, verbose=1, device="auto")
    model.learn(total_timesteps=100_000)
    model_path = os.path.join(FILE_DIR, "cart_pole_gz_train_ppo")
    model.save(model_path)
    # The running obs-normalization statistics are part of the trained
    # policy's expected input distribution - a deployed/evaluated policy
    # must feed it observations normalized with these same stats, so they
    # are saved alongside the model (see evaluate_policy.py).
    vecnorm_path = os.path.join(FILE_DIR, "vecnormalize.pkl")
    venv.save(vecnorm_path)
    venv.close()
    print(f"Training complete. Saved model to {model_path}.zip")
    print(f"Saved VecNormalize stats to {vecnorm_path}")


if __name__ == "__main__":
    main()
