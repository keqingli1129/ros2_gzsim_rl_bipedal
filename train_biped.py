import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_scorer import BipedScorer, HEIGHT_DROP_LIMIT, PITCH_LIMIT

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOTAL_TIMESTEPS = 5_000_000
SEED = 42


class EntCoefDecayCallback(BaseCallback):
    """Linearly decays model.ent_coef from initial_value to 0 over
    training. SB3 has no built-in schedule for ent_coef (only
    learning_rate) - PPO.train() reads self.ent_coef fresh each update,
    so overwriting it here works as a manual one. A constant ent_coef
    held for a full 5M-timestep run (see this repo's
    docs/superpowers/specs/2026-08-03-entropy-coefficient-decay-design.md)
    kept action std at 0.899 even at the end, decoupling the deterministic
    policy infer.py evaluates from what PPO's stochastic rollouts actually
    optimized."""

    def __init__(self, initial_value, total_timesteps):
        super().__init__()
        self.initial_value = initial_value
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        self.model.ent_coef = self.initial_value * max(0.0, 1.0 - progress)
        return True


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
    # ent_coef=0.01 (decayed to 0 via EntCoefDecayCallback below): the
    # first 1M-timestep run (ent_coef=0, no decay possible) converged
    # confidently on a ~1.2s-survival local optimum well before finding a
    # real gait. A later run holding ent_coef=0.01 constant for the full
    # 5M timesteps overcorrected - std never came down, so infer.py's
    # deterministic policy was worse than either prior run despite higher
    # training-time reward. See this repo's
    # docs/superpowers/specs/2026-08-02-entropy-coefficient-tuning-design.md
    # and .../2026-08-03-entropy-coefficient-decay-design.md.
    model = PPO("MlpPolicy", venv, verbose=1, device="auto", ent_coef=0.01, seed=SEED)
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=EntCoefDecayCallback(0.01, TOTAL_TIMESTEPS))
    model_path = os.path.join(FILE_DIR, "biped_ppo")
    model.save(model_path)
    vecnorm_path = os.path.join(FILE_DIR, "biped_vecnormalize.pkl")
    venv.save(vecnorm_path)
    venv.close()
    print(f"Training complete. Saved model to {model_path}.zip")
    print(f"Saved VecNormalize stats to {vecnorm_path}")


if __name__ == "__main__":
    main()
