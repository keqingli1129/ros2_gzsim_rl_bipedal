"""Loads biped_ppo.zip and prints what's actually recoverable from a saved
SB3 model.

IMPORTANT: most of the fields in the training-log table (ep_rew_mean,
approx_kl, explained_variance, fps, iterations, loss, n_updates, ...) are
per-iteration statistics that SB3's Logger prints to stdout during
model.learn() - they are NOT written into the saved .zip and can't be
recovered after the fact. train_biped.py doesn't pass tensorboard_log= or
configure a CSV logger, so this run's full history only ever existed in
scrollback/terminal output. To capture it for a future run, add e.g.
PPO(..., tensorboard_log=os.path.join(FILE_DIR, "tb_logs")) and view with
`tensorboard --logdir tb_logs`.

What follows is the subset of hyperparameters/state that a saved model
DOES retain, with comments on how each relates to a row in that table.
"""
import os

import numpy as np
from stable_baselines3 import PPO

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(FILE_DIR, "biped_ppo.zip")


def main():
    model = PPO.load(MODEL_PATH, device="cpu")

    # time/total_timesteps: cumulative env steps at save time. Only the
    # final value survives - not the per-iteration progression.
    print(f"total_timesteps (final): {model.num_timesteps}")

    # time/iterations: rollout-collection cycles = total_timesteps /
    # (n_steps * n_envs). Reconstructable as a count, not as a per-iteration
    # history.
    iterations = model.num_timesteps // (model.n_steps * model.n_envs)
    print(f"n_steps (per env, per rollout): {model.n_steps}")
    print(f"n_envs: {model.n_envs}")
    print(f"implied iterations completed: {iterations}")

    # train/clip_range: PPO's trust-region clipping hyperparameter (epsilon).
    # Stored as a schedule function of training progress (1.0 = start,
    # 0.0 = end) rather than a plain float; train_biped.py never schedules
    # it, so it's constant across the run.
    print(f"clip_range: {model.clip_range(1.0)}")

    # train/learning_rate: also a schedule function; constant here too since
    # train_biped.py doesn't pass a lr schedule to PPO().
    print(f"learning_rate: {model.lr_schedule(1.0)}")

    # ent_coef: entropy-bonus coefficient. train_biped.py's EntCoefDecayCallback
    # linearly decays this from 0.01 to 0.0 over training, so this is
    # whatever value it landed on at the final timestep before save() -
    # not the 0.01 starting value.
    print(f"ent_coef (final, post-decay): {model.ent_coef}")

    # Not printed in the log table shown, but the two other PPO-return
    # hyperparameters that shape value_loss/explained_variance behavior.
    print(f"gamma: {model.gamma}")
    print(f"gae_lambda: {model.gae_lambda}")

    # train/std: current action-distribution standard deviation. SB3's
    # default DiagGaussian policy stores log_std, one value per action
    # dimension (this env's 4 actuated joints: hip_L, knee_L, hip_R, knee_R).
    log_std = model.policy.log_std.detach().cpu().numpy()
    std = np.exp(log_std)
    print(f"action std per dim [hip_L, knee_L, hip_R, knee_R]: {std}")
    print(f"action std (mean, matches training log's scalar 'std'): {std.mean():.4f}")

    # NOT recoverable from the saved .zip - these existed only as stdout
    # during model.learn() for this run:
    #   rollout/ep_len_mean, rollout/ep_rew_mean
    #   time/fps, time/time_elapsed
    #   train/approx_kl, train/clip_fraction, train/entropy_loss,
    #   train/explained_variance, train/loss, train/n_updates,
    #   train/policy_gradient_loss, train/value_loss


if __name__ == "__main__":
    main()
