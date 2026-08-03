# Entropy coefficient tuning for train_biped.py — design

Date: 2026-08-02
Status: approved

## Purpose

The 1M-timestep training run (`biped_ppo.zip`, produced 2026-08-02) plateaued:
`rollout/ep_len_mean` climbed steadily early on but flattened at ~233 steps
(~1.17s) for the last several hundred logged iterations, matching what
`infer.py` shows in the live GUI (falls consistently around 1.2-1.3s).
`explained_variance` (0.853) and the collapsed action `std` (0.79 → 0.236
over the run) both indicate the policy converged confidently to this
strategy rather than still exploring.

## Root cause

`train_biped.py`'s `PPO("MlpPolicy", venv, verbose=1, device="auto")` never
sets `ent_coef`, so it defaults to SB3's `0.0` — no entropy bonus in the
loss. `train/entropy_loss` (SB3's `-mean(entropy)`, logged independent of
`ent_coef`) climbed steadily from `-5.67` to `+0.16` across the run, meaning
the Gaussian action distribution's `std` kept shrinking monotonically with
nothing discouraging it. This is a standard way for PPO to lock into a local
optimum (survive briefly, don't fall immediately) before discovering a
qualitatively different, better strategy (an actual forward gait).

## Change

Add `ent_coef=0.01` to the `PPO(...)` call in `train_biped.py`:

```python
model = PPO("MlpPolicy", venv, verbose=1, device="auto", ent_coef=0.01)
```

Nothing else changes: same `total_timesteps=1_000_000`, same reward
function (`biped_scorer.py` untouched), same network architecture (SB3's
default `MlpPolicy` sizing), same observation/action spaces.

## Scope

Per user's explicit choice: apply `ent_coef` only this round (not also the
learning-rate schedule that was considered alongside it), to isolate
whether this one change measurably affects the plateau before trying
anything else. Budget stays near 1M timesteps (user's choice — not a longer
run), since the goal right now is validating whether this lever helps at
all, not maximizing final walking quality.

## Verification

Retrain (`uv run python train_biped.py`), then compare the new run's
`rollout/ep_len_mean` trajectory against this run's (plateaued ~233 steps)
— specifically whether it plateaus later, higher, or not at all within the
same 1M-timestep budget. Then run `infer.py` to observe the resulting
behavior directly. No automated test — this is a training-quality
judgment call, not something a `verify_*.py` script asserts against.

## Out of scope

- Reward-function changes (`biped_scorer.py`'s `on_post_update`).
- Learning-rate schedule, network architecture, or any other PPO
  hyperparameter besides `ent_coef`.
- Increasing `total_timesteps` beyond 1M this round.
