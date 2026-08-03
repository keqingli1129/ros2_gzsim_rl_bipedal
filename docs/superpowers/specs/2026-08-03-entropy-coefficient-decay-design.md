# Entropy coefficient decay + seeding — design

Date: 2026-08-03
Status: approved

Follow-up to `2026-08-02-entropy-coefficient-tuning-design.md`.

## What the constant `ent_coef=0.01` run showed

The 5M-timestep run (constant `ent_coef=0.01`) ended with `train/std=0.899`
— barely below a near-random policy on a `[-1,1]`-bounded action space —
and `entropy_loss=-4.66`, meaning exploration never wound down over the
full run. `infer.py`'s `model.predict(deterministic=True)` uses the mean
action, but PPO's rollout collection (and therefore what it actually
optimized) sampled from that wide stochastic distribution the whole time.
Result: the deterministic policy fell in 0.75-1.0s in the live GUI — worse
than even the original `ent_coef=0` run's 1.2-1.3s — despite this run
logging higher training-time reward. A constant `ent_coef` has no way to
let exploration taper off as training progresses.

## Change

**1. Linear ent_coef decay via callback.** SB3's `PPO` only supports a
schedule for `learning_rate`, not `ent_coef` — but `PPO.train()` reads
`self.ent_coef` fresh every update, so overwriting it via callback works
as a manual schedule:

```python
from stable_baselines3.common.callbacks import BaseCallback

class EntCoefDecayCallback(BaseCallback):
    """Linearly decays model.ent_coef from initial_value to 0 over
    training. SB3 has no built-in schedule for ent_coef (only
    learning_rate) - PPO.train() reads self.ent_coef fresh each update,
    so overwriting it here works as a manual one. A constant ent_coef
    held for the full run (see 2026-08-02's follow-up run) kept action
    std at 0.899 even at the end, decoupling the deterministic policy
    infer.py evaluates from what PPO's stochastic rollouts actually
    optimized."""

    def __init__(self, initial_value, total_timesteps):
        super().__init__()
        self.initial_value = initial_value
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        self.model.ent_coef = self.initial_value * max(0.0, 1.0 - progress)
        return True
```

**2. Seed the PPO model.** `PPO(..., seed=SEED)` seeds the policy's initial
weights and SB3's internal RNG usage, reducing (not eliminating — see
caveat below) run-to-run variance, so future comparisons between training
runs are less confounded than the 1M-vs-5M comparison was.

**Caveat, explicitly accepted:** `CustomBipedGzTrain.reset()`
(`train_biped.py`) ignores its `seed` argument, and
`BipedScorer.reset()`'s symmetry-breaking torque impulse
(`np.random.uniform`, unseeded) is untouched by this — so runs won't be
*fully* reproducible, just meaningfully less noisy than a zero-seed
comparison. Fixing that fully is out of scope for this round (would mean
threading a seed through `BipedScorer`/`CustomBipedGzTrain`, not requested).

## Wiring

```python
TOTAL_TIMESTEPS = 5_000_000
SEED = 42
...
model = PPO("MlpPolicy", venv, verbose=1, device="auto", ent_coef=0.01, seed=SEED)
model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=EntCoefDecayCallback(0.01, TOTAL_TIMESTEPS))
```

`ent_coef=0.01` passed to `PPO(...)` remains the *starting* value the
callback decays from (SB3 requires a valid initial float even though the
callback overwrites it every step).

## Scope

- Budget stays at 5,000,000 timesteps (same overnight window as before).
- No reward-function, network-architecture, or learning-rate changes.
- No seeding of the environment's own reset noise (see caveat above).

## Verification

Retrain, then compare against both prior runs:
- 1M run (`ent_coef=0`): plateaued `ep_len_mean=233`.
- 5M run (constant `ent_coef=0.01`): `ep_len_mean=378` at end, `std=0.899`
  (never decayed), deterministic `infer.py` behavior far worse than
  training metrics suggested (0.75-1.0s falls).

Success looks like: `std` decaying toward a small value by the end of
training (confirming the callback actually took effect — check
`train/std` in the logged output), and `infer.py`'s deterministic
behavior actually resembling what the final training-time `ep_len_mean`
suggested, not falling short of it the way the constant-`ent_coef` run
did.
