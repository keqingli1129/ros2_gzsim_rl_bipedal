"""Exercise GzCartPoleScorer's step/reset against the real generated world.

The original version only ran 20 steps (100ms) of action=1. That window was
entirely inside the ~590ms free-fall the old z=2 spawn pose produced, so it
asserted nothing about grounded behaviour. The idle check below deliberately
runs 300 steps (1.5s) - long past that former free-fall window - because
under the old geometry the pole's mispositioned collision cylinder speared
the ground on landing and slammed pole_joint to its limit, terminating the
episode. On a correctly grounded world, an untouched cart-pole just stands
there.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gz_scorer import GzCartPoleScorer

IDLE_STEPS = 300  # 1.5s at 5ms/step

scorer = GzCartPoleScorer()
obs, _info = scorer.reset()
assert obs.shape == (4,)
assert all(abs(v) < 1e-3 for v in obs), \
    f"env should reset to a motionless upright state, got {obs}"

# --- idle: no force at all, for far longer than the old free-fall window ---
for i in range(IDLE_STEPS):
    obs, _reward, terminated, _truncated, _info = scorer.step(None)
    assert not terminated, (
        f"episode terminated at idle step {i + 1} ({(i + 1) * 5}ms) with "
        f"obs={obs} - an unactuated cart-pole standing on the ground should "
        "never fall over on its own; this means the model is falling, "
        "sinking, or its collision geometry is fouling the ground plane"
    )
idle_obs = obs
assert all(abs(v) < 1e-2 for v in idle_obs), \
    f"state drifted while idle for {IDLE_STEPS} steps: {idle_obs}"

# --- actuated: push right and confirm the cart actually moves right ---
scorer.reset()
for _ in range(20):
    obs, reward, terminated, truncated, _info = scorer.step(1)
    assert not terminated, "should not fall over in 20 steps of 5ms each"

assert obs[1] > 0.5, \
    f"cart should be moving right briskly after 100ms at max effort, got {obs[1]}"
scorer.close()
print(
    f"PASS: idle for {IDLE_STEPS} steps stayed grounded and upright "
    f"(obs={idle_obs}); after 20 steps of action=1, cart_vel={obs[1]:.3f}"
)
