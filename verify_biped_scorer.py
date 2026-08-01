"""Exercise BipedScorer's step/reset against the real biped.sdf world.

Mirrors verify_scorer.py's structure for the cart-pole precedent, adapted
for the biped's genuinely different stability profile: a short idle
window (not an indefinite one) is the right check here, since standing
unactuated is an inherently unstable equilibrium for this two-legged,
narrow-footed design (measured separately: falls over within 2-3s with
zero control) - unlike cart-pole's naturally-stable base.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from biped_scorer import BipedScorer

IDLE_STEPS = 100  # 500ms at 5ms/step - well inside the measured ~1s stable window
ZERO_ACTION = np.zeros(4, dtype=np.float32)
HIP_L_PUSH = np.array([60.0, 0.0, 0.0, 0.0], dtype=np.float32)

scorer = BipedScorer()
obs, _info = scorer.reset()
assert obs.shape == (13,)
assert all(abs(v) < 1e-3 for v in obs), \
    f"env should reset to a motionless, upright, standing state, got {obs}"

# --- idle: no torque at all, for the measured-stable window ---
for i in range(IDLE_STEPS):
    obs, _reward, terminated, _truncated, _info = scorer.step(ZERO_ACTION)
    assert not terminated, (
        f"episode terminated at idle step {i + 1} ({(i + 1) * 5}ms) with "
        f"obs={obs} - the biped should stay upright for at least "
        f"{IDLE_STEPS * 5}ms unactuated (measured stable window is ~1s); "
        "this means the standing pose itself regressed"
    )
idle_obs = obs
assert abs(idle_obs[1]) < 0.1 and abs(idle_obs[3]) < 0.1, (
    f"state drifted more than expected while idle for {IDLE_STEPS} steps: "
    f"torso_z_pos={idle_obs[1]:.4f}, torso_pitch={idle_obs[3]:.4f}"
)

# --- actuated: push left hip forward and confirm it actually moves ---
scorer.reset()
for _ in range(20):
    obs, reward, terminated, truncated, _info = scorer.step(HIP_L_PUSH)
    assert not terminated, "should not fall over in 20 steps of 5ms each"

assert obs[5] > 0.3, \
    f"hip_L should have swung forward briskly after 100ms at max torque, got {obs[5]}"
scorer.close()
print(
    f"PASS: idle for {IDLE_STEPS} steps stayed upright "
    f"(torso_z_pos={idle_obs[1]:+.4f}, torso_pitch={idle_obs[3]:+.4f}); "
    f"after 20 steps of hip_L torque, hip_L_pos={obs[5]:.3f}"
)
