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
HIP_L_PUSH = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

scorer = BipedScorer()
obs, _info = scorer.reset()
assert obs.shape == (13,)
# reset() now injects a small random torque impulse (+/-0.05 normalized,
# see BipedScorer.reset) over its first 5ms to break the otherwise
# bit-identical left/right symmetry every episode would start from, so the
# reset observation is no longer exactly zero. Measured empirically over 50
# resets: max |component| observed was ~0.078 (mean ~0.026). 0.15 gives
# ~2x headroom above that measured max while still meaningfully checking
# the robot resets to something close to upright/motionless, not literally
# exact zero and not some other, badly-perturbed state.
assert all(abs(v) < 0.15 for v in obs), \
    f"env should reset close to a motionless, upright, standing state (small " \
    f"intentional reset noise aside), got {obs}"

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
# torso_z_pos's tolerance (0.1) is unaffected by the new reset noise -
# measured max |torso_z_pos| after IDLE_STEPS idle steps across 180 resets
# was ~0.011, well inside 0.1 already. torso_pitch's tolerance needed
# widening though: reset()'s new small random torque impulse (see
# BipedScorer.reset) leaves a residual pitch rate that the biped's
# inherently unstable standing pose then amplifies over IDLE_STEPS*5ms of
# unactuated idling. Measured empirically over 180 resets: max
# |torso_pitch| observed after idling was ~0.234 rad, never terminating.
# 0.35 gives real headroom above that measured max while staying well
# inside PITCH_LIMIT (0.6 rad), so this still meaningfully catches the
# biped falling over, not just any drift at all.
assert abs(idle_obs[1]) < 0.1 and abs(idle_obs[3]) < 0.35, (
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
actuated_hip_L_pos = obs[5]

# --- termination: drive hard on one hip only until the biped actually falls,
# and check the fall penalty/latch around the step that first trips it.
# Measured empirically (see task report): max torque on hip_L alone with all
# other joints slack reliably terminates by step 25 (pitch crosses
# PITCH_LIMIT); 150 gives ~6x headroom above that so this isn't flaky.
TERMINATION_STEP_BUDGET = 150
scorer.reset()
typical_reward = None
terminated_step = None
first_terminal_reward = None
for i in range(TERMINATION_STEP_BUDGET):
    obs, reward, terminated, truncated, _info = scorer.step(HIP_L_PUSH)
    if typical_reward is None and not terminated:
        # grab a representative in-episode, pre-fall reward to compare against
        typical_reward = reward
    if terminated:
        terminated_step = i + 1
        first_terminal_reward = reward
        break

assert terminated_step is not None, (
    f"episode never terminated within {TERMINATION_STEP_BUDGET} steps of "
    "sustained max one-sided hip_L torque - HEIGHT_DROP_LIMIT/PITCH_LIMIT "
    "termination logic may be broken"
)

# FALL_PENALTY (5.0) is subtracted only on the step that first sets
# terminated=True, so that step's reward should be sharply lower than a
# typical non-terminal step from the same run. A margin of 3.0 is looser
# than the full 5.0 penalty (control-cost/velocity terms shift reward
# step-to-step too, so matching -5.0 exactly would be flaky) but still tight
# enough that a missing/miswired penalty would fail this assertion.
assert first_terminal_reward <= typical_reward - 3.0, (
    f"reward on the terminating step ({first_terminal_reward:.4f}) is not "
    f"low enough vs. a typical non-terminal step in this run "
    f"({typical_reward:.4f}) - FALL_PENALTY may not be getting applied"
)

# --- latch: terminated must still read True on the very next step ---
obs, reward, terminated, truncated, _info = scorer.step(ZERO_ACTION)
assert terminated, (
    "terminated flipped back to False on the step right after first tripping "
    "- the latch in on_post_update should hold it True once set"
)

scorer.close()
print(
    f"PASS: idle for {IDLE_STEPS} steps stayed upright "
    f"(torso_z_pos={idle_obs[1]:+.4f}, torso_pitch={idle_obs[3]:+.4f}); "
    f"after 20 steps of hip_L torque, hip_L_pos={actuated_hip_L_pos:.3f}; "
    f"terminated at step {terminated_step} with reward={first_terminal_reward:.4f} "
    f"(typical non-terminal reward {typical_reward:.4f}), and latch held true "
    "on the following step"
)
