"""T1: do the two checkpoints actually produce different actions on the same observation?

If they do, the evaluation chain discarded the difference (harness defect).
If they do not, the two runs converged to functionally equivalent actors.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
WATCH = Path(r'D:\G1Climbing\.dsh-watch')

import sys
sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO  # noqa: E402

A = REPO / 'outputs/calibrated_geometry_ppo131072_20261029_wide125_v1/policy.zip'
B = REPO / 'outputs/calibrated_geometry_ppo131072_20261029_oracle_v1/policy.zip'

pol_a = PPO.load(A, device='cpu')
pol_b = PPO.load(B, device='cpu')


def weights(policy):
    return policy.policy.state_dict()


wa, wb = weights(pol_a), weights(pol_b)
diffs = {}
for key in wa:
    if key in wb:
        diffs[key] = float(np.abs(wa[key].numpy() - wb[key].numpy()).max())
print('actor/critic weight differences (max abs):')
for key in sorted(diffs, key=lambda k: -diffs[k])[:8]:
    print('  %-40s %.6e' % (key, diffs[key]))

# same observation -> same action?
traj = WATCH / 'retrained_all24/arm1/eval/after_training_13.npz'
with np.load(traj) as data:
    obs = data['observations']
print('\nobservation shape:', obs.shape)
picks = [0, 500, 2000, 4000, 5500]
print('%-8s %12s %12s %12s %s' % ('step', 'max|da|', 'a_sample', 'b_sample', 'identical'))
for i in picks:
    if i >= obs.shape[0]:
        continue
    row = obs[i].astype(np.float32)
    a, _ = pol_a.predict(row, deterministic=True)
    b, _ = pol_b.predict(row, deterministic=True)
    d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
    print('%-8d %12.3e %12.5f %12.5f %s' % (i, d, float(np.asarray(a).max()), float(np.asarray(b).max()),
                                            np.array_equal(np.asarray(a), np.asarray(b))))

# how much do the 12 geometry features vary inside this trajectory?
geom = obs[:, 136:148]
print('\ngeometry features over the trajectory: min %.5f max %.5f std %.5f' % (
    geom.min(), geom.max(), geom.std()))
print('geometry features at t=0:', np.round(obs[0, 136:148], 5))
print('geometry features at t=-1:', np.round(obs[-1, 136:148], 5))
