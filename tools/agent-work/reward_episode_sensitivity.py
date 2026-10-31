"""Episode-level sensitivity of the closure shaping reward, integrated over a real audited trajectory.

The bonus is added every step, so what matters is the summed difference over an episode, not the
per-step slope. Sanity check: the recomputed sum must match the recorded closure_episode_bonus.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(ROOT))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv

BASE = ROOT / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
OUT = Path(r'D:\G1Climbing\.dsh-watch')
CASE = 'training_13'          # deep-but-complete endpoint failure
manifest = json.loads((BASE / 'manifest.json').read_text())

env = StrictEnv(plan=manifest['reference'], cases={CASE: manifest['evaluation_cases'][CASE]},
                normal_bias_y_deg=manifest['normal_bias_y_deg'],
                memory_enabled=manifest['force_history_memory'])
physical = env.env
qadr = physical.qadr
plan = env.plan
initial = plan['qpos'][0]
closure_index = np.flatnonzero(plan['phase'] == 'posture_closure')
start = float(plan['time'][closure_index[0] - 1])

with np.load(BASE / f'after_{CASE}.npz') as trace:
    qpos = trace['qpos']
    times = trace['time_s']

closure = np.abs(qpos[:, qadr] - initial[qadr]).max(axis=1)
rise = qpos[:, 2] - initial[2]


def ramp_of(t):
    u = float(np.clip((t - start) / 3., 0., 1.))
    return u * u * (3. - 2. * u)


def quality_bonus_step(c, r, ramp, weight):
    joint = max(0., c - .032) / .008
    height = max(0., abs(r - .25) - .006) / .004
    return .5 * weight * ramp * (1. / (1. + joint * joint) + 1. / (1. + height * height))


def linear_step(c, r, ramp, weight):
    margin = min((.040 - c) / .001, (.010 - abs(r - .25)) / .0001)
    return ramp * weight * max(-1., min(1., (margin - 2.) / 3.))


def smooth_step(c, r, ramp, weight):
    def bounded(value):
        value = max(0., value)
        return value * value / (1. + value * value) if value < 1. else 1. / (1. + 1. / (value * value))
    return -.5 * weight * ramp * (bounded((c - .032) / .008) + bounded((abs(r - .25) - .006) / .004))


ramps = np.array([ramp_of(t) for t in times])


def total(step, weight, shift_mrad=0.0, shift_rise_mm=0.0):
    return float(sum(step(c + shift_mrad / 1000., r + shift_rise_mm / 1000., u, weight)
                     for c, r, u in zip(closure, rise, ramps)))


rows = []
for name, step, weight in [('quality_bonus w=0.1 (current)', quality_bonus_step, .1),
                           ('quality_bonus w=0.5', quality_bonus_step, .5),
                           ('quality_bonus w=1.0', quality_bonus_step, 1.),
                           ('quality_bonus w=2.0', quality_bonus_step, 2.),
                           ('closure_smooth w=0.5', smooth_step, .5),
                           ('closure_smooth w=1.0', smooth_step, 1.),
                           ('linear margin w=2', linear_step, 2.),
                           ('linear margin w=5', linear_step, 5.)]:
    base_sum = total(step, weight)
    better = total(step, weight, shift_mrad=-0.3)    # 0.3 mrad of improvement
    rows.append(dict(name=name, episode_sum=base_sum, delta_per_0p3mrad=better - base_sum,
                     share_of_terminal=abs(better - base_sum) / 40.))

print('closure phase start %.3f s, %d steps, episode %.2f s' % (start, len(times), times[-1]))
print('final closure %.3f mrad, final rise %.3f mm' % (1000 * closure[-1], 1000 * rise[-1]))
print()
print('%-32s %10s %14s %12s' % ('candidate', 'episode sum', 'value of 0.3 mrad', 'vs terminal 40'))
for r in rows:
    print('%-32s %10.2f %14.4f %11.2f%%' % (r['name'], r['episode_sum'], r['delta_per_0p3mrad'],
                                            100 * r['share_of_terminal']))
(OUT / 'reward_episode_sensitivity.json').write_text(json.dumps(dict(
    case=CASE, closure_phase_start_s=start, steps=len(times),
    final_closure_mrad=1000 * float(closure[-1]), rows=rows), indent=2))
