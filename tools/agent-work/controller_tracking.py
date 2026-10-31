"""Diagnose the controller's terminal tracking error: steady-state droop or transient lag?

Compares the achieved state with the reference it was asked to track, for the zero-residual run
(no learned residual, no calibration bias). If the error is flat over time it is static droop
(integral / feedforward fixes it); if it grows towards the endpoint it is lag (gain / lead fixes it).
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
ZERO = Path(r'D:\G1Climbing\.dsh-watch\zero_residual/ref_only/eval')
CASE = 'training_13'

manifest = json.loads((BASE / 'manifest.json').read_text())
env = StrictEnv(plan=manifest['reference'], cases={CASE: manifest['evaluation_cases'][CASE]},
                normal_bias_y_deg=manifest['normal_bias_y_deg'],
                memory_enabled=manifest['force_history_memory'])
model = env.env.model
qadr = np.asarray(env.env.qadr)
plan = env.plan

with np.load(Path(manifest['reference'])) as reference:
    ref_t, ref_q = reference['time'], reference['qpos']
with np.load(ZERO / f'after_{CASE}.npz') as trace:
    t, q = trace['time_s'], trace['qpos']

# joint-space tracking error vs the requested reference
ref_interp = np.stack([np.interp(t, ref_t, ref_q[:, k]) for k in qadr], axis=1)
joint_error = np.degrees(q[:, qadr] - ref_interp)          # degrees
max_abs = np.max(np.abs(joint_error), axis=1)

# foot position error via forward kinematics
scratch = mujoco.MjData(model)
feet = [model.body(n).id for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]


def foot_positions(state):
    scratch.qpos[:] = state
    mujoco.mj_kinematics(model, scratch)
    return scratch.xpos[feet].copy()


foot_achieved = np.stack([foot_positions(s) for s in q])
foot_reference = np.stack([foot_positions(s) for s in ref_interp])
foot_error = foot_achieved - foot_reference           # metres, 3-D

windows = [('70-80 s', 70, 80), ('90-100 s', 90, 100), ('111.7-113.66 s', 111.7, 113.66)]
print('case', CASE, ' steps', len(t))
print('%-18s %10s %10s %10s %10s' % ('window', 'max_joint°', 'rms_joint°', '|L foot| mm', '|R foot| mm'))
for label, lo, hi in windows:
    mask = (t >= lo) & (t <= hi)
    print('%-18s %10.3f %10.3f %10.2f %10.2f' % (
        label, np.abs(joint_error[mask]).max(), np.sqrt((joint_error[mask] ** 2).mean()),
        1000 * np.linalg.norm(foot_error[mask, 0], axis=1).mean(),
        1000 * np.linalg.norm(foot_error[mask, 1], axis=1).mean()))

mask = t > 111.7
mean_err = 1000 * foot_error[mask].mean(axis=0)
print('\nendpoint mean foot error (mm), left / right:')
print('   left  dx=%+6.2f  dy=%+6.2f  dz=%+6.2f' % tuple(mean_err[0]))
print('   right dx=%+6.2f  dy=%+6.2f  dz=%+6.2f' % tuple(mean_err[1]))
print('   std over the last 2 s: %.2f / %.2f mm' % (
    1000 * foot_error[mask, 0].std(), 1000 * foot_error[mask, 1].std()))

# per-joint steady offset at the endpoint (which joints carry the error)
steady = joint_error[mask].mean(axis=0)
order = np.argsort(-np.abs(steady))[:6]
names = [model.joint(int(j)).name for j in env.env.joints]
print('\nlargest steady joint offsets at the endpoint (deg):')
for i in order:
    print('   %-28s %+6.3f' % (names[i], steady[i]))

slope = np.polyfit(t[mask], max_abs[mask], 1)[0]
print('\nmax joint error slope over the last 2 s: %+.4f deg/s (≈0 means steady state)' % slope)
flat = np.abs(max_abs[mask].std())
print('max joint error std over the last 2 s: %.4f deg' % flat)
print('\nverdict:', 'steady-state droop (flat error)' if flat < 0.02 else 'still varying near the endpoint')
