"""Pre-hardware check 3: kinematic sensitivity of the adopted reference's swing clearance.

The reference keeps only a few millimetres of sole-to-rung clearance in mid-swing under the displaced
geometry used by the evaluation cases. A real G1 will have joint-zero calibration error (~0.1-0.5 deg)
and link-length error (~0.5-1%). This measures at what error the reference starts scraping.

Measured under the case geometry (rung offsets applied) and in mid-swing only (>=1 s away from both
contacts), because the clearance at the contacts is zero by design.
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
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
CASES = ['training_13', 'training_00', 'interior_00']
ZERO_ERRORS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
LENGTH_ERRORS = [0.0, 0.005, 0.01, 0.02]
GUARD = 5   # nodes (0.1 s) excluded at each end of the airborne window

manifest = json.loads((BASE / 'manifest.json').read_text())
with np.load(PLAN) as data:
    times = data['time'].copy()
    qpos = data['qpos'].copy()

rows = []
for case in CASES:
    env = StrictEnv(plan=str(PLAN), cases={case: manifest['evaluation_cases'][case]},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    env.reset(options={'case': case})
    physical = env.env
    model = physical.model
    qadr = np.asarray(physical.qadr)
    joint_ids = np.asarray(physical.joints)
    names = [model.joint(int(j)).name for j in joint_ids]
    leg = [i for i, n in enumerate(names) if n.startswith('left_')
           and any(k in n for k in ('hip', 'knee', 'ankle'))]
    foot = model.body('left_ankle_roll_link').id
    gname = [model.geom(g).name or '' for g in range(model.ngeom)]
    foot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot
                  and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    rung_geoms = [g for g in range(model.ngeom) if gname[g].startswith('rung_')
                  and 'arm' not in gname[g] and 'brace' not in gname[g]]
    scratch = mujoco.MjData(model)
    knee_pos = model.body('left_knee_link').pos.copy()
    ankle_pos = model.body('left_ankle_pitch_link').pos.copy()

    def clearance(state):
        scratch.qpos[:] = state
        mujoco.mj_forward(model, scratch)
        mujoco.mj_collision(model, scratch)
        return min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.25, None)
                   for g1 in foot_geoms for g2 in rung_geoms)

    series = np.array([clearance(qpos[i]) for i in range(len(times))])
    contact = series < 0.0005
    target = int(np.argmin(abs(times - 77.5)))
    left, right = target, target
    while left > 0 and not contact[left - 1]:
        left -= 1
    while right < len(times) - 1 and not contact[right + 1]:
        right += 1
    mid = range(min(left + GUARD, right), max(right - GUARD, left + GUARD) + 1)
    baseline = min(series[i] for i in mid)
    print('%-13s swing %.1f-%.1f s  mid-swing baseline clearance %.2f mm'
          % (case, times[left], times[right], 1000 * baseline))

    for degrees in ZERO_ERRORS:
        delta = np.radians(degrees)
        worst = 1e9
        for i in mid:
            state = qpos[i].copy()
            state[qadr[leg]] += delta
            worst = min(worst, clearance(state))
        rows.append(dict(case=case, kind='zero_deg', value=degrees, clearance_mm=1000 * worst))
    for eps in LENGTH_ERRORS:
        model.body_pos[model.body('left_knee_link').id] = knee_pos * (1 + eps)
        model.body_pos[model.body('left_ankle_pitch_link').id] = ankle_pos * (1 + eps)
        worst = min(clearance(qpos[i]) for i in mid)
        model.body_pos[model.body('left_knee_link').id] = knee_pos
        model.body_pos[model.body('left_ankle_pitch_link').id] = ankle_pos
        rows.append(dict(case=case, kind='length', value=eps, clearance_mm=1000 * worst))
    env.close()

print('\n%-13s %-12s %10s %14s' % ('case', 'perturbation', 'clearance', 'vs baseline'))
baseline_by_case = {}
for row in rows:
    if row['kind'] == 'zero_deg' and row['value'] == 0.0:
        baseline_by_case[row['case']] = row['clearance_mm']
for row in rows:
    label = ('%.2f deg' % row['value']) if row['kind'] == 'zero_deg' else ('%+.1f%% length' % (100 * row['value']))
    base = baseline_by_case[row['case']]
    print('%-13s %-12s %10.2f %+14.2f' % (row['case'], label, row['clearance_mm'],
                                          row['clearance_mm'] - base))
Path(r'D:\G1Climbing\.dsh-watch\prehw_kinematic_sensitivity.json').write_text(json.dumps(rows, indent=2))
print('\nwrote .dsh-watch/prehw_kinematic_sensitivity.json')
