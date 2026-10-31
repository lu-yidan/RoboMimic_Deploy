"""Compare swing clearance between the current reference and the rebuilt candidate, under a case geometry."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
CANDIDATE = Path(r'D:\G1Climbing\.dsh-watch\rebuild/plan_swing_open_v3/plan.npz')
CASES = ['training_13', 'training_09', 'training_03']
manifest = json.loads((BASE / 'manifest.json').read_text())
parent = Path(manifest['reference'])


def measure(plan_path, case):
    env = StrictEnv(plan=str(plan_path), cases={case: manifest['evaluation_cases'][case]},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    env.reset(options={'case': case})
    physical = env.env
    model = physical.model
    with np.load(plan_path) as data:
        times, qpos = data['time'].copy(), data['qpos'].copy()
    gname = [model.geom(g).name or '' for g in range(model.ngeom)]
    foot = model.body('left_ankle_roll_link').id
    foot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot
                  and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    rung_geoms = [g for g in range(model.ngeom) if gname[g].startswith('rung_')
                  and 'arm' not in gname[g] and 'brace' not in gname[g]]
    scratch = mujoco.MjData(model)
    series = []
    for state in qpos:
        scratch.qpos[:] = state
        mujoco.mj_forward(model, scratch)
        mujoco.mj_collision(model, scratch)
        series.append(min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.25, None)
                          for g1 in foot_geoms for g2 in rung_geoms))
    series = np.array(series)
    contact = series < 0.0005
    target = int(np.argmin(abs(times - 77.5)))
    left = target
    while left > 0 and not contact[left - 1]:
        left -= 1
    right = target
    while right < len(times) - 1 and not contact[right + 1]:
        right += 1
    window = series[left:right + 1]
    return dict(case=case, gap=(float(times[left]), float(times[right])),
                at_77p5_mm=1000 * float(series[target]),
                swing_min_mm=1000 * float(window[:-3].min()) if len(window) > 3 else None,
                swing_median_mm=1000 * float(np.median(window)))


for case in CASES:
    before = measure(parent, case)
    after = measure(CANDIDATE, case)
    print('%-13s  at 77.5 s: %6.2f -> %6.2f mm   swing min: %6.2f -> %6.2f mm   median: %6.2f -> %6.2f'
          % (case, before['at_77p5_mm'], after['at_77p5_mm'],
             before['swing_min_mm'], after['swing_min_mm'],
             before['swing_median_mm'], after['swing_median_mm']))
