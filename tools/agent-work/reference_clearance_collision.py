"""Collision-only clearance of the left sole to rung_01 along the reference (per case).

Excludes the visual mesh (contype=0/conaffinity=0). Reports the clearance at the moment the physical
rollouts died (77.5675 s) and the minimum over the swing window before the landing contact.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(ROOT))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv

BASE = ROOT / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
OUT = Path(r'D:\G1Climbing\.dsh-watch')
manifest = json.loads((BASE / 'manifest.json').read_text())
cases = {k: v for k, v in manifest['evaluation_cases'].items() if k.startswith('training_')}
after = {row['case']: row for row in json.loads((BASE / 'after.json').read_text())}

env = StrictEnv(plan=manifest['reference'], cases=cases,
                normal_bias_y_deg=manifest['normal_bias_y_deg'],
                memory_enabled=manifest['force_history_memory'])
model, data, plan = env.env.model, env.env.data, env.plan
geom_name = [model.geom(g).name or '' for g in range(model.ngeom)]
left = model.body('left_ankle_roll_link').id
left_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == left
              and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
rung_geoms = [g for g in range(model.ngeom)
              if geom_name[g].startswith('rung_')
              and 'arm' not in geom_name[g] and 'brace' not in geom_name[g]]

times = plan['time']
window = np.flatnonzero((times >= 70.0) & (times <= 81.0))
probe = int(np.argmin(abs(times - 77.5675)))
rows = []
for case in cases:
    env.reset(seed=20261022, options={'case': case})
    per_sample = []
    at_7757 = None
    for i in window:
        data.qpos[:] = plan['qpos'][i]
        mujoco.mj_forward(model, data)
        d = min(mujoco.mj_geomDistance(model, data, g1, g2, 0.20, None)
                for g1 in left_geoms for g2 in rung_geoms)
        per_sample.append(float(d))
        if i == probe:
            at_7757 = float(d)
    rows.append(dict(case=case, clearance_at_77p57_mm=1000 * at_7757,
                     min_clearance_70_81_mm=1000 * min(per_sample),
                     baseline_success=bool(after[case]['success']),
                     baseline_reason=after[case]['reason']))

rows.sort(key=lambda r: r['clearance_at_77p57_mm'])
print('%-12s  clearance@77.57s  min(70-81s)  baseline' % 'case')
for r in rows:
    print('%-12s  %+9.3f mm      %+9.3f    %s' % (r['case'], r['clearance_at_77p57_mm'],
                                                  r['min_clearance_70_81_mm'],
                                                  'PASS' if r['baseline_success'] else r['baseline_reason']))
(OUT / 'reference_clearance_collision.json').write_text(json.dumps(dict(rows=rows), indent=2))
values = [r['clearance_at_77p57_mm'] for r in rows]
print('\nclearance at 77.57 s: %.2f .. %.2f mm (all %s)' % (
    min(values), max(values), 'positive' if min(values) > 0 else 'NOT all positive'))
