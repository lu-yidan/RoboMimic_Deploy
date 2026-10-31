"""Re-derive every quantitative claim in the report from the raw artefacts and assert it.

Any FAIL here means the report is wrong somewhere; run after any report edit.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
WATCH = Path(r'D:\G1Climbing\.dsh-watch')
BASE = ROOT / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PROTO = ROOT / 'outputs/calibrated_geometry_curriculum_protocol_20261027_v1/protocol.json'
SEVEN = ['training_00', 'training_01', 'training_02', 'training_09',
         'training_13', 'training_14', 'training_15']

results = []


def check(name, actual, expected, tol=1e-3):
    ok = (abs(actual - expected) <= tol) if isinstance(expected, (int, float)) else (actual == expected)
    results.append((ok, name, actual, expected))


after = {r['case']: r for r in json.loads((BASE / 'after.json').read_text())}
protocol = json.loads(PROTO.read_text())
failures = sorted(k for k, v in after.items() if not v['success'])
check('baseline successes == 17', sum(v['success'] for v in after.values()), 17)
check('baseline cases == 24', len(after), 24)
check('the 7 failures are exactly the SEVEN list', failures, SEVEN)

center = protocol['center']['rung_01']
xs = {k: 1000 * (v['rung_01'][0] - center[0]) for k, v in protocol['training_cases'].items()
      if k.startswith('training_')}
zs = {k: 1000 * (v['rung_01'][2] - center[2]) for k, v in protocol['training_cases'].items()
      if k.startswith('training_')}
check('x span mm == 1.444', max(xs.values()) - min(xs.values()), 1.444, 0.002)
check('z span um == 139.3', 1000 * (max(zs.values()) - min(zs.values())), 139.3, 0.2)

feas = json.loads((WATCH / 'feasibility_summary.json').read_text())
check('feasible x window low', feas['feasible_x_window_mm'][0], -0.429, 0.002)
check('feasible x window high', feas['feasible_x_window_mm'][1], 0.454, 0.002)
check('passed new cases == 9', feas['passed'], 9)
rows = {r['case']: r for r in feas['rows']}
check('T03 closure 38.02', rows['training_03']['closure_mrad'], 38.02, 0.01)
check('T04 closure 39.76', rows['training_04']['closure_mrad'], 39.76, 0.01)
check('T11 closure 39.99', rows['training_11']['closure_mrad'], 39.99, 0.01)
check('T12 closure 38.33', rows['training_12']['closure_mrad'], 38.33, 0.01)
rise = [r['rise_mm'] for r in feas['rows'] if r['rise_mm'] is not None]
check('rise min 259.93', min(rise), 259.93, 0.02)
check('rise max 260.03', max(rise), 260.03, 0.02)

clearance = json.loads((WATCH / 'reference_clearance_collision.json').read_text())
values = [r['clearance_at_77p57_mm'] for r in clearance['rows']]
mins = [r['min_clearance_70_81_mm'] for r in clearance['rows']]
check('clearance at 77.57 s all positive', min(values) > 0, True)
check('tightest clearance +2.62 mm', min(values), 2.622, 0.01)
check('widest clearance +3.70 mm', max(values), 3.697, 0.01)
check('window minimum +1.39 mm', min(mins), 1.385, 0.01)
tight = sorted(clearance['rows'], key=lambda r: r['clearance_at_77p57_mm'])[:2]
check('two tightest are training_00/01', sorted(r['case'] for r in tight),
      ['training_00', 'training_01'])

sens = json.loads((WATCH / 'reward_episode_sensitivity.json').read_text())
current = next(r for r in sens['rows'] if 'current' in r['name'])
check('current shaping: 0.3 mrad worth 0.29', current['delta_per_0p3mrad'], 0.29, 0.005)
check('current shaping share 0.7%', 100 * current['share_of_terminal'], 0.73, 0.05)

diag = json.loads((BASE / 'failure_endpoint_diagnosis.json').read_text())['rows']
check('T02 fails on rise only (rise 260.026)', diag['training_02']['rise_max_mm'], 260.026, 0.001)
check('T02 closure is legal (39.286 < 40)', diag['training_02']['closure_mrad'], 39.286, 0.001)

print('%-46s %-22s %-22s' % ('claim', 'actual', 'expected'))
for ok, name, actual, expected in results:
    print('%-46s %-22s %-22s %s' % (name, actual, expected, 'PASS' if ok else '<<< FAIL'))
failed = [r for r in results if not r[0]]
print('\n%d checks, %d failed' % (len(results), len(failed)))
raise SystemExit(1 if failed else 0)
