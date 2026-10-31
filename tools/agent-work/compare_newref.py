"""Compare the old reference and the rebuilt (8 mm swing lift) reference over all 24 cases."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

WATCH = Path(r'D:\G1Climbing\.dsh-watch')
BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
NEW = WATCH / 'rebuild/plan_swing_open_v3/plan.npz'
EVAL = WATCH / 'newref_test/run/eval'

old = {r['case']: r for r in json.loads((WATCH / 'geometry_verdict.json').read_text(encoding='utf-8-sig'))['rows']}
manifest = json.loads((BASE / 'manifest.json').read_text())
env = StrictEnv(plan=str(NEW), cases={'training_13': manifest['evaluation_cases']['training_13']},
                normal_bias_y_deg=manifest['normal_bias_y_deg'],
                memory_enabled=manifest['force_history_memory'])
qadr = np.asarray(env.env.qadr)
initial = np.asarray(env.plan['qpos'][0])
end = float(env.plan['time'][-1])
reasons = {r['case']: r['reason'] for r in json.loads((EVAL / 'after.json').read_text())}

rows = []
for case, ref in sorted(old.items()):
    path = EVAL / f'after_{case}.npz'
    if not path.exists():
        rows.append(dict(case=case, missing=True))
        continue
    with np.load(path) as trace:
        q, t = trace['qpos'], trace['time_s']
    mask = (t > end - 2) & (t <= end + 1e-7)
    closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
    rise = 1000 * float((q[mask, 2] - initial[2]).max())
    reason = reasons[case]
    physical_fail = reason == 'unintended_or_swing_contact' or abs(rise - 250) > 11.0
    rows.append(dict(case=case, old_closure=ref['baseline_closure_mrad'], new_closure=closure,
                     old_rise=ref['baseline_rise_max_mm'], new_rise=rise,
                     reason=reason, physical_fail=physical_fail,
                     delta=closure - ref['baseline_closure_mrad'] if ref['baseline_closure_mrad'] else None))

print('%-13s %9s %9s %8s   %9s %9s  %s' % ('case', 'old cl', 'new cl', 'delta', 'old rise', 'new rise', 'verdict'))
for r in rows:
    if r.get('missing'):
        print('%-13s  (no trajectory)' % r['case'])
        continue
    flag = 'PHYS-FAIL' if r['physical_fail'] else ('pass' if r['reason'] == 'success' else 'closure-only')
    fmt = lambda v, spec='.3f': '—' if v is None else format(v, spec)
    print('%-13s %9s %9s %8s   %9s %9s  %s' % (
        r['case'], fmt(r['old_closure']), fmt(r['new_closure']), fmt(r['delta'], '+.3f'),
        fmt(r['old_rise']), fmt(r['new_rise']), flag))

deltas = [r['delta'] for r in rows if r.get('delta') is not None]
new_cl = [r['new_closure'] for r in rows if 'new_closure' in r]
old_cl = [r['old_closure'] for r in rows if r.get('old_closure') is not None]
print('\ncases: %d   improved: %d   worsened: %d' % (
    len(deltas), sum(1 for d in deltas if d < 0), sum(1 for d in deltas if d > 0)))
print('closure mean:  old %.3f  ->  new %.3f  (%+.3f)' % (np.mean(old_cl), np.mean(new_cl), np.mean(new_cl) - np.mean(old_cl)))
print('closure P95 :  old %.3f  ->  new %.3f' % (np.percentile(old_cl, 95), np.percentile(new_cl, 95)))
print('closure max :  old %.3f  ->  new %.3f' % (max(old_cl), max(new_cl)))
print('physical failures: %s' % [r['case'] for r in rows if r.get('physical_fail')])
print('old-gate passes: %d/24' % sum(1 for r in rows if r.get('reason') == 'success'))
(WATCH / 'newref_profile.json').write_text(json.dumps(rows, indent=2))
