"""A1: fixed-calibration bias sweep on the audited reference.

Rebuilds the audited evaluation for scaled versions of the fixed knee/ankle action bias
(the only nonzero entries of `closure_action_bias.normalized_bias`), keeping the policy,
reference, controller and acceptance untouched. Purpose:

  * split the ~11 mrad endpoint-closure offset into "calibration contribution" and "tracking";
  * test the residual policy's leverage: the bias is itself a constant knee/ankle offset within
    the same 26D action bounds, so the spread across scales bounds what a learned policy can do.

Closure/rise are recomputed from the recorded trajectories with the same formula the audit uses
(max |q_endpoint - q_start| over the 29 joints; max root rise over the last 2 s).
scale 1.00 must reproduce the audited baseline numbers — that is the self-check.
"""
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from tools.evaluate_closure_action_bias_v2 import evaluate_parallel  # noqa: E402
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
OUT = Path(r'D:\G1Climbing\.dsh-watch\calibration_sweep')
CASES = ['training_02', 'training_09', 'training_13', 'training_14', 'training_15',
         'training_03', 'boundary_07']
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
BASELINE_CLOSURE = {'training_02': 39.286, 'training_09': 40.072, 'training_13': 40.212,
                    'training_14': 40.446, 'training_15': 40.350, 'training_03': 38.024,
                    'boundary_07': 39.898}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_frame():
    manifest = json.loads((BASE / 'manifest.json').read_text())
    env = StrictEnv(plan=manifest['reference'], cases={'nominal': {}},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = env.plan['qpos'][0]
    end = float(env.plan['time'][-1])
    return qadr, initial, end


def make_arm(scale):
    arm = OUT / f'scale_{scale:.2f}'
    exp = arm / 'experiment'
    if arm.exists():
        shutil.rmtree(arm)
    exp.mkdir(parents=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    spec = manifest['closure_action_bias']
    spec['normalized_bias'] = [float(v) * scale for v in spec['normalized_bias']]
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))
    return exp


def measure(arm, cases, qadr, initial, end):
    rows = []
    for case in cases:
        path = arm / 'eval' / f'after_{case}.npz'
        if not path.exists():
            rows.append(dict(case=case, missing=True))
            continue
        with np.load(path) as trace:
            qpos, times = trace['qpos'], trace['time_s']
            actions = trace['actions']
            mask = (times > end - 2) & (times <= end + 1e-7)
            closure = float(np.abs(qpos[mask][:, qadr] - initial[qadr]).max()) if mask.any() else float('nan')
            rise = float((qpos[mask, 2] - initial[2]).max()) if mask.any() else float('nan')
            complete = abs(float(times[-1]) - end) < 1e-6
        rows.append(dict(case=case, closure_mrad=1000 * closure, rise_mm=1000 * rise,
                         complete=bool(complete),
                         mean_abs_action=float(np.abs(actions).mean()),
                         max_abs_action=float(np.abs(actions).max())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qadr, initial, end = reference_frame()
    summary = []
    for scale in SCALES:
        print('\n===== bias scale %.2f =====' % scale, flush=True)
        try:
            exp = make_arm(scale)
            result = evaluate_parallel(exp, OUT / f'scale_{scale:.2f}' / 'eval', CASES, 4)
            success = {r['case']: bool(r['success']) for r in result['trajectories']}
            rows = measure(OUT / f'scale_{scale:.2f}', CASES, qadr, initial, end)
            for row in rows:
                row['success'] = success.get(row['case'])
            summary.append(dict(scale=scale, rows=rows,
                                successes=sum(1 for v in success.values() if v)))
            for row in rows:
                print('  %-13s success=%-5s closure=%7.3f mrad  rise=%8.3f mm  |a|=%0.4f'
                      % (row['case'], row.get('success'), row.get('closure_mrad', float('nan')),
                         row.get('rise_mm', float('nan')), row.get('mean_abs_action', float('nan'))),
                      flush=True)
        except Exception as exc:
            summary.append(dict(scale=scale, error=str(exc), trace=traceback.format_exc()[-800:]))
            print('ARM FAILED %.2f: %s' % (scale, exc), flush=True)
        (OUT / 'sweep.json').write_text(json.dumps(summary, indent=2))
    print('\nwrote', OUT / 'sweep.json')
    check = next((a for a in summary if a.get('scale') == 1.0 and a.get('rows')), None)
    if check:
        drift = {r['case']: r['closure_mrad'] - BASELINE_CLOSURE[r['case']]
                 for r in check['rows'] if r['case'] in BASELINE_CLOSURE}
        print('self-check vs audited baseline (mrad):',
              {k: round(v, 4) for k, v in drift.items()})


if __name__ == '__main__':
    main()
