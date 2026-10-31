"""Confirm the usable authority gain (1.25x) on four cases against their 1.0x baselines."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROPOSED = Path(r'D:\G1Climbing\.dsh-watch\proposed')
WIDE = Path(r'D:\G1Climbing\.dsh-watch\wide')
REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(r'D:\G1Climbing\.dsh-watch')))

import tools.evaluate_closure_action_bias_v2 as ev  # noqa: E402
import numpy as np  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT = Path(r'D:\G1Climbing\.dsh-watch\authority_confirm')
GAIN = 1.25
CASES = ['training_13', 'training_14', 'training_00', 'interior_00']
BASELINE_1X = {'training_13': 39.866, 'training_14': 40.092, 'training_00': 39.905, 'interior_00': 34.700}
WORKERS = 4


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    log = directory.parent / f'{case}_worker.log'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(0.015 * GAIN)
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / 'patch_marker_confirm.txt')
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def main():
    exp = OUT / 'experiment'
    if OUT.exists():
        shutil.rmtree(OUT)
    exp.mkdir(parents=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest['reference'] = str(PLAN)
    manifest['reference_sha256'] = __import__('calibration_sweep').sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': __import__('calibration_sweep').sha(exp / 'policy.zip'),
         'evaluation_pending': True}, indent=2))
    marker = WIDE / 'patch_marker_confirm.txt'
    if marker.exists():
        marker.unlink()
    result = ev.evaluate_parallel(exp, OUT / 'eval', CASES, WORKERS)

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    env = StrictEnv(plan=str(PLAN), cases={'training_13': manifest['evaluation_cases']['training_13']},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = np.asarray(env.plan['qpos'][0])
    end = float(env.plan['time'][-1])

    print('\n%-13s %10s %10s %8s %-8s' % ('case', '1.0x', '1.25x', 'delta', 'verdict'))
    deltas = []
    for row in result['trajectories']:
        case = row['case']
        with np.load(OUT / 'eval' / f'after_{case}.npz') as trace:
            q, t = trace['qpos'], trace['time_s']
        mask = (t > end - 2) & (t <= end + 1e-7)
        closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
        delta = closure - BASELINE_1X[case]
        deltas.append(delta)
        print('%-13s %10.3f %10.3f %+8.3f %-8s' % (case, BASELINE_1X[case], closure, delta,
                                                    'PASS' if row['success'] else row['reason'][:8]))
    print('\nmean delta over %d cases: %+.3f mrad' % (len(deltas), float(np.mean(deltas))))
    print('clusters patched:', sum(1 for _ in marker.open()) if marker.exists() else 0)


if __name__ == '__main__':
    main()
