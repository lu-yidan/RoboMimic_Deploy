"""Verify the interpolated failure threshold: run synthetic rung displacements at z = 19 and 20 mm.

The 24-case suite only covers z = 5.0-15.4 mm, and 16 of the 24 sit within 0.1 mm of each other. The
fitted relation predicts the 40 mrad line is crossed at z ~= 19.1 mm. This tests that prediction on cases
the suite never covered, which is also the first genuine out-of-sample geometry test in the project.
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROPOSED = Path(r'D:\G1Climbing\.dsh-watch\proposed')
WIDE = Path(r'D:\G1Climbing\.dsh-watch\wide')
REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(r'D:\G1Climbing\.dsh-watch')))

import numpy as np  # noqa: E402
import tools.evaluate_closure_action_bias_v2 as ev  # noqa: E402
from calibration_sweep import sha  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT = Path(r'D:\G1Climbing\.dsh-watch\displacement_fine')
LEVELS = [16.0, 16.5, 18.0]      # mm; 19/20 are outside the covered range
AUTHORITY = 0.01875
X_MM = -1.5                              # representative lateral offset of the training family


def case_name(z):
    return 'disp%02d' % int(round(z * 10))


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(AUTHORITY)
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / ('marker_disp_%s.txt' % settings['name']))
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, z):
    exp = arm / 'experiment'
    if arm.exists():
        shutil.rmtree(arm)
    exp.mkdir(parents=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest['reference'] = str(PLAN)
    manifest['reference_sha256'] = sha(PLAN)
    manifest['evaluation_cases'] = {case_name(z): {'rung_01': [X_MM / 1000., 0.0, z / 1000.]}}
    manifest['training_cases'] = manifest['evaluation_cases']
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))
    (arm / 'settings.json').write_text(json.dumps(dict(name=case_name(z), z_mm=z), indent=2))
    return exp


def one(z):
    arm = OUT / ('z%03d' % int(round(z * 10)))
    exp = build(arm, z)
    result = ev.evaluate_parallel(exp, arm / 'eval', [case_name(z)], 1)
    return z, result['trajectories'][0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(LEVELS)) as pool:
        results = list(pool.map(one, LEVELS))

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    manifest = json.loads((BASE / 'manifest.json').read_text())
    rows = []
    print('\n%-8s %10s %10s %-10s %s' % ('z mm', 'closure', 'rise mm', 'verdict', 'physical'))
    for z, row in sorted(results):
        arm = OUT / ('z%03d' % int(round(z * 10)))
        path = arm / 'eval' / f'after_{case_name(z)}.npz'
        closure = rise = None
        if path.exists():
            env = StrictEnv(plan=str(PLAN), cases={case_name(z): {'rung_01': [X_MM / 1000., 0., z / 1000.]}},
                            normal_bias_y_deg=manifest['normal_bias_y_deg'],
                            memory_enabled=manifest['force_history_memory'])
            qadr = np.asarray(env.env.qadr)
            initial = np.asarray(env.plan['qpos'][0])
            end = float(env.plan['time'][-1])
            with np.load(path) as trace:
                q, t = trace['qpos'], trace['time_s']
            mask = (t > end - 2) & (t <= end + 1e-7)
            if mask.any():
                closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
                rise = 1000 * float((q[mask, 2] - initial[2]).max())
        peaks = row.get('peaks') or {}
        physical = 'ok' if (row['success'] or row['reason'] == 'cycle_endpoint_failure') else row['reason']
        print('%-8.1f %10s %10s %-10s %s' % (
            z, '%.3f' % closure if closure else '—', '%.1f' % rise if rise else '—',
            'PASS' if row['success'] else str(row['reason'])[:10], physical))
        rows.append(dict(z_mm=z, closure=closure, rise=rise, success=row['success'],
                         reason=row['reason'], peaks=peaks))
    (OUT / 'result.json').write_text(json.dumps(rows, indent=2))
    print('\nwrote', OUT / 'result.json')


if __name__ == '__main__':
    main()
