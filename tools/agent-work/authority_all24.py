"""All 24 cases at 1.25x residual authority, against the audited 1.0x per-case baseline."""
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

import numpy as np  # noqa: E402
import tools.evaluate_closure_action_bias_v2 as ev  # noqa: E402
from calibration_sweep import sha  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
BASELINE_AUDIT = REPO / 'outputs/swing_lift_phase_sine_policy_20261029_v2/reference_chain_audit.json'
OUT = Path(r'D:\G1Climbing\.dsh-watch\authority_all24')
GAIN = 1.25
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
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / 'patch_marker_all24.txt')
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
    manifest['reference_sha256'] = sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))

    baseline = {r['case']: r['new_closure_mrad'] for r in
                json.loads(BASELINE_AUDIT.read_text(encoding='utf-8-sig'))['rows']}
    cases = sorted(baseline)
    marker = WIDE / 'patch_marker_all24.txt'
    if marker.exists():
        marker.unlink()
    result = ev.evaluate_parallel(exp, OUT / 'eval', cases, WORKERS)

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    env = StrictEnv(plan=str(PLAN), cases={'training_13': manifest['evaluation_cases']['training_13']},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = np.asarray(env.plan['qpos'][0])
    end = float(env.plan['time'][-1])

    rows = []
    for row in result['trajectories']:
        case = row['case']
        path = OUT / 'eval' / f'after_{case}.npz'
        with np.load(path) as trace:
            q, t = trace['qpos'], trace['time_s']
        mask = (t > end - 2) & (t <= end + 1e-7)
        closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
        rise = 1000 * float((q[mask, 2] - initial[2]).max())
        rows.append(dict(case=case, baseline=baseline[case], closure=closure,
                         delta=closure - baseline[case], rise=rise,
                         success=row['success'], reason=row['reason']))

    rows.sort(key=lambda r: r['delta'])
    print('\n%-13s %9s %9s %8s %8s %-10s' % ('case', '1.0x', '1.25x', 'delta', 'rise', 'verdict'))
    for r in rows:
        print('%-13s %9.3f %9.3f %+8.3f %8.2f %-10s' % (r['case'], r['baseline'], r['closure'],
                                                         r['delta'], r['rise'],
                                                         'PASS' if r['success'] else str(r['reason'])[:10]))
    deltas = [r['delta'] for r in rows]
    closures = [r['closure'] for r in rows]
    fails = [r['case'] for r in rows if not r['success']]
    print('\nmean 1.0x %.3f  ->  1.25x %.3f  (delta %+.3f)' % (
        np.mean([r['baseline'] for r in rows]), np.mean(closures), np.mean(deltas)))
    print('P95 %.3f -> %.3f   max %.3f -> %.3f' % (
        np.percentile([r['baseline'] for r in rows], 95), np.percentile(closures, 95),
        max(r['baseline'] for r in rows), max(closures)))
    print('improved %d / worsened %d' % (sum(1 for d in deltas if d < 0), sum(1 for d in deltas if d > 0)))
    print('failures:', fails if fails else 'none')
    print('patched clusters:', sum(1 for _ in marker.open()) if marker.exists() else 0)
    (OUT / 'result.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
