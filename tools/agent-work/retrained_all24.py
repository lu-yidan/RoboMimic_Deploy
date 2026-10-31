"""Evaluate the retrained 1.25x policy on the adopted reference: two 12-case arms, 4 workers each."""
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

TRAINED = REPO / 'outputs/calibrated_geometry_ppo131072_20261029_wide125_v1'
BASELINE_AUDIT = REPO / 'outputs/swing_lift_phase_sine_policy_20261029_v2/reference_chain_audit.json'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT = Path(r'D:\G1Climbing\.dsh-watch\retrained_all24')
GAIN = 1.25
WORKERS = 4


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(0.015 * GAIN)
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / ('marker_retrained_%s.txt' % settings['arm']))
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, cases, manifest, qadr_ref):
    exp = arm / 'experiment'
    if arm.exists():
        shutil.rmtree(arm)
    exp.mkdir(parents=True)
    local = dict(manifest)
    local['reference'] = str(PLAN)
    local['reference_sha256'] = sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(local, indent=2))
    shutil.copy2(TRAINED / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(TRAINED / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))
    (arm / 'settings.json').write_text(json.dumps(dict(arm=arm.name, cases=cases), indent=2))
    return exp


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = json.loads((TRAINED / 'manifest.json').read_text())
    baseline = {r['case']: r['new_closure_mrad'] for r in
                json.loads(BASELINE_AUDIT.read_text(encoding='utf-8-sig'))['rows']}
    cases = sorted(baseline)
    halves = [cases[:len(cases) // 2], cases[len(cases) // 2:]]

    def run(args):
        arm, chunk = args
        exp = build(arm, chunk, manifest, None)
        return ev.evaluate_parallel(exp, arm / 'eval', chunk, WORKERS)['trajectories']

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [(OUT / 'arm0', halves[0]), (OUT / 'arm1', halves[1])]))
    trajectories = [row for chunk in results for row in chunk]

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    env = StrictEnv(plan=str(PLAN), cases={'training_13': manifest['evaluation_cases']['training_13']},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = np.asarray(env.plan['qpos'][0])
    end = float(env.plan['time'][-1])

    rows = []
    for row in trajectories:
        case = row['case']
        closure = None
        for arm in ('arm0', 'arm1'):
            path = OUT / arm / 'eval' / f'after_{case}.npz'
            if path.exists():
                with np.load(path) as trace:
                    q, t = trace['qpos'], trace['time_s']
                mask = (t > end - 2) & (t <= end + 1e-7)
                if mask.any():
                    closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
                break
        rows.append(dict(case=case, frozen_1x=baseline[case], closure=closure,
                         success=row['success'], reason=row['reason']))

    rows.sort(key=lambda r: (r['closure'] is None, r['closure'] or 0))
    print('\n%-13s %11s %14s %8s %-10s' % ('case', 'frozen 1.0x', 'trained 1.25x', 'delta', 'verdict'))
    for r in rows:
        delta = '' if r['closure'] is None else '%+.2f' % (r['closure'] - r['frozen_1x'])
        print('%-13s %11.3f %14s %8s %-10s' % (
            r['case'], r['frozen_1x'], '%.3f' % r['closure'] if r['closure'] else '—', delta,
            'PASS' if r['success'] else str(r['reason'])[:10]))
    done = [r['closure'] for r in rows if r['closure']]
    print('\ncompleted %d/24  mean %.3f  P95 %.3f  max %.3f' % (
        len(done), np.mean(done), np.percentile(done, 95), max(done)))
    print('frozen 1.0x : mean 38.115  P95 39.852  max 40.092')
    print('frozen 1.25x: mean 37.920  P95 38.895  max 38.943')
    print('failures:', [r['case'] for r in rows if not r['success']])
    (OUT / 'result.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
