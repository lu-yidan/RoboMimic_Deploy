"""Generic 24-case evaluation for a trained run at 1.25x authority.

usage: eval_all24.py <run directory> <output directory> <label>
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

TRAINED = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LABEL = sys.argv[3] if len(sys.argv) > 3 else 'run'
BASELINE_AUDIT = REPO / 'outputs/swing_lift_phase_sine_policy_20261029_v2/reference_chain_audit.json'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
GAIN = 1.25
WORKERS = 4


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / (case + '_worker_payload.json')
    request.write_text(json.dumps(payload), encoding='utf-8')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(WIDE) + ';' + str(REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(0.015 * GAIN)
    if os.environ.get('G1_ACTION_MASK'):
        environment['G1_ACTION_MASK'] = os.environ['G1_ACTION_MASK']
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / ('marker_' + LABEL + '_' + settings['arm'] + '.txt'))
    log = directory.parent / (case + '_worker.log')
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True,
                       cwd=str(REPO), env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, manifest):
    exp = arm / 'experiment'
    if arm.exists():
        shutil.rmtree(arm)
    exp.mkdir(parents=True)
    local = dict(manifest)
    local['reference'] = str(PLAN)
    local['reference_sha256'] = sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(local), encoding='utf-8')
    shutil.copy2(TRAINED / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(TRAINED / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}), encoding='utf-8')
    (arm / 'settings.json').write_text(json.dumps({'arm': arm.name}), encoding='utf-8')
    return exp


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = json.loads((TRAINED / 'manifest.json').read_text())
    baseline = {}
    for row in json.loads(BASELINE_AUDIT.read_text(encoding='utf-8-sig'))['rows']:
        baseline[row['case']] = row['new_closure_mrad']
    blind = {}
    for row in json.loads(Path(r'D:\G1Climbing\.dsh-watch\retrained_all24\result.json').read_text()):
        blind[row['case']] = row['closure']
    cases = sorted(baseline)
    half = len(cases) // 2
    chunks = [cases[:half], cases[half:]]

    def run(args):
        arm, chunk = args
        exp = build(arm, manifest)
        return ev.evaluate_parallel(exp, arm / 'eval', chunk, WORKERS)['trajectories']

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [(OUT / 'arm0', chunks[0]), (OUT / 'arm1', chunks[1])]))
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
            path = OUT / arm / 'eval' / ('after_' + case + '.npz')
            if path.exists():
                with np.load(path) as trace:
                    q, t = trace['qpos'], trace['time_s']
                mask = (t > end - 2) & (t <= end + 1e-7)
                if mask.any():
                    closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
                break
        rows.append(dict(case=case, frozen_1x=baseline[case], blind_125=blind.get(case),
                         test=closure, success=row['success'], reason=row['reason']))

    rows.sort(key=lambda r: (r['test'] is None, r['test'] or 0))
    header = '[%s] %-13s %11s %11s %11s %8s %s'
    print(header % (LABEL, 'case', 'frozen 1.0x', 'blind 1.25x', LABEL, 'dvs blind', 'verdict'))
    for r in rows:
        delta = '' if (r['test'] is None or r['blind_125'] is None) else '%+.2f' % (r['test'] - r['blind_125'])
        print(header % (LABEL, r['case'], round(r['frozen_1x'], 3),
                        round(r['blind_125'], 3) if r['blind_125'] else '-',
                        round(r['test'], 3) if r['test'] else '-', delta,
                        'PASS' if r['success'] else str(r['reason'])[:10]))
    done = [r['test'] for r in rows if r['test']]
    print('')
    print('[%s] completed %d/24  mean %.3f  P95 %.3f  max %.3f' % (
        LABEL, len(done), np.mean(done), np.percentile(done, 95), max(done)))
    print('[%s] blind 1.25x reference: mean 37.905  P95 38.800  max 38.844' % LABEL)
    print('[%s] failures: %s' % (LABEL, [r['case'] for r in rows if not r['success']]))
    (OUT / 'result.json').write_text(json.dumps(rows), encoding='utf-8')


if __name__ == '__main__':
    main()
