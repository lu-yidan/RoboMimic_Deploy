"""Pre-hardware check 2: how much does the frozen policy depend on ideal support forces?

Four arms on two cases, at the adopted 1.25x authority:
    nominal (forces visible) | zero | noise (std 0.15) | stale (held from episode start)
A large drop for zero/noise means the policy leans on ground-truth force feedback that hardware will
not provide, and it must be retrained with realistic sensing before December.
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
OUT = Path(r'D:\G1Climbing\.dsh-watch\force_ablation')
MODES = [('nominal', {}), ('zero', {'G1_FORCE_MODE': 'zero'}),
         ('noise', {'G1_FORCE_MODE': 'noise'}), ('stale', {'G1_FORCE_MODE': 'drop'})]
CASES = ['training_13', 'training_00']
AUTHORITY = 0.01875


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(AUTHORITY)
    environment['G1_RESIDUAL_MARKER'] = str(settings['marker'])
    environment.update(settings['extra'])
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, name, extra):
    exp = arm / 'experiment'
    if arm.exists():
        shutil.rmtree(arm)
    exp.mkdir(parents=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest['reference'] = str(PLAN)
    manifest['reference_sha256'] = sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))
    marker = WIDE / ('marker_%s.txt' % name)
    if marker.exists():
        marker.unlink()
    (arm / 'settings.json').write_text(json.dumps(dict(name=name, extra=extra, marker=str(marker)), indent=2))
    return exp


def one(item):
    name, extra = item
    arm = OUT / name
    exp = build(arm, name, extra)
    result = ev.evaluate_parallel(exp, arm / 'eval', CASES, 2)
    return dict(name=name, trajectories=result['trajectories'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(MODES)) as pool:
        results = list(pool.map(one, MODES))

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    manifest = json.loads((BASE / 'manifest.json').read_text())
    env = StrictEnv(plan=str(PLAN), cases={'training_13': manifest['evaluation_cases']['training_13']},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = np.asarray(env.plan['qpos'][0])
    end = float(env.plan['time'][-1])

    summary = {}
    for result in results:
        for row in result['trajectories']:
            case = row['case']
            path = OUT / result['name'] / 'eval' / f'after_{case}.npz'
            closure = None
            if path.exists():
                with np.load(path) as trace:
                    q, t = trace['qpos'], trace['time_s']
                mask = (t > end - 2) & (t <= end + 1e-7)
                closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
            summary.setdefault(case, {})[result['name']] = dict(
                success=row['success'], reason=row['reason'], closure=closure)

    print('\n%-13s %-8s %10s %-26s' % ('case', 'forces', 'closure', 'verdict'))
    for case in CASES:
        base = summary[case]['nominal']['closure']
        for name, _ in MODES:
            entry = summary[case][name]
            delta = '' if (entry['closure'] is None or base is None) else '%+.2f' % (entry['closure'] - base)
            print('%-13s %-8s %10s %-26s %s' % (
                case, name,
                '%.2f' % entry['closure'] if entry['closure'] else '—',
                'PASS' if entry['success'] else str(entry['reason'])[:24], delta))
        print()
    (OUT / 'result.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
