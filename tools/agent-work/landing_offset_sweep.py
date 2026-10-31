"""Landing-tolerance sweep: how far can the foot be off before the closed loop fails?"""
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
OUT = Path(r'D:\G1Climbing\.dsh-watch\landing_offset')
OFFSETS = [-0.5, -0.25, 0.0, 0.25, 0.5]
CASES = ['training_13']
AUTHORITY = 0.01875


def tag(value):
    return ('%+.2f' % value).replace('.', 'p').replace('+', 'p').replace('-', 'm')


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(AUTHORITY)
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / ('marker_landing_%s.txt' % settings['tag']))
    if settings['offset']:
        environment['G1_LANDING_OFFSET_DEG'] = repr(settings['offset'])
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, offset):
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
    arm.mkdir(parents=True, exist_ok=True)
    (arm / 'settings.json').write_text(json.dumps(dict(offset=offset, tag=tag(offset)), indent=2))
    return exp


def one(offset):
    arm = OUT / tag(offset)
    exp = build(arm, offset)
    result = ev.evaluate_parallel(exp, arm / 'eval', CASES, 1)
    return dict(offset=offset, trajectory=result['trajectories'][0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(OFFSETS)) as pool:
        results = list(pool.map(one, OFFSETS))

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    manifest = json.loads((BASE / 'manifest.json').read_text())
    env = StrictEnv(plan=str(PLAN), cases={'training_13': manifest['evaluation_cases']['training_13']},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = np.asarray(env.plan['qpos'][0])
    end = float(env.plan['time'][-1])

    print('\n%-12s %8s %10s %10s %-26s' % ('offset deg', '~mm', 'closure', 'rise', 'verdict'))
    baseline = None
    for result in sorted(results, key=lambda r: r['offset']):
        path = OUT / tag(result['offset']) / 'eval' / f"after_{CASES[0]}.npz"
        closure = rise = None
        if path.exists():
            with np.load(path) as trace:
                q, t = trace['qpos'], trace['time_s']
            mask = (t > end - 2) & (t <= end + 1e-7)
            closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
            rise = 1000 * float((q[mask, 2] - initial[2]).max())
        if result['offset'] == 0.0:
            baseline = closure
        row = result['trajectory']
        print('%-12s %8.1f %10s %10s %-26s %s' % (
            '%+.2f' % result['offset'], 10.9 * abs(result['offset']),
            '%.3f' % closure if closure else '—', '%.2f' % rise if rise else '—',
            'PASS' if row['success'] else str(row['reason'])[:24],
            '' if (closure is None or baseline is None) else '%+.2f' % (closure - baseline)))
    (OUT / 'result.json').write_text(json.dumps(
        [dict(offset=r['offset'], **{k: r['trajectory'][k] for k in ('success', 'reason', 'time_s')})
         for r in results], indent=2))


if __name__ == '__main__':
    main()
