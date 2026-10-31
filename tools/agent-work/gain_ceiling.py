"""Achievable-gain ceiling: closure spread under full-authority residual noise.

Four noise seeds at sigma = 0.015 (the full residual authority), two cases, all other settings at the
adopted configuration. If the spread is ~0.4 mrad, training has already delivered the ceiling.
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
OUT = Path(r'D:\G1Climbing\.dsh-watch\gain_ceiling')
SEEDS = [1, 2, 3, 4]
SIGMA = 0.015
AUTHORITY = 0.01875
CASES = ['training_13', 'training_14']


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(AUTHORITY)
    environment['G1_NOISE_SIGMA'] = repr(SIGMA)
    environment['G1_NOISE_SEED'] = repr(settings['seed'])
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / ('marker_noise_seed%d.txt' % settings['seed']))
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, seed):
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
    (arm / 'settings.json').write_text(json.dumps(dict(seed=seed), indent=2))
    return exp


def one(seed):
    arm = OUT / ('seed%d' % seed)
    exp = build(arm, seed)
    return seed, ev.evaluate_parallel(exp, arm / 'eval', CASES, 2)['trajectories']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        results = list(pool.map(one, SEEDS))

    from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
    manifest = json.loads((BASE / 'manifest.json').read_text())
    env = StrictEnv(plan=str(PLAN), cases={'training_13': manifest['evaluation_cases']['training_13']},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    qadr = np.asarray(env.env.qadr)
    initial = np.asarray(env.plan['qpos'][0])
    end = float(env.plan['time'][-1])

    table = {}
    for seed, rows in results:
        for row in rows:
            case = row['case']
            path = OUT / ('seed%d' % seed) / 'eval' / f'after_{case}.npz'
            closure = None
            if path.exists():
                with np.load(path) as trace:
                    q, t = trace['qpos'], trace['time_s']
                mask = (t > end - 2) & (t <= end + 1e-7)
                if mask.any():
                    closure = 1000 * float(np.abs(q[mask][:, qadr] - initial[qadr]).max())
            table.setdefault(case, {})[seed] = dict(closure=closure, success=row['success'],
                                                    reason=row['reason'])

    print('\n%-13s %-30s %10s %10s %8s' % ('case', 'closures per seed', 'mean', 'std', 'range'))
    for case, per_seed in table.items():
        values = [per_seed[s]['closure'] for s in SEEDS if per_seed[s]['closure']]
        print('%-13s %-30s %10.3f %10.3f %8.3f' % (
            case, ' '.join('%.2f' % v for v in values), np.mean(values), np.std(values),
            max(values) - min(values)))
    all_values = [v['closure'] for per in table.values() for v in per.values() if v['closure']]
    print('\npooled: mean %.3f  std %.3f  (full-authority residual noise sigma=%.3f rad)' % (
        np.mean(all_values), np.std(all_values), SIGMA))
    print('reference gains: training over three runs delivered ~0.4 mrad; 1.25x authority bought ~0.85 mrad')
    (OUT / 'result.json').write_text(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
