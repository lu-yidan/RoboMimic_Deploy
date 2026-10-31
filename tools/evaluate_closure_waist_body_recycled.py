"""Independent processes for deterministic policy evaluation and exact replay checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_composite_policy import ClosureCompositePolicy
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target, body_target_offset
from training.bar_geometry_observation import BarGeometryObservation, OBSERVATION_CONTRACT
from training.full_cycle_residual_env import FullCycleResidualEnv as BaseEnv, ROOT, CONTRACT as BASE_CONTRACT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv, CONTRACT as STRICT_CONTRACT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from tools.evaluate_closure_waist_body import evaluate_case


def run_case_in_fresh_process(payload):
    directory=Path(payload[-1]);case=payload[3]
    request=directory.parent/f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload,indent=2))
    log=directory.parent/f'{case}_worker.log'
    with log.open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-m','tools.evaluate_closure_waist_body_recycled','--worker-payload',str(request.resolve())],
            stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    return json.loads((directory/'result.json').read_text())


def evaluate_parallel(experiment, output, cases, workers, verify_recorded=False):
    experiment, output = Path(experiment), Path(output)
    manifest = json.loads((experiment/'manifest.json').read_text())
    summary = json.loads((experiment/'summary.json').read_text())
    checkpoint = experiment/'policy.zip'
    if manifest['observation_contract'] != OBSERVATION_CONTRACT:
        raise ValueError('Expected148-input geometry contract')
    if manifest['contract'] not in (BASE_CONTRACT, STRICT_CONTRACT):
        raise ValueError('Unknown environment contract')
    if sha(checkpoint) != summary['checkpoint_sha256'] or sha(manifest['reference']) != manifest['reference_sha256']:
        raise ValueError('Experiment checkpoint or reference changed')
    for relative, expected in manifest['sources'].items():
        if relative.replace('\\', '/').startswith('training/'):
            if sha(experiment/'source_snapshot'/relative) != expected or sha(ROOT/relative) != expected:
                raise ValueError('Controller/environment source changed: '+relative)
    cases = list(cases or manifest['evaluation_cases'])
    if len(set(cases)) != len(cases) or not set(cases) <= set(manifest['evaluation_cases']):
        raise ValueError('Select distinct declared cases')
    if not 1 <= workers <= 4:
        raise ValueError('Use 1..4 independent evaluation workers')
    output.mkdir(parents=True, exist_ok=False)
    payloads = [(manifest, str(checkpoint.resolve()), summary['checkpoint_sha256'], case, str((output/case).resolve())) for case in cases]
    results = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case_in_fresh_process, payload) for payload in payloads]
        for future in as_completed(futures):
            row = future.result()
            results[row['case']] = row
            print(json.dumps(dict(case=row['case'], success=row['success'], reason=row['reason'])), flush=True)
    elapsed = time.perf_counter()-started
    comparisons = []
    for case in cases:
        produced = output/case/f'after_{case}.npz'
        shutil.copy2(produced, output/produced.name)
        if verify_recorded:
            original = experiment/f'after_{case}.npz'
            previous = next(row for row in summary['after'] if row['case'] == case)
            if sha(original) != previous['trajectory_sha256']:
                raise ValueError('Original trajectory hash changed')
            checks, differences = {}, {}
            with np.load(original) as a, np.load(produced) as b:
                for key in ('qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards'):
                    equal_shape = a[key].shape == b[key].shape
                    checks[key] = bool(equal_shape and np.array_equal(a[key], b[key]))
                    differences[key] = float(abs(a[key]-b[key]).max()) if equal_shape else None
            comparisons.append(dict(case=case, passed=all(checks.values()), checks=checks, maximum_differences=differences,
                                    original_sha256=sha(original), parallel_sha256=sha(produced)))
    ordered = [results[case] for case in cases]
    (output/'after.json').write_text(json.dumps(ordered, indent=2))
    result = dict(parent_experiment=str(experiment.resolve()), checkpoint_sha256=summary['checkpoint_sha256'],
        parent_manifest_sha256=sha(experiment/'manifest.json'), workers=workers, cases=cases,
        wall_seconds=elapsed, trajectories=ordered, comparisons=comparisons,
        exact_reproduction_passed=all(r['passed'] for r in comparisons) if verify_recorded else None,
        worker_evaluator_sha256=sha('tools/evaluate_closure_waist_body.py'),process_lifecycle='One fresh Python process per case; no native simulator retained between cases',source_sha256=sha(__file__), scope='Complete time-zero physics with explicit waist and body reference additions. Original success gates. Search-initialized actor, not a PPO improvement.')
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    (output/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    if verify_recorded and not result['exact_reproduction_passed']:
        raise AssertionError('Parallel evaluation differs from recorded serial evaluation')
    return result


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker-payload':
        evaluate_case(json.loads(Path(sys.argv[2]).read_text()))
        raise SystemExit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--verify-recorded', action='store_true')
    args = parser.parse_args()
    evaluate_parallel(args.experiment, args.output, args.cases, args.workers, args.verify_recorded)
