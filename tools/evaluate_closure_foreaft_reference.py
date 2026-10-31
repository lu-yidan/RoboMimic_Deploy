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


from training.closure_foreaft_reference import install_adaptive_reference


def evaluate_case(payload):
    manifest, checkpoint, expected_hash, case, directory = payload
    torch.set_num_threads(1)
    if sha(checkpoint) != expected_hash:
        raise ValueError('Checkpoint changed before worker loaded it')
    classes = {BASE_CONTRACT: BaseEnv, STRICT_CONTRACT: StrictEnv}
    env = classes[manifest['contract']](plan=manifest['reference'], cases={case: manifest['evaluation_cases'][case]},
        normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    if sha(env.env.bundle/'manifest.json') != manifest['bundle_manifest_sha256']:
        raise ValueError('Worker scene bundle changed')
    model = ClosureCompositePolicy(PPO.load(checkpoint, device='cpu'), manifest)
    observer = BarGeometryObservation(env, mode=manifest['geometry_mode'])
    obs, _ = observer.reset(seed=20261022, options={'case': case})
    modifier = manifest['controller_modifier']
    gain = float(np.clip((float(obs[138])-.625)/.125, 0., 1.))
    waist = install_waist_target(env, modifier['waist_bias'], geometry_gain=gain,
        ramp_seconds=modifier['waist_ramp_seconds'], record=True)
    drop = gain*modifier['body_drop_m']
    install_body_target(env, drop)
    adaptive=install_adaptive_reference(env,modifier,geometry_gain=gain,record=True)
    adaptive['set_action'](manifest['adaptive_reference_command'])
    states, velocities, times = [env.env.data.qpos.copy()], [env.env.data.qvel.copy()], [0.]
    actions, observations, torques, rewards = [], [], [], []
    started = time.perf_counter()
    while True:
        observations.append(obs.copy())
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = observer.step(action)
        states.append(env.env.data.qpos.copy())
        velocities.append(env.env.data.qvel.copy())
        times.append(float(env.env.data.time))
        actions.append(action)
        torques.append(env.last_torque.copy())
        rewards.append(reward)
        if terminated or truncated:
            break
    directory = Path(directory)
    directory.mkdir()
    path = directory/f'after_{case}.npz'
    np.savez_compressed(path, qpos=states, qvel=velocities, time_s=times, actions=actions,
                        observations=observations, torques=torques, rewards=rewards,
                        waist_reference_trace=np.asarray(waist['trace']).reshape(-1,10),
                        body_reference_trace=np.array([body_target_offset(t,drop) for t in times]),
                        controller_geometry_gain=np.array(gain),adaptive_reference_trace=np.asarray(adaptive['trace']).reshape(-1,21))
    result = dict(**info, return_sum=float(sum(rewards)), wall_seconds=time.perf_counter()-started,
                  trajectory_sha256=sha(path), controller_geometry_gain=gain, maximum_waist_offset_rad=waist['maximum_offset_rad'],adaptive_reference_command=adaptive['command'].tolist())
    (directory/'result.json').write_text(json.dumps(result, indent=2))
    env.close()
    return result



def run_case_in_fresh_process(payload):
    directory=Path(payload[-1]);case=payload[3]
    request=directory.parent/f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload,indent=2))
    log=directory.parent/f'{case}_worker.log'
    with log.open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-m','tools.evaluate_closure_foreaft_reference','--worker-payload',str(request.resolve())],
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
        worker_evaluator_sha256=sha(__file__),process_lifecycle='One fresh Python process per case; no native simulator retained between cases',source_sha256=sha(__file__), scope='Complete time-zero physics with a frozen composite teacher and four explicit constant reference commands. Original success gates; no new actor learning.')
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
