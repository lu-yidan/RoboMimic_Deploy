"""Full time-zero evaluation of a 10 Hz reference actor and frozen 50 Hz teacher."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from training.full_cycle_residual_env import ROOT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation
from training.closure_composite_policy import ClosureCompositePolicy
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target, body_target_offset
from training.closure_foreaft_reference import install_adaptive_reference, SCALES
from tools.compare_audited_closure_candidates import sha

FEEDBACK_CONTRACT = 'seeded-reference-153obs-5delta-10hz-v1'


def evaluate_case(payload):
    m, checkpoint, checkpoint_hash, case, directory = payload
    torch.set_num_threads(1)
    assert sha(checkpoint) == checkpoint_hash
    assert sha(m['teacher_checkpoint']) == m['teacher_checkpoint_sha256']
    assert m['observation_contract'] == FEEDBACK_CONTRACT
    assert m['feedback_hold_steps'] == 5 and m['suffix_start_seconds'] == 99.66
    env = FullCycleResidualEnv(plan=m['reference'], cases={case: m['evaluation_cases'][case]},
        normal_bias_y_deg=m['normal_bias_y_deg'], memory_enabled=m['force_history_memory'])
    assert sha(env.env.bundle/'manifest.json') == m['bundle_manifest_sha256']
    observer = BarGeometryObservation(env, mode=m['geometry_mode'])
    feedback = PPO.load(checkpoint, device='cpu')
    assert feedback.observation_space.shape == (153,) and feedback.action_space.shape == (5,)
    teacher = ClosureCompositePolicy(PPO.load(m['teacher_checkpoint'], device='cpu'), m)
    obs, _ = observer.reset(seed=20261022, options={'case': case})
    c = m['controller_modifier']
    gain = float(np.clip((float(obs[138])-.625)/.125, 0., 1.))
    waist = install_waist_target(env, c['waist_bias'], geometry_gain=gain,
                                ramp_seconds=c['waist_ramp_seconds'], record=True)
    drop = gain*c['body_drop_m']
    install_body_target(env, drop)
    extra = install_adaptive_reference(env, c, geometry_gain=gain, record=True)
    seed = np.asarray(m['adaptive_reference_command'], dtype=float)
    extra['set_action'](seed)
    data = dict(qpos=[env.env.data.qpos.copy()], qvel=[env.env.data.qvel.copy()], time_s=[0.],
                actions=[], observations=[], torques=[], rewards=[], feedback_actions=[],
                feedback_observations=[], feedback_time_s=[], feedback_teacher_indices=[])
    boundary = round(m['suffix_start_seconds']/.02)
    started = time.perf_counter()
    try:
        while True:
            index = len(data['actions'])
            if index >= boundary and (index-boundary) % m['feedback_hold_steps'] == 0:
                feedback_obs = np.r_[obs, extra['residual']/SCALES].astype(np.float32)
                delta = feedback.predict(feedback_obs, deterministic=True)[0]
                command = np.clip(seed+m['feedback_delta_scale']*delta.astype(float), -1., 1.)
                extra['set_action'](command)
                data['feedback_observations'].append(feedback_obs.copy())
                data['feedback_actions'].append(delta.copy())
                data['feedback_time_s'].append(float(env.env.data.time))
                data['feedback_teacher_indices'].append(index)
            data['observations'].append(obs.copy())
            action = teacher.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, info = observer.step(action)
            data['actions'].append(action.copy())
            data['torques'].append(env.last_torque.copy())
            data['rewards'].append(float(reward))
            data['qpos'].append(env.env.data.qpos.copy())
            data['qvel'].append(env.env.data.qvel.copy())
            data['time_s'].append(float(env.env.data.time))
            if done or truncated:
                break
        arrays = {k: np.asarray(v) for k, v in data.items()}
        arrays['feedback_actions'] = arrays['feedback_actions'].reshape(-1, 5)
        arrays['feedback_observations'] = arrays['feedback_observations'].reshape(-1, 153)
        arrays['feedback_teacher_indices'] = arrays['feedback_teacher_indices'].astype(np.int64)
        arrays.update(waist_reference_trace=np.asarray(waist['trace']).reshape(-1, 10),
                      body_reference_trace=np.asarray([body_target_offset(t, drop) for t in data['time_s']]),
                      adaptive_reference_trace=np.asarray(extra['trace']).reshape(-1, 21),
                      controller_geometry_gain=np.array(gain))
        output = Path(directory)
        output.mkdir()
        path = output/f'after_{case}.npz'
        np.savez_compressed(path, **arrays)
        result = dict(**info, trajectory_sha256=sha(path), return_sum=float(sum(data['rewards'])),
                      wall_seconds=time.perf_counter()-started, feedback_steps=len(data['feedback_actions']),
                      controller_geometry_gain=gain, feedback_contract=FEEDBACK_CONTRACT,
                      actor_scope='Frozen 26D teacher plus learned five-dimensional reference deltas at 10 Hz')
        (output/'result.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def fresh_worker(payload):
    directory = Path(payload[-1])
    request = directory.parent/f'{payload[3]}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    with (directory.parent/f'{payload[3]}_worker.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'tools.evaluate_seeded_reference_feedback',
                        '--worker-payload', str(request.resolve())], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((directory/'result.json').read_text())


def evaluate_parallel(experiment, output, workers=2):
    experiment, output = Path(experiment), Path(output)
    m = json.loads((experiment/'manifest.json').read_text())
    s = json.loads((experiment/'summary.json').read_text())
    assert m['observation_contract'] == FEEDBACK_CONTRACT
    assert sha(experiment/'policy.zip') == s['checkpoint_sha256']
    assert sha(m['reference']) == m['reference_sha256']
    for name, digest in m['sources'].items():
        assert sha(experiment/'source_snapshot'/name) == digest
        if name.replace('\\', '/').startswith('training/'):
            assert sha(ROOT/name) == digest
    output.mkdir(exist_ok=False)
    rows = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fresh_worker, (m, str((experiment/'policy.zip').resolve()),
                s['checkpoint_sha256'], case, str((output/case).resolve()))) for case in m['evaluation_cases']]
        for job in as_completed(jobs):
            row = job.result()
            rows[row['case']] = row
            print(json.dumps({k: row[k] for k in ('case', 'success', 'reason')}), flush=True)
    ordered = [rows[case] for case in m['evaluation_cases']]
    for row in ordered:
        name = f"after_{row['case']}.npz"
        shutil.copy2(output/row['case']/name, output/name)
    result = dict(trajectories=ordered, wall_seconds=time.perf_counter()-started,
                  workers=workers, source_sha256=sha(__file__))
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    (output/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    return result


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker-payload':
        evaluate_case(json.loads(Path(sys.argv[2]).read_text()))
    else:
        p = argparse.ArgumentParser()
        p.add_argument('--experiment', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--workers', type=int, default=2, choices=(1, 2))
        a = p.parse_args()
        evaluate_parallel(a.experiment, a.output, a.workers)
