"""Repeat an audited one-cycle actor on a continuous, audited multi-cycle plan.

Only remaining-time input is converted from whole-episode to cycle-local time.
There are no physics resets, force assists or state replacements at the seam.
The first cycle must exactly reproduce the parent's deterministic trajectory.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from deploy_mujoco.run_trained_full_cycle import load_verified, sha
from training.cycle_clock import cycle_index
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints


def cycle_local_observation(observation, time_s, cycle_ends):
    ends = np.asarray(cycle_ends, dtype=float)
    if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(np.r_[0., ends]) <= 0):
        raise ValueError('Increasing positive cycle boundaries required')
    if np.asarray(observation).shape != (136,):
        raise ValueError('Expected 136-dimensional observation')
    index = min(cycle_index(time_s, ends), len(ends)-1)
    start = 0. if index == 0 else ends[index-1]
    result = np.asarray(observation).copy()
    # 29+29+3+3+6+29+4 = 103; sin, cos, remaining-time occupy 103:106.
    result[105] = max(0., 1.-(time_s-start)/(ends[index]-start))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zero', action='store_true', help='Matched zero-residual baseline; no learned actions')
    args = parser.parse_args()
    torch.set_num_threads(1)
    original_env, model, manifest, original_summary = load_verified(args.experiment, 'nominal')
    original_plan = original_env.plan
    original_env.close()
    if manifest['evaluation_cases']['nominal']:
        raise ValueError('This transfer probe requires nominal geometry')
    env = FullCycleResidualEnv(plan=args.plan, cases={'nominal': {}},
        normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    if len(env.ends) < 2 or abs(env.ends[0]-original_plan['time'][-1]) > 1e-8:
        raise ValueError('At least two cycles with matching first horizon required')
    for key in ('qpos', 'time', 'phase', 'active'):
        if not np.array_equal(original_plan[key], env.plan[key][:len(original_plan[key])]):
            raise ValueError('First-cycle reference changed: '+key)
    parent_label = 'zero' if args.zero else 'after'
    original_record = next(row for row in original_summary[parent_label] if row['case'] == 'nominal')
    original_path = args.experiment/f'{parent_label}_nominal.npz'
    if not original_record['success'] or sha(original_path) != original_record['trajectory_sha256']:
        raise ValueError('Completed matching parent nominal trajectory required')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(experiment=str(args.experiment.resolve()), checkpoint_sha256=sha(args.experiment/'policy.zip'),
        parent_audit_sha256=sha(args.experiment/'independent_training_audit.json'),
        plan=str(args.plan.resolve()), plan_sha256=sha(args.plan), runner_sha256=sha(__file__),
        policy_mode='zero' if args.zero else 'learned', cycle_count=len(env.ends),
        adaptation='Only observation[105] remaining time becomes cycle-local; all other observations and actions unchanged.',
        scope='Continuous nominal repeated-cycle probe on a scripted reference. Completion and height require trajectory evidence; not retraining, randomization or autonomous climbing.')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    obs, _ = env.reset(seed=20261018, options={'case': 'nominal'})
    q, v, ts = [env.env.data.qpos.copy()], [env.env.data.qvel.copy()], [0.]
    observations, raw_observations, actions, torques, rewards = [], [], [], [], []
    start = time.perf_counter()
    while True:
        raw_observations.append(obs.copy())
        adapted = cycle_local_observation(obs, float(env.env.data.time), env.ends)
        observations.append(adapted)
        action = np.zeros(26, dtype=np.float32) if args.zero else model.predict(adapted, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        torques.append(env.last_torque.copy())
        rewards.append(reward)
        q.append(env.env.data.qpos.copy())
        v.append(env.env.data.qvel.copy())
        ts.append(float(env.env.data.time))
        if terminated or truncated:
            break
    arrays = dict(qpos=np.array(q), qvel=np.array(v), time_s=np.array(ts), observations=np.array(observations),
                  raw_observations=np.array(raw_observations), actions=np.array(actions), torques=np.array(torques), rewards=np.array(rewards))
    np.savez_compressed(args.output/'trajectory.npz', **arrays)
    with np.load(original_path) as original:
        n = len(original['actions'])
        checks = {key: bool(np.array_equal(arrays[key][:n+1], original[key])) for key in ('qpos', 'qvel', 'time_s')}
        checks.update({key: bool(np.array_equal(arrays[key][:n], original[key])) for key in ('observations', 'actions', 'torques')})
    endpoints = audit_endpoints(env.env, arrays, env.plan)
    result = dict(**info, first_cycle_exact_checks=checks, first_cycle_exact=all(checks.values()),
        independent_endpoints=endpoints, accepted=bool(info['success'] and all(checks.values()) and all(row['passed'] for row in endpoints)),
        wall_seconds=time.perf_counter()-start, trajectory_sha256=sha(args.output/'trajectory.npz'),
        reward_comparison_note='First-cycle terminal reward is absent because the repeated-cycle episode continues; rewards are not used for exact prefix comparison.')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    env.close()
    if not all(checks.values()):
        raise AssertionError('First-cycle learned trajectory did not reproduce exactly')


if __name__ == '__main__':
    main()
