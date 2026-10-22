"""Run an audited full-cycle residual checkpoint in MuJoCo, with no robot link.

This simulation entry point deliberately keeps the new 136D/26D policy separate
from the original hardware deployment action/observation contract.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from training.full_cycle_residual_env import FullCycleResidualEnv as BaseEnv, ROOT, CONTRACT as BASE_CONTRACT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv, CONTRACT as STRICT_CONTRACT
from tools.audit_multirung_cycle import audit_endpoints


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified(experiment, case):
    manifest = json.loads((experiment/'manifest.json').read_text())
    summary = json.loads((experiment/'summary.json').read_text())
    audit = json.loads((experiment/'independent_training_audit.json').read_text())
    if not audit['passed']:
        raise ValueError('Checkpoint experiment integrity audit must pass first')
    if (audit['experiment_manifest_sha256'] != sha(experiment/'manifest.json')
            or audit['experiment_summary_sha256'] != sha(experiment/'summary.json')):
        raise ValueError('Experiment metadata changed after the independent audit')
    if manifest['contract'] not in (BASE_CONTRACT, STRICT_CONTRACT):
        raise ValueError('Unsupported policy contract')
    if case not in manifest['evaluation_cases']:
        raise ValueError('Choose a declared evaluation geometry')
    policy = experiment/'policy.zip'
    if sha(policy) != summary['checkpoint_sha256'] or sha(policy) != audit['checkpoint_sha256']:
        raise ValueError('Policy checkpoint hash mismatch')
    if sha(manifest['reference']) != manifest['reference_sha256']:
        raise ValueError('Reference hash mismatch')
    for relative, expected in manifest['sources'].items():
        if sha(experiment/'source_snapshot'/relative) != expected:
            raise ValueError('Frozen experiment source changed: '+relative)
        if relative.replace('\\', '/').startswith('training/') and sha(ROOT/relative) != expected:
            raise ValueError('Current environment differs from trained environment: '+relative)
    cls = StrictEnv if manifest['contract'] == STRICT_CONTRACT else BaseEnv
    env = cls(plan=manifest['reference'], cases={case: manifest['evaluation_cases'][case]},
              normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    if sha(env.env.bundle/'manifest.json') != manifest['bundle_manifest_sha256']:
        env.close()
        raise ValueError('Scene bundle differs from trained environment')
    model = PPO.load(policy, device='cpu')
    if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
        env.close()
        raise ValueError('Policy/environment dimension mismatch')
    return env, model, manifest, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--viewer', action='store_true', help='Open a passive MuJoCo simulation viewer')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    env, model, manifest, training_summary = load_verified(args.experiment, args.case)
    args.output.mkdir(parents=True)
    provenance = dict(experiment=str(args.experiment.resolve()), checkpoint_sha256=training_summary['checkpoint_sha256'],
        experiment_manifest_sha256=sha(args.experiment/'manifest.json'),
        integrity_audit_sha256=sha(args.experiment/'independent_training_audit.json'),
        runner_sha256=sha(__file__), contract=manifest['contract'], case=args.case,
        offsets=manifest['evaluation_cases'][args.case], simulation_only=True,
        scope='Full-cycle learned residual over the audited scripted reference; privileged contact controller remains active.')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    obs, info = env.reset(seed=20261018, options={'case': args.case})
    (args.output/'scene_variant.json').write_text(json.dumps(env.variant, indent=2))
    q, v, times = [env.env.data.qpos.copy()], [env.env.data.qvel.copy()], [0.]
    actions, observations, torques, rewards = [], [], [], []
    started = time.perf_counter()
    with ExitStack() as stack:
        viewer = None
        if args.viewer:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(env.env.model, env.env.data))
        while True:
            if viewer is not None and not viewer.is_running():
                info.update(reason='viewer_closed', success=False)
                break
            step_started = time.perf_counter()
            observations.append(obs.copy())
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            torques.append(env.last_torque.copy())
            q.append(env.env.data.qpos.copy())
            v.append(env.env.data.qvel.copy())
            times.append(float(env.env.data.time))
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0., .02-(time.perf_counter()-step_started)))
            if len(actions) % 500 == 0:
                print(json.dumps(dict(time_s=times[-1], reason=info['reason'])), flush=True)
            if terminated or truncated:
                break
    path = args.output/'trajectory.npz'
    np.savez_compressed(path, qpos=q, qvel=v, time_s=times, actions=actions,
                        observations=observations, torques=torques, rewards=rewards)
    endpoints = audit_endpoints(env.env, dict(qpos=np.asarray(q), qvel=np.asarray(v), time_s=np.asarray(times)), env.plan)
    accepted = bool(info['success'] and all(row['passed'] for row in endpoints))
    result = dict(runtime=info, independently_accepted=accepted, independent_endpoints=endpoints,
        wall_seconds=time.perf_counter()-started, trajectory_sha256=sha(path),
        simulation_only=True, autonomous_climbing=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    env.close()


if __name__ == '__main__':
    main()
