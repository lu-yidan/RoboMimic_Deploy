"""Reproducible small PPO experiment on the complete climbing cycle.

Preserves zero/before/after trajectories and never overwrites an experiment.
The initial policy mean is zero, so its deterministic evaluation is explicitly
the zero-residual controller, not evidence of learned climbing.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from training.full_cycle_residual_env import FullCycleResidualEnv, ROOT, CONTRACT
from tools.full_cycle_experiment_io import numpy_json, reuse_zero_evaluations

CASES = {'nominal': {}, 'foot_z_plus5': {'rung_01': [0., 0., .005]},
         'foot_x_minus10': {'rung_01': [-.01, 0., 0.]}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(model, out, label, cases):
    env = FullCycleResidualEnv(cases=cases)
    result = []
    for name in cases:
        obs, _ = env.reset(seed=20261018, options={'case': name})
        states, velocities, times = [env.env.data.qpos.copy()], [env.env.data.qvel.copy()], [0.]
        actions, observations, torques, rewards = [], [], [], []
        start = time.perf_counter()
        while True:
            observations.append(obs.copy())
            action = np.zeros(26, dtype=np.float32) if model is None else model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            torques.append(env.last_torque.copy())
            states.append(env.env.data.qpos.copy())
            velocities.append(env.env.data.qvel.copy())
            times.append(float(env.env.data.time))
            if terminated or truncated:
                break
        path = out/f'{label}_{name}.npz'
        np.savez_compressed(path, qpos=states, qvel=velocities, time_s=times,
                            actions=actions, observations=observations, torques=torques, rewards=rewards)
        result.append(dict(**info, return_sum=float(sum(rewards)), wall_seconds=time.perf_counter()-start,
                           trajectory_sha256=digest(path)))
        (out/f'{label}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(evaluation=label, **result[-1])), flush=True)
    env.close()
    return result


class Progress(BaseCallback):
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.start = time.perf_counter()

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                with (self.out/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=self.num_timesteps, **info), default=numpy_json)+'\n')
        if self.num_timesteps % 256 == 0:
            status = dict(training_steps=self.num_timesteps, wall_seconds=time.perf_counter()-self.start)
            (self.out/'training_status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
        if self.num_timesteps % 2048 == 0:
            self.model.save(self.out/f'checkpoint_{self.num_timesteps}')
        return not (self.out/'STOP_REQUESTED').exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--gate', type=Path, required=True, help='Passed full two-cycle zero-replay summary')
    parser.add_argument('--reuse-zero', type=Path, help='Identical frozen zero-policy evaluations to verify and reuse')
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    if not gate['exact_match'] or not gate['completed_requested_duration'] or not gate['info']['success']:
        raise ValueError('Full-cycle exact replay gate has not passed')
    if gate['source_sha256'] != digest(ROOT/'training/full_cycle_residual_env.py'):
        raise ValueError('Environment changed after the replay gate')
    if args.steps < 256 or args.steps % 256:
        raise ValueError('Steps must be a positive multiple of 256')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = Monitor(FullCycleResidualEnv(cases=CASES), str(args.output/'episodes'))
    config = dict(n_steps=256, batch_size=64, n_epochs=5, learning_rate=1e-4,
                  gamma=.9995, gae_lambda=.95,
                  policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-3.))
    model = PPO('MlpPolicy', env, seed=args.seed, device='cpu', verbose=0, **config)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    model.set_logger(configure(str(args.output), ['csv']))
    sources = {}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and path.suffix == '.py' and '.venv' not in path.parts:
            relative = path.relative_to(ROOT)
            target = args.output/'source_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            sources[str(relative)] = digest(target)
    manifest = dict(contract=CONTRACT, seed=args.seed, requested_steps=args.steps, config=config,
        sources=sources, training_cases=CASES, evaluation_cases=CASES,
        normal_bias_y_deg=-3., force_history_memory=True,
        reference_sha256=env.unwrapped.plan_hash, reference=str(env.unwrapped.plan_path),
        bundle_manifest_sha256=digest(env.unwrapped.env.bundle/'manifest.json'),
        gate=str(args.gate), gate_sha256=digest(args.gate), torch=torch.__version__,
        sb3=stable_baselines3.__version__, platform=platform.platform(), processor=platform.processor(),
        policy_initialization='Exactly zero deterministic action mean; stochastic std exp(-3).',
        scope='One-seed full-cycle residual learning on an audited scripted trajectory. Ideal forces and privileged low-level contact geometry. No held-out generalization claim.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    model.save(args.output/'policy_before')
    zero = (reuse_zero_evaluations(args.reuse_zero, args.output, manifest, ROOT) if args.reuse_zero
            else evaluate(None, args.output, 'zero', CASES))
    # Prove equivalence of the zero-mean initial policy on every recorded observation.
    initial_difference = 0.
    for name in CASES:
        with np.load(args.output/f'zero_{name}.npz') as trajectory:
            initial_difference = max(initial_difference, float(abs(model.predict(trajectory['observations'], deterministic=True)[0]).max()))
    if initial_difference != 0.:
        raise AssertionError('Initial deterministic policy is not exactly zero')
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=Progress(args.output))
    elapsed = time.perf_counter()-started
    model.save(args.output/'policy')
    if (args.output/'STOP_REQUESTED').exists():
        stopped = dict(actual_steps=model.num_timesteps, requested_steps=args.steps,
            training_wall_seconds=elapsed, stopped_by_request=True,
            checkpoint_sha256=digest(args.output/'policy.zip'), zero=zero,
            after_evaluation_pending=True, scope='Stopped at a policy step; no new post-training evaluation launched.')
        (args.output/'stopped_summary.json').write_text(json.dumps(stopped, indent=2))
        print(json.dumps(stopped), flush=True)
        env.close()
        return
    after = evaluate(model, args.output, 'after', CASES)
    restored = PPO.load(args.output/'policy.zip', device='cpu')
    difference = 0.
    for name in CASES:
        with np.load(args.output/f'after_{name}.npz') as trajectory:
            x = model.predict(trajectory['observations'], deterministic=True)[0]
            y = restored.predict(trajectory['observations'], deterministic=True)[0]
            difference = max(difference, float(abs(x-y).max()))
    if difference != 0.:
        raise AssertionError('Checkpoint reload changes actions')
    episode_path = args.output/'training_episodes.jsonl'
    episodes = [json.loads(line) for line in episode_path.read_text().splitlines()] if episode_path.exists() else []
    summary = dict(actual_steps=model.num_timesteps, training_wall_seconds=elapsed,
        zero=zero, after=after, before_equals_zero=True, checkpoint_reload_max_difference=difference,
        checkpoint_sha256=digest(args.output/'policy.zip'), full_cycle_training_interface=True,
        completed_training_episodes=len(episodes), successful_training_episodes=sum(bool(x['success']) for x in episodes),
        maximum_completed_episode_time_s=max((x['time_s'] for x in episodes), default=None),
        autonomous_climbing_policy=False, held_out_generalization_validated=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    env.close()


if __name__ == '__main__':
    main()
