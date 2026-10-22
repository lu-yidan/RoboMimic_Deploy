"""Continue an audited strict checkpoint with independent complete-cycle workers.

Policy and optimizer are restored exactly. Physics and random-number streams
restart explicitly; this is not a bitwise uninterrupted simulation continuation.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from training.full_cycle_residual_env import ROOT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv, CONTRACT
from training.balanced_case_reset import BalancedCaseReset
from training.endpoint_potential_wrapper import EndpointPotentialWrapper
from tools.full_cycle_experiment_io import numpy_json, reuse_zero_evaluations
from tools.parallel_full_cycle_evaluation import evaluate_parallel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_difference(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            return float('inf')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return float('inf')
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return float('inf')
        return max((state_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return float('inf')
        return max((state_difference(x, y) for x, y in zip(a, b)), default=0.)
    return 0. if a == b else float('inf')


def make_env(manifest, rank, output, weight):
    torch.set_num_threads(1)
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=manifest['training_cases'],
        normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    env = EndpointPotentialWrapper(env, gamma=manifest['config']['gamma'], weight=weight)
    return Monitor(BalancedCaseReset(env), str(Path(output)/f'episodes_worker_{rank}'))


class VectorProgress(BaseCallback):
    def __init__(self, output, parent_steps):
        super().__init__()
        self.output, self.parent_steps = Path(output), parent_steps
        self.started = time.perf_counter()
        self.case_samples = {}
        self.episodes = []
        self.reward_sums = {}

    def _on_step(self):
        for rank, info in enumerate(self.locals['infos']):
            case = info['case']
            self.case_samples[case] = self.case_samples.get(case, 0)+1
            totals = self.reward_sums.setdefault(rank, [0., 0.])
            totals[0] += info['base_reward']
            totals[1] += info['shaping_reward']
            if 'episode' in info:
                info = dict(info, base_return_sum=totals[0], shaping_return_sum=totals[1])
                self.reward_sums[rank] = [0., 0.]
                record = dict(global_step=self.num_timesteps, worker=rank, **info)
                self.episodes.append(record)
                with (self.output/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, default=numpy_json)+'\n')
        additional = self.num_timesteps-self.parent_steps
        if additional % 1024 == 0:
            status = dict(total_steps=self.num_timesteps, additional_steps=additional,
                          case_samples=self.case_samples, completed_episodes=len(self.episodes),
                          wall_seconds=time.perf_counter()-self.started)
            (self.output/'training_status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
        if additional % 8192 == 0:
            self.model.save(self.output/f'checkpoint_additional_{additional}')
        return not (self.output/'STOP_REQUESTED').exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--potential-gate', type=Path, required=True)
    parser.add_argument('--weight', type=float, choices=(0., 4.), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vector-gate', type=Path, required=True)
    parser.add_argument('--evaluation-gate', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=69632, help='Additional collected transitions')
    parser.add_argument('--environments', type=int, choices=(1, 4), default=4)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    gate = json.loads(args.potential_gate.read_text())
    if not gate['passed'] or not gate['exact_observations_states_torques_times_and_original_info']:
        raise ValueError('Complete paired physics gate required')
    for path, expected in gate['sources'].items():
        if sha(path) != expected:
            raise ValueError('Potential gate source changed: '+path)
    protocol = json.loads(args.protocol.read_text())
    for source, expected in protocol['sources'].items():
        if sha(source) != expected:
            raise ValueError('Preregistered source changed: '+source)
    if sha(args.parent/'policy.zip') != protocol['parent_checkpoint_sha256']:
        raise ValueError('Preregistered checkpoint changed')
    if (args.steps, args.seed, args.environments) != (protocol['steps'], protocol['seed'], protocol['environments']):
        raise ValueError('Matched protocol mismatch')
    if str(args.parent.resolve()) != protocol['parent']:
        raise ValueError('Matched parent mismatch')
    parent = json.loads((args.parent/'manifest.json').read_text())
    previous = json.loads((args.parent/'summary.json').read_text())
    audit = json.loads((args.parent/'independent_training_audit.json').read_text())
    if parent['contract'] != CONTRACT or not audit['passed']:
        raise ValueError('An audited strict full-cycle parent is required')
    if audit['experiment_manifest_sha256'] != sha(args.parent/'manifest.json') or audit['experiment_summary_sha256'] != sha(args.parent/'summary.json'):
        raise ValueError('Parent metadata changed after audit')
    if sha(args.parent/'policy.zip') != previous['checkpoint_sha256'] or sha(args.parent/'policy.zip') != audit['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    vector_gate = json.loads(args.vector_gate.read_text())
    matching = [r for r in vector_gate['results'] if r['environments'] == args.environments]
    if not matching or not all(r['exact_final_state_match'] and all(c['success'] for c in r['acceptance']) for r in matching):
        raise ValueError('Complete-cycle vectorization gate did not pass')
    evaluation_gate = json.loads(args.evaluation_gate.read_text())
    if not evaluation_gate['exact_reproduction_passed'] or evaluation_gate['source_sha256'] != sha(ROOT/'tools/parallel_full_cycle_evaluation.py'):
        raise ValueError('Exact parallel evaluation gate did not pass or source changed')
    for relative, expected in parent['sources'].items():
        if relative.replace('\\', '/').startswith('training/'):
            if sha(args.parent/'source_snapshot'/relative) != expected or sha(ROOT/relative) != expected:
                raise ValueError('Parent environment changed: '+relative)
    if args.steps < 256*args.environments or args.steps % (256*args.environments):
        raise ValueError('Additional steps must be a positive multiple of 256 per environment')
    if parent['config']['gamma'] != gate['gamma']:
        raise ValueError('PPO and shaping discounts must match')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output/'matched_protocol.json')
    torch.set_num_threads(1)
    factories = [partial(make_env, parent, rank, str(args.output.resolve()), args.weight) for rank in range(args.environments)]
    env = DummyVecEnv(factories) if args.environments == 1 else SubprocVecEnv(factories, start_method='spawn')
    model = PPO.load(args.parent/'policy.zip', env=env, device='cpu', force_reset=True)
    if model.gamma != parent['config']['gamma']:
        raise ValueError('Loaded PPO discount differs from shaping')
    original = PPO.load(args.parent/'policy.zip', device='cpu')
    policy_difference = state_difference(model.policy.state_dict(), original.policy.state_dict())
    optimizer_difference = state_difference(model.policy.optimizer.state_dict(), original.policy.optimizer.state_dict())
    if policy_difference != 0. or optimizer_difference != 0.:
        raise AssertionError('Parent policy or optimizer did not restore exactly')
    parent_steps = int(model.num_timesteps)
    model.set_random_seed(args.seed)
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
            sources[str(relative)] = sha(target)
    manifest = {key: parent[key] for key in ('contract', 'config', 'training_cases', 'evaluation_cases',
        'normal_bias_y_deg', 'force_history_memory', 'reference', 'reference_sha256', 'bundle_manifest_sha256')}
    manifest.update(training_reward_contract='strict-v2-plus-endpoint-potential-v1',
        shaping_weight=args.weight, shaping_gamma=model.gamma,
        evaluation_reward_contract=CONTRACT, potential_gate_sha256=sha(args.potential_gate),
        protocol_sha256=sha(args.protocol), critic_initialization='Exact parent critic and optimizer, identical in both arms',
        zero_reuse_scope='Frozen strict evaluator and all parent environment sources unchanged; wrapper is training-only',
        sources=sources, seed=args.seed, parent_experiment=str(args.parent.resolve()),
        parent_checkpoint_sha256=previous['checkpoint_sha256'], parent_steps=parent_steps,
        additional_steps_requested=args.steps, environments=args.environments,
        rollout_transitions=int(model.n_steps*args.environments), vector_gate_sha256=sha(args.vector_gate),
        evaluation_gate_sha256=sha(args.evaluation_gate),
        reset_sampling='Balanced shuffled case blocks independently per worker; worker seeds use seed plus rank',
        continuation='Exact policy and optimizer restoration; fresh physics resets and seeded random streams; not uninterrupted simulation',
        initial_policy_max_difference=policy_difference, initial_optimizer_max_difference=optimizer_difference,
        scope='Matched training-only potential experiment; unchanged strict physics, actor inputs and evaluation reward/acceptance.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    model.save(args.output/'policy_before')
    zero = reuse_zero_evaluations(args.parent, args.output, manifest, ROOT)
    before = previous['after']
    for row in before:
        path = args.parent/f"after_{row['case']}.npz"
        if sha(path) != row['trajectory_sha256']:
            raise ValueError('Parent before-evaluation trajectory changed')
        shutil.copy2(path, args.output/f"before_{row['case']}.npz")
    (args.output/'before.json').write_text(json.dumps(before, indent=2))
    callback = VectorProgress(args.output, parent_steps)
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
    elapsed = time.perf_counter()-started
    model.save(args.output/'policy')
    env.close()
    summary = dict(actual_steps=int(model.num_timesteps), parent_steps=parent_steps,
        additional_steps=int(model.num_timesteps-parent_steps), training_wall_seconds=elapsed,
        zero=zero, before=before, before_equals_parent_after=True,
        checkpoint_sha256=sha(args.output/'policy.zip'), case_samples=callback.case_samples,
        completed_training_episodes=len(callback.episodes),
        successful_training_episodes=sum(bool(row['success']) for row in callback.episodes),
        full_cycle_training_interface=True, autonomous_climbing_policy=False, held_out_generalization_validated=False,
        evaluation_pending=True)
    if (args.output/'STOP_REQUESTED').exists():
        summary['stopped_by_request'] = True
        (args.output/'stopped_summary.json').write_text(json.dumps(summary, indent=2))
        return
    # The parallel evaluator reads this checkpoint metadata; the pending flag
    # prevents treating this interim file as completed evaluation evidence.
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    evaluation = evaluate_parallel(args.output, args.output/'parallel_after', None, workers=3)
    after = evaluation['trajectories']
    for row in after:
        filename = f"after_{row['case']}.npz"
        shutil.copy2(args.output/'parallel_after'/filename, args.output/filename)
    (args.output/'after.json').write_text(json.dumps(after, indent=2))
    restored = PPO.load(args.output/'policy.zip', device='cpu')
    reload_difference = state_difference(model.policy.state_dict(), restored.policy.state_dict())
    if reload_difference != 0.:
        raise AssertionError('Saved policy differs from in-memory policy')
    summary.update(after=after, checkpoint_reload_max_difference=reload_difference, evaluation_pending=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(actual_steps=model.num_timesteps, additional_steps=model.num_timesteps-parent_steps,
                         training_wall_seconds=elapsed, completed=True)), flush=True)


if __name__ == '__main__':
    main()
