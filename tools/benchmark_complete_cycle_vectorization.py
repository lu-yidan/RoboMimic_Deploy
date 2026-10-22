"""Benchmark one complete nominal cycle, preserving terminal-state evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tools.benchmark_full_cycle_vectorization import SnapshotEnv


class CompleteSnapshotEnv(SnapshotEnv):
    def step(self, action):
        result = super().step(action)
        _, _, terminated, truncated, info = result
        if terminated or truncated:
            self.terminal_snapshot = super().snapshot()
            self.terminal_snapshot.update(success=bool(info['success']),
                minimum_margin_rad=info['minimum_native_joint_margin_rad'],
                peaks=info['peaks'].copy(), cycle_results=list(info['cycle_results']))
        return result

    def snapshot(self):
        return getattr(self, 'terminal_snapshot', super().snapshot())


def make_env():
    torch.set_num_threads(1)
    return CompleteSnapshotEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--environments', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.environments[0] != 1 or any(n not in (1, 2, 4) for n in args.environments):
        raise ValueError('Counts must start at 1 and stay in 1,2,4')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    results, reference = [], None
    for count in args.environments:
        startup = time.perf_counter()
        env = DummyVecEnv([make_env]) if count == 1 else SubprocVecEnv([make_env]*count, start_method='spawn')
        env.seed(20261018)
        env.reset()
        horizon = float(env.get_attr('horizon')[0])
        total_steps = int(round(horizon/.02))
        actions = np.zeros((count, 26), dtype=np.float32)
        for _ in range(10):
            _, _, done, _ = env.step(actions)
            if np.any(done):
                raise AssertionError('Unexpected termination during warmup')
        startup = time.perf_counter()-startup
        started = time.perf_counter()
        for step in range(10, total_steps):
            _, _, done, infos = env.step(actions)
            expected_end = step == total_steps-1
            if bool(np.all(done)) != expected_end or (np.any(done) and not np.all(done)):
                raise AssertionError('Unexpected episode duration')
            if (step+1) % 1000 == 0:
                print(json.dumps(dict(environments=count, policy_step=step+1, wall_seconds=time.perf_counter()-started)), flush=True)
        elapsed = time.perf_counter()-started
        snapshots = env.env_method('snapshot')
        env.close()
        if not all(item['success'] for item in snapshots):
            raise AssertionError('Full-cycle runtime acceptance failed')
        if reference is None:
            reference = snapshots[0]
        difference = max(float(abs(item[key]-reference[key]).max()) for item in snapshots for key in ('qpos', 'qvel'))
        times_match = all(item['time_s'] == reference['time_s'] for item in snapshots)
        if difference != 0. or not times_match:
            raise AssertionError('Parallel final physics state differs from serial')
        np.savez_compressed(args.output/f'terminal_states_{count}.npz',
                            qpos=[s['qpos'] for s in snapshots], qvel=[s['qvel'] for s in snapshots],
                            time_s=[s['time_s'] for s in snapshots])
        result = dict(environments=count, total_steps_per_env=total_steps, measured_transitions=(total_steps-10)*count,
            startup_and_warmup_seconds=startup, rollout_seconds=elapsed,
            transitions_per_second=(total_steps-10)*count/elapsed, exact_final_state_match=True,
            maximum_final_state_difference=difference, simulation_time_per_env_s=reference['time_s'],
            acceptance=[{k: s[k] for k in ('success', 'minimum_margin_rad', 'peaks', 'cycle_results')} for s in snapshots])
        results.append(result)
        (args.output/'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    summary = dict(results=results, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Complete nominal cycle, zero residual. Includes SB3 terminal auto-reset cost. Other research jobs run concurrently; timing is an engineering estimate, not isolated PPO performance. Final states compared exactly, not every intermediate state.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
