"""Short, state-checked CPU vectorization benchmark, not a full-cycle estimate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


class SnapshotEnv(FullCycleResidualEnv):
    def snapshot(self):
        return dict(qpos=self.env.data.qpos.copy(), qvel=self.env.data.qvel.copy(), time_s=float(self.env.data.time))


def make_env():
    torch.set_num_threads(1)
    return SnapshotEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--environments', type=int, nargs='+', default=[1, 2, 4])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 16 or args.steps > 1000 or args.environments[0] != 1 or any(n not in (1, 2, 4) for n in args.environments):
        raise ValueError('Use 16..1000 steps; environment counts start at 1 and stay in 1,2,4')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    results, reference = [], None
    for count in args.environments:
        startup = time.perf_counter()
        env = DummyVecEnv([make_env]) if count == 1 else SubprocVecEnv([make_env]*count, start_method='spawn')
        env.seed(20261018)
        env.reset()
        actions = np.zeros((count, 26), dtype=np.float32)
        for _ in range(10):
            _, _, done, _ = env.step(actions)
            if np.any(done):
                raise AssertionError('Unexpected task termination in warmup')
        startup = time.perf_counter()-startup
        start = time.perf_counter()
        for _ in range(args.steps):
            _, _, done, _ = env.step(actions)
            if np.any(done):
                raise AssertionError('Unexpected task termination in benchmark')
        elapsed = time.perf_counter()-start
        snapshots = env.env_method('snapshot')
        env.close()
        if reference is None:
            reference = snapshots[0]
        difference = max(float(abs(snapshot[key]-reference[key]).max()) for snapshot in snapshots for key in ('qpos', 'qvel'))
        times_match = all(snapshot['time_s'] == reference['time_s'] for snapshot in snapshots)
        if difference != 0. or not times_match:
            raise AssertionError('Parallel environment changes the zero-policy physics state')
        result = dict(environments=count, vector_steps=args.steps, transitions=args.steps*count,
                      startup_and_warmup_seconds=startup, rollout_seconds=elapsed,
                      transitions_per_second=args.steps*count/elapsed,
                      simulation_time_per_env_s=snapshots[0]['time_s'],
                      exact_state_match=True, maximum_state_difference=difference)
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.output/'results.json').write_text(json.dumps(results, indent=2))
    summary = dict(results=results, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Early 5.32 s only with default steps; zero residual, CPU, 10 warmup steps. Concurrent research jobs may affect absolute timing. Not a full-cycle or PPO throughput claim.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
