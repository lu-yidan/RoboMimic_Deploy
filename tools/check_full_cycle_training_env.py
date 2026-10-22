"""Replay zero residual and compare every saved state with the accepted runner."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT/'outputs/periodic_force_history_two_cycles_v1/persistence__nominal')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--strict-endpoints', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads((args.baseline/'manifest.json').read_text())
    cfg = manifest['config']
    environment_class = FullCycleResidualEnv
    source_relative = 'training/full_cycle_residual_env.py'
    if args.strict_endpoints:
        from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
        environment_class = StrictEnv
        source_relative = 'training/strict_full_cycle_residual_env.py'
    env = environment_class(plan=cfg['plan'], cases={'nominal': cfg['rung_offsets']},
        normal_bias_y_deg=cfg['window_normal_bias_y_deg'], memory_enabled=cfg['window_memory_enabled'])
    source = ROOT/source_relative
    (args.output/'environment_source.py').write_bytes(source.read_bytes())
    obs, info = env.reset(seed=20261018)
    assert env.observation_space.contains(obs)
    states, velocities, times = [env.env.data.qpos.copy()], [env.env.data.qvel.copy()], [0.]
    started = time.perf_counter()
    count = 0
    horizon = args.seconds or env.horizon
    while env.env.data.time < horizon-1e-7:
        obs, reward, terminated, truncated, info = env.step(np.zeros(26))
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        states.append(env.env.data.qpos.copy())
        velocities.append(env.env.data.qvel.copy())
        times.append(float(env.env.data.time))
        count += 1
        if count % 100 == 0:
            print(json.dumps(dict(time_s=times[-1], wall_s=time.perf_counter()-started, **info['peaks'])), flush=True)
        if terminated or truncated:
            break
    elapsed = time.perf_counter()-started
    np.savez_compressed(args.output/'trajectory.npz', qpos=states, qvel=velocities, time_s=times)
    with np.load(args.baseline/'trajectory.npz') as original:
        n = len(times)
        aligned = bool(n <= len(original['time_s']) and np.max(abs(original['time_s'][:n]-times)) < 1e-8)
        qdiff = float(np.max(abs(original['qpos'][:n]-states))) if aligned else None
        vdiff = float(np.max(abs(original['qvel'][:n]-velocities))) if aligned else None
    summary = dict(info=info, aligned=aligned, qpos_max_difference=qdiff, qvel_max_difference=vdiff,
        exact_match=aligned and qdiff == 0. and vdiff == 0., completed_requested_duration=abs(times[-1]-horizon)<1e-7,
        wall_seconds=elapsed, policy_steps=count, policy_steps_per_second=count/elapsed,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), baseline=str(args.baseline),
        base_environment_sha256=hashlib.sha256((ROOT/'training/full_cycle_residual_env.py').read_bytes()).hexdigest(),
        baseline_manifest_sha256=hashlib.sha256((args.baseline/'manifest.json').read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    env.close()


if __name__ == '__main__':
    main()
