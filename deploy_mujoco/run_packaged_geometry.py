"""Replay one audited candidate case from a relocatable simulation package."""
import argparse
from contextlib import ExitStack
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'release_manifest.json').read_text())
    if manifest['format'] != 'g1-climb-simulation-candidate-v1' or not manifest['simulation_only']:
        raise ValueError('Unsupported package')
    for name, digest in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Package integrity mismatch: '+name)
    for name, version in manifest['runtime_versions'].items():
        if metadata.version(name) != version:
            raise ValueError('Runtime version differs: '+name)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = verify_package(root)
    if args.verify_only:
        print(json.dumps(dict(verified=True, files=len(manifest['files']), package_root=str(root))))
        return
    if args.output is None:
        parser.error('--output required unless --verify-only')
    if args.output.exists():
        raise FileExistsError(args.output)
    # Local code is imported only after its package hash has been checked.
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from training.strict_full_cycle_residual_env import FullCycleResidualEnv
    from training.bar_geometry_observation import BarGeometryObservation
    from tools.audit_multirung_cycle import audit_endpoints
    torch.set_num_threads(1)
    base = FullCycleResidualEnv(plan=root/manifest['reference'], cases={manifest['case']: manifest['offsets']},
                               normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    observer = BarGeometryObservation(base, mode=manifest['geometry_mode'])
    try:
        model = PPO.load(root/manifest['checkpoint'], device='cpu')
        assert model.observation_space.shape == observer.observation_space.shape == (148,)
        assert model.action_space.shape == observer.action_space.shape == (26,)
        observation, info = observer.reset(seed=20261022, options={'case': manifest['case']})
        args.output.mkdir(parents=True)
        e = base.env
        q, v, times = [e.data.qpos.copy()], [e.data.qvel.copy()], [float(e.data.time)]
        actions, observations, torques, rewards = [], [], [], []
        started = time.perf_counter()
        with ExitStack() as stack:
            viewer = None
            if args.viewer:
                import mujoco.viewer
                viewer = stack.enter_context(mujoco.viewer.launch_passive(e.model, e.data))
            while True:
                if viewer is not None and not viewer.is_running():
                    info = dict(info, reason='viewer_closed', success=False)
                    break
                step_start = time.perf_counter()
                observations.append(observation.copy())
                action = model.predict(observation, deterministic=True)[0]
                observation, reward, done, truncated, info = observer.step(action)
                actions.append(action)
                rewards.append(reward)
                torques.append(base.last_torque.copy())
                q.append(e.data.qpos.copy())
                v.append(e.data.qvel.copy())
                times.append(float(e.data.time))
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0., .02-(time.perf_counter()-step_start)))
                if len(actions) % 1000 == 0:
                    print(json.dumps(dict(time_s=times[-1])), flush=True)
                if done or truncated:
                    break
        arrays = dict(qpos=np.asarray(q), qvel=np.asarray(v), time_s=np.asarray(times), actions=np.asarray(actions),
                      observations=np.asarray(observations), torques=np.asarray(torques), rewards=np.asarray(rewards))
        trajectory = args.output/'trajectory.npz'
        np.savez_compressed(trajectory, **arrays)
        endpoints = audit_endpoints(e, arrays, base.plan)
        differences, equal = {}, {}
        with np.load(root/manifest['golden_trajectory']) as golden:
            for name, value in arrays.items():
                same_shape = value.shape == golden[name].shape
                equal[name] = bool(same_shape and np.array_equal(value, golden[name]))
                differences[name] = float(abs(value-golden[name]).max()) if same_shape else None
        passed = bool(info['success'] and all(row['passed'] for row in endpoints) and all(equal.values()))
        result = dict(passed=passed, runtime=info, independent_endpoints=endpoints, exact_arrays=equal,
                      maximum_differences=differences, trajectory_sha256=sha(trajectory),
                      release_manifest_sha256=sha(root/'release_manifest.json'), package_root=str(root),
                      wall_seconds=time.perf_counter()-started, simulation_only=True,
                      scope='Portable candidate package replay of one declared audited case. Not an autonomous policy or hardware release.')
        (args.output/'reproduction_audit.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if not passed:
            raise AssertionError('Packaged replay did not exactly reproduce the audited case')
    finally:
        observer.close()


if __name__ == '__main__':
    main()
