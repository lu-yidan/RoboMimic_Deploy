"""Validate a successful sampled suffix from physical time zero, without caches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation
from tools.continue_vector_full_cycle_training import sha
from tools.audit_multirung_cycle import audit_endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = json.loads(args.record.with_suffix('.json').read_text())
    if not metadata['success'] or metadata['training_segment'] != 'closure_suffix_only':
        raise ValueError('A completed successful training suffix is required')
    assert sha(args.record) == metadata['trajectory_sha256']
    experiment = args.record.parent.parent
    m = json.loads((experiment / 'manifest.json').read_text())
    for source, digest in m['sources'].items():
        if source.replace('\\', '/').startswith('training/'):
            assert sha(source) == digest, source
    parent = Path(m['parent_experiment'])
    audit = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed']
    case = metadata['case']
    row = next(r for r in audit['trajectories'] if r['label'] == 'after' and r['case'] == case)
    prefix_path = parent / f'after_{case}.npz'
    assert sha(prefix_path) == row['trajectory_sha256']
    count = round(m['suffix_start_seconds'] / .02)
    with np.load(prefix_path) as a, np.load(args.record) as b:
        prefix = {k: a[k] for k in a.files}
        suffix = {k: b[k] for k in b.files}
    expected = {}
    for key in ('qpos', 'qvel', 'time_s', 'observations', 'actions', 'torques', 'rewards'):
        sk = 'base_rewards' if key == 'rewards' else key
        expected[key] = np.concatenate((prefix[key][:count], suffix[sk]), axis=0)
    assert abs(expected['time_s'][0]) < 1e-10 and len(expected['actions']) == 5683
    args.output.mkdir(exist_ok=False)
    base = FullCycleResidualEnv(plan=m['reference'], cases=m['evaluation_cases'],
        normal_bias_y_deg=m['normal_bias_y_deg'], memory_enabled=m['force_history_memory'])
    env = BarGeometryObservation(base, mode=m['geometry_mode'])
    obs, _ = env.reset(seed=20261022, options={'case': case})
    recorded = dict(qpos=[base.env.data.qpos.copy()], qvel=[base.env.data.qvel.copy()],
        time_s=[float(base.env.data.time)], observations=[], actions=[], torques=[], rewards=[])
    for action in expected['actions']:
        recorded['observations'].append(obs.copy())
        obs, reward, done, truncated, info = env.step(action)
        recorded['actions'].append(action.copy())
        recorded['qpos'].append(base.env.data.qpos.copy())
        recorded['qvel'].append(base.env.data.qvel.copy())
        recorded['time_s'].append(float(base.env.data.time))
        recorded['torques'].append(base.last_torque.copy())
        recorded['rewards'].append(reward)
        if done or truncated:
            break
    arrays = {k: np.asarray(v) for k, v in recorded.items()}
    np.savez_compressed(args.output / 'trajectory.npz', **arrays)
    differences = {k: float(abs(arrays[k] - expected[k]).max())
        if arrays[k].shape == expected[k].shape else None for k in expected}
    tolerance = dict(qpos=1e-7, qvel=1e-5, time_s=1e-8, observations=1e-5,
                     actions=0., torques=1e-3, rewards=1e-6)
    replay_match = all(differences[k] is not None and differences[k] <= tolerance[k] for k in tolerance)
    endpoints = audit_endpoints(base.env, arrays, base.plan)
    result = dict(**info, starts_at_zero=True, sampled_actions_not_deterministic_policy=True,
        source_record=str(args.record.resolve()), source_record_sha256=sha(args.record),
        source_record_metadata_sha256=sha(args.record.with_suffix('.json')),
        source_prefix_sha256=sha(prefix_path), source_sha256=sha(__file__),
        trajectory_sha256=sha(args.output / 'trajectory.npz'), replay_match=replay_match,
        differences=differences, tolerances=tolerance, independent_endpoints=endpoints,
        validated_sample_success=bool(info['success'] and replay_match and all(r['passed'] for r in endpoints)))
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    (args.output / 'replay_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == '__main__':
    main()
