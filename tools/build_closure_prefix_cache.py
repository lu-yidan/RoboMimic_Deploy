"""Cache genuinely simulated prefixes, proving exact serialized continuation.

Caches are trusted local Python objects for training resets only. They must
never be used to claim a complete-cycle evaluation began at physical time zero.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation
from tools.continue_vector_full_cycle_training import sha


PARENT = Path('outputs/closure_margin_control98304_v1')
START = 99.66


def build(payload):
    case, output = payload
    torch.set_num_threads(1)
    output = Path(output)
    m = json.loads((PARENT / 'manifest.json').read_text())
    audit = json.loads((PARENT / 'independent_training_audit.json').read_text())
    assert audit['passed'] and sha(PARENT / 'manifest.json') == audit['experiment_manifest_sha256']
    row = next(r for r in audit['trajectories'] if r['case'] == case and r['label'] == 'after')
    path = PARENT / f'after_{case}.npz'
    assert sha(path) == row['trajectory_sha256']
    base = FullCycleResidualEnv(plan=m['reference'], cases=m['evaluation_cases'],
        normal_bias_y_deg=m['normal_bias_y_deg'], memory_enabled=m['force_history_memory'])
    env = BarGeometryObservation(base, mode='oracle')
    obs, _ = env.reset(seed=20261022, options={'case': case})
    maximum = dict(qpos=0., qvel=0., torques=0., observations=0., rewards=0.)
    restored = None
    with np.load(path) as z:
        for i, action in enumerate(z['actions']):
            if i == round(START / .02):
                assert abs(base.env.data.time - START) < 1e-7
                target = output / f'{case}.pkl.gz'
                # Compression and pickle roundtrip include solver and controller state.
                with gzip.open(target, 'wb') as stream:
                    pickle.dump(env, stream, protocol=pickle.HIGHEST_PROTOCOL)
                with gzip.open(target, 'rb') as stream:
                    cached = pickle.load(stream)
                restored = copy.deepcopy(cached)
                assert restored.unwrapped.env.model is not base.env.model
                assert restored.unwrapped.env.data is not base.env.data
                assert restored.unwrapped.contact.d is restored.unwrapped.env.data
                assert restored.unwrapped.contact.m is restored.unwrapped.env.model
                assert restored.unwrapped.baseline.d is restored.unwrapped.env.data
                cached.close()
            maximum['observations'] = max(maximum['observations'], float(abs(obs - z['observations'][i]).max()))
            obs, reward, done, truncated, info = env.step(action)
            for key, value in [('qpos', base.env.data.qpos), ('qvel', base.env.data.qvel), ('torques', base.last_torque)]:
                expected = z[key][i if key == 'torques' else i + 1]
                maximum[key] = max(maximum[key], float(abs(value - expected).max()))
            maximum['rewards'] = max(maximum['rewards'], abs(reward - float(z['rewards'][i])))
            assert not truncated and done == (i == len(z['actions']) - 1)
            if restored is not None:
                ro, rr, rd, rt, ri = restored.step(action)
                assert np.array_equal(ro, obs) and rr == reward and rd == done and rt == truncated
                assert np.array_equal(restored.unwrapped.env.data.qpos, base.env.data.qpos)
                assert np.array_equal(restored.unwrapped.env.data.qvel, base.env.data.qvel)
                assert np.array_equal(restored.unwrapped.last_torque, base.last_torque)
                assert ri == info
    assert all(v == 0. for v in maximum.values())
    assert info['success'] == row['experiment_success']
    result = dict(case=case, passed=True, prefix_steps=round(START / .02), suffix_steps=700,
        cache_sha256=sha(target), source_trajectory_sha256=sha(path), exact_replay_maxima=maximum,
        continuation_exact=True, complete_cycle_success=info['success'])
    (output / f'{case}.json').write_text(json.dumps(result, indent=2))
    env.close()
    restored.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    m = json.loads((PARENT / 'manifest.json').read_text())
    sources = dict(m['sources'])
    for p, digest in sources.items():
        if p.replace('\\', '/').startswith('training/'):
            assert sha(p) == digest
    sources[str(Path(__file__).resolve())] = sha(__file__)
    protocol = dict(parent=str(PARENT.resolve()), parent_checkpoint_sha256=sha(PARENT / 'policy.zip'),
        reference=m['reference'], reference_sha256=m['reference_sha256'], sources=sources,
        cases=m['evaluation_cases'], start_seconds=START, horizon_seconds=113.66,
        python_version=sys.version, scope='Trusted local reset caches from continuous parent-policy prefixes, never full-cycle evaluation shortcuts.')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(build, [(case, str(args.output.resolve())) for case in m['evaluation_cases']]))
    result = dict(passed=all(r['passed'] for r in rows), rows=rows,
        protocol_sha256=sha(args.output / 'protocol.json'), source_sha256=sha(__file__))
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
