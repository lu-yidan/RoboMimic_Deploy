"""Cache genuinely simulated prefixes, checking bounded solver-restart differences.

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
START = 70.2


def build_impl(payload):
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
    continuation = dict(qpos=0., qvel=0., torques=0., observations=0.)
    tolerances = dict(qpos=1e-7, qvel=1e-5, torques=1e-3, observations=1e-5)
    restored = None
    with np.load(path) as archive:
        z = {key: archive[key] for key in archive.files}
        for i, action in enumerate(z['actions']):
            if i == round(START / .02):
                assert abs(base.env.data.time - START) < 1e-7
                target = output / f'{case}.pkl.gz'
                # Discard only cached OSQP solver objects in the COPY, never in the live run.
                snapshot = copy.deepcopy(env, {id(base.baseline.cache): {}, id(base.contact.cache): {}, id(base.contact.intervention.cache): {}})
                with gzip.open(target, 'wb', compresslevel=1) as stream:
                    pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
                snapshot.close()
                del snapshot
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
                assert rd == done and rt == truncated
                for key, a, b in [('qpos', restored.unwrapped.env.data.qpos, base.env.data.qpos),
                                  ('qvel', restored.unwrapped.env.data.qvel, base.env.data.qvel),
                                  ('torques', restored.unwrapped.last_torque, base.last_torque),
                                  ('observations', ro, obs)]:
                    continuation[key] = max(continuation[key], float(abs(a - b).max()))
                assert ri['reason'] == info['reason'] and ri['success'] == info['success']
    assert all(v == 0. for v in maximum.values())
    assert info['success'] == row['experiment_success']
    result = dict(case=case, passed=all(continuation[k] <= tolerances[k] for k in tolerances), prefix_steps=round(START / .02), suffix_steps=round((113.66-START)/.02),
        continuation_differences=continuation, continuation_tolerances=tolerances,
        cache_sha256=sha(target), source_trajectory_sha256=sha(path), exact_replay_maxima=maximum,
        continuation_exact=all(v == 0. for v in continuation.values()), solver_cache_restarted=True, complete_cycle_success=info['success'])
    (output / f'{case}.json').write_text(json.dumps(result, indent=2))
    env.close()
    restored.close()
    return result


def build(payload):
    try:
        return build_impl(payload)
    except Exception:
        import traceback
        case, output = payload
        (Path(output) / f'{case}_error.txt').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases',nargs='+',default=['foot_z_plus15'])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    m = json.loads((PARENT / 'manifest.json').read_text())
    assert set(args.cases)<=set(m['evaluation_cases']) and len(args.cases)==len(set(args.cases))
    cases={k:m['evaluation_cases'][k] for k in args.cases}
    sources = dict(m['sources'])
    for p, digest in sources.items():
        if p.replace('\\', '/').startswith('training/'):
            assert sha(p) == digest
    sources[str(Path(__file__).resolve())] = sha(__file__)
    protocol = dict(parent=str(PARENT.resolve()), parent_checkpoint_sha256=sha(PARENT / 'policy.zip'),
        reference=m['reference'], reference_sha256=m['reference_sha256'], sources=sources,
        solver_cache_restarted=True, cases=cases, start_seconds=START, horizon_seconds=113.66,
        python_version=sys.version, scope='Landing-window cache at 70.2 seconds, before left-foot airborne retarget. Trusted local reset only; not a complete-cycle evaluation or new training result.')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=1) as pool:
        rows = list(pool.map(build, [(case, str(args.output.resolve())) for case in cases]))
    result = dict(passed=all(r['passed'] for r in rows), rows=rows,
        protocol_sha256=sha(args.output / 'protocol.json'), source_sha256=sha(__file__))
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result['passed'], 'Solver-restart differences exceed the declared cache gate'


if __name__ == '__main__':
    main()
