"""Collect reference-delta exploration episodes; do not update either actor."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import sha
from training.seeded_reference_delta_env import SeededReferenceDeltaEnv
from training.held_reference_command import HeldReferenceCommand


def worker(payload):
    rank, seed_path, protocol, output = payload
    torch.set_num_threads(1)
    model = PPO.load(Path(seed_path)/'policy_initial.zip', device='cpu')
    env = SeededReferenceDeltaEnv(Path(protocol['worker_shards'])/f'worker_{rank}',
        controller_modifier=protocol['controller_modifier'],
        teacher_checkpoint=protocol['teacher_checkpoint'], teacher_sha256=protocol['teacher_sha256'],
        seed_command=protocol['seed_command'], delta_scale=protocol['delta_scale'], weight=.1)
    held = HeldReferenceCommand(env, repeat=protocol['hold_steps'], physical_gamma=protocol['physical_gamma'])
    assert abs(model.gamma-held.macro_gamma) < 1e-15
    names = list(protocol['cases'])
    rows = []
    try:
        for episode, split in enumerate(('train', 'validation', 'test')):
            for case in env.cases:
                index = names.index(case)*3+episode
                sampling_seed = 20321025+index
                torch.manual_seed(sampling_seed)
                np.random.seed(sampling_seed)
                obs, _ = held.reset(seed=sampling_seed, options={'case': case})
                base = env.env.unwrapped
                record = dict(observations=[], actions=[], rewards=[],
                              qpos=[base.env.data.qpos.copy()], qvel=[base.env.data.qvel.copy()],
                              time_s=[float(base.env.data.time)])
                while True:
                    record['observations'].append(obs.copy())
                    action = model.predict(obs, deterministic=False)[0]
                    obs, reward, done, truncated, info = held.step(action)
                    record['actions'].append(action.copy())
                    record['rewards'].append(float(reward))
                    record['qpos'].append(base.env.data.qpos.copy())
                    record['qvel'].append(base.env.data.qvel.copy())
                    record['time_s'].append(float(base.env.data.time))
                    if done or truncated:
                        break
                arrays = {k: np.asarray(v) for k, v in record.items()}
                assert all(np.isfinite(v).all() for v in arrays.values())
                path = Path(output)/f'episode_{index:03d}.npz'
                np.savez_compressed(path, **arrays)
                row = dict(case=case, index=index, split=split, seed=sampling_seed,
                    file=path.name, steps=len(arrays['actions']), physical_teacher_steps=info['macro_teacher_steps_total'],
                    success=bool(info['success']), info=info, trajectory_sha256=sha(path))
                path.with_suffix('.json').write_text(json.dumps(row, indent=2))
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('case', 'split', 'steps', 'success')}), flush=True)
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads((a.seed/'summary.json').read_text())
    protocol = json.loads((a.seed/'protocol.json').read_text())
    assert gate['passed'] and gate['zero_mean_exact']
    assert gate['checkpoint_sha256'] == sha(a.seed/'policy_initial.zip')
    assert gate['protocol_sha256'] == sha(a.seed/'protocol.json')
    for name, digest in protocol['sources'].items():
        assert sha(name) == digest
    assert sha(protocol['teacher_checkpoint']) == protocol['teacher_sha256']
    a.output.mkdir(exist_ok=False)
    sources = dict(protocol['sources'], **{str(Path(__file__).relative_to(Path.cwd())): sha(__file__)})
    experiment = dict(seed=str(a.seed.resolve()), seed_checkpoint_sha256=gate['checkpoint_sha256'],
        seed_protocol_sha256=gate['protocol_sha256'], teacher_sha256=protocol['teacher_sha256'],
        cases=protocol['cases'], sources=sources, gamma=protocol['config']['gamma'],
        split=dict(train=1, validation=1, test=1), episodes_per_case=3,
        scope='24 stochastic five-reference-delta episodes with frozen teacher and zero-mean feedback actor. Recorded states at 10 Hz; teacher 50 Hz and physics 2000 Hz unchanged. No actor updates or complete-from-zero performance claim.')
    (a.output/'protocol.json').write_text(json.dumps(experiment, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(worker, (rank, str(a.seed.resolve()), protocol, str(a.output.resolve()))) for rank in range(4)]
        for future in as_completed(jobs):
            rows.extend(future.result())
    assert len(rows) == 24 and len({r['index'] for r in rows}) == 24
    result = dict(passed=True, rows=sorted(rows, key=lambda r: r['index']),
                  macro_steps=sum(r['steps'] for r in rows),
                  physical_teacher_steps=sum(r['physical_teacher_steps'] for r in rows),
                  successes=sum(r['success'] for r in rows),
                  protocol_sha256=sha(a.output/'protocol.json'))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    for name in sources:
        target = a.output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
