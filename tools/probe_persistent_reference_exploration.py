"""Compare episode-persistent bounded exploration on known active cases, without learning."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.seeded_reference_delta_env import SeededReferenceDeltaEnv
from training.held_reference_command import HeldReferenceCommand
from tools.continue_vector_full_cycle_training import sha


def worker(payload):
    rank, seed_path, protocol, output = payload
    torch.set_num_threads(1)
    model = PPO.load(Path(seed_path)/'policy_initial.zip', device='cpu')
    original = {k: v.clone() for k, v in model.policy.state_dict().items()}
    env = SeededReferenceDeltaEnv(Path(protocol['worker_shards'])/f'worker_{rank}',
        controller_modifier=protocol['controller_modifier'],
        teacher_checkpoint=protocol['teacher_checkpoint'], teacher_sha256=protocol['teacher_sha256'],
        seed_command=protocol['seed_command'], delta_scale=protocol['delta_scale'], weight=.1)
    held = HeldReferenceCommand(env, repeat=protocol['hold_steps'], physical_gamma=protocol['physical_gamma'])
    cases = [case for case in env.cases if case in protocol['active_cases']]
    assert len(cases) == 1
    case = cases[0]
    index = list(protocol['active_cases']).index(case)
    rows = []
    try:
        for sigma in protocol['sigmas']:
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(sigma)))
            for trial in range(protocol['trials']):
                sampling_seed = protocol['sampling_seed_base']+index*10+trial
                torch.manual_seed(sampling_seed)
                np.random.seed(sampling_seed)
                obs, _ = held.reset(seed=sampling_seed, options={'case': case})
                base = env.env.unwrapped.env
                record = dict(observations=[], actions=[], rewards=[],
                    qpos=[base.data.qpos.copy()], qvel=[base.data.qvel.copy()], time_s=[float(base.data.time)])
                persistent_action = model.predict(obs, deterministic=False)[0].copy()
                while True:
                    record['observations'].append(obs.copy())
                    action = persistent_action.copy()
                    obs, reward, done, truncated, info = held.step(action)
                    record['actions'].append(action.copy())
                    record['rewards'].append(float(reward))
                    record['qpos'].append(base.data.qpos.copy())
                    record['qvel'].append(base.data.qvel.copy())
                    record['time_s'].append(float(base.data.time))
                    if done or truncated:
                        break
                arrays = {key: np.asarray(value) for key, value in record.items()}
                assert all(np.isfinite(value).all() for value in arrays.values())
                assert np.array_equal(arrays['actions'], np.broadcast_to(persistent_action, arrays['actions'].shape))
                with torch.no_grad():
                    distribution = model.policy.get_distribution(torch.as_tensor(arrays['observations'], dtype=torch.float32)).distribution
                    assert torch.equal(distribution.mean, torch.zeros_like(distribution.mean))
                    assert torch.allclose(distribution.stddev, torch.full_like(distribution.stddev, sigma), atol=1e-7, rtol=1e-6)
                assert all(torch.equal(original[key], model.policy.state_dict()[key]) for key in original if key != 'log_std')
                path = Path(output)/f'case{index}_sigma{sigma:g}_trial{trial}.npz'
                np.savez_compressed(path, **arrays)
                row = dict(case=case, sigma=sigma, trial=trial, seed=sampling_seed,
                    file=path.name, steps=len(arrays['actions']), teacher_steps=info['macro_teacher_steps_total'],
                    success=bool(info['success']), reason=info['reason'], info=info,
                    discounted_return=float(sum(model.gamma**i*r for i, r in enumerate(arrays['rewards']))),
                    trajectory_sha256=sha(path))
                path.with_suffix('.json').write_text(json.dumps(row, indent=2))
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('case', 'sigma', 'trial', 'success', 'reason')}), flush=True)
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, choices=(1, 2, 4), default=2)
    a = p.parse_args()
    protocol = json.loads((a.seed/'protocol.json').read_text())
    seed = json.loads((a.seed/'summary.json').read_text())
    assert seed['passed'] and seed['zero_mean_exact']
    assert sha(a.seed/'policy_initial.zip') == seed['checkpoint_sha256']
    assert sha(a.seed/'protocol.json') == seed['protocol_sha256']
    assert sha(protocol['teacher_checkpoint']) == protocol['teacher_sha256']
    assert sha(Path(protocol['worker_shards'])/'summary.json') == protocol['shards_summary_sha256']
    for source, digest in protocol['sources'].items():
        assert sha(source) == digest
    active = {case: offsets for case, offsets in protocol['cases'].items() if offsets['rung_01'][2] > .0125}
    assert len(active) == 4 and 'z15.1' in active
    protocol.update(active_cases=active, sigmas=[.1, .3, .6], trials=3,
        sampling_seed_base=20351025, source_sha256=sha(__file__),
        initial_checkpoint_sha256=seed['checkpoint_sha256'], workers=a.workers,
        exploration_temporal_contract='One Gaussian action sampled at episode start, held for all 140 macro decisions; identical first draw and seeds to the independent-action probe.',
        scope='36 cached final-14-second exploration episodes on four known nonzero-gain cases. '
              'Paired random seeds across three Gaussian standard deviations; initial mean actor remains zero. Episode-constant exploration, not the stochastic Markov policy used in PPO. '
              'All delta and physical bounds unchanged; no gradient updates, final-checkpoint selection, '
              'full time-zero validation, independent geometry result or learned-policy claim.')
    a.output.mkdir(exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (a.output/'persistent_probe_source.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(worker, (rank, str(a.seed.resolve()), protocol, str(a.output.resolve()))) for rank in range(4)]
        for future in as_completed(futures):
            rows.extend(future.result())
    assert len(rows) == 36
    counts = []
    for case in active:
        for sigma in protocol['sigmas']:
            group = [row for row in rows if row['case'] == case and row['sigma'] == sigma]
            assert len(group) == 3 and {row['trial'] for row in group} == {0, 1, 2}
            counts.append(dict(case=case, sigma=sigma, successes=sum(row['success'] for row in group), episodes=3))
    result = dict(completed=True, rows=rows, counts=counts,
        macro_steps=sum(row['steps'] for row in rows), teacher_steps=sum(row['teacher_steps'] for row in rows),
        protocol_sha256=sha(a.output/'protocol.json'), original_mean_actor_unchanged=True,
        scope=protocol['scope'])
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(completed=True, counts=counts)), flush=True)


if __name__ == '__main__':
    main()
