"""Collect eight fresh stochastic episodes and evaluate, never fit, the critic."""
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
    env = SeededReferenceDeltaEnv(Path(protocol['worker_shards'])/f'worker_{rank}',
        controller_modifier=protocol['controller_modifier'],
        teacher_checkpoint=protocol['teacher_checkpoint'], teacher_sha256=protocol['teacher_sha256'],
        seed_command=protocol['seed_command'], delta_scale=protocol['delta_scale'], weight=.1)
    held = HeldReferenceCommand(env, repeat=protocol['hold_steps'], physical_gamma=protocol['physical_gamma'])
    rows = []
    try:
        for case in env.cases:
            index = list(protocol['cases']).index(case)
            sampling_seed = 20341025+index
            torch.manual_seed(sampling_seed)
            np.random.seed(sampling_seed)
            obs, _ = held.reset(seed=sampling_seed, options={'case': case})
            base = env.env.unwrapped.env
            record = dict(observations=[], actions=[], rewards=[],
                qpos=[base.data.qpos.copy()], qvel=[base.data.qvel.copy()], time_s=[float(base.data.time)])
            while True:
                record['observations'].append(obs.copy())
                action = model.predict(obs, deterministic=False)[0]
                obs, reward, done, truncated, info = held.step(action)
                record['actions'].append(action.copy())
                record['rewards'].append(float(reward))
                record['qpos'].append(base.data.qpos.copy())
                record['qvel'].append(base.data.qvel.copy())
                record['time_s'].append(float(base.data.time))
                if done or truncated:
                    break
            arrays = {k: np.asarray(v) for k, v in record.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            path = Path(output)/f'episode_{index:03d}.npz'
            np.savez_compressed(path, **arrays)
            row = dict(case=case, seed=sampling_seed, file=path.name, steps=len(arrays['actions']),
                teacher_steps=info['macro_teacher_steps_total'], success=bool(info['success']),
                info=info, trajectory_sha256=sha(path))
            path.with_suffix('.json').write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(json.dumps(dict(case=case, steps=row['steps'], success=row['success'])), flush=True)
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    for name in ('seed', 'critic', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    sp = json.loads((a.seed/'protocol.json').read_text())
    seed = json.loads((a.seed/'summary.json').read_text())
    critic = json.loads((a.critic/'summary.json').read_text())
    cp = json.loads((a.critic/'protocol.json').read_text())
    assert seed['passed'] and critic['passed'] and critic['frozen_actor_and_distribution_exact']
    assert sha(a.seed/'policy_initial.zip') == seed['checkpoint_sha256'] == cp['seed_checkpoint_sha256']
    assert sha(a.critic/'policy_initial.zip') == critic['checkpoint_sha256']
    assert sha(a.critic/'protocol.json') == critic['protocol_sha256']
    assert sha(a.seed/'protocol.json') == seed['protocol_sha256']
    assert sha(Path(sp['worker_shards'])/'summary.json') == sp['shards_summary_sha256']
    for source, digest in sp['sources'].items():
        assert sha(source) == digest
    initial = PPO.load(a.seed/'policy_initial.zip', device='cpu')
    model = PPO.load(a.critic/'policy_initial.zip', device='cpu')
    frozen = [k for k in initial.policy.state_dict() if not k.startswith(('mlp_extractor.value_net.', 'value_net.'))]
    assert all(torch.equal(initial.policy.state_dict()[k], model.policy.state_dict()[k]) for k in frozen)
    assert initial.gamma == model.gamma == sp['config']['gamma']
    a.output.mkdir(exist_ok=False)
    protocol = dict(seed_summary_sha256=sha(a.seed/'summary.json'), critic_summary_sha256=sha(a.critic/'summary.json'),
        initial_checkpoint_sha256=seed['checkpoint_sha256'], critic_checkpoint_sha256=critic['checkpoint_sha256'],
        sampling_seed_base=20341025, episodes=8, cases=sp['cases'], workers=a.workers,
        sources={**sp['sources'], str(Path(__file__)): sha(__file__)},
        gates=dict(explained_variance_min=.8, mse_ratio_max=.5),
        scope='Fresh sampled trajectories on eight known geometries, validation only; no critic/actor updates or checkpoint selection.')
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(worker, (rank, str(a.seed.resolve()), sp, str(a.output.resolve()))) for rank in range(4)]
        for future in as_completed(futures):
            rows.extend(future.result())
    assert len(rows) == 8 and {r['case'] for r in rows} == set(sp['cases'])
    observations, returns = [], []
    for row in rows:
        path = a.output/row['file']
        assert sha(path) == row['trajectory_sha256']
        with np.load(path) as z:
            observations.append(z['observations'].copy())
            rewards = z['rewards'].copy()
        target = np.empty_like(rewards)
        value = 0.
        for i in range(len(rewards)-1, -1, -1):
            value = rewards[i]+model.gamma*value
            target[i] = value
        returns.append(target)
    obs = torch.as_tensor(np.concatenate(observations), dtype=torch.float32)
    target = np.concatenate(returns)
    with torch.no_grad():
        before = initial.policy.predict_values(obs).numpy().ravel().astype(float)
        after = model.policy.predict_values(obs).numpy().ravel().astype(float)
        old_d, new_d = initial.policy.get_distribution(obs).distribution, model.policy.get_distribution(obs).distribution
        exact = torch.equal(old_d.mean, new_d.mean) and torch.equal(old_d.stddev, new_d.stddev)
    metric = lambda value: dict(mse=float(np.mean((target-value)**2)), explained_variance=float(1-np.var(target-value)/np.var(target)))
    bm, am = metric(before), metric(after)
    result = dict(passed=bool(exact and am['explained_variance'] >= .8 and am['mse'] <= .5*bm['mse']),
        before=bm, after=am, rows=rows, macro_steps=len(obs), teacher_steps=sum(r['teacher_steps'] for r in rows),
        frozen_actor_and_distribution_exact=bool(exact), protocol_sha256=sha(a.output/'protocol.json'),
        critic_checkpoint_sha256=critic['checkpoint_sha256'], critic_summary_sha256=sha(a.critic/'summary.json'))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
