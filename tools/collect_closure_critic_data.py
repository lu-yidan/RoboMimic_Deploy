"""Collect fixed-policy exploratory suffix returns with episode-level splits."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_curriculum_env_v2 import ClosureCurriculumEnvV2
from training.closure_exploration_policy import ClosureExplorationPolicy
from tools.continue_vector_full_cycle_training import sha


PARENT = Path('outputs/closure_margin_control98304_v1')
CACHE = Path('outputs/closure_prefix_cache_20261024_v4')


def collect(payload):
    case, rank, output = payload
    torch.set_num_threads(1)
    model = PPO.load(PARENT / 'policy.zip', device='cpu',
                     custom_objects={'policy_class': ClosureExplorationPolicy})
    env = ClosureCurriculumEnvV2(CACHE)
    rows = []
    for episode in range(6):
        seed = 20261024 + rank * 100 + episode
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed, options={'case': case})
        base = env.env.unwrapped
        arrays = dict(observations=[], actions=[], rewards=[], base_rewards=[],
            qpos=[base.env.data.qpos.copy()], qvel=[base.env.data.qvel.copy()],
            time_s=[float(base.env.data.time)], torques=[])
        while True:
            arrays['observations'].append(obs.copy())
            action = model.predict(obs, deterministic=False)[0]
            obs, reward, done, truncated, info = env.step(action)
            arrays['actions'].append(action.copy())
            arrays['rewards'].append(reward)
            arrays['base_rewards'].append(info['base_reward'])
            arrays['qpos'].append(base.env.data.qpos.copy())
            arrays['qvel'].append(base.env.data.qvel.copy())
            arrays['time_s'].append(float(base.env.data.time))
            arrays['torques'].append(base.last_torque.copy())
            if done or truncated:
                break
        name = f'{case}_{episode:02d}'
        path = Path(output) / (name + '.npz')
        np.savez_compressed(path, **arrays)
        row = dict(case=case, episode=episode, seed=seed,
            split='train' if episode < 4 else 'validation', steps=len(arrays['actions']),
            success=info['success'], reason=info['reason'], file=path.name,
            trajectory_sha256=sha(path), info=info)
        (Path(output) / (name + '.json')).write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(json.dumps(dict(case=case, episode=episode, success=info['success'])), flush=True)
        del base
    env.close()
    return rows


def main():
    out = Path('outputs/closure_critic_data_20261024_v1')
    out.mkdir(exist_ok=False)
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    cache_gate = json.loads((CACHE / 'summary.json').read_text())
    assert cache_gate['passed']
    sources = {name: sha(name) for name in (
        str(Path(__file__).resolve()), 'training/closure_curriculum_env.py',
        'training/closure_curriculum_env_v2.py', 'training/closure_exploration_policy.py',
        'training/closure_quality_reward.py')}
    protocol = dict(parent_checkpoint_sha256=sha(PARENT / 'policy.zip'),
        cache_summary_sha256=sha(CACHE / 'summary.json'), cases=manifest['evaluation_cases'],
        episodes_per_case=6, training_episodes_per_case=4, validation_episodes_per_case=2,
        sources=sources, scope='Fixed parent actor with terminal exploration. Entire episodes split before collection. Known-case return calibration, not independent climbing generalization.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(collect, [(case, i, str(out.resolve())) for i, case in enumerate(manifest['evaluation_cases'])]))
    rows = [row for group in groups for row in group]
    result = dict(passed=len(rows) == 30, rows=rows, protocol_sha256=sha(out / 'protocol.json'))
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'collector_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(completed=len(rows), success=sum(r['success'] for r in rows))), flush=True)


if __name__ == '__main__':
    main()
