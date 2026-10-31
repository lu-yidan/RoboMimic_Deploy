"""Validate a frozen calibrated critic on fresh episodes, without fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import sha


def main():
    p = argparse.ArgumentParser()
    for name in ('seed', 'critic', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    gate = json.loads((a.critic / 'summary.json').read_text())
    dataset = json.loads((a.dataset / 'summary.json').read_text())
    protocol = json.loads((a.dataset / 'protocol.json').read_text())
    assert gate['passed'] and dataset['passed']
    assert gate['checkpoint_sha256'] == sha(a.critic / 'policy_initial.zip')
    assert protocol['seed_checkpoint_sha256'] == sha(a.seed / 'policy_initial.zip')
    assert dataset['protocol_sha256'] == sha(a.dataset / 'protocol.json')
    assert len(dataset['rows']) == 8
    assert all(r['split'] == 'fresh_validation' for r in dataset['rows'])
    model = PPO.load(a.critic / 'policy_initial.zip', device='cpu')
    seed = PPO.load(a.seed / 'policy_initial.zip', device='cpu')
    names = [k for k in seed.policy.state_dict()
             if not k.startswith(('mlp_extractor.value_net.', 'value_net.'))]
    assert all(torch.equal(seed.policy.state_dict()[k], model.policy.state_dict()[k])
               for k in names)
    arrays = {k: [] for k in ('targets', 'before', 'after')}
    rows, evidence = [], {}

    def metrics(target, value):
        return dict(mse=float(np.mean((target - value) ** 2)),
                    explained_variance=float(1 - np.var(target - value) / np.var(target)))

    for row in dataset['rows']:
        path = a.dataset / row['file']
        assert sha(path) == row['trajectory_sha256']
        evidence[str(path.resolve())] = sha(path)
        with np.load(path) as z:
            obs = torch.as_tensor(z['observations'].copy(), dtype=torch.float32)
            rewards = z['rewards'].copy()
        target = np.empty(len(rewards), dtype=np.float64)
        total = 0.
        for i in range(len(rewards) - 1, -1, -1):
            total = float(rewards[i]) + model.gamma * total
            target[i] = total
        with torch.no_grad():
            before = seed.policy.predict_values(obs).numpy().ravel()
            after = model.policy.predict_values(obs).numpy().ravel()
            d0 = seed.policy.get_distribution(obs).distribution
            d1 = model.policy.get_distribution(obs).distribution
            assert torch.equal(d0.mean, d1.mean) and torch.equal(d0.stddev, d1.stddev)
        for key, value in zip(arrays, (target, before, after)):
            arrays[key].append(value)
        rows.append(dict(case=row['case'], samples=len(target),
                         before=metrics(target, before), after=metrics(target, after)))
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    before = metrics(arrays['targets'], arrays['before'])
    after = metrics(arrays['targets'], arrays['after'])
    passed = after['explained_variance'] >= .8 and after['mse'] <= .5 * before['mse']
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output / 'predictions.npz', **arrays)
    result = dict(passed=bool(passed), before=before, after=after, rows=rows,
                  frozen_actor_and_distribution_exact=True, evidence=evidence,
                  critic_summary_sha256=sha(a.critic / 'summary.json'),
                  critic_checkpoint_sha256=sha(a.critic / 'policy_initial.zip'),
                  dataset_summary_sha256=sha(a.dataset / 'summary.json'),
                  dataset_protocol_sha256=sha(a.dataset / 'protocol.json'),
                  source_sha256=sha(__file__),
                  scope='Eight new stochastic episodes on known geometries. No updates or checkpoint selection on these data. Not policy-performance or unseen-geometry generalization evidence.')
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2))
    (a.output / 'validation_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError('Fresh-episode critic validation failed')


if __name__ == '__main__':
    main()
