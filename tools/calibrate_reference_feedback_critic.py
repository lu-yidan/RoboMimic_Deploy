"""Prepare a value-only initialization; never update the feedback actor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha, state_difference


def main():
    p = argparse.ArgumentParser()
    for name in ('seed', 'dataset', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    seed = json.loads((a.seed/'summary.json').read_text())
    sp = json.loads((a.seed/'protocol.json').read_text())
    data = json.loads((a.dataset/'summary.json').read_text())
    dp = json.loads((a.dataset/'protocol.json').read_text())
    assert seed['passed'] and data['passed']
    assert sha(a.seed/'policy_initial.zip') == seed['checkpoint_sha256']
    assert sha(a.seed/'protocol.json') == seed['protocol_sha256']
    assert sha(a.dataset/'protocol.json') == data['protocol_sha256']
    sampled_seed = Path(dp['seed'])
    assert sha(sampled_seed/'policy_initial.zip') == dp['seed_checkpoint_sha256']
    assert sha(sampled_seed/'protocol.json') == dp['seed_protocol_sha256']
    assert seed['previous_checkpoint_sha256'] == dp['seed_checkpoint_sha256']
    assert seed['previous_parameters_exact']
    assert sp['teacher_sha256'] == dp['teacher_sha256'] and sp['cases'] == dp['cases']
    for source, digest in dp['sources'].items():
        assert sha(source) == digest
    for source, digest in sp['sources'].items():
        assert sha(source) == digest
    model = PPO.load(a.seed/'policy_initial.zip', device='cpu')
    sampled = PPO.load(sampled_seed/'policy_initial.zip', device='cpu')
    assert state_difference(model.policy.state_dict(), sampled.policy.state_dict()) == 0
    assert model.gamma == sampled.gamma == dp['gamma']
    assert model.action_space == sampled.action_space
    original = {k: v.clone() for k, v in model.policy.state_dict().items()}
    observations, targets, splits, cases, evidence = [], [], [], [], {}
    for row in data['rows']:
        path = a.dataset/row['file']
        assert sha(path) == row['trajectory_sha256']
        evidence[str(path)] = sha(path)
        with np.load(path) as z:
            obs, rewards = z['observations'].copy(), z['rewards'].copy()
        returns = np.empty_like(rewards)
        running = 0.
        for i in range(len(rewards)-1, -1, -1):
            running = float(rewards[i])+model.gamma*running
            returns[i] = running
        observations.append(obs)
        targets.append(returns)
        splits.extend([row['split']]*len(returns))
        cases.extend([row['case']]*len(returns))
    obs = torch.as_tensor(np.concatenate(observations), dtype=torch.float32)
    target = torch.as_tensor(np.concatenate(targets), dtype=torch.float32).reshape(-1, 1)
    assert obs.shape[1] == 153 and torch.isfinite(obs).all() and torch.isfinite(target).all()
    ids = {split: torch.tensor(np.flatnonzero(np.asarray(splits) == split))
           for split in ('train', 'validation', 'test')}
    assert all(len(v) == 1120 for v in ids.values())
    with torch.no_grad():
        # Metadata-only space correction must preserve the sampled distribution.
        d1 = model.policy.get_distribution(obs).distribution
        d2 = sampled.policy.get_distribution(obs).distribution
        assert torch.equal(d1.mean, d2.mean) and torch.equal(d1.stddev, d2.stddev)
        features = model.policy.extract_features(obs).detach()
        before = model.policy.predict_values(obs).detach()
    protocol = dict(max_updates=12000, batch_size=256, learning_rate=.0003, patience_updates=2000,
        selection='minimum validation MSE at 200-update intervals; test inspected only after selection',
        gate=dict(validation_explained_variance_min=.8, validation_mse_ratio_max=.5, test_mse_improvement=True),
        seed_checkpoint_sha256=seed['checkpoint_sha256'], dataset_seed_sha256=dp['seed_checkpoint_sha256'],
        metadata_migration=dict(all_network_parameters_exact=True, all_sampled_means_and_std_exact=True),
        dataset_protocol_sha256=sha(a.dataset/'protocol.json'), dataset_summary_sha256=sha(a.dataset/'summary.json'),
        sources={str(Path(__file__)): sha(__file__)}, evidence=evidence,
        scope='Value-only initialization for a possible later PPO run. Known-case exploration data; '
              'current fresh-critic PPO continues unchanged. No action improvement or independent geometry claim.')
    a.output.mkdir(exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    params = list(model.policy.mlp_extractor.value_net.parameters())+list(model.policy.value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=protocol['learning_rate'])

    def predict(x=features):
        return model.policy.value_net(model.policy.mlp_extractor.forward_critic(x))

    def metrics(value, split):
        index = ids[split]
        residual = target[index]-value[index]
        return dict(mse=float((residual**2).mean()),
            explained_variance=float(1-residual.var(unbiased=False)/target[index].var(unbiased=False)))

    best, best_loss, best_update = original, metrics(before, 'validation')['mse'], 0
    history = []
    for update in range(1, protocol['max_updates']+1):
        index = ids['train'][torch.randint(len(ids['train']), (protocol['batch_size'],))]
        loss = ((predict(features[index])-target[index])**2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.)
        optimizer.step()
        if update % 200 == 0:
            with torch.no_grad():
                value = predict()
                row = dict(update=update, train=metrics(value, 'train'), validation=metrics(value, 'validation'))
            history.append(row)
            print(json.dumps(row), flush=True)
            if row['validation']['mse'] < best_loss:
                best_loss, best_update = row['validation']['mse'], update
                best = {k: v.clone() for k, v in model.policy.state_dict().items()}
            if update-best_update >= protocol['patience_updates']:
                break
    model.policy.load_state_dict(best)
    frozen = [k for k in original if not k.startswith(('mlp_extractor.value_net.', 'value_net.'))]
    assert all(torch.equal(original[k], model.policy.state_dict()[k]) for k in frozen)
    for parameter in params:
        model.policy.optimizer.state.pop(parameter, None)
    with torch.no_grad():
        after = predict().detach()
        distribution = model.policy.get_distribution(obs).distribution
        assert torch.equal(distribution.mean, d1.mean) and torch.equal(distribution.stddev, d1.stddev)
    model.save(a.output/'policy_initial')
    restored = PPO.load(a.output/'policy_initial.zip', device='cpu')
    assert state_difference(model.policy.state_dict(), restored.policy.state_dict()) == 0
    assert state_difference(model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()) == 0
    validation = metrics(after, 'validation')
    passed = (validation['explained_variance'] >= .8 and validation['mse'] <= .5*metrics(before, 'validation')['mse']
              and metrics(after, 'test')['mse'] < metrics(before, 'test')['mse'])
    np.savez_compressed(a.output/'value_targets.npz', observations=obs.numpy(), returns=target.numpy(),
        values_before=before.numpy(), values_after=after.numpy(), cases=np.asarray(cases), splits=np.asarray(splits))
    result = dict(passed=bool(passed), selected_update=best_update, updates=update,
        validation=validation, validation_before=metrics(before, 'validation'),
        train=metrics(after, 'train'), test=metrics(after, 'test'), test_before=metrics(before, 'test'),
        history=history, frozen_actor_and_distribution_exact=True, protocol_sha256=sha(a.output/'protocol.json'),
        checkpoint_sha256=sha(a.output/'policy_initial.zip'), value_targets_sha256=sha(a.output/'value_targets.npz'))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main()
