"""Fit only critic heads to known deterministic suffix returns under new reward.

Actor/geometry features and action distribution parameters stay unchanged.
These are in-sample return targets, not evidence of better policy performance.
"""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_exploration_policy import ClosureExplorationPolicy
from training.closure_quality_reward import quality_bonus
from training.full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha, state_difference


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    parent = Path('outputs/closure_margin_control98304_v1')
    out = Path('outputs/closure_critic_warmstart_20261024_v2')
    m = json.loads((parent / 'manifest.json').read_text())
    audit = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed'] and sha(parent / 'policy.zip') == audit['checkpoint_sha256']
    assert sha(parent / 'manifest.json') == audit['experiment_manifest_sha256']
    model = PPO.load(parent / 'policy.zip', device='cpu',
                     custom_objects={'policy_class': ClosureExplorationPolicy})
    original = {k: v.clone() for k, v in model.policy.state_dict().items()}
    dataset = Path('outputs/closure_critic_data_20261024_v1')
    data_summary = json.loads((dataset / 'summary.json').read_text())
    assert data_summary['passed']
    assert data_summary['protocol_sha256'] == sha(dataset / 'protocol.json')
    observations, targets, cases, evidence, splits = [], [], [], {}, []
    for row in data_summary['rows']:
        path = dataset / row['file']
        assert sha(path) == row['trajectory_sha256']
        evidence[str(path)] = sha(path)
        with np.load(path) as z:
            episode_obs, rewards = z['observations'].copy(), z['rewards'].copy()
        returns = np.empty_like(rewards)
        running = 0.
        for j in range(len(rewards)-1, -1, -1):
            running = rewards[j] + model.gamma * running
            returns[j] = running
        observations.append(episode_obs)
        targets.append(returns)
        cases.extend([row['case']] * len(returns))
        splits.extend([row['split']] * len(returns))
    obs = torch.as_tensor(np.concatenate(observations), dtype=torch.float32)
    target = torch.as_tensor(np.concatenate(targets), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        features = model.policy.extract_features(obs).detach()
        before = model.policy.predict_values(obs).detach()
    params = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=.001)

    def predict():
        return model.policy.value_net(model.policy.mlp_extractor.forward_critic(features))

    def metrics(values):
        residual = target - values
        return dict(mse=float((residual ** 2).mean()),
            explained_variance=float(1 - residual.var(unbiased=False) / target.var(unbiased=False)))

    train_ids = torch.tensor(np.flatnonzero(np.asarray(splits) == 'train'))
    val_ids = torch.tensor(np.flatnonzero(np.asarray(splits) == 'validation'))
    def split_metrics(values, ids):
        residual = target[ids] - values[ids]
        return dict(mse=float((residual**2).mean()),
                    explained_variance=float(1-residual.var(unbiased=False)/target[ids].var(unbiased=False)))
    history = [dict(updates=0, train=split_metrics(before, train_ids), validation=split_metrics(before, val_ids))]
    best_loss = float('inf')
    best_state = None
    best_update = 0
    for update in range(1, 6001):
        ids = train_ids[torch.randint(len(train_ids), (256,))]
        value = model.policy.value_net(model.policy.mlp_extractor.forward_critic(features[ids]))
        loss = ((value - target[ids]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.)
        optimizer.step()
        if update % 200 == 0:
            with torch.no_grad():
                row = dict(updates=update, train=split_metrics(predict(), train_ids), validation=split_metrics(predict(), val_ids))
            history.append(row)
            print(json.dumps(row), flush=True)
            if row['validation']['mse'] < best_loss:
                best_loss = row['validation']['mse']
                best_state = {k: v.clone() for k, v in model.policy.state_dict().items()}
                best_update = update
            if update - best_update >= 1000:
                break
    model.policy.load_state_dict(best_state)
    with torch.no_grad():
        after = predict().detach()
    frozen = [k for k in original if not k.startswith(('mlp_extractor.value_net.', 'value_net.'))]
    assert all(torch.equal(original[k], model.policy.state_dict()[k]) for k in frozen)
    # Clear stale Adam moments only for the deliberately refitted critic heads.
    for parameter in params:
        model.policy.optimizer.state.pop(parameter, None)
    out.mkdir(exist_ok=False)
    model.save(out / 'policy_initial')
    restored = PPO.load(out / 'policy_initial.zip', device='cpu')
    assert state_difference(model.policy.state_dict(), restored.policy.state_dict()) == 0.
    assert state_difference(model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()) == 0.
    np.savez_compressed(out / 'value_targets.npz', observations=obs.numpy(), returns=target.numpy(),
        values_before=before.numpy(), values_after=after.numpy(), cases=np.asarray(cases), splits=np.asarray(splits))
    final = split_metrics(after, val_ids)
    result = dict(passed=bool(final['explained_variance'] >= .98 and final['mse'] <= 1.),
        samples=len(obs), selected_update=best_update, train=split_metrics(after, train_ids), validation_before=split_metrics(before, val_ids), critic_gradient_updates=update, before=metrics(before), after=final, history=history,
        frozen_actor_and_features_exact=True, critic_optimizer_moments_cleared=True,
        parent_checkpoint_sha256=sha(parent / 'policy.zip'), checkpoint_sha256=sha(out / 'policy_initial.zip'),
        targets_sha256=sha(out / 'value_targets.npz'), evidence=evidence,
        sources={p: sha(p) for p in [str(Path(__file__).resolve()), 'training/closure_quality_reward.py',
                                  'training/closure_exploration_policy.py']},
        scope='Critic-only fixed-policy exploratory return calibration. 20 whole episodes fit, 10 whole episodes validation used to select checkpoint; validation is not an independent test set. Actor and feature parameters frozen. Five known conditions only. Full-cycle policy evaluation remains mandatory.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'warmstart_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
