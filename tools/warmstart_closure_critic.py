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
    out = Path('outputs/closure_critic_warmstart_20261024_v1')
    m = json.loads((parent / 'manifest.json').read_text())
    audit = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed'] and sha(parent / 'policy.zip') == audit['checkpoint_sha256']
    assert sha(parent / 'manifest.json') == audit['experiment_manifest_sha256']
    model = PPO.load(parent / 'policy.zip', device='cpu',
                     custom_objects={'policy_class': ClosureExplorationPolicy})
    original = {k: v.clone() for k, v in model.policy.state_dict().items()}
    env = FullCycleResidualEnv(plan=m['reference'], cases=m['evaluation_cases'])
    env.reset(options={'case': 'foot_z_plus15'})
    qadr = env.env.qadr.copy()
    env.close()
    observations, targets, cases, evidence = [], [], [], {}
    count = round(99.66 / .02)
    for case in m['evaluation_cases']:
        path = parent / f'after_{case}.npz'
        record = next(r for r in audit['trajectories'] if r['label'] == 'after' and r['case'] == case)
        assert sha(path) == record['trajectory_sha256']
        evidence[str(path)] = sha(path)
        with np.load(path) as z:
            obs, q, ts, rewards = (z[k].copy() for k in ('observations', 'qpos', 'time_s', 'rewards'))
        shaped = rewards[count:].copy()
        for j, i in enumerate(range(count, len(rewards))):
            closure = float(abs(q[i + 1, qadr] - q[0, qadr]).max())
            rise = float(q[i + 1, 2] - q[0, 2])
            u = float(np.clip((ts[i + 1] - 99.66) / 3., 0., 1.))
            shaped[j] += quality_bonus(closure, rise, u * u * (3 - 2 * u))
        returns = np.empty_like(shaped)
        running = 0.
        for j in range(len(shaped) - 1, -1, -1):
            running = shaped[j] + model.gamma * running
            returns[j] = running
        observations.append(obs[count:])
        targets.append(returns)
        cases.extend([case] * len(returns))
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

    history = [dict(updates=0, **metrics(before))]
    for update in range(1, 6001):
        ids = torch.randint(len(obs), (256,))
        value = model.policy.value_net(model.policy.mlp_extractor.forward_critic(features[ids]))
        loss = ((value - target[ids]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.)
        optimizer.step()
        if update % 200 == 0:
            with torch.no_grad():
                row = dict(updates=update, **metrics(predict()))
            history.append(row)
            if row['explained_variance'] >= .98 and row['mse'] <= .25:
                break
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
        values_before=before.numpy(), values_after=after.numpy(), cases=np.asarray(cases))
    final = metrics(after)
    result = dict(passed=bool(final['explained_variance'] >= .98 and final['mse'] <= .25),
        samples=len(obs), critic_gradient_updates=update, before=metrics(before), after=final, history=history,
        frozen_actor_and_features_exact=True, critic_optimizer_moments_cleared=True,
        parent_checkpoint_sha256=sha(parent / 'policy.zip'), checkpoint_sha256=sha(out / 'policy_initial.zip'),
        targets_sha256=sha(out / 'value_targets.npz'), evidence=evidence,
        sources={p: sha(p) for p in [str(Path(__file__).resolve()), 'training/closure_quality_reward.py',
                                  'training/closure_exploration_policy.py']},
        scope='Critic-only supervised warm start on all known deterministic parent suffixes under positive quality reward. In-sample fit, not held-out value accuracy or improved policy success. Exploratory returns may differ; subsequent PPO and complete-cycle evaluation are mandatory.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'warmstart_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
