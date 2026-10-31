"""Fit only critic heads to sampled landing-window returns under the declared reward.

Actor/geometry features and action distribution parameters stay unchanged.
These are in-sample return targets, not evidence of better policy performance.
"""
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',type=Path,required=True);args=parser.parse_args()
    seed=args.seed;seed_gate=json.loads((seed/'summary.json').read_text())
    parent=Path(seed_gate['parent']);out=args.output
    m = json.loads((parent / 'manifest.json').read_text())
    audit = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed'] and sha(parent / 'policy.zip') == audit['checkpoint_sha256']
    assert sha(parent / 'manifest.json') == audit['experiment_manifest_sha256']
    seed_gate = json.loads((seed/'summary.json').read_text())
    assert seed_gate['passed'] and sha(seed/'policy_initial.zip') == seed_gate['checkpoint_sha256']
    model = PPO.load(seed/'policy_initial.zip', device='cpu')
    original = {k: v.clone() for k, v in model.policy.state_dict().items()}
    resume_summary=json.loads((args.resume/'summary.json').read_text())
    assert resume_summary['seed_checkpoint_sha256']==sha(seed/'policy_initial.zip')
    assert resume_summary['checkpoint_sha256']==sha(args.resume/'policy_initial.zip')
    model=PPO.load(args.resume/'policy_initial.zip',device='cpu')
    frozen_names=[k for k in original if not k.startswith(('mlp_extractor.value_net.','value_net.'))]
    assert all(torch.equal(original[k],model.policy.state_dict()[k]) for k in frozen_names)
    dataset=args.dataset
    data_summary = json.loads((dataset / 'summary.json').read_text())
    assert data_summary['passed']
    assert data_summary['protocol_sha256'] == sha(dataset / 'protocol.json')
    data_protocol=json.loads((dataset/'protocol.json').read_text())
    assert data_protocol['seed_checkpoint_sha256']==sha(seed/'policy_initial.zip')
    for path,value in data_protocol['sources'].items():assert sha(path)==value
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
    optimizer = torch.optim.Adam(params, lr=.0003)

    def predict():
        return model.policy.value_net(model.policy.mlp_extractor.forward_critic(features))

    def metrics(values):
        residual = target - values
        return dict(mse=float((residual ** 2).mean()),
            explained_variance=float(1 - residual.var(unbiased=False) / target.var(unbiased=False)))

    train_ids = torch.tensor(np.flatnonzero(np.asarray(splits) == 'train'))
    val_ids = torch.tensor(np.flatnonzero(np.asarray(splits) == 'validation'))
    test_ids = torch.tensor(np.flatnonzero(np.asarray(splits) == 'test'))
    def split_metrics(values, ids):
        residual = target[ids] - values[ids]
        return dict(mse=float((residual**2).mean()),
                    explained_variance=float(1-residual.var(unbiased=False)/target[ids].var(unbiased=False)))
    history = [dict(updates=0, train=split_metrics(before, train_ids), validation=split_metrics(before, val_ids))]
    best_loss = split_metrics(before,val_ids)['mse']
    best_state = {k:v.clone() for k,v in model.policy.state_dict().items()}
    best_update = 0
    for update in range(1, 24001):
        ids = train_ids[torch.randint(len(train_ids), (1024,))]
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
            if update - best_update >= 4000:
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
    result = dict(resume_checkpoint_sha256=sha(args.resume/'policy_initial.zip'),resume_summary_sha256=sha(args.resume/'summary.json'),calibration_protocol=dict(max_updates=24000,batch_size=1024,learning_rate=.0003,patience_updates=4000,selection='validation_mse'),dataset_protocol_sha256=sha(dataset/'protocol.json'),dataset_summary_sha256=sha(dataset/'summary.json'),passed=bool(final['explained_variance'] >= .8 and final['mse'] <= .5*split_metrics(before,val_ids)['mse'] and split_metrics(after,test_ids)['mse'] < split_metrics(before,test_ids)['mse']),
        test=split_metrics(after,test_ids),test_before=split_metrics(before,test_ids),
        seed_checkpoint_sha256=sha(seed/'policy_initial.zip'),
        samples=len(obs), selected_update=best_update, train=split_metrics(after, train_ids), validation_before=split_metrics(before, val_ids), critic_gradient_updates=update, before=metrics(before), after=final, history=history,
        frozen_actor_and_features_exact=True, critic_optimizer_moments_cleared=True,
        parent_checkpoint_sha256=sha(parent / 'policy.zip'), checkpoint_sha256=sha(out / 'policy_initial.zip'),
        targets_sha256=sha(out / 'value_targets.npz'), evidence=evidence,
        sources={p: sha(p) for p in [str(Path(__file__).resolve()), 'training/closure_quality_reward.py',
                                  'training/closure_exploration_policy.py', 'training/closure_seeded_policy.py', 'training/closure_combined_curriculum_env.py','training/landing_controller_curriculum_env.py','training/closure_foreaft_reference.py']},
        scope='Search-seeded actor held fixed. Twenty-four fresh episodes: eight fit, eight validation episodes select checkpoint, eight test episodes assessed only afterward. This reuses previously inspected validation/test episodes as development diagnostics; no fresh held-out claim. Eight known cases, 70.2-second start, same declared fixed controller and sampling distribution. Calibrator test does not establish policy generalization. Full-cycle policy evaluation remains mandatory.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'warmstart_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
