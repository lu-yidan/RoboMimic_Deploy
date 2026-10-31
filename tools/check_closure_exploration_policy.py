"""Verify phase-conditioned PPO distributions and saved deterministic actions."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_exploration_policy import ClosureExplorationPolicy
from tools.continue_vector_full_cycle_training import sha, state_difference


def main():
    torch.set_num_threads(1)
    parent = Path('outputs/closure_margin_control98304_v1')
    audit = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed'] and sha(parent / 'policy.zip') == audit['checkpoint_sha256']
    original = PPO.load(parent / 'policy.zip', device='cpu')
    changed = PPO.load(parent / 'policy.zip', device='cpu',
                       custom_objects={'policy_class': ClosureExplorationPolicy})
    assert state_difference(original.policy.state_dict(), changed.policy.state_dict()) == 0.
    assert state_difference(original.policy.optimizer.state_dict(), changed.policy.optimizer.state_dict()) == 0.
    with np.load(parent / 'after_foot_z_plus15.npz') as z:
        sample = torch.as_tensor(z['observations'][-1:]).repeat(5, 1)
    sample[:, 105] = 1. - torch.tensor([0., 99., 99.66, 101.16, 113.66]) / 113.66
    with torch.no_grad():
        before = original.policy.get_distribution(sample).distribution
        after = changed.policy.get_distribution(sample).distribution
        assert torch.equal(before.mean, after.mean)
        ratios = after.stddev / before.stddev
        assert torch.allclose(ratios[:3], torch.ones_like(ratios[:3]), atol=1e-5)
        assert torch.allclose(ratios[-1, :12], torch.full((12,), 4.))
        assert torch.equal(ratios[:, 12:], torch.ones_like(ratios[:, 12:]))
        actions, values, logp = changed.policy(sample)
        ev, ep, entropy = changed.policy.evaluate_actions(sample, actions)
        assert torch.equal(values, ev) and torch.equal(logp, ep)
        manual = torch.distributions.Normal(after.mean, after.stddev).log_prob(actions).sum(-1)
        assert torch.allclose(manual, logp, atol=1e-6)
        assert torch.isfinite(entropy).all()
    # Sampled rollout actions are detached, matching PPO's stored actions.
    changed.policy.optimizer.zero_grad()
    _, lp, ent = changed.policy.evaluate_actions(sample, actions.detach())
    (-lp.mean() - .001 * ent.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in changed.policy.parameters() if p.grad is not None)
    changed.policy.optimizer.zero_grad()
    rows = []
    for row in audit['trajectories']:
        if row['label'] != 'after':
            continue
        path = parent / f"after_{row['case']}.npz"
        assert sha(path) == row['trajectory_sha256']
        maximum = 0.
        with np.load(path) as z:
            for obs, expected in zip(z['observations'], z['actions']):
                action = changed.predict(obs, deterministic=True)[0]
                maximum = max(maximum, float(abs(action - expected).max()))
        assert maximum == 0.
        rows.append(dict(case=row['case'], deterministic_action_max_difference=maximum))
    out = Path('outputs/closure_exploration_gate_20261024_v1')
    out.mkdir(exist_ok=False)
    changed.save(out / 'policy_initial')
    restored = PPO.load(out / 'policy_initial.zip', device='cpu')
    assert isinstance(restored.policy, ClosureExplorationPolicy)
    assert state_difference(changed.policy.state_dict(), restored.policy.state_dict()) == 0.
    assert torch.equal(restored.policy.get_distribution(sample).distribution.stddev, after.stddev)
    alpha = .95 ** 4  # Four controller updates hold each 50 Hz actor command.
    filtered_factor = np.sqrt((1 - alpha) / (1 + alpha))
    result = dict(passed=True, trajectories=rows, initial_parameter_max_difference=0.,
        initial_optimizer_max_difference=0., phase_std_ratios=ratios.tolist(),
        source_sha256=sha(__file__), policy_source_sha256=sha('training/closure_exploration_policy.py'),
        checkpoint_sha256=sha(out / 'policy_initial.zip'), parent_checkpoint_sha256=sha(parent / 'policy.zip'),
        reference_noise_std_rad_before=float(.015 * np.exp(original.policy.log_std.detach().numpy()[4]) * filtered_factor),
        reference_noise_std_rad_after=float(4 * .015 * np.exp(original.policy.log_std.detach().numpy()[4]) * filtered_factor),
        scope='Analytic stationary filtered reference noise assumes fixed mean and independent actor samples; not actual joint motion. Deterministic replay matches saved observations, not a second complete dynamic rollout. Training distribution intentionally changes only terminal leg exploration; PPO likelihood and entropy remain consistent.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'gate_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
