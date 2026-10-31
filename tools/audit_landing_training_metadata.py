"""Verify landing-window training identity and record inherited reward-label correction."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha, state_difference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    m = json.loads((a.experiment/'manifest.json').read_text())
    s = json.loads((a.experiment/'summary.json').read_text())
    assert m['training_segment'] == 'left_foot_landing_and_closure'
    source = Path(m['critic_initialization'])/'policy_initial.zip'
    assert sha(source) == m['initial_checkpoint_sha256']
    initial = PPO.load(source, device='cpu')
    before = PPO.load(a.experiment/'policy_before.zip', device='cpu')
    after = PPO.load(a.experiment/'policy.zip', device='cpu')
    assert sha(a.experiment/'policy.zip') == s['checkpoint_sha256']
    assert state_difference(initial.policy.state_dict(), before.policy.state_dict()) == 0
    assert state_difference(initial.policy.optimizer.state_dict(), before.policy.optimizer.state_dict()) == 0
    assert before.num_timesteps == s['parent_steps']
    assert after.num_timesteps-before.num_timesteps == s['additional_steps'] == m['training_additional_steps']
    assert before.n_envs == m['environments'] == 4
    assert before.observation_space.shape == (148,) and before.action_space.shape == (26,)
    evidence = {}
    for name in ('tools/train_landing_window_curriculum_v2.py', 'training/landing_controller_curriculum_env.py',
                 'training/closure_combined_curriculum_env.py', 'training/closure_curriculum_env_v3.py', 'training/closure_quality_reward.py'):
        expected = next(value for key, value in m['sources'].items() if key.replace('\\', '/') == name)
        assert sha(a.experiment/'source_snapshot'/name) == expected == sha(name)
        evidence[name] = expected
    assert m['quality_bonus_weight'] == .1 and m['suffix_start_seconds'] == 70.2
    training_source=(a.experiment/'source_snapshot/tools/train_landing_window_curriculum_v2.py').read_text()
    assert "model.set_random_seed(20261024)" in training_source and m['seed'] == 20261024
    result = dict(verified=True, manifest_sha256=sha(a.experiment/'manifest.json'),
                  summary_sha256=sha(a.experiment/'summary.json'), source_evidence=evidence,
                  inherited_reward_label=m['training_reward_contract'],
                  actual_reward='Base strict task reward plus nonnegative ClosureQualityReward, weight 0.1.',
                  actual_steps=s['additional_steps'], explicit_runtime_seed=m['seed'], inherited_saved_model_seed=before.seed,
                  initial_policy_and_optimizer_exact=True, source_sha256=sha(__file__),
                  scope='Metadata correction only. Frozen sources implement positive shaping; the inherited minus-terminal-cost label is obsolete. No simulation, reward, thresholds or existing manifests changed.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
