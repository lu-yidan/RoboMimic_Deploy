"""Record active feedback-training settings without rewriting frozen manifests."""
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
    assert m['observation_contract'] == 'seeded-reference-153obs-5delta-10hz-v1'
    before = PPO.load(a.experiment/'policy_before.zip', device='cpu')
    warm = m['initial_critic'].startswith('value-only')
    initial = Path(m['critic_initialization'] if warm else m['feedback_initialization'])/'policy_initial.zip'
    assert sha(initial) == m['initial_checkpoint_sha256']
    source = PPO.load(initial, device='cpu')
    assert state_difference(before.policy.state_dict(), source.policy.state_dict()) == 0
    assert state_difference(before.policy.optimizer.state_dict(), source.policy.optimizer.state_dict()) == 0
    assert before.num_timesteps == source.num_timesteps == 0
    for name in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'target_kl', 'learning_rate', 'seed'):
        assert getattr(before, name) == m['config'][name]
    assert before.n_envs == m['environments'] == 4
    actual = dict(random_seed=m['config']['seed'], parent_feedback_steps=0,
        requested_macro_steps=m['feedback_training_macro_steps'],
        maximum_teacher_transitions=m['maximum_teacher_transitions'],
        feedback_decision_hz=10, teacher_hz=50, control_hz=200, physics_hz=2000,
        training_window_s=[99.66, 113.66], quality_bonus_weight=.1,
        reward='Base teacher-step reward plus positive ClosureQualityReward; sum five rewards with physical gamma powers.',
        physical_gamma=m['feedback_physical_gamma'], macro_gamma=before.gamma,
        initial_critic=m['initial_critic'])
    inherited = {key: m.get(key) for key in
        ('seed', 'parent_steps', 'additional_steps_requested', 'training_additional_steps', 'training_reward_contract')}
    evidence = {}
    for source in ('training/closure_curriculum_env.py', 'training/closure_curriculum_env_v3.py',
                   'training/closure_quality_reward.py', 'training/held_reference_command.py'):
        expected = next(value for key, value in m['sources'].items() if key.replace('\\', '/') == source)
        assert sha(source) == expected and sha(a.experiment/'source_snapshot'/source) == expected
        evidence[source] = expected
    result = dict(verified=True, actual_execution=actual, inherited_legacy_fields=inherited,
        has_legacy_conflicts=True, manifest_sha256=sha(a.experiment/'manifest.json'),
        restored_initial_checkpoint_sha256=sha(a.experiment/'policy_before.zip'),
        original_initial_checkpoint_sha256=sha(initial), source_evidence=evidence, source_sha256=sha(__file__),
        scope='Frozen manifest inherited obsolete parent training fields. Active settings are verified against '
              'the restored initial checkpoint, feedback-specific fields and frozen environment sources. '
              'This is an explicit metadata correction record, not a change to training, thresholds or original audit files. '
              'Final executed counts must come from summary.json, not the inherited legacy fields.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
