"""Verify the two feedback runs differ in value initialization, not actor/settings."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha, state_difference


def main():
    p = argparse.ArgumentParser()
    for name in ('cold', 'warm', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    manifests = [json.loads((path/'manifest.json').read_text()) for path in (a.cold, a.warm)]
    cold, warm = [PPO.load(path/'policy_before.zip', device='cpu') for path in (a.cold, a.warm)]
    keys = ('config', 'contract', 'observation_contract', 'reference_sha256', 'bundle_manifest_sha256',
            'teacher_checkpoint_sha256', 'prefix_checkpoint_sha256', 'controller_modifier',
            'adaptive_reference_command', 'feedback_hold_steps', 'feedback_delta_scale',
            'feedback_physical_gamma', 'feedback_training_macro_steps', 'maximum_teacher_transitions',
            'training_episode_seconds', 'environments', 'training_cases', 'evaluation_cases')
    checks = {key: manifests[0][key] == manifests[1][key] for key in keys}
    checks['same_initialization_protocol'] = sha(a.cold/'initialization_protocol.json') == sha(a.warm/'initialization_protocol.json')
    state_cold, state_warm = cold.policy.state_dict(), warm.policy.state_dict()
    frozen = [key for key in state_cold if not key.startswith(('mlp_extractor.value_net.', 'value_net.'))]
    value = [key for key in state_cold if key not in frozen]
    checks['actor_and_distribution_exact'] = all(torch.equal(state_cold[key], state_warm[key]) for key in frozen)
    checks['critic_is_different'] = any(not torch.equal(state_cold[key], state_warm[key]) for key in value)
    checks['optimizer_states_exact'] = state_difference(cold.policy.optimizer.state_dict(), warm.policy.optimizer.state_dict()) == 0
    checks['same_initial_step_count'] = cold.num_timesteps == warm.num_timesteps == 0
    checks['actual_seed_matches'] = cold.seed == warm.seed == manifests[0]['config']['seed']
    sources = {key.replace('\\', '/'): digest for key, digest in manifests[0]['sources'].items()
               if key.replace('\\', '/').startswith('training/')}
    other = {key.replace('\\', '/'): digest for key, digest in manifests[1]['sources'].items()
             if key.replace('\\', '/').startswith('training/')}
    checks['same_environment_sources'] = sources == other
    result = dict(passed=all(checks.values()), checks=checks,
        manifest_sha256=[sha(path/'manifest.json') for path in (a.cold, a.warm)],
        initial_checkpoint_sha256=[sha(path/'policy_before.zip') for path in (a.cold, a.warm)],
        source_sha256=sha(__file__),
        caveats=['Only one paired random seed; no statistical superiority claim.',
                 'Evaluation order prioritizes boundary failures in the warm run; case membership is unchanged.',
                 'Legacy top-level training fields are stale; use config, feedback fields and separate metadata audits.',
                 'Same initialization distribution and settings do not imply identical trajectories after policy updates.'],
        scope='Experimental-design check before outcome comparison; not a dynamics or task-success certificate.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()
