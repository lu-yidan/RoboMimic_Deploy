"""Require equal budgets and audited original-task evaluations before comparison."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import state_difference
from tools.summarize_full_cycle_experiment import summarize, sha


MATCHED_FIELDS = ('parent_checkpoint_sha256', 'parent_steps', 'config', 'seed',
    'environments', 'rollout_transitions', 'additional_steps_requested',
    'training_cases', 'evaluation_cases', 'normal_bias_y_deg', 'force_history_memory',
    'reference_sha256', 'bundle_manifest_sha256', 'training_reward_contract',
    'evaluation_reward_contract', 'shaping_gamma', 'protocol_sha256',
    'potential_gate_sha256', 'sources', 'critic_initialization')


def validate_pair(manifests, summaries):
    if [m['shaping_weight'] for m in manifests] != [0., 4.]:
        raise ValueError('Expected unshaped then shaped arms')
    for field in MATCHED_FIELDS:
        if manifests[0][field] != manifests[1][field]:
            raise ValueError('Unmatched field: '+field)
    for m, s in zip(manifests, summaries):
        if s.get('evaluation_pending') or s.get('stopped_by_request'):
            raise ValueError('Unfinished experiment')
        if s['additional_steps'] != m['additional_steps_requested']:
            raise ValueError('Incomplete requested budget')
        if s['actual_steps'] != s['parent_steps']+s['additional_steps'] or s['parent_steps'] != m['parent_steps']:
            raise ValueError('Inconsistent continuation budget')
        if m['shaping_gamma'] != m['config']['gamma']:
            raise ValueError('Discount mismatch')
    if summaries[0]['actual_steps'] != summaries[1]['actual_steps']:
        raise ValueError('Unequal training budgets')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--shaped', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directories = [args.control, args.shaped]
    manifests = [json.loads((d/'manifest.json').read_text()) for d in directories]
    summaries = [json.loads((d/'summary.json').read_text()) for d in directories]
    validate_pair(manifests, summaries)
    torch.set_num_threads(1)
    initials = [PPO.load(d/'policy_before.zip', device='cpu') for d in directories]
    parent_path = Path(manifests[0]['parent_experiment'])/'policy.zip'
    if sha(parent_path) != manifests[0]['parent_checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    parent = PPO.load(parent_path, device='cpu')
    for model in initials:
        if state_difference(model.policy.state_dict(), parent.policy.state_dict()) != 0.:
            raise ValueError('Initial actor/critic differs')
        if state_difference(model.policy.optimizer.state_dict(), parent.policy.optimizer.state_dict()) != 0.:
            raise ValueError('Initial optimizer differs')
    for d, m in zip(directories, manifests):
        if sha(d/'matched_protocol.json') != m['protocol_sha256']:
            raise ValueError('Saved protocol changed')
    comparisons = [summarize(d) for d in directories]
    cases = []
    for a, b in zip(comparisons[0]['cases'], comparisons[1]['cases']):
        if a['case'] != b['case']:
            raise ValueError('Case ordering differs')
        cases.append(dict(case=a['case'], in_training_set=a['in_training_set'],
            control=a['after'], shaped=b['after'],
            both_accepted=bool(a['after']['accepted'] and b['after']['accepted'])))
    result = dict(control_accepted=comparisons[0]['after_accepted'],
        shaped_accepted=comparisons[1]['after_accepted'], case_count=len(cases),
        success_count_difference=comparisons[1]['after_accepted']-comparisons[0]['after_accepted'],
        fixed_cases=[c['case'] for c in cases if c['shaped']['accepted'] and not c['control']['accepted']],
        regressed_cases=[c['case'] for c in cases if c['control']['accepted'] and not c['shaped']['accepted']],
        additional_steps_per_arm=summaries[0]['additional_steps'],
        initial_actor_critic_optimizer_exact=True, matched_fields=MATCHED_FIELDS, cases=cases,
        coverage=[dict(completed_by_case=c['completed_by_case'], missing=c['training_cases_without_completed_episode']) for c in comparisons],
        evidence={str(d): {name:sha(d/name) for name in ['manifest.json','summary.json','independent_training_audit.json','policy_before.zip','policy.zip','matched_protocol.json']} for d in directories},
        source_sha256=sha(__file__),
        scope='One matched seed on known cases. Original strict acceptance and reward used for evaluation. Shaped training returns are not evaluation performance. Unequal-duration peaks are not ranked.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2))
    (args.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('control_accepted','shaped_accepted','case_count','fixed_cases','regressed_cases')}))


if __name__ == '__main__':
    main()
