"""Summarize audited matched cases, preserving failures and unequal horizons."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(experiment):
    manifest = json.loads((experiment/'manifest.json').read_text())
    summary = json.loads((experiment/'summary.json').read_text())
    audit = json.loads((experiment/'independent_training_audit.json').read_text())
    if not audit['passed'] or summary.get('evaluation_pending'):
        raise ValueError('Completed and independently audited experiment required')
    for field, filename in [('experiment_manifest_sha256', 'manifest.json'),
                            ('experiment_summary_sha256', 'summary.json'),
                            ('checkpoint_sha256', 'policy.zip')]:
        if audit[field] != sha(experiment/filename):
            raise ValueError('Evidence changed after independent audit: '+filename)
    acceptance = {(row['label'], row['case']): row for row in audit['trajectories']}
    cases = []
    for case in manifest['evaluation_cases']:
        row = {'case': case, 'in_training_set': case in manifest['training_cases']}
        for label in ('zero', 'after'):
            original = next(item for item in summary[label] if item['case'] == case)
            check = acceptance[label, case]
            trajectory = experiment/f'{label}_{case}.npz'
            if sha(trajectory) != check['trajectory_sha256']:
                raise ValueError('Trajectory changed after audit: '+str(trajectory))
            with np.load(trajectory) as states:
                action_peak = float(np.max(np.abs(states['actions'])))
                # The observation stores residual/.015, not radians.
                residual_peak = .015*float(np.max(np.abs(states['observations'][:, -26:])))
            row[label] = dict(accepted=check['independently_accepted'],
                runtime_success=original['success'], reason=original['reason'],
                simulation_time_s=original['time_s'], return_sum=original['return_sum'],
                base_peak_m=original['peaks']['base'],
                minimum_native_joint_margin_rad=original['minimum_native_joint_margin_rad'],
                max_abs_normalized_action=action_peak,
                max_sampled_filtered_residual_rad=residual_peak,
                endpoint_checks=check['independent_endpoints'])
        row['equal_duration'] = abs(row['zero']['simulation_time_s']-row['after']['simulation_time_s']) < 1e-7
        row['paired_full_success'] = row['zero']['accepted'] and row['after']['accepted']
        # Peak/margin changes on a shorter failed trajectory are not performance gains.
        row['accepted_pair_deltas'] = ({
            'base_peak_m': row['after']['base_peak_m']-row['zero']['base_peak_m'],
            'minimum_native_joint_margin_rad': row['after']['minimum_native_joint_margin_rad']-row['zero']['minimum_native_joint_margin_rad']
        } if row['paired_full_success'] else None)
        cases.append(row)
    episodes_file = experiment/'training_episodes.jsonl'
    episodes = [json.loads(line) for line in episodes_file.read_text().splitlines()] if episodes_file.exists() else []
    counts = Counter(row['case'] for row in episodes)
    successes = Counter(row['case'] for row in episodes if row['success'])
    result = dict(experiment=str(experiment.resolve()), contract=manifest['contract'],
        seed=manifest['seed'], actual_steps=summary['actual_steps'],
        additional_steps=summary.get('additional_steps', summary['actual_steps']),
        training_wall_seconds=summary['training_wall_seconds'],
        case_count=len(cases), zero_accepted=sum(row['zero']['accepted'] for row in cases),
        after_accepted=sum(row['after']['accepted'] for row in cases),
        fixed_cases=[row['case'] for row in cases if not row['zero']['accepted'] and row['after']['accepted']],
        regressed_cases=[row['case'] for row in cases if row['zero']['accepted'] and not row['after']['accepted']],
        completed_episodes=len(episodes), completed_by_case=dict(counts), successful_by_case=dict(successes),
        training_cases_without_completed_episode=sorted(set(manifest['training_cases'])-set(counts)),
        cases=cases, audit_sha256=sha(experiment/'independent_training_audit.json'), source_sha256=sha(__file__),
        scope='Single-seed declared-case comparison; evaluation-only geometries are not blind held-out tests. Residual peaks sampled at actor inputs (50 Hz), not all controller steps.')
    (experiment/'case_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('actual_steps', 'case_count', 'zero_accepted',
          'after_accepted', 'fixed_cases', 'regressed_cases', 'completed_by_case')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    summarize(parser.parse_args().experiment)
