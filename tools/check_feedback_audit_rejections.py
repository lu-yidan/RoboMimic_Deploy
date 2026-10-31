"""Negative controls for the feedback auditor; original artifacts remain untouched."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import sha
from tools.audit_feedback_reference_commands import feedback_checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    m = json.loads((a.experiment/'manifest.json').read_text())
    gate = json.loads((a.experiment/'zero_feedback_gate.json').read_text())
    assert gate['passed'] and gate['manifest_sha256'] == sha(a.experiment/'manifest.json')
    s = json.loads((a.experiment/'summary.json').read_text())
    row = s['after'][0]
    path = a.experiment/f"after_{row['case']}.npz"
    assert sha(path) == row['trajectory_sha256']
    with np.load(path) as z:
        original = {k: z[k].copy() for k in z.files}
    policy = PPO.load(a.experiment/'policy.zip', device='cpu')
    clean, _ = feedback_checks(m, original, policy)
    assert all(clean.values())
    rows = []
    mutations = [('feedback_action', 'feedback_actor_replay'),
                 ('filter_observation', 'feedback_observations'),
                 ('decision_index', 'feedback_schema'),
                 ('filter_trace', 'feedback_filter'),
                 ('held_command', 'feedback_command_hold')]
    for label, required_rejection in mutations:
        data = {k: v.copy() for k, v in original.items()}
        index = round(99.66/.005)+4
        if label == 'feedback_action':
            data['feedback_actions'][0, 1] += .1
        elif label == 'filter_observation':
            data['feedback_observations'][0, -1] += .02
        elif label == 'decision_index':
            data['feedback_teacher_indices'][0] += 1
        elif label == 'filter_trace':
            data['adaptive_reference_trace'][index, 11] += 1e-5
        else:
            data['adaptive_reference_trace'][index, 2] += .001
        checks, _ = feedback_checks(m, data, policy)
        rejected = checks.get(required_rejection) is False
        rows.append(dict(mutation=label, rejected=rejected, required_check=required_rejection,
                         failed_checks=[k for k, value in checks.items() if not value]))
    assert all(row['rejected'] for row in rows)
    a.output.mkdir(exist_ok=False)
    result = dict(passed=True, clean_record_passed=True, rows=rows,
        input_trajectory_sha256=sha(path), zero_gate_sha256=sha(a.experiment/'zero_feedback_gate.json'),
        source_sha256=sha(__file__), auditor_sha256=sha('tools/audit_feedback_reference_commands.py'),
        scope='In-memory corruptions are rejected; original trajectory and checkpoints are not modified. This verifies audit sensitivity, not new task performance.')
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'check_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
