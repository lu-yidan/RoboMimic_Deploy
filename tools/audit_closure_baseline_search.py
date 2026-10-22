"""Verify the unchanged prefix and select fixed baselines on training cases only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_candidate(candidates, zero_count):
    eligible = [row for row in candidates if row['accepted'] > zero_count]
    if not eligible:
        return 0.
    return min(eligible, key=lambda row: (-row['accepted'], row['worst_endpoint_error_m'], row['residual_rad']))['residual_rad']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--search', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.search/'protocol.json').read_text())
    rules = json.loads((args.search/'selection_protocol.json').read_text())
    summary = json.loads((args.search/'summary.json').read_text())
    manifest = protocol['manifest']
    if sha(args.experiment/'manifest.json') != protocol['parent_manifest_sha256']:
        raise ValueError('Parent experiment changed')
    parent_audit = json.loads((args.experiment/'independent_training_audit.json').read_text())
    if (not parent_audit['passed'] or parent_audit['experiment_manifest_sha256'] != sha(args.experiment/'manifest.json')
            or parent_audit['experiment_summary_sha256'] != sha(args.experiment/'summary.json')):
        raise ValueError('Complete matching parent audit required for independent zero baseline')
    if sha(args.search/'protocol.json') != summary['protocol_sha256'] or sha(args.search/'baseline_source.py') != protocol['source_sha256']:
        raise ValueError('Baseline protocol or source changed')
    if rules['candidates_rad'] != protocol['candidates_rad'] or rules['selection_cases'] != protocol['selection_cases']:
        raise ValueError('Selection rules differ from experiment')
    if 'selection_protocol_sha256' in protocol and sha(args.search/'selection_protocol.json') != protocol['selection_protocol_sha256']:
        raise ValueError('Predeclared selection rules changed')
    if not all(sha(args.search/'source_snapshot'/relative) == expected for relative, expected in manifest['sources'].items()):
        raise ValueError('Frozen experiment sources changed')
    expected = {(residual, case) for residual in protocol['candidates_rad'] for case in protocol['selection_cases']}
    actual = [(row['residual_rad'], row['case']) for row in summary['results']]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Candidate search incomplete or duplicated')
    zero = {row['case']: row for row in json.loads((args.experiment/'zero.json').read_text())}
    prefix_checks = []
    for row in summary['results']:
        residual, case = row['residual_rad'], row['case']
        path = args.search/f"{'minus' if residual < 0 else 'plus'}3mrad_{case}"/'trajectory.npz'
        baseline = args.experiment/f'zero_{case}.npz'
        if sha(path) != row['trajectory_sha256'] or sha(baseline) != zero[case]['trajectory_sha256']:
            raise ValueError('Saved trajectory changed')
        with np.load(path) as a, np.load(baseline) as b:
            first_nonzero = np.flatnonzero(np.any(a['actions'] != 0., axis=1))
            cutoff = int(first_nonzero[0]) if len(first_nonzero) else len(a['actions'])
            checks = {key: bool(np.array_equal(a[key][:cutoff+1], b[key][:cutoff+1])) for key in ('qpos', 'qvel', 'time_s')}
            checks.update({key: bool(np.array_equal(a[key][:cutoff], b[key][:cutoff])) for key in ('actions', 'observations', 'torques', 'rewards')})
            prefix_checks.append(dict(case=case, residual_rad=residual, prefix_steps=cutoff,
                intervention_time_s=float(a['time_s'][cutoff]), checks=checks, passed=all(checks.values())))
    if not all(row['passed'] for row in prefix_checks):
        raise AssertionError('Fixed intervention changed the preceding zero-action trajectory')
    audited_zero = {row['case']: row for row in parent_audit['trajectories'] if row['label'] == 'zero'}
    for case in protocol['selection_cases']:
        if audited_zero[case]['trajectory_sha256'] != zero[case]['trajectory_sha256']:
            raise ValueError('Zero trajectory differs from independent audit')
    zero_count = sum(bool(audited_zero[case]['independently_accepted']) for case in protocol['selection_cases'])
    if zero_count != rules['zero_baseline_accepted_count']:
        raise ValueError('Declared zero baseline count differs')
    candidates = []
    for residual in protocol['candidates_rad']:
        rows = [row for row in summary['results'] if row['residual_rad'] == residual]
        errors = [endpoint['effector_error_m'] for row in rows for endpoint in row['endpoint_checks']]
        worst = max(errors) if errors and all(error is not None for error in errors) else float('inf')
        candidates.append(dict(residual_rad=residual, accepted=sum(row['independently_accepted'] for row in rows),
            worst_endpoint_error_m=worst))
    selected = select_candidate(candidates, zero_count)
    result = dict(prefix_integrity_passed=True, prefix_checks=prefix_checks, zero_accepted=zero_count,
        candidates=candidates, selected_residual_rad=selected, improves_training_success_count=selected != 0.,
        selection_protocol_sha256=sha(args.search/'selection_protocol.json'),
        parent_audit_sha256=sha(args.experiment/'independent_training_audit.json'),
        summary_sha256=sha(args.search/'summary.json'), source_sha256=sha(__file__),
        scope='Complete declared training-case search. Prefix identical to zero baseline. No evaluation-only geometries used for selection; no generalization claim.')
    (args.search/'selection_audit.json').write_text(json.dumps(result, indent=2))
    (args.search/'selection_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
