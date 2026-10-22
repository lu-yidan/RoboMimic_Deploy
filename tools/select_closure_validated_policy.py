"""Apply the frozen second-stage selection rule to independently audited runs."""
import argparse
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def read(path):
    return json.loads(Path(path).read_text())


def verified_successes(path, expected_cases):
    path = Path(path)
    manifest = read(path / 'manifest.json')
    audit = read(path / 'independent_training_audit.json')
    assert audit['passed']
    for file, key in [('policy.zip', 'checkpoint_sha256'), ('manifest.json', 'experiment_manifest_sha256'), ('summary.json', 'experiment_summary_sha256')]:
        assert sha(path / file) == audit[key]
    assert manifest['evaluation_cases'] == expected_cases
    rows = audit['trajectories']
    assert len(rows) == 2 * len(expected_cases)
    for case in expected_cases:
        for label in ('zero', 'after'):
            found = [r for r in rows if r['case'] == case and r['label'] == label]
            assert len(found) == 1
            assert found[0]['passed']
            assert sha(path / f'{label}_{case}.npz') == found[0]['trajectory_sha256']
    return {r['case'] for r in rows if r['label'] == 'after' and r['independently_accepted']}


def choose(parent_successes, candidate_successes):
    """Namespaced case sets prevent accidental known/grid case collisions."""
    qualifying = {arm: successes for arm, successes in candidate_successes.items()
                  if parent_successes < successes}
    if not qualifying:
        return 'parent'
    return sorted(qualifying, key=lambda arm: (-len(qualifying[arm]), arm != 'control', arm))[0]


def verify_task(path, parent_manifest):
    manifest = read(Path(path) / 'manifest.json')
    for key in ('contract', 'observation_contract', 'geometry_mode', 'bundle_manifest_sha256',
                'reference_sha256', 'normal_bias_y_deg', 'force_history_memory'):
        assert manifest[key] == parent_manifest[key], f'Task mismatch: {path}: {key}'
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/closure_policy_selection_v1'))
    args = parser.parse_args()
    root = Path('outputs/closure_policy_validation_protocol_v1')
    protocol = read(root / 'protocol.json')
    assert sha('outputs/support_reference_validation_v1/protocol.json') == protocol['grid_protocol_sha256']
    assert sha('outputs/closure_margin_comparison_protocol_v1/protocol.json') == protocol['training_protocol_sha256']
    parent = Path(protocol['parent'])
    parent_manifest = read(parent / 'manifest.json')
    known_cases = parent_manifest['evaluation_cases']
    grid_jobs = read('outputs/support_reference_validation_v1/protocol.json')['jobs']
    grid_cases = {job[4]: job[5] for job in grid_jobs}
    assert len(grid_cases) == len(grid_jobs)
    assert len(known_cases) == 5 and len(grid_cases) == 12
    known = verified_successes(parent, known_cases)
    assert known == set(protocol['parent_known_successes'])
    parent_grid = Path(protocol['parent_grid_evaluation'])
    verify_task(parent_grid, parent_manifest)
    assert sha(parent_grid / 'policy.zip') == sha(parent / 'policy.zip') == protocol['parent_checkpoint_sha256']
    grid = verified_successes(parent_grid, grid_cases)
    baseline = {'known:' + c for c in known} | {'grid:' + c for c in grid}
    records, candidates = {}, {}
    evidence = {str(root / 'protocol.json'): sha(root / 'protocol.json')}
    for path in (parent, parent_grid):
        evidence[str(path / 'independent_training_audit.json')] = sha(path / 'independent_training_audit.json')
    for arm in ('control', 'penalty'):
        candidate = Path(f'outputs/closure_margin_{arm}98304_v1')
        assert candidate in [Path(p) for p in protocol['candidates']]
        dispatch_path = root / f'{arm}_dispatch.json'
        dispatch = read(dispatch_path)
        manifest, summary = verify_task(candidate, parent_manifest), read(candidate / 'summary.json')
        assert not summary['evaluation_pending'] and summary['additional_steps'] == 98304
        assert manifest['parent_checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
        assert manifest['protocol_sha256'] == protocol['training_protocol_sha256']
        assert dispatch['selection_protocol_sha256'] == sha(root / 'protocol.json')
        assert dispatch['policy_sha256'] == sha(candidate / 'policy.zip')
        assert dispatch['audit_sha256'] == sha(candidate / 'independent_training_audit.json')
        successes = verified_successes(candidate, known_cases)
        assert dispatch['known_successes'] == sorted(successes)
        eligible = known <= successes
        assert dispatch['eligible'] == eligible
        record = dict(eligible=eligible, known_passed=len(successes), lost_known=sorted(known-successes))
        evidence[str(dispatch_path)] = sha(dispatch_path)
        evidence[str(candidate / 'independent_training_audit.json')] = sha(candidate / 'independent_training_audit.json')
        if eligible:
            evaluation = Path(f'outputs/closure_{arm}_policy_validation_v1')
            verify_task(evaluation, parent_manifest)
            assert sha(evaluation / 'validation_dispatch.json') == sha(dispatch_path)
            assert sha(evaluation / 'policy.zip') == sha(candidate / 'policy.zip')
            grid_successes = verified_successes(evaluation, grid_cases)
            combined = {'known:' + c for c in successes} | {'grid:' + c for c in grid_successes}
            candidates[arm] = combined
            record.update(grid_passed=len(grid_successes), added=sorted(combined-baseline), lost=sorted(baseline-combined), qualifies=baseline < combined)
            evidence[str(evaluation / 'independent_training_audit.json')] = sha(evaluation / 'independent_training_audit.json')
        records[arm] = record
    selected = choose(baseline, candidates)
    selected_path = parent if selected == 'parent' else Path(f'outputs/closure_margin_{selected}98304_v1')
    result = dict(selected=selected, selected_experiment=str(selected_path), checkpoint_sha256=sha(selected_path / 'policy.zip'), parent_known_passed=len(known), parent_grid_passed=len(grid), candidates=records, evidence=evidence, source_sha256=sha(__file__), scope=protocol['scope'])
    args.output.mkdir(exist_ok=False)
    (args.output / 'selection.json').write_text(json.dumps(result, indent=2))
    (args.output / 'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
