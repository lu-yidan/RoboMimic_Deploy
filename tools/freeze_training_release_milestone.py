"""Freeze completed reward comparisons, model selection and portable replay."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    root = Path('outputs/research_progress_20261018')
    previous = root/'milestone_v8.json'
    target = root/'milestone_v9.json'
    if target.exists():
        raise FileExistsError(target)
    old = json.loads(previous.read_text())
    completed = ['closure_control_policy_validation_v1', 'closure_penalty_policy_validation_v1', 'closure_smooth_penalty98304_v1']
    for name in completed:
        p = Path('outputs')/name
        audit = json.loads((p/'independent_training_audit.json').read_text())
        summary = json.loads((p/'summary.json').read_text())
        assert audit['passed'] and not summary['evaluation_pending']
        assert sha(p/'summary.json') == audit['experiment_summary_sha256']
        assert sha(p/'policy.zip') == audit['checkpoint_sha256']
    required = ['outputs/closure_policy_selection_v1/selection.json', 'outputs/closure_smooth_comparison_v1/comparison.json',
                'outputs/closure_control_package_selection_receipt_v1.json']
    assert all(Path(p).is_file() for p in required)
    runs = completed + ['closure_policy_validation_protocol_v1', 'closure_policy_selection_v1',
                        'closure_smooth_followup_protocol_v1', 'closure_smooth_comparison_v1',
                        'closure_control_grid_deploy_replay_v1', 'closure_control_simulation_package_v1',
                        'closure_candidate_archive_check_v1', 'closure_training_endpoint_figure_v1', 'closure_training_endpoint_figure_v2']
    evidence = dict(old['evidence'])
    for name in runs:
        directory = Path('outputs')/name
        assert directory.is_dir(), name
        for path in directory.rglob('*'):
            if 'source_snapshot' in path.parts:
                continue
            if path.is_file() and path.suffix in ('.json', '.png', '.pdf'):
                evidence[str(path)] = sha(path)
    for name in required + ['outputs/closure_control_simulation_candidate_v1.zip', 'outputs/closure_control_simulation_candidate_v1.zip.manifest.json']:
        evidence[name] = sha(name)
    # Existing run directories received additive diagnostics after v8.
    for name in ('support_reference_policy_transfer_v1', 'closure_margin_control98304_v1', 'closure_margin_penalty98304_v1'):
        file = Path('outputs')/name/'endpoint_slack_diagnostic.json'
        evidence[str(file)] = sha(file)
    sources = set(old['sources'])
    for directory in ('tools', 'training', 'tests', 'deploy_mujoco'):
        for pattern in ('*closure*.py', '*geometry_candidate*.py', '*packaged_geometry*.py', '*selected_simulation*.py', '*training_release*.py'):
            sources.update(str(p) for p in Path(directory).glob(pattern))
    hashes = {}
    for name in sorted(sources):
        destination = root/'source_snapshot_v9'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
        hashes[name] = sha(destination)
    report = root/'milestone_v9_report.md'
    report.write_bytes(Path('docs/baselines/full_cycle_training_20261018.md').read_bytes())
    result = dict(sources=hashes, evidence=evidence, report=str(report), report_sha256=sha(report),
                  previous_milestone_sha256=sha(previous), completed_directories=runs,
                  scope='Completed three98304-step continuation experiments, original two-arm frozen model selection, smooth-cost follow-up against explicitly reused control, all declared final evaluations, and exact portable candidate replay. Includes failures. No active training in this milestone, no hardware or independent-test generalization claim.')
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(path=str(target), sources=len(hashes), evidence=len(evidence))))


if __name__ == '__main__':
    main()
