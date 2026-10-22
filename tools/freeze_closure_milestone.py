"""Extend completed evidence snapshots; never overwrite previous milestones."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    root = Path('outputs/research_progress_20261018')
    previous_path = root/'milestone_v7.json'
    target = root/'milestone_v8.json'
    if target.exists():
        raise FileExistsError(target)
    previous = json.loads(previous_path.read_text())
    runs = ['prior_reference_validation_v1', 'reference_validation_comparison_v1',
            'window_acceleration_qp_calibration_v2', 'closure_ankle_authority_v1',
            'closure_margin_reward_gate_v1', 'closure_margin_reward_gate_v2',
            'closure_margin_comparison_protocol_v1', 'closure_margin_control98304_v1',
            'closure_margin_penalty98304_v1', 'closure_margin_comparison_v1',
            'support_policy_exploration_diagnostic_v1', 'closure_cost_saturation_v1',
            'closure_smooth_reward_gate_v1', 'support_parent_policy_validation_v1',
            'support_policy_grid_deploy_replay_v1', 'closure_reward_signal_figure_v1',
            'runtime_capture_20261022_v1']
    for name in ('closure_margin_control98304_v1', 'closure_margin_penalty98304_v1', 'support_parent_policy_validation_v1'):
        assert json.loads((Path('outputs')/name/'independent_training_audit.json').read_text())['passed']
        assert not json.loads((Path('outputs')/name/'summary.json').read_text())['evaluation_pending']
    assert json.loads(Path('outputs/support_policy_grid_deploy_replay_v1/deployment_reproduction_audit.json').read_text())['passed']
    sources = set(previous['sources'])
    for directory in ('tools', 'tests', 'training'):
        for pattern in ('*closure*.py', '*policy_validation*.py', '*endpoint_slack*.py', '*simulation_runtime*.py'):
            sources.update(str(p) for p in Path(directory).glob(pattern))
    sources.update(['tools/compare_reference_validations.py', 'tools/plot_policy_validation.py', 'tools/freeze_closure_milestone.py'])
    # Some comparisons predate this snapshot's naming; include only actual sources.
    sources = {source for source in sources if Path(source).is_file()}
    evidence = dict(previous['evidence'])
    for name in runs:
        directory = Path('outputs')/name
        assert directory.is_dir(), name
        for path in directory.rglob('*.json'):
            if 'source_snapshot' in path.parts:
                continue
            evidence[str(path)] = sha(path)
            record = json.loads(path.read_text())
            if isinstance(record, dict) and isinstance(record.get('sources'), dict):
                sources.update(source for source in record['sources'] if Path(source).suffix == '.py' and Path(source).is_file())
        for path in directory.rglob('*'):
            if path.suffix.lower() in ('.png', '.pdf', '.mp4'):
                evidence[str(path)] = sha(path)
    hashes = {}
    for source in sorted(sources):
        dest = root/'source_snapshot_v8'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(source).read_bytes())
        hashes[source] = sha(dest)
    report = root/'milestone_v8_report.md'
    report.write_bytes(Path('docs/baselines/full_cycle_training_20261018.md').read_bytes())
    result = dict(sources=hashes, evidence=evidence, report=str(report), report_sha256=sha(report),
                  previous_milestone_sha256=sha(previous_path), completed_run_directories=runs,
                  scope='Completed capped-reward matched4/5 versus4/5; parent-policy12-grid10/12 versus zero9/12; exact deployment replay and video; rejected QPv2 and ankle-only action; completed diagnostics and smooth reward validation. Interrupted gatev1 retained as failed evidence. Active trained-policy grid evaluations and smooth-cost training excluded. Current source copies are distinct from each experiment original snapshots.')
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(path=str(target), sources=len(hashes), evidence=len(evidence))))


if __name__ == '__main__':
    main()
