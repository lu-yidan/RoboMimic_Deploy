"""Immutable Oct22 source/report/evidence inventory, preserving old snapshots."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/research_progress_20261018');version='v5'
    target=root/f'milestone_{version}.json'
    if target.exists():raise FileExistsError(target)
    previous=json.loads((root/'milestone_v4.json').read_text())
    sources=set(previous['sources'])
    names=['bar_geometry_features','bar_geometry_observation']
    sources.update('training/'+n+'.py' for n in names)
    names=['train_bar_geometry_ablation','evaluate_bar_geometry_policy','audit_bar_geometry_training',
        'compare_bar_geometry_ablation','migrate_bar_geometry_policy','validate_bar_geometry_full_cycle',
        'diagnose_geometry_input_use','diagnose_bar_collision_geometry','probe_swing_clearance_geometry',
        'evaluate_geometric_swing_residual','audit_geometric_swing_residual','close_geometry_calibration',
        'evaluate_reference_transfer','audit_reference_transfer','compare_reference_transfer','freeze_geometry_milestone']
    sources.update('tools/'+n+'.py' for n in names)
    sources.update(['tests/test_bar_geometry_features.py','tests/test_bar_geometry_comparison.py'])
    hashes={}
    for source in sorted(sources):
        dest=root/f'source_snapshot_{version}'/source;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(source).read_bytes());hashes[source]=sha(dest)
    evidence={}
    runs=['bar_geometry_ablation_protocol_v1','bar_geometry_full_cycle_gate_v1','bar_geometry_policy_migration_v1',
        'bar_geometry_blind98304_v1','bar_geometry_oracle98304_v1','bar_geometry_comparison_v1',
        'geometric_swing_amplitude_calibration_v1','geometric_swing_residual_plus10_v2',
        'geometric_swing_half_plus10_v1','geometric_swing_threequarter_plus10_v1','highrod_strict_reference_transfer_v1']
    for run in runs:
        for file in (Path('outputs')/run).rglob('*.json'):
            if 'source_snapshot' in file.parts:continue
            evidence[str(file)]=sha(file)
    report=root/f'milestone_{version}_report.md';report.write_bytes(Path('docs/baselines/full_cycle_training_20261018.md').read_bytes())
    result=dict(sources=hashes,evidence=evidence,report=str(report),report_sha256=sha(report),
        scope='Completed information ablation, rejected geometric residual candidates, and six strict reference-transfer runs. Current source copies; each run retains original frozen sources. Height mismatch experiment is running and excluded from completed evidence.')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(dict(path=str(target),sources=len(hashes),evidence=len(evidence))))


if __name__=='__main__':main()
