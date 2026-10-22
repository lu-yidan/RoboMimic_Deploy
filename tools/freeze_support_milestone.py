"""Save completed margin investigations without modifying previous evidence."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/research_progress_20261018');version='v7'
    target=root/f'milestone_{version}.json'
    if target.exists():raise FileExistsError(target)
    previous=json.loads((root/'milestone_v6.json').read_text())
    sources=set(previous['sources'])
    runs=['retimed_unload_calibration_v1','smooth_unload_calibration_v1','highrod_reference_retarget_probe_v1',
        'support_retarget_plan_v1','support_retarget_calibration_v1','support_endpoint075_plan_v1','support_endpoint075_calibration_v1',
        'support_reference_policy_transfer_v1','support_policy_deploy_replay_v1','bounded_terminal_geometry_probe_v1',
        'terminal_sole_tangency_probe_v1','terminal_sole_tangency_cap8_probe_v1','terminal_contact_location_probe_v1',
        'terminal_contact_location_height10_probe_v1','terminal_contact_minimax_v1','support_contact10_plan_v1','support_contact15_plan_v1',
        'support_contact_terminal_calibration_v1','landing_contact_plan_v1','landing_contact_plan_v2','landing_contact_plan_v3','landing_contact_plan_v4',
        'landing_contact_calibration_v1','landing_contact_calibration_v2','contact_acceleration_diagnostic_v1',
        'contact_acceleration_qp_probe_v1','contact_acceleration_qp_soft_probe_v1','window_acceleration_qp_calibration_v1',
        'support_reference_validation_v1']
    for run in runs:
        for manifest in (Path('outputs')/run).rglob('manifest.json'):
            if 'source_snapshot' in manifest.parts:continue
            record=json.loads(manifest.read_text())
            for source in record.get('sources',{}):
                file=Path(source)
                if file.is_file() and file.suffix=='.py':sources.add(str(file))
    for pattern in ('*margin*.py','*highrod*.py','*support*.py','*terminal*.py','*landing*.py','*contact_acceleration*.py','*reference_policy*.py','*policy_transfer*.py','*window_acceleration*.py','*unload*.py'):
        sources.update(str(p) for p in Path('tools').glob(pattern))
    sources.update(['training/native_margin_reward.py','training/temporal_base_reference_env.py','training/temporal_leg_gain_env.py',
        'deploy_mujoco/run_geometry_full_cycle.py','tests/test_native_margin_reward.py','tests/test_native_margin_comparison.py',
        'tests/test_temporal_base_reference.py','tests/test_temporal_leg_gain.py','README_zh.md',
        'training/dynamics_trace_env.py','training/retimed_unload_env.py','training/smooth_unload_reference_env.py',
        'tests/test_retimed_unload.py','tests/test_smooth_unload_reference.py'])
    assert (Path('outputs/temporal_leg_gain_calibration_v1')/'selection.json').is_file()
    sources.update(str(p) for p in Path('training').glob('*acceleration*.py'))
    sources.update(['training/closure_margin_reward.py','tests/test_closure_margin_reward.py','tests/test_closure_margin_comparison.py','tests/test_window_acceleration_roundoff.py','tools/freeze_support_milestone.py'])
    assert Path('outputs/support_reference_validation_v1/independent_comparison.json').is_file()
    evidence=dict(previous['evidence'])
    for run in runs:
        for file in (Path('outputs')/run).rglob('*.json'):
            if 'source_snapshot' not in file.parts:evidence[str(file)]=sha(file)
    hashes={}
    for source in sorted(sources):
        dest=root/f'source_snapshot_{version}'/source;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(source).read_bytes());hashes[source]=sha(dest)
    report=root/f'milestone_{version}_report.md';report.write_bytes(Path('docs/baselines/full_cycle_training_20261018.md').read_bytes())
    result=dict(sources=hashes,evidence=evidence,report=str(report),report_sha256=sha(report),previous_milestone_sha256=sha(root/'milestone_v6.json'),
        scope='Completed support-reference4of5, unchanged-policy transfer/deployment, rejected terminal/landing references, static contact optimization, contact acceleration diagnostics, offline QP and failed runtime QPv1, twelve-case reference validation. Includes current source copies; each experiment retains originals. Active QPv2, closure reward gate and upcoming training excluded from completed evidence.')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(dict(path=str(target),sources=len(hashes),evidence=len(evidence))))


if __name__=='__main__':main()
