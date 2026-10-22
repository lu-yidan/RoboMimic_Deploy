"""Save completed margin investigations without modifying previous evidence."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/research_progress_20261018');version='v6'
    target=root/f'milestone_{version}.json'
    if target.exists():raise FileExistsError(target)
    previous=json.loads((root/'milestone_v5.json').read_text())
    sources=set(previous['sources'])
    runs=['highrod_height_mismatch_v1','highrod_existing_oracle_policy_v1','native_margin_reward_gate_v1',
        'native_margin_comparison_protocol_v1','native_margin_control98304_v1','native_margin_penalty98304_v1',
        'native_margin_comparison_v1','highrod_limit_posture_probe_v1','highrod_limit_posture_locked_waist_v1',
        'load_margin_fixed_residual_v1','load_margin_base_reference_v1','geometry_policy_deploy_replay_v1',
        'temporal_leg_gain_calibration_v1','highrod_unbiased_height_v1','highrod_dynamics_trace_v1',
        'smooth_unload_reference_gate_v1','retimed_unload_factor1.5_plan_v1','retimed_unload_factor2_plan_v1']
    for run in runs:
        for manifest in (Path('outputs')/run).rglob('manifest.json'):
            if 'source_snapshot' in manifest.parts:continue
            record=json.loads(manifest.read_text())
            for source in record.get('sources',{}):
                file=Path(source)
                if file.is_file() and file.suffix=='.py':sources.add(str(file))
    for pattern in ('*margin*.py','*highrod*.py','*temporal_leg*.py','*base_reference*.py','*actor_sources*.py','*unload*.py'):
        sources.update(str(p) for p in Path('tools').glob(pattern))
    sources.update(['training/native_margin_reward.py','training/temporal_base_reference_env.py','training/temporal_leg_gain_env.py',
        'deploy_mujoco/run_geometry_full_cycle.py','tests/test_native_margin_reward.py','tests/test_native_margin_comparison.py',
        'tests/test_temporal_base_reference.py','tests/test_temporal_leg_gain.py','README_zh.md',
        'training/dynamics_trace_env.py','training/retimed_unload_env.py','training/smooth_unload_reference_env.py',
        'tests/test_retimed_unload.py','tests/test_smooth_unload_reference.py'])
    assert (Path('outputs/temporal_leg_gain_calibration_v1')/'selection.json').is_file()
    evidence={}
    for run in runs:
        for file in (Path('outputs')/run).rglob('*.json'):
            if 'source_snapshot' not in file.parts:evidence[str(file)]=sha(file)
    hashes={}
    for source in sorted(sources):
        dest=root/f'source_snapshot_{version}'/source;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(source).read_bytes());hashes[source]=sha(dest)
    report=root/f'milestone_{version}_report.md';report.write_bytes(Path('docs/baselines/full_cycle_training_20261018.md').read_bytes())
    result=dict(sources=hashes,evidence=evidence,report=str(report),report_sha256=sha(report),previous_milestone_sha256=sha(root/'milestone_v5.json'),
        scope='Completed tolerance, policy transfer, matched reward training, static IK, rejected fixed/base-reference/gain candidates, simulation packaging, normal-bias control and exact dynamics instrumentation. Retiming/smoothing geometry gates completed; their active dynamic experiments excluded. Each run retains original sources; these are current source copies.')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(dict(path=str(target),sources=len(hashes),evidence=len(evidence))))


if __name__=='__main__':main()
