"""Freeze a conditional follow-up only after the original comparison is complete."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    comparison=Path('outputs/closure_margin_comparison_v1/comparison.json');first=json.loads(comparison.read_text())
    if first['penalty_passed']==5:
        print('First penalty achieved all known cases; follow-up not activated.');return
    control=Path('outputs/closure_margin_control98304_v1');penalty=Path('outputs/closure_margin_penalty98304_v1')
    for p in (control,penalty):
        audit=json.loads((p/'independent_training_audit.json').read_text());assert audit['passed']
        for file,key in [('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256'),('policy.zip','checkpoint_sha256')]:assert sha(p/file)==audit[key]
    gate=Path('outputs/closure_smooth_reward_gate_v1');g=json.loads((gate/'summary.json').read_text());assert g['passed']
    for source,digest in g['sources'].items():assert sha(source)==digest
    old=Path('outputs/closure_margin_comparison_protocol_v1/protocol.json');protocol=json.loads(old.read_text())
    source_names=['tools/train_closure_smooth_followup.py','tools/audit_bar_geometry_training.py','tools/evaluate_bar_geometry_policy.py','training/closure_smooth_reward.py','training/bar_geometry_observation.py','training/strict_full_cycle_residual_env.py','training/full_cycle_residual_env.py','training/bar_geometry_features.py','tools/compare_closure_smooth_followup.py','tools/prepare_closure_smooth_protocol.py']
    protocol.update(gate=str(gate.resolve()),gate_summary_sha256=sha(gate/'summary.json'),sources={p:sha(p) for p in source_names},
        control_reuse=str(control.resolve()),control_reuse_hashes={file:sha(control/file) for file in ('manifest.json','summary.json','policy.zip','policy_before.zip','independent_training_audit.json')},
        previous_protocol_sha256=sha(old),previous_comparison_sha256=sha(comparison),
        saturation_diagnostic_sha256=sha('outputs/closure_cost_saturation_v1/summary.json'),
        scope='Conditional known-case follow-up after capped-cost evaluation. Same parent, optimizer, seed, physics, observations and98304-step budget. Explicitly reuse the completed zero-cost control; its outcome was known when this protocol was frozen. Not a fresh prospective two-arm or statistical generalization claim.')
    protocol['closure_cost_parameters']=dict(protocol['closure_cost_parameters'],transform='squared_normalized_error/(1+squared_normalized_error)')
    output=Path('outputs/closure_smooth_followup_protocol_v1');output.mkdir(exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'protocol_source.py').write_bytes(Path(__file__).read_bytes());print(output/'protocol.json')


if __name__=='__main__':main()
