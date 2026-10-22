"""Freeze a prospective equal-budget reward comparison after required gates."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    parent=Path('outputs/highrod_existing_oracle_policy_v1');gate=Path('outputs/native_margin_reward_gate_v1')
    a=json.loads((parent/'independent_training_audit.json').read_text());g=json.loads((gate/'summary.json').read_text());m=json.loads((parent/'manifest.json').read_text())
    assert a['passed'] and g['passed'] and a['checkpoint_sha256']==sha(parent/'policy.zip') and a['experiment_summary_sha256']==sha(parent/'summary.json')
    actor=json.loads((parent/'actor_source_provenance.json').read_text())
    assert actor['sha256']==sha(actor['source']) and actor['checkpoint_sha256']==sha(parent/'policy.zip')
    out=Path('outputs/native_margin_comparison_protocol_v1');out.mkdir(exist_ok=False)
    sources={p:sha(p) for p in ['tools/train_native_margin_comparison.py','tools/audit_bar_geometry_training.py',
        'tools/evaluate_bar_geometry_policy.py','training/native_margin_reward.py','training/bar_geometry_observation.py',
        'training/strict_full_cycle_residual_env.py','training/full_cycle_residual_env.py','training/bar_geometry_features.py',
        'tools/compare_native_margin_training.py','tools/prepare_native_margin_protocol.py']}
    protocol=dict(parent=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),parent_audit_sha256=sha(parent/'independent_training_audit.json'),
        gate=str(gate.resolve()),gate_summary_sha256=sha(gate/'summary.json'),sources=sources,reference_sha256=m['reference_sha256'],
        actor_source_provenance_sha256=sha(parent/'actor_source_provenance.json'),
        training_cases=m['evaluation_cases'],evaluation_cases=m['evaluation_cases'],steps=98304,environments=4,seed=20261022,
        weights=[0.,.01],margin_width_rad=.02,selection='Compare all five strict case outcomes after both complete equal budgets; retain failures and regressions. No threshold adjustment.',
        scope='Known five height cases after measuring zero/existing-policy baselines. Both arms share exact pretrained actor, critic, optimizer and input information. This is a new fixed-reference task, not a repair of old benchmark labels. Margin penalty changes objective; no policy-invariance claim. No held-out cases or physical hardware.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'protocol_source.py').write_bytes(Path(__file__).read_bytes());print(str(out/'protocol.json'))


if __name__=='__main__':main()
