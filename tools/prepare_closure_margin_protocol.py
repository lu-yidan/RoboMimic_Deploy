"""Freeze a prospective equal-budget reward comparison after required gates."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    parent=Path('outputs/support_reference_policy_transfer_v1');gate=Path('outputs/closure_margin_reward_gate_v2')
    a=json.loads((parent/'independent_training_audit.json').read_text());g=json.loads((gate/'summary.json').read_text());m=json.loads((parent/'manifest.json').read_text())
    assert a['passed'] and g['passed'] and a['checkpoint_sha256']==sha(parent/'policy.zip') and a['experiment_summary_sha256']==sha(parent/'summary.json')
    lineage=json.loads((parent/'policy_lineage_audit.json').read_text());assert lineage['passed'] and lineage['checkpoint_sha256']==sha(parent/'policy.zip')
    from stable_baselines3 import PPO
    model=PPO.load(parent/'policy.zip',device='cpu')
    config={key:getattr(model,key) for key in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','target_kl')}
    config.update(learning_rate=float(model.lr_schedule(1.)),net_arch=model.policy.net_arch,feature_class=type(model.policy.features_extractor).__name__)
    out=Path('outputs/closure_margin_comparison_protocol_v1');out.mkdir(exist_ok=False)
    sources={p:sha(p) for p in ['tools/train_closure_margin_comparison.py','tools/audit_bar_geometry_training.py',
        'tools/evaluate_bar_geometry_policy.py','training/closure_margin_reward.py','training/bar_geometry_observation.py',
        'training/strict_full_cycle_residual_env.py','training/full_cycle_residual_env.py','training/bar_geometry_features.py',
        'tools/compare_closure_margin_training.py','tools/prepare_closure_margin_protocol.py']}
    protocol=dict(parent=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),parent_audit_sha256=sha(parent/'independent_training_audit.json'),
        gate=str(gate.resolve()),gate_summary_sha256=sha(gate/'summary.json'),sources=sources,reference_sha256=m['reference_sha256'],
        policy_lineage_audit_sha256=sha(parent/'policy_lineage_audit.json'),config=config,
        training_cases=m['evaluation_cases'],evaluation_cases=m['evaluation_cases'],steps=98304,environments=4,seed=20261022,
        weights=[0.,.02],closure_cost_parameters=dict(joint_free_rad=.032,joint_scale_rad=.008,height_free_m=.006,height_scale_m=.004,ramp_seconds=3.,joint_fraction=.5),selection='Compare all five strict case outcomes after both complete equal budgets; retain failures and regressions. No threshold adjustment.',
        scope='Known five height cases after measuring zero/existing-policy baselines. Both arms share exact pretrained actor, critic, optimizer and input information. This is a new fixed-reference task, not a repair of old benchmark labels. Closure penalty changes objective; no policy-invariance claim. No held-out cases or physical hardware.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'protocol_source.py').write_bytes(Path(__file__).read_bytes());print(str(out/'protocol.json'))


if __name__=='__main__':main()
