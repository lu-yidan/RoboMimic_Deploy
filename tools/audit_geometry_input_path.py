"""Record which rod-offset channels can reach the current frozen actors."""
import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.bar_geometry_features import BarGeometryFeatures
from tools.validate_swing_reference import ROOT,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--sensitivity',type=Path,required=True);p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    e=a.experiment.resolve();m=read(e/'manifest.json');s=read(a.sensitivity);follow=read(a.followup)
    assert len(s['rows'])==16 and s['original_actions_exact']
    assert s['checkpoint_sha256']==sha(e/'policy.zip')
    assert follow['estimated_offsets']=={'rung_06':[0.,0.,0.]} and follow['new_training_steps']==0
    assert all(r['maximum_absolute_normalized_action_difference']==0 for c in s['rows'] for r in c['cycles'].values())
    torch.set_num_threads(1);actors={}
    for label,path in [('prefix',Path(m['prefix_checkpoint'])),('suffix',e/'policy.zip')]:
        model=PPO.load(path,device='cpu');extractor=model.policy.pi_features_extractor
        assert type(extractor) is BarGeometryFeatures
        weights=extractor.geometry_projection.weight.detach().cpu().numpy()
        assert weights.shape==(136,12) and extractor.geometry_projection.bias is None
        actors[label]=dict(checkpoint_sha256=sha(path),policy_class=type(model.policy).__name__,
            feature_class=type(extractor).__name__,shares_actor_critic_features=model.policy.share_features_extractor,
            projection_max_absolute_by_input=abs(weights).max(axis=0).tolist(),
            exactly_zero_geometry_slots=[i for i in range(4) if np.count_nonzero(weights[:,3*i:3*i+3])==0],
            hand_slots_exactly_zero=bool(np.count_nonzero(weights[:,6:12])==0))
    assert all(v['hand_slots_exactly_zero'] for v in actors.values())
    sources=('training/bar_geometry_features.py','training/closure_composite_policy.py',
        'training/closure_action_bias_policy_v2.py','training/closure_exploration_policy.py',
        'training/cycle_relative_bar_observation.py','tools/evaluate_compiled_hook_pair.py',
        'tools/evaluate_compiled_timed_hook.py','tools/probe_policy_geometry_sensitivity.py',
        'tools/audit_geometry_input_path.py')
    report=dict(actors=actors,experiment_manifest_sha256=sha(e/'manifest.json'),
        sensitivity_summary_sha256=sha(a.sensitivity),superseded_followup_protocol_sha256=sha(a.followup),
        sources={r:sha(ROOT/r) for r in sources},
        finding='Both frozen actors have exactly zero projection weights for geometry slots 2 and 3 (zero-based), which represent current/future hand rods. The observation override for rung_06 uses only those slots in this two-cycle protocol. Composite switching uses time index 105 and fixed calibration uses index 138, neither overridden.',
        decision='Do not execute the predeclared two-cycle rung_06 zero-estimate follow-up: this input-only ablation cannot distinguish these actors at fixed state. Preserve the protocol as an abandoned design, not as an executed or successful perception experiment.',
        retained_experiment='The current 16-case physical-scene three-arm comparison still changes real geometry and remains informative about controller behavior.',
        boundaries=['No claim that joint/state/contact feedback is ignored.',
            'No claim that all geometry is ignored: first-foot Z (prefix) and X/Z (suffix) have nonzero input weights.',
            'This proof concerns hand-role slots and the declared two-cycle experiment. Rung_06 can become a foot-role input in later cycles.',
            'No policy weights, controller parameters, observations in ongoing runs, or prior success reports were changed.'],
        new_training_steps=0,new_physics_steps=0,automatic_promotion=False)
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'summary.json',report)
    (a.output/'source_snapshot').mkdir()
    for relative in sources:
        target=a.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    print({k:v['projection_max_absolute_by_input'] for k,v in actors.items()},flush=True)


if __name__=='__main__':main()
