"""From-zero five-case validation of one locally predicted action bias."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_seeded_policy import ClosureSeededPolicy
from tools.continue_vector_full_cycle_training import sha, state_difference
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_closure_curriculum_policy import evaluate_parallel
from tools.audit_closure_curriculum_training import audit
from tools.audit_closure_composite_prefix import audit as audit_prefix


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--search',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path.cwd();parent=Path('outputs/closure_margin_control98304_v1')
    search=json.loads((args.search/'summary.json').read_text())
    assert search['completed'] and search['best']['success'] and search['best']['complete']
    best=search['best'];bias=np.asarray(search['basis'])@np.asarray(best['parameters'])
    assert abs(bias).max()<=1.+1e-9;bias=np.clip(bias,-1.,1.)
    duration=float(best['ramp_seconds'])
    original=PPO.load(parent/'policy.zip',device='cpu');model=PPO.load(parent/'policy.zip',device='cpu')
    model.policy_class=ClosureSeededPolicy
    model.policy_kwargs=dict(model.policy_kwargs,closure_seed_bias=bias.tolist(),seed_ramp_seconds=duration)
    model.policy=ClosureSeededPolicy(model.observation_space,model.action_space,model.lr_schedule,use_sde=model.use_sde,**model.policy_kwargs).to(model.device)
    missing,unexpected=model.policy.load_state_dict(original.policy.state_dict(),strict=False)
    assert missing==['closure_seed_bias'] and not unexpected
    model.policy.optimizer.load_state_dict(original.policy.optimizer.state_dict())
    assert state_difference({k:v for k,v in model.policy.state_dict().items() if k!='closure_seed_bias'},original.policy.state_dict())==0.
    trajectory=Path(best['directory'])/best['file'];assert sha(trajectory)==best['trajectory_sha256']
    with np.load(trajectory) as z:
        reproduced=np.asarray([model.predict(o,deterministic=True)[0] for o in z['observations']])
        assert np.array_equal(reproduced,z['actions']), 'Seeded policy must exactly reproduce selected probe actions'
    out=args.output;out.mkdir(exist_ok=False)
    model.save(out/'policy')
    gate=dict(passed=True,seed_bias=bias.tolist(),seed_ramp_seconds=duration,search_sha256=sha(args.search/'summary.json'),
      selected_trial=best['trial'],selected_trajectory_sha256=sha(trajectory),actor_parameter_difference=0.,selected_action_replay_exact=True)
    (out/'initialization_gate.json').write_text(json.dumps(gate,indent=2))
    manifest=json.loads((parent/'manifest.json').read_text())
    parent_audit=json.loads((parent/'independent_training_audit.json').read_text())
    assert parent_audit['passed'] and sha(parent/'policy.zip')==parent_audit['checkpoint_sha256']
    for rel,digest in manifest['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(root/rel)==digest
    shutil.copytree(parent/'source_snapshot',out/'source_snapshot')
    for name in ('training/closure_seeded_policy.py','training/closure_exploration_policy.py','training/closure_composite_policy.py','tools/evaluate_closure_curriculum_policy.py','tools/audit_closure_curriculum_training.py','tools/validate_closure_search_seed.py'):
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());manifest['sources'][name]=sha(name)
    manifest.update(parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
        suffix_start_seconds=99.66,horizon_seconds=113.66,
        prefix_checkpoint=str((out/'prefix_policy.zip').resolve()),prefix_checkpoint_sha256=sha(parent/'policy.zip'),
        evaluation_only=True,training_additional_steps=0,seed_gate_sha256=sha(out/'initialization_gate.json'),search_simulator_steps=sum(r['info']['time_s']>=113.66 for r in search['rows'])*700,
        seed_bias=gate['seed_bias'],seed_ramp_seconds=gate['seed_ramp_seconds'],
        scope='Passing suffix search candidate instantiated as a fixed mean shift before any new PPO; existing geometry-gated action bias. Five known cases, full time-zero physics and original success gates. Not a learned improvement or held-out validation.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(parent/'policy.zip',out/'prefix_policy.zip')
    # policy.zip was saved after exact selected-action replay above.
    zero=reuse_zero_evaluations(parent,out,manifest,root)
    summary=dict(zero=zero,checkpoint_sha256=sha(out/'policy.zip'),evaluation_pending=True,additional_steps=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    after=evaluate_parallel(out,out/'parallel_after',None,workers=3)['trajectories']
    for row in after:
        name=f"after_{row['case']}.npz";shutil.copy2(out/'parallel_after'/name,out/name)
    (out/'after.json').write_text(json.dumps(after,indent=2))
    summary.update(after=after,evaluation_pending=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(out);audit_prefix(out)
    print(json.dumps(dict(success=sum(r['success'] for r in after),cases=len(after))),flush=True)


if __name__=='__main__':main()
