"""Create equal-capacity observation-ablation initial policy; optimizer reset explicit."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation, ROWS
from training.bar_geometry_features import BarGeometryFeatures
from deploy_mujoco.run_trained_full_cycle import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((args.parent/'manifest.json').read_text())
    summary=json.loads((args.parent/'summary.json').read_text())
    audit=json.loads((args.parent/'independent_training_audit.json').read_text())
    if not audit['passed'] or sha(args.parent/'summary.json')!=audit['experiment_summary_sha256'] or sha(args.parent/'policy.zip')!=audit['checkpoint_sha256']:
        raise ValueError('Completed audited parent required')
    env=BarGeometryObservation(FullCycleResidualEnv(plan=manifest['reference'],cases=manifest['training_cases']),mode='oracle')
    config=copy.deepcopy(manifest['config'])
    config['policy_kwargs']['features_extractor_class']=BarGeometryFeatures
    model=PPO('MlpPolicy',env,seed=20261019,device='cpu',**config)
    parent=PPO.load(args.parent/'policy.zip',device='cpu')
    original=parent.policy.state_dict()
    state=model.policy.state_dict()
    extra=[]
    for key in state:
        if key in original:
            if state[key].shape!=original[key].shape:raise ValueError('Changed pretrained parameter shape')
            state[key]=original[key].clone()
        elif key.endswith('geometry_projection.weight'):
            # SB3 orthogonal initialization may have overwritten constructor zeros.
            state[key]=torch.zeros_like(state[key]);extra.append(key)
        else:raise ValueError('Unexpected new parameter: '+key)
    if not set(original)<=set(state) or not extra:raise ValueError('Incomplete migration')
    model.policy.load_state_dict(state,strict=True)
    if model.policy.optimizer.state:raise ValueError('Expected fresh optimizer in both future arms')
    args.output.mkdir(exist_ok=False)
    rows=[]
    try:
        for row in summary['after']:
            path=args.parent/f"after_{row['case']}.npz"
            if sha(path)!=row['trajectory_sha256']:raise ValueError('Parent trajectory changed')
            geom=np.array([manifest['evaluation_cases'][row['case']].get(r,[0.,0.,0.]) for r in ROWS],dtype=np.float32).reshape(-1)/.02
            count=0
            with np.load(path) as z, torch.no_grad():
                for obs,recorded in zip(z['observations'],z['actions']):
                    old=parent.predict(obs,deterministic=True)[0]
                    assert np.array_equal(old,recorded)
                    for mode,features in [('blind',np.zeros(12,dtype=np.float32)),('oracle',geom)]:
                        augmented=np.r_[obs,features].astype(np.float32)
                        action=model.predict(augmented,deterministic=True)[0]
                        assert np.array_equal(old,action)
                        old_value=parent.policy.predict_values(torch.as_tensor(obs[None,:]))
                        value=model.policy.predict_values(torch.as_tensor(augmented[None,:]))
                        assert torch.equal(old_value,value)
                    count+=1
            rows.append(dict(case=row['case'],samples=count,exact_actor_and_critic_both_modes=True,trajectory_sha256=sha(path)))
            print(json.dumps(rows[-1]),flush=True)
        model.save(args.output/'policy_initial')
        loaded=PPO.load(args.output/'policy_initial.zip',device='cpu')
        assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        sources=[__file__,'training/bar_geometry_features.py','training/bar_geometry_observation.py']
        result=dict(passed=True,parent=str(args.parent.resolve()),parent_checkpoint_sha256=sha(args.parent/'policy.zip'),
            parent_audit_sha256=sha(args.parent/'independent_training_audit.json'),initial_checkpoint_sha256=sha(args.output/'policy_initial.zip'),
            new_projection_keys=extra,observations=148,features=136,actions=26,results=rows,
            optimizer='Fresh empty optimizer for both future arms; not uninterrupted optimizer continuation.',
            training_steps_after_migration=0,sources={p:sha(p) for p in sources},
            scope='All parent deterministic evaluation inputs replayed: initial actor and critic exactly preserved in blind/oracle modes. No oracle learning or robustness claim.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
        for p in sources:(args.output/Path(p).name).write_bytes(Path(p).read_bytes())
    finally:env.close()


if __name__=='__main__':main()
