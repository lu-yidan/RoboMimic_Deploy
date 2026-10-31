"""Small-step nonlinear action-space search with physical acceptance of every step.

Known-case diagnostic/teacher search. Cached suffix success is never sufficient
for adoption; from-zero full-cycle regression remains mandatory.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from stable_baselines3 import PPO
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from tools.continue_vector_full_cycle_training import sha


def main():
    torch.set_num_threads(1)
    source=Path('outputs/closure_action_authority_20261024_v1')
    old=json.loads((source/'summary.json').read_text())
    protocol=json.loads((source/'protocol.json').read_text())
    assert old['completed'] and old['protocol_sha256']==sha(source/'protocol.json')
    out=Path('outputs/closure_action_trust_region_20261024_v1');out.mkdir(exist_ok=False)
    parent=Path(protocol['parent'])
    assert sha(parent/'policy.zip')==protocol['parent_checkpoint_sha256']
    model=PPO.load(parent/'policy.zip',device='cpu')
    env=ClosureCurriculumEnvV3('outputs/closure_prefix_cache_20261024_v4')
    basis=np.asarray(protocol['basis'])
    declaration=dict(source_protocol_sha256=sha(source/'protocol.json'),parent_checkpoint_sha256=sha(parent/'policy.zip'),
      start_parameters=[0.,0.,-.5,0.,0.,0.],maximum_iterations=4,finite_difference_step=.1,
      initial_radius=.25,maximum_radius=.4,minimum_radius=.025,parameter_bounds=[-1.,1.],
      source_sha256=sha(__file__),environment_sha256=sha('training/closure_curriculum_env_v3.py'),
      scope='Local nonlinear search on known +15mm case, original 26D action bounds. Extra simulator samples. No controller change. From-zero five-case evaluation required before adoption.')
    (out/'protocol.json').write_text(json.dumps(declaration,indent=2))
    rows=[]
    def rollout(p,label):
        obs,_=env.reset(options={'case':'foot_z_plus15'})
        base=env.env.unwrapped
        trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],base_rewards=[])
        while True:
            t=float(base.env.data.time);u=float(np.clip((t-99.66)/3.,0.,1.));ramp=u*u*(3-2*u)
            action=np.clip(model.predict(obs,deterministic=True)[0]+ramp*(basis@p),-1.,1.).astype(np.float32)
            trace['observations'].append(obs.copy());trace['actions'].append(action.copy())
            obs,reward,done,truncated,info=env.step(action)
            trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy())
            trace['time_s'].append(float(base.env.data.time));trace['torques'].append(base.last_torque.copy());trace['base_rewards'].append(info['base_reward'])
            if done or truncated:break
        arrays={k:np.asarray(v) for k,v in trace.items()}
        tail=(arrays['time_s']>111.66)&(arrays['time_s']<=113.6600001)
        q=arrays['qpos'][tail]
        complete=len(arrays['actions'])==700 and len(q)>=99
        if complete:
            joint=q[:,base.env.qadr]-base.env.initial[base.env.qadr]
            rise=q[:,2]-base.env.initial[2]
            features=np.r_[joint.mean(axis=0),rise.mean()-.25]
            worst=np.r_[np.max(abs(joint),axis=0),np.max(abs(rise-.25))]
            cost=objective(worst)
        else:
            features=np.full(30,100.);worst=features;cost=1e8
        index=len(rows);path=out/f'trial_{index:02d}.npz';np.savez_compressed(path,**arrays)
        row=dict(trial=index,label=label,parameters=p.tolist(),success=info['success'],complete=complete,
          cost=float(cost),features=features.tolist(),worst=worst.tolist(),info=info,
          trajectory_sha256=sha(path))
        rows.append(row);(out/f'trial_{index:02d}.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(dict(trial=index,label=label,cost=float(cost),success=info['success'],closure_mrad=float(max(worst[:29])*1000),height_error_mm=float(worst[-1]*1000))),flush=True)
        return row
    limits=np.r_[np.full(29,.039),.0098]
    scales=np.r_[np.full(29,.005),.001]
    def objective(features):
        excess=np.maximum(0.,abs(features)-limits)/scales
        return float(excess@excess)
    p=np.asarray(declaration['start_parameters']);current=rollout(p,'initial')
    radius=.25;history=[]
    for iteration in range(4):
        if current['success']:break
        y=np.asarray(current['features']);columns=[]
        for j in range(6):
            delta=.1 if p[j]+.1<=1. else -.1
            probe=p.copy();probe[j]+=delta
            row=rollout(probe,f'iteration_{iteration}_derivative_{j}')
            if not row['complete']:raise RuntimeError('Derivative probe terminated early; search requires revised safe region')
            columns.append((np.asarray(row['features'])-y)/delta)
        jac=np.column_stack(columns)
        def predicted(d):return objective(y+jac@d)+1e-5*float(d@d)
        bounds=[(max(-radius,-1.-v),min(radius,1.-v)) for v in p]
        fit=minimize(predicted,np.zeros(6),method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':300})
        if not fit.success:raise RuntimeError(fit.message)
        proposal=p+fit.x;candidate=rollout(proposal,f'iteration_{iteration}_proposal')
        expected=max(1e-12,objective(y)-objective(y+jac@fit.x))
        improvement=current['cost']-candidate['cost'];ratio=improvement/expected
        accepted=bool(candidate['complete'] and improvement>0.)
        history.append(dict(iteration=iteration,radius=radius,accepted=accepted,improvement=improvement,ratio=ratio,trial=candidate['trial']))
        if accepted:p=proposal;current=candidate
        radius=min(.4,radius*1.5) if accepted and ratio>.75 else radius*.5 if ratio<.25 else radius
        if radius<.025:break
    best=min((r for r in rows if r['complete']),key=lambda r:r['cost'])
    result=dict(completed=True,rows=rows,history=history,best=best,groups=protocol['groups'],basis=protocol['basis'],
      protocol_sha256=sha(out/'protocol.json'),scope=declaration['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2));env.close()
    print(json.dumps(dict(best_trial=best['trial'],cost=best['cost'],success=best['success'],parameters=best['parameters'])),flush=True)


if __name__=='__main__':main()
