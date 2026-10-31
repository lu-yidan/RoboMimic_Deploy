"""Bounded local margin search around a passing combined controller.
Only the known +15 mm development case is used; every proposal uses real physics.
"""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from stable_baselines3 import PPO
from scipy.optimize import linprog
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target
from tools.continue_vector_full_cycle_training import sha

ENV=None
MODEL=None
BASE_BIAS=None

def rollout(payload):
    global ENV,MODEL,BASE_BIAS
    theta,output,index=payload;theta=np.asarray(theta,dtype=float);out=Path(output)
    if ENV is None:
        torch.set_num_threads(1)
        ENV=ClosureCurriculumEnvV3('outputs/closure_prefix_cache_20261024_v4')
        MODEL=PPO.load('outputs/closure_seed_policy_20261024_v1/policy_initial.zip',device='cpu')
        BASE_BIAS=MODEL.policy.closure_seed_bias.detach().cpu().numpy().copy()
    bias=BASE_BIAS.copy()
    bias[4]+=.05*theta[0];bias[10]+=.05*theta[1]
    bias[[0,6]]+=.05*theta[2];bias[[3,9]]+=.05*theta[3]
    waist=np.array([0.,-.4+.05*theta[4],0.]);drop=.00019+.00005*theta[5]
    assert abs(bias).max()<=1.+1e-10 and abs(waist).max()<=1. and 0.<=drop<=.001
    with torch.no_grad():MODEL.policy.closure_seed_bias.copy_(torch.as_tensor(bias,dtype=torch.float64))
    obs,_=ENV.reset(options={'case':'foot_z_plus15'});base=ENV.env.unwrapped
    state=install_waist_target(base,waist,ramp_seconds=1.,record=True);install_body_target(base,drop)
    trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],base_rewards=[])
    while True:
        trace['observations'].append(obs.copy());action=MODEL.predict(obs,deterministic=True)[0]
        obs,reward,done,truncated,info=ENV.step(action)
        trace['actions'].append(action.copy());trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy())
        trace['time_s'].append(float(base.env.data.time));trace['torques'].append(base.last_torque.copy());trace['base_rewards'].append(info['base_reward'])
        if done or truncated:break
    arrays={k:np.asarray(v) for k,v in trace.items()};arrays['waist_reference_trace']=np.asarray(state['trace']).reshape(-1,10)
    tail=(arrays['time_s']>111.66)&(arrays['time_s']<=113.6600001);q=arrays['qpos'][tail]
    complete=len(arrays['actions'])==700 and len(q)>=99
    if not complete:raise RuntimeError('Local search left complete-suffix region')
    joint=q[:,base.env.qadr]-base.env.initial[base.env.qadr];height=q[:,2]-base.env.initial[2]-.25
    features=np.column_stack([joint,height]);arrays['endpoint_features']=features
    limits=np.r_[np.full(29,.04),.01];scales=np.r_[np.full(29,.001),.0001]
    score=float(((limits-abs(features))/scales).min())
    path=out/f'trial_{index:03d}.npz';np.savez_compressed(path,**arrays)
    row=dict(trial=index,theta=theta.tolist(),seed_bias=bias.tolist(),waist_bias=waist.tolist(),body_drop_m=drop,
        success=info['success'],complete=complete,margin_score=score,closure_rad=float(abs(joint).max()),rise_max_m=float(height.max()+.25),
        info=info,trajectory_sha256=sha(path))
    (out/f'trial_{index:03d}.json').write_text(json.dumps(row,indent=2))
    print(json.dumps({k:row[k] for k in ('trial','success','margin_score','closure_rad','rise_max_m')}),flush=True)
    return row


def main():
    out=Path('outputs/closure_combined_margin_search_20261024_v1');out.mkdir(exist_ok=False)
    parent=Path('outputs/closure_waist_body_fullcycle_20261024_v1')
    a=json.loads((parent/'independent_training_audit.json').read_text());assert a['passed'] and sum(r['independently_accepted'] for r in a['trajectories'] if r['label']=='after')==5
    sources=[str(Path(__file__).resolve()),'training/closure_waist_target.py','training/closure_body_target.py','training/closure_curriculum_env_v3.py','training/closure_seeded_policy.py']
    protocol=dict(parent_audit_sha256=sha(parent/'independent_training_audit.json'),parent_manifest_sha256=sha(parent/'manifest.json'),
        source_sha256={p:sha(p) for p in sources},seed_policy_sha256=sha('outputs/closure_seed_policy_20261024_v1/policy_initial.zip'),
        theta_mean=[0.]*6,theta_low=[-4.,-4.,-4.,-4.,-4.,-3.8],theta_high=[0.,4.,4.,4.,4.,4.],
        finite_difference_step=.5,initial_trust_radius=1.,maximum_iterations=3,maximum_proposal_retries=3,
        scope='Known +15 mm cached-suffix local margin search. Fixed original task gates and motor limits. Adds waist/body controller authority; not PPO or held-out performance.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];history=[];cache={}
    def key(p):return tuple(np.round(p,12))
    def feature(row):
        with np.load(out/f"trial_{row['trial']:03d}.npz") as z:return z['endpoint_features'].ravel()
    with ProcessPoolExecutor(max_workers=2) as pool:
        def batch(parameters):
            future=[]
            for p in parameters:
                if key(p) not in cache:
                    idx=len(rows)+len(future);future.append((key(p),pool.submit(rollout,(list(p),str(out.resolve()),idx))))
            for k,f in future:
                row=f.result();rows.append(row);cache[k]=row
            return [cache[key(p)] for p in parameters]
        theta=np.zeros(6);current=batch([theta])[0]
        with np.load(parent/'after_foot_z_plus15.npz') as old,np.load(out/'trial_000.npz') as new:
            assert np.max(abs(old['qpos'][4983:]-new['qpos']))<1e-10
            assert np.max(abs(old['qvel'][4983:]-new['qvel']))<1e-8
        low=np.asarray(protocol['theta_low']);high=np.asarray(protocol['theta_high']);radius=1.
        for iteration in range(3):
            probes=[];steps=[]
            for j in range(6):
                step=.5 if theta[j]+.5<=high[j] else -.5
                probe=theta.copy();probe[j]+=step;assert low[j]<=probe[j]<=high[j]
                probes.append(probe);steps.append(step)
            measured=batch(probes);y=feature(current)
            jac=np.column_stack([(feature(row)-y)/step for row,step in zip(measured,steps)])
            limits=np.tile(np.r_[np.full(29,.04),.01],len(y)//30);scales=np.tile(np.r_[np.full(29,.001),.0001],len(y)//30)
            accepted=False
            for retry in range(3):
                bounds=[(max(-radius,l-v),min(radius,h-v)) for l,h,v in zip(low,high,theta)]+[(None,None)]
                A=np.vstack([np.column_stack([jac,scales]),np.column_stack([-jac,scales])]);b=np.r_[limits-y,limits+y]
                fit=linprog(np.r_[np.zeros(6),-1.],A_ub=A,b_ub=b,bounds=bounds,method='highs')
                if not fit.success:raise RuntimeError(fit.message)
                proposal=theta+fit.x[:6];candidate=batch([proposal])[0]
                accepted=bool(candidate['success'] and candidate['margin_score']>current['margin_score']+1e-5)
                history.append(dict(iteration=iteration,retry=retry,radius=radius,trial=candidate['trial'],predicted_margin=float(fit.x[-1]),accepted=accepted))
                if accepted:theta=proposal;current=candidate;radius=min(1.5,radius*1.25);break
                radius*=.5
            if not accepted or current['margin_score']>=.75:break
    best=max(rows,key=lambda r:(r['success'],r['margin_score']))
    (out/'summary.json').write_text(json.dumps(dict(completed=True,rows=rows,best=best,history=history,protocol_sha256=sha(out/'protocol.json')),indent=2))
    print(json.dumps(dict(best_trial=best['trial'],margin_score=best['margin_score'],success=best['success'])),flush=True)

if __name__=='__main__':main()
