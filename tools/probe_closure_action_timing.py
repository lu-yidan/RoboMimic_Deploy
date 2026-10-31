"""Compare ramp durations for one fixed bounded action vector on real suffix physics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from tools.continue_vector_full_cycle_training import sha

ENV=None
POLICY=None

def run(payload):
    global ENV,POLICY
    duration,bias,parameters,out=payload
    torch.set_num_threads(1)
    if ENV is None:
        ENV=ClosureCurriculumEnvV3('outputs/closure_prefix_cache_20261024_v4')
        POLICY=PPO.load('outputs/closure_margin_control98304_v1/policy.zip',device='cpu')
    obs,_=ENV.reset(options={'case':'foot_z_plus15'})
    base=ENV.env.unwrapped
    trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],base_rewards=[])
    while True:
        t=113.66*(1.-float(obs[105]));u=float(np.clip((t-99.66)/duration,0.,1.));ramp=u*u*(3-2*u)
        action=np.clip(POLICY.predict(obs,deterministic=True)[0]+ramp*bias,-1.,1.).astype(np.float32)
        trace['observations'].append(obs.copy());trace['actions'].append(action.copy())
        obs,reward,done,truncated,info=ENV.step(action)
        trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy())
        trace['time_s'].append(float(base.env.data.time));trace['torques'].append(base.last_torque.copy());trace['base_rewards'].append(info['base_reward'])
        if done or truncated:break
    arrays={k:np.asarray(v) for k,v in trace.items()}
    tail=(arrays['time_s']>111.66)&(arrays['time_s']<=113.6600001)
    q=arrays['qpos'][tail];complete=len(arrays['actions'])==700 and len(q)>=99
    if complete:
        joint=q[:,base.env.qadr]-base.env.initial[base.env.qadr];rise=q[:,2]-base.env.initial[2]
        worst=np.r_[np.max(abs(joint),axis=0),np.max(abs(rise-.25))]
        excess=np.maximum(0.,worst-np.r_[np.full(29,.039),.0098])/np.r_[np.full(29,.005),.001]
        cost=float(excess@excess)
        metrics=dict(closure_rad=float(abs(joint).max()),rise_min_m=float(rise.min()),rise_max_m=float(rise.max()))
    else:cost=1e8;metrics={}
    path=Path(out)/f'ramp_{duration:g}s.npz';np.savez_compressed(path,**arrays)
    row=dict(ramp_seconds=duration,parameters=parameters,complete=complete,cost=cost,success=info['success'],
      metrics=metrics,info=info,trajectory_sha256=sha(path),file=path.name)
    path.with_suffix('.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(dict(duration=duration,success=info['success'],cost=cost,**metrics)),flush=True)
    return row


def main():
    source=Path('outputs/closure_action_trust_region_20261024_v2')
    search=json.loads((source/'summary.json').read_text());assert search['completed']
    p=search['best']['parameters'];basis=np.asarray(search['basis']);bias=basis@np.asarray(p)
    assert abs(bias).max()<=1.+1e-9;bias=np.clip(bias,-1.,1.)
    out=Path('outputs/closure_action_timing_20261024_v1');out.mkdir(exist_ok=False)
    durations=[1.,2.,3.,4.,5.,6.,8.]
    protocol=dict(durations=durations,parameters=p,basis=basis.tolist(),groups=search['groups'],
      source_search_sha256=sha(source/'summary.json'),parent_checkpoint_sha256=sha('outputs/closure_margin_control98304_v1/policy.zip'),
      cache_summary_sha256=sha('outputs/closure_prefix_cache_20261024_v4/summary.json'),
      sources={n:sha(n) for n in [str(Path(__file__).resolve()),'training/closure_curriculum_env_v3.py']},
      scope='Fixed action amplitude, seven declared ramp durations; time derived from existing observation to match deployment. Known-case cached suffix screening only.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,(d,bias,p,str(out.resolve()))) for d in durations]
        for future in as_completed(futures):rows.append(future.result())
    rows.sort(key=lambda r:r['ramp_seconds'])
    best=min(rows,key=lambda r:(not r['success'],r['cost']))
    result=dict(completed=True,rows=rows,best=best,basis=basis.tolist(),groups=search['groups'],protocol_sha256=sha(out/'protocol.json'),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(best_duration=best['ramp_seconds'],success=best['success'],cost=best['cost'])),flush=True)


if __name__=='__main__':main()
