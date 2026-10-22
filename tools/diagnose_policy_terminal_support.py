"""Deterministic replay with read-only diagnostics of the exact runtime guard."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from training.biased_load_transfer_residual_env import BiasedLoadTransferResidualEnv,scenario_name
from training.climb_robustness import GROUPS,BODIES


def trial(run,out,label):
    manifest=json.loads((run/'manifest.json').read_text());env=BiasedLoadTransferResidualEnv(manifest['suite'])
    case='row01_x_-10mm';angle=3.;name=scenario_name(case,angle)
    obs,_=env.reset(seed=manifest['seed'],options=dict(case=case,normal_bias_y_deg=angle))
    model=None if label=='zero' else PPO.load(run/'policy.zip',device='cpu')
    original=env.guard.sample;rows=[];e=env.env
    def sample(t):
        original(t)
        if t<61.25:return
        metrics=e.contact_metrics();groups=[]
        for i,group in enumerate(GROUPS):
            contacts=[c for c in metrics['contacts'] if c['group']==group and np.linalg.norm(c['force_on_robot_world_n'])>1.]
            force=sum(c['force_on_robot_world_n'][2] for c in contacts if c['external_geom']==env.guard.expected[i])
            body=e.model.body(BODIES[i]).id;local=np.array([.09,0.,-.035] if i<2 else [0.,0.,.022])
            point=e.data.xpos[body]+e.data.xmat[body].reshape(3,3)@local;jac=np.zeros((3,e.model.nv))
            mujoco.mj_jac(e.model,e.data,jac,None,point,body);speed=float(np.linalg.norm(jac@e.data.qvel))
            groups.append(dict(group=group,force_n=float(force),speed_m_s=speed,rods=[c['external_geom'] for c in contacts],expected=env.guard.expected[i],valid=env.guard.history[-1][1][i],force_ok=bool(force>=10.),speed_ok=bool(speed<=.02),identity_ok=bool(contacts and all(c['external_geom']==env.guard.expected[i] for c in contacts))))
        rows.append(dict(time_s=float(t),groups=groups))
    env.guard.sample=sample
    states=[e.data.qpos.copy()];velocities=[e.data.qvel.copy()]
    for _ in range(env.max_steps):
        action=np.zeros(6) if model is None else model.predict(obs,deterministic=True)[0]
        obs,_,term,trunc,info=env.step(action);states.append(e.data.qpos.copy());velocities.append(e.data.qvel.copy())
        if term or trunc:break
    old=dict(np.load(run/f'{label}_{name}.npz'))
    delta=float(np.max(abs(np.asarray(states)-old['qpos'])));vdelta=float(np.max(abs(np.asarray(velocities)-old['qvel'])))
    if max(delta,vdelta)>1e-9:raise AssertionError(f'Replay changed trajectory {delta} {vdelta}')
    terminal=[row for row in rows if row['time_s']>=61.31-1e-8]
    stats={}
    for i,group in enumerate(GROUPS):
        values=[row['groups'][i] for row in terminal]
        stats[group]=dict(min_force_n=min(r['force_n'] for r in values),max_speed_m_s=max(r['speed_m_s'] for r in values),invalid_samples=sum(not r['valid'] for r in values),force_failures=sum(not r['force_ok'] for r in values),speed_failures=sum(not r['speed_ok'] for r in values),identity_failures=sum(not r['identity_ok'] for r in values))
    result=dict(label=label,success=info['success'],reason=info['reason'],qpos_replay_max_difference=delta,qvel_replay_max_difference=vdelta,terminal_samples=len(terminal),groups=stats,rows=rows,scope='Read-only extra guard diagnostics; replay states compared against previously audited trajectory. Left foot is intentionally unloaded; required supports are rightfoot and both hooks.')
    (out/(label+'.json')).write_text(json.dumps(result,indent=2));env.close();return {k:v for k,v in result.items() if k!='rows'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    results=[trial(a.run,a.out,label) for label in ['zero','after']]
    result=dict(results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
