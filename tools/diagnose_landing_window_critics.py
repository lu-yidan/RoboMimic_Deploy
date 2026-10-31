"""Assess existing critics on the longer, audited landing-to-closure window."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.closure_quality_reward import quality_bonus
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
 verified=load_run(a.parent);m=json.loads((a.parent/'manifest.json').read_text());base=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases'])
 first=np.flatnonzero(base.plan['phase']=='posture_closure')[0];start=float(base.plan['time'][first-1]);assert abs(start-99.66)<1e-8
 models={
  'original_fullcycle_critic':a.parent/'policy.zip',
  'short_window_warmstart':Path('outputs/closure_combined_critic_warmstart_20261024_v1/policy_initial.zip'),
  'eightcase_trained_critic':Path('outputs/closure_eightcase_curriculum98304_20261024_v1/policy.zip')}
 policies={name:PPO.load(path,device='cpu') for name,path in models.items()};rows=[]
 try:
  for case in m['evaluation_cases']:
   base.reset(options={'case':case});e=base.env
   with np.load(a.parent/f'after_{case}.npz') as z:
    index=round(70.2/.02);observations=z['observations'][index:].copy();times=z['time_s'][index:-1].copy();q=z['qpos'][index+1:];end_times=z['time_s'][index+1:];rewards=z['rewards'][index:].copy()
   for i,(pose,t) in enumerate(zip(q,end_times)):
    u=float(np.clip((t-start)/3.,0.,1.));rewards[i]+=quality_bonus(float(abs(pose[e.qadr]-e.initial[e.qadr]).max()),float(pose[2]-e.initial[2]),u*u*(3-2*u),weight=.1)
   returns=np.zeros_like(rewards);value=0.
   for i in range(len(rewards)-1,-1,-1):value=float(rewards[i])+.9995*value;returns[i]=value
   for name,model in policies.items():
    with torch.no_grad():predicted=model.policy.predict_values(torch.as_tensor(observations)).cpu().numpy().ravel()
    for label,mask in [('landing_70_99',times<99.66-1e-6),('closure_99_113',times>=99.66-1e-6)]:
     target=returns[mask];estimate=predicted[mask];variance=float(np.var(target))
     rows.append(dict(case=case,critic=name,window=label,samples=int(mask.sum()),mse=float(np.mean((estimate-target)**2)),
      explained_variance=1-float(np.var(estimate-target))/variance if variance>0 else None,mean_bias=float(np.mean(estimate-target))))
 finally:base.close()
 a.output.mkdir(exist_ok=False);result=dict(rows=rows,parent_audit_sha256=verified['audit_sha256'],models={k:dict(path=str(v.resolve()),sha256=sha(v)) for k,v in models.items()},source_sha256=sha(__file__),
  scope='Critic predictions against deterministic fixed-seed trajectory returns at gamma 0.9995 and quality weight 0.1. Off-policy diagnostic, not stochastic value calibration or policy performance.')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({name:{window:float(np.mean([r['mse'] for r in rows if r['critic']==name and r['window']==window])) for window in ('landing_70_99','closure_99_113')} for name in models}))

if __name__=='__main__':main()
