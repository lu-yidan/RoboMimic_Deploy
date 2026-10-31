"""Collect landing-window episodes after a real physical prefix on every reset."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.live_prefix_landing_env import LivePrefixLandingEnv
from tools.compare_audited_closure_candidates import load_run,sha


def collect_worker(payload):
 rank,seed_dir,parent,m,output=payload;torch.set_num_threads(1)
 cases={key:m['evaluation_cases'][key] for key in list(m['evaluation_cases'])[rank::4]}
 env=LivePrefixLandingEnv(parent,cases=cases,weight=.1)
 model=PPO.load(Path(seed_dir)/'policy_initial.zip',device='cpu');rows=[];names=list(m['evaluation_cases'])
 try:
  for episode,split in enumerate(('train','validation','test')):
   for case in env.cases:
    index=names.index(case)*3+episode;seed=20301024+index;torch.manual_seed(seed);np.random.seed(seed)
    obs,_=env.reset(seed=seed,options={'case':case});base=env.env.unwrapped
    trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],rewards=[],base_rewards=[])
    while True:
     trace['observations'].append(obs.copy());action=model.predict(obs,deterministic=False)[0]
     obs,reward,done,truncated,info=env.step(action)
     trace['actions'].append(action.copy());trace['rewards'].append(reward);trace['base_rewards'].append(info['base_reward'])
     trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy());trace['time_s'].append(float(base.env.data.time))
     if done or truncated:break
    arrays={k:np.asarray(v) for k,v in trace.items()};assert all(np.isfinite(v).all() for v in arrays.values())
    file=f'episode_{index:03d}.npz';target=Path(output)/file;np.savez_compressed(target,**arrays)
    row=dict(case=case,episode=episode,index=index,seed=seed,split=split,file=file,steps=len(arrays['actions']),success=info['success'],info=info,trajectory_sha256=sha(target))
    target.with_suffix('.json').write_text(json.dumps(row,indent=2));rows.append(row);print(json.dumps({k:row[k] for k in ('case','split','steps','success')}),flush=True)
 finally:env.close()
 return rows


def main():
 p=argparse.ArgumentParser()
 for name in ('parent','seed','live-gate','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();verified=load_run(a.parent);m=json.loads((a.parent/'manifest.json').read_text());seed=json.loads((a.seed/'summary.json').read_text())
 assert seed['passed'] and sha(a.seed/'policy_initial.zip')==seed['checkpoint_sha256'] and seed['parent_checkpoint_sha256']==verified['checkpoint_sha256']
 assert seed['controller_modifier']==m['controller_modifier']
 for name,digest in seed['sources'].items():assert sha(name)==digest
 gate=json.loads((a.live_gate/'summary.json').read_text());assert gate['passed']
 protocol=json.loads((a.live_gate/'protocol.json').read_text())
 assert gate['protocol_sha256']==sha(a.live_gate/'protocol.json') and protocol['parent_audit_sha256']==verified['audit_sha256']
 assert {r['case'] for r in gate['rows']}==set(m['evaluation_cases'])
 for name,digest in protocol['sources'].items():assert sha(name)==digest
 a.output.mkdir(exist_ok=False)
 sources={name:sha(name) for name in ('tools/collect_live_prefix_landing_data.py','training/live_prefix_landing_env.py','training/full_cycle_residual_env.py','training/strict_full_cycle_residual_env.py','training/closure_quality_reward.py','training/closure_foreaft_reference.py','training/closure_composite_policy.py')}
 protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=verified['audit_sha256'],seed_checkpoint_sha256=sha(a.seed/'policy_initial.zip'),seed_gate_sha256=sha(a.seed/'summary.json'),
  cases=list(m['evaluation_cases']),episodes_per_case=3,split=dict(train=1,validation=1,test=1),source_sha256=sha(__file__),sources=sources,
  controller_modifier=m['controller_modifier'],fixed_reference_command=m.get('adaptive_reference_command'),start_seconds=70.2,horizon_seconds=113.66,
  live_reset_gate_sha256=sha(a.live_gate/'summary.json'),reset_method='Real time-zero frozen-prefix simulation on every reset; no serialized state restore.',
  scope='24 stochastic landing-window episodes, one fit/validation/test episode per known case. Real prefix simulation on every reset. Same sampling seed schedule as prior cached collection, under the new fixed reference. Critic calibration and sampling feasibility only; no policy generalization or PPO updates.')
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(collect_worker,(rank,str(a.seed.resolve()),str(a.parent.resolve()),m,str(a.output.resolve()))) for rank in range(4)]
  for future in as_completed(jobs):rows.extend(future.result())
 assert len(rows)==24 and len({r['index'] for r in rows})==24
 result=dict(passed=True,rows=sorted(rows,key=lambda r:r['index']),steps=sum(r['steps'] for r in rows),successes=sum(r['success'] for r in rows),protocol_sha256=sha(a.output/'protocol.json'),
  success_by_case={case:sum(r['success'] for r in rows if r['case']==case) for case in m['evaluation_cases']})
 (a.output/'summary.json').write_text(json.dumps(result,indent=2))
 for name in sources:
  dest=a.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
