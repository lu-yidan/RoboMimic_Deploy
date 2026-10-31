"""Validate a longer landing-window reset against the audited fixed search seed."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser()
 for name in ('parent','cache','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(1);verified=load_run(a.parent)
 m=json.loads((a.parent/'manifest.json').read_text());cp=json.loads((a.cache/'protocol.json').read_text());cs=json.loads((a.cache/'summary.json').read_text())
 assert cs['passed'] and cs['protocol_sha256']==sha(a.cache/'protocol.json')
 assert abs(cp['start_seconds']-70.2)<1e-8 and cp['reference_sha256']==m['reference_sha256']
 assert all(m['evaluation_cases'][case]==offset for case,offset in cp['cases'].items())
 a.output.mkdir(exist_ok=False)
 tolerance=dict(qpos=1e-7,qvel=1e-5,torques=1e-3,observations=1e-5)
 protocol=dict(parent_audit_sha256=verified['audit_sha256'],cache_summary_sha256=sha(a.cache/'summary.json'),tolerance=tolerance,
  scope='Longer training reset and closed-loop seed replay gate only. No learned policy or new full-cycle success claim.')
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
 env=ClosureCombinedCurriculumEnv(a.cache,controller_modifier=m['controller_modifier'],weight=0.)
 teacher=PPO.load(a.parent/'policy.zip',device='cpu');rows=[]
 try:
  for case in cp['cases']:
   obs,_=env.reset(options={'case':case});base=env.env.unwrapped
   arrays=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],torques=[],observations=[])
   while True:
    arrays['observations'].append(obs.copy());action=teacher.predict(obs,deterministic=True)[0]
    obs,_,done,truncated,info=env.step(action)
    arrays['qpos'].append(base.env.data.qpos.copy());arrays['qvel'].append(base.env.data.qvel.copy());arrays['torques'].append(base.last_torque.copy())
    if done or truncated:break
   arrays={k:np.asarray(v) for k,v in arrays.items()};start=round(cp['start_seconds']/.02)
   with np.load(a.parent/f'after_{case}.npz') as original:
    differences={k:float(abs(v-original[k][start:]).max()) for k,v in arrays.items()}
   passed=all(differences[k]<=tolerance[k] for k in tolerance) and info['success']==verified['rows'][case]['success']
   np.savez_compressed(a.output/f'{case}.npz',**arrays)
   rows.append(dict(case=case,passed=bool(passed),differences=differences,task_success=info['success'],trajectory_sha256=sha(a.output/f'{case}.npz')))
   print(json.dumps(rows[-1]),flush=True)
  sources={name:sha(name) for name in ('tools/check_landing_window_seed.py','training/closure_combined_curriculum_env.py','training/closure_curriculum_env_v3.py')}
  result=dict(passed=all(r['passed'] for r in rows),rows=rows,sources=sources,protocol_sha256=sha(a.output/'protocol.json'))
  (a.output/'summary.json').write_text(json.dumps(result,indent=2))
  for name in sources:
   dest=a.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
  assert result['passed']
 finally:env.close()

if __name__=='__main__':main()
