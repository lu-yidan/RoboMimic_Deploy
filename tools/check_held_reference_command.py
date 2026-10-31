"""Verify action holding against identical commands at every teacher step."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.closure_legtrim_curriculum_env import ClosureLegtrimCurriculumEnv
from training.held_reference_command import HeldReferenceCommand
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser()
 for name in ('parent','cache','env-gate','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(1);verified=load_run(a.parent);m=json.loads((a.parent/'manifest.json').read_text())
 gate=json.loads((a.env_gate/'summary.json').read_text());assert gate['passed'] and gate['teacher_sha256']==sha(a.parent/'policy.zip')
 for name,digest in gate['sources'].items():assert sha(name)==digest
 assert gate['cache_summary_sha256']==sha(a.cache/'summary.json')
 a.output.mkdir(exist_ok=False)
 env=ClosureLegtrimCurriculumEnv(a.cache,controller_modifier=m['controller_modifier'],teacher_checkpoint=a.parent/'policy.zip',teacher_sha256=sha(a.parent/'policy.zip'))
 original_step=env.step;records=[]
 def read_step(action):
  result=original_step(action);base=env.env.unwrapped
  records.append(dict(qpos=base.env.data.qpos.copy(),qvel=base.env.data.qvel.copy(),time=float(base.env.data.time),reward=float(result[1]),observation=result[0].copy()))
  return result
 env.step=read_step
 held=HeldReferenceCommand(env,repeat=5,physical_gamma=.9995)
 rows=[]
 try:
  for label,command in [('zero',np.zeros(2)),('probe',np.array([-.25,.25]))]:
   results=[]
   for mode in ('raw','held'):
    records.clear();runner=env if mode=='raw' else held;runner.reset(options={'case':'foot_z_plus15'})
    reward_values=[]
    while True:
     _,reward,done,truncated,info=runner.step(command);reward_values.append(reward)
     if done or truncated:break
    arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
    target=a.output/f'{label}_{mode}.npz';np.savez_compressed(target,**arrays)
    gamma=.9995 if mode=='raw' else .9995**5
    total=float(sum(gamma**i*r for i,r in enumerate(reward_values)))
    results.append((arrays,dict(steps=len(reward_values),teacher_steps=len(records),return_sum=total,success=info['success'],trajectory_sha256=sha(target))))
   raw,macro=results
   differences={k:float(abs(raw[0][k]-macro[0][k]).max()) for k in raw[0]}
   checks=dict(all_samples_identical=all(value==0. for value in differences.values()),discounted_return=abs(raw[1]['return_sum']-macro[1]['return_sum'])<1e-10,
    same_success=raw[1]['success']==macro[1]['success'],cadence=raw[1]['teacher_steps']==macro[1]['teacher_steps']==700,macro_steps=macro[1]['steps']==140)
   rows.append(dict(label=label,passed=all(checks.values()),checks=checks,differences=differences,raw=raw[1],held=macro[1]));print(json.dumps(rows[-1]),flush=True)
  sources={name:sha(name) for name in ('training/held_reference_command.py','tools/check_held_reference_command.py')}
  report=dict(passed=all(r['passed'] for r in rows),rows=rows,sources=sources,environment_gate_sha256=sha(a.env_gate/'summary.json'),
   scope='Identical held command yields identical 50 Hz teacher trajectories and physically discounted returns. No trained 10 Hz feedback performance claim.')
  (a.output/'summary.json').write_text(json.dumps(report,indent=2))
  for name in sources:
   dest=a.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
  assert report['passed']
 finally:env.close()

if __name__=='__main__':main()
