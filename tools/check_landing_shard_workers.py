"""Validate disjoint four-worker resets and first policy transitions before training."""
import argparse
from functools import partial
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from training.landing_controller_curriculum_env import LandingControllerCurriculumEnv
from training.balanced_case_reset import BalancedCaseReset
from tools.compare_audited_closure_candidates import load_run,sha


def make_env(directory,modifier,command):
 torch.set_num_threads(1)
 env=LandingControllerCurriculumEnv(Path(directory),controller_modifier=modifier,reference_command=command,weight=.1)
 env.loaded_cache_names=list(env.caches)
 return BalancedCaseReset(env)


def main():
 p=argparse.ArgumentParser()
 for name in ('parent','shards','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(1);verified=load_run(a.parent);m=json.loads((a.parent/'manifest.json').read_text())
 shards=json.loads((a.shards/'summary.json').read_text());assert shards['passed'] and len(shards['workers'])==4
 for r in shards['workers']:assert sha(a.shards/f"worker_{r['worker']}"/'summary.json')==r['summary_sha256']
 a.output.mkdir(exist_ok=False);teacher=PPO.load(a.parent/'policy.zip',device='cpu');rows=[]
 env=SubprocVecEnv([partial(make_env,str((a.shards/f'worker_{rank}').resolve()),m['controller_modifier'],m.get('adaptive_reference_command')) for rank in range(4)],start_method='spawn')
 try:
  loaded=env.get_attr('loaded_cache_names');assert all(len(c)==2 for c in loaded) and len(set(k for cases in loaded for k in cases))==8
  env.seed(20261024)
  for cycle in range(2):
   obs=env.reset();cases=[r['case'] for r in env.reset_infos]
   actions=np.asarray([teacher.predict(row,deterministic=True)[0] for row in obs])
   following,rewards,done,infos=env.step(actions)
   assert not done.any()
   for rank,case in enumerate(cases):
    assert case in loaded[rank]
    with np.load(a.parent/f'after_{case}.npz') as trace:
     index=round(70.2/.02)
     difference_before=float(abs(obs[rank]-trace['observations'][index]).max())
     difference_after=float(abs(following[rank]-trace['observations'][index+1]).max())
    row=dict(worker=rank,cycle=cycle,case=case,reset_difference=difference_before,next_difference=difference_after,time_s=infos[rank]['time_s'],
     passed=difference_before<1e-5 and difference_after<1e-5 and abs(infos[rank]['time_s']-70.22)<1e-7)
    rows.append(row);print(json.dumps(row),flush=True)
  assert len({r['case'] for r in rows})==8
  sources={name:sha(name) for name in ('tools/check_landing_shard_workers.py','training/landing_controller_curriculum_env.py','training/closure_foreaft_reference.py','training/closure_combined_curriculum_env.py','training/closure_curriculum_env_v3.py','training/balanced_case_reset.py')}
  result=dict(passed=all(r['passed'] for r in rows),rows=rows,loaded_cache_names=loaded,loaded_templates=sum(map(len,loaded)),sources=sources,
   parent_audit_sha256=verified['audit_sha256'],shards_summary_sha256=sha(a.shards/'summary.json'),scope='Four-worker reset scheduling and first-transition checks only; full-suffix replay is a separate gate. Eight resident immutable templates, excluding live working buffers and Python overhead.')
  (a.output/'summary.json').write_text(json.dumps(result,indent=2))
  for name in sources:
   dest=a.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
  assert result['passed']
 finally:env.close()

if __name__=='__main__':main()
