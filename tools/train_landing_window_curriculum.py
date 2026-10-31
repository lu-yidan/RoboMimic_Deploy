"""Train the landing-to-closure window; keep full time-zero evaluation mandatory."""
import argparse
from collections import deque
from functools import partial
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from training.landing_controller_curriculum_env import LandingControllerCurriculumEnv
from training.balanced_case_reset import BalancedCaseReset
from tools.compare_audited_closure_candidates import load_run,sha
from tools.continue_vector_full_cycle_training import state_difference
from tools.train_closure_curriculum_critic import BonusProgress
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_candidate_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def make_env(manifest,output,rank):
 torch.set_num_threads(1)
 env=LandingControllerCurriculumEnv(Path(manifest['worker_shards'])/f'worker_{rank}',controller_modifier=manifest['controller_modifier'],reference_command=manifest['adaptive_reference_command'],weight=.1,
  record_dir=Path(output)/f'episodes_worker_{rank}_traces')
 return Monitor(BalancedCaseReset(env),str(Path(output)/f'episodes_worker_{rank}'))


def main():
 p=argparse.ArgumentParser()
 for name in ('parent','grid','critic','dataset','shards','window-gate','worker-gate','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--steps',type=int,default=98304);a=p.parse_args();assert a.steps>0 and a.steps%1024==0
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[1];verified=load_run(a.parent);grid=load_run(a.grid)
 assert len(verified['rows'])==8 and sum(r['success'] for r in verified['rows'].values())>=7
 assert len(grid['rows'])==12 and all(r['success'] for r in grid['rows'].values())
 assert json.loads((a.grid/'regression.json').read_text())['lost']==[]
 assert grid['checkpoint_sha256']==verified['checkpoint_sha256'] and grid['adaptive_reference_command']==verified['adaptive_reference_command']
 m=json.loads((a.parent/'manifest.json').read_text());before=json.loads((a.parent/'summary.json').read_text())
 critic=json.loads((a.critic/'summary.json').read_text());data=json.loads((a.dataset/'protocol.json').read_text())
 assert critic['passed'] and critic['checkpoint_sha256']==sha(a.critic/'policy_initial.zip')
 assert critic['dataset_protocol_sha256']==sha(a.dataset/'protocol.json') and critic['dataset_summary_sha256']==sha(a.dataset/'summary.json')
 assert data['parent_audit_sha256']==verified['audit_sha256'] and data['fixed_reference_command']==m['adaptive_reference_command']
 for name,digest in critic['evidence'].items():assert sha(name)==digest
 for name,digest in critic['sources'].items():assert sha(name)==digest
 window=json.loads((a.window_gate/'summary.json').read_text());workers=json.loads((a.worker_gate/'summary.json').read_text());assert window['passed'] and workers['passed']
 assert json.loads((a.window_gate/'protocol.json').read_text())['parent_audit_sha256']==verified['audit_sha256']
 assert workers['parent_audit_sha256']==verified['audit_sha256'] and workers['shards_summary_sha256']==sha(a.shards/'summary.json')
 for gate in (window,workers):
  for name,digest in gate['sources'].items():assert sha(name)==digest
 a.output.mkdir(exist_ok=False);shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot');shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
 sources=dict(m['sources'])
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if not file:continue
  file=Path(file).resolve()
  if file.suffix!='.py' or not file.is_relative_to(root):continue
  relative=file.relative_to(root)
  if relative.parts[0] not in ('training','tools'):continue
  dest=a.output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes());sources[str(relative)]=sha(file)
 m.update(sources=sources,parent_experiment=str(a.parent.resolve()),parent_manifest_sha256=sha(a.parent/'manifest.json'),parent_checkpoint_sha256=sha(a.parent/'policy.zip'),
  prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),suffix_start_seconds=70.2,training_episode_seconds=43.46,
  training_cases=m['evaluation_cases'],evaluation_only=False,training_additional_steps=a.steps,additional_steps_requested=a.steps,environments=4,seed=20261024,
  require_inactive_case_exact_parent=False,training_segment='left_foot_landing_and_closure',complete_cycle_training=False,
  worker_shards=str(a.shards.resolve()),worker_shards_sha256=sha(a.shards/'summary.json'),initial_checkpoint_sha256=sha(a.critic/'policy_initial.zip'),
  critic_initialization=str(a.critic.resolve()),critic_gate_sha256=sha(a.critic/'summary.json'),quality_bonus_weight=.1,
  window_gate_sha256=sha(a.window_gate/'summary.json'),worker_gate_sha256=sha(a.worker_gate/'summary.json'),
  scope='PPO on 70.2-113.66 seconds over a fixed searched controller, including left-foot landing. Four disjoint two-case workers, unchanged physical/task gates. New training window and critic calibration; not an equal-initialization comparison to the old 14-second experiments.')
 m.pop('new_training_steps',None)
 (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
 env=SubprocVecEnv([partial(make_env,m,str(a.output.resolve()),rank) for rank in range(4)],start_method='spawn')
 model=PPO.load(a.critic/'policy_initial.zip',env=env,device='cpu',force_reset=True);initial=PPO.load(a.critic/'policy_initial.zip',device='cpu')
 assert state_difference(model.policy.state_dict(),initial.policy.state_dict())==0 and state_difference(model.policy.optimizer.state_dict(),initial.policy.optimizer.state_dict())==0
 for key in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','target_kl'):assert getattr(model,key)==m['config'][key]
 parent_steps=int(model.num_timesteps);model.set_random_seed(20261024);model.set_logger(configure(str(a.output),['csv']));model.ep_info_buffer=deque(maxlen=100);model.ep_success_buffer=deque(maxlen=100)
 model.save(a.output/'policy_before');callback=BonusProgress(a.output,parent_steps,4);started=time.perf_counter()
 try:model.learn(total_timesteps=a.steps,reset_num_timesteps=False,callback=callback)
 except Exception as exc:
  model.save(a.output/'policy_interrupted');(a.output/'interrupted_status.json').write_text(json.dumps(dict(completed=False,reason=str(exc),steps=int(model.num_timesteps-parent_steps)),indent=2));raise
 finally:env.close()
 model.save(a.output/'policy.zip');summary=dict(parent_steps=parent_steps,actual_steps=int(model.num_timesteps),additional_steps=int(model.num_timesteps-parent_steps),training_wall_seconds=time.perf_counter()-started,
  before=before['after'],checkpoint_sha256=sha(a.output/'policy.zip'),case_samples=callback.case_samples,completed_training_episodes=len(callback.episodes),successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),evaluation_pending=True)
 if (a.output/'STOP_REQUESTED').exists():
  summary['stopped_by_request']=True;(a.output/'stopped_summary.json').write_text(json.dumps(summary,indent=2));return
 assert summary['additional_steps']==a.steps
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
 rows=evaluate_parallel(a.output,a.output/'parallel_after',None,workers=2)['trajectories']
 for row in rows:
  name=f"after_{row['case']}.npz";shutil.copy2(a.output/'parallel_after'/name,a.output/name)
 summary.update(after=rows,evaluation_pending=False);(a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
 audit(a.output);audit_prefix(a.output);audit_fixed(a.output);audit_added(a.output)
 print(json.dumps(dict(completed=True,additional_steps=summary['additional_steps'],successes=sum(r['success'] for r in rows))),flush=True)

if __name__=='__main__':main()
