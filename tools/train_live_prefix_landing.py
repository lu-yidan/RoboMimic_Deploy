"""Short PPO feasibility run with an actual physical prefix on every reset."""
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
from training.live_prefix_landing_env import LivePrefixLandingEnv
from training.balanced_case_reset import BalancedCaseReset
from tools.compare_audited_closure_candidates import load_run,sha
from tools.continue_vector_full_cycle_training import state_difference
from tools.train_closure_curriculum_critic import BonusProgress
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_candidate_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def make_env(parent,manifest,output,rank):
    torch.set_num_threads(1)
    cases={k:manifest['evaluation_cases'][k] for k in list(manifest['evaluation_cases'])[rank::4]}
    env=LivePrefixLandingEnv(parent,cases=cases,weight=.1)
    return Monitor(BalancedCaseReset(env),str(Path(output)/f'episodes_worker_{rank}'))


def main():
    p=argparse.ArgumentParser()
    for key in ('parent','grid','seed','live-gate','dataset','data-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=18432);a=p.parse_args()
    assert a.steps>=18432 and a.steps%1024==0
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    verified=load_run(a.parent);grid=load_run(a.grid)
    assert len(verified['rows'])==8 and all(r['success'] for r in verified['rows'].values())
    assert len(grid['rows'])==12 and all(r['success'] for r in grid['rows'].values())
    for key in ('checkpoint_sha256','reference_sha256','controller_modifier','adaptive_reference_command'):assert verified[key]==grid[key]
    m=json.loads((a.parent/'manifest.json').read_text());before=json.loads((a.parent/'summary.json').read_text())
    seed=json.loads((a.seed/'summary.json').read_text());assert seed['passed'] and seed['checkpoint_sha256']==sha(a.seed/'policy_initial.zip')
    assert seed['parent_checkpoint_sha256']==verified['checkpoint_sha256'] and seed['controller_modifier']==m['controller_modifier']
    for path,digest in seed['sources'].items():assert sha(path)==digest
    live=json.loads((a.live_gate/'summary.json').read_text());lp=json.loads((a.live_gate/'protocol.json').read_text())
    assert live['passed'] and live['protocol_sha256']==sha(a.live_gate/'protocol.json') and lp['parent_audit_sha256']==verified['audit_sha256']
    assert {r['case'] for r in live['rows']}==set(m['evaluation_cases'])
    for path,digest in lp['sources'].items():assert sha(path)==digest
    data=json.loads((a.dataset/'summary.json').read_text());dp=json.loads((a.dataset/'protocol.json').read_text());gate=json.loads((a.data_audit/'summary.json').read_text())
    assert data['passed'] and gate['passed'] and gate['complete_dataset'] and len(data['rows'])==24
    assert gate['source_sha256']==sha('tools/audit_live_landing_data.py') and gate['seed_checkpoint_sha256']==seed['checkpoint_sha256']
    assert len(gate['rows'])==24 and all(r['passed'] for r in gate['rows'])
    assert {r['index']:r['trajectory_sha256'] for r in gate['rows']}=={r['index']:r['trajectory_sha256'] for r in data['rows']}
    assert data['protocol_sha256']==gate['dataset_protocol_sha256']==sha(a.dataset/'protocol.json')
    assert gate['dataset_summary_sha256']==sha(a.dataset/'summary.json')
    assert dp['parent_audit_sha256']==verified['audit_sha256'] and dp['seed_checkpoint_sha256']==seed['checkpoint_sha256']
    assert dp['fixed_reference_command']==m['adaptive_reference_command']
    assert set(data['success_by_case'])==set(m['evaluation_cases']) and min(data['success_by_case'].values())>=1
    for path,digest in dp['sources'].items():assert sha(path)==digest
    for row in data['rows']:assert sha(a.dataset/row['file'])==row['trajectory_sha256']
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
    m.update(sources=sources,parent_experiment=str(a.parent.resolve()),parent_manifest_sha256=sha(a.parent/'manifest.json'),
             parent_checkpoint_sha256=sha(a.parent/'policy.zip'),prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),
             suffix_start_seconds=70.2,training_episode_seconds=43.46,training_cases=m['evaluation_cases'],evaluation_only=False,
             training_additional_steps=a.steps,additional_steps_requested=a.steps,training_step_unit='50 Hz suffix transitions; prefix reset simulation excluded',
             environments=4,seed=20261024,require_inactive_case_exact_parent=False,training_segment='left_foot_landing_and_closure',complete_cycle_training=False,
             initial_checkpoint_sha256=seed['checkpoint_sha256'],initialization=str(a.seed.resolve()),quality_bonus_weight=.1,
             live_prefix_gate_sha256=sha(a.live_gate/'summary.json'),sample_data_audit_sha256=sha(a.data_audit/'summary.json'),
             training_reward_contract='strict-v2-plus-positive-closure-quality-v1',
             training_reward_description='Base strict task reward plus nonnegative closure quality bonus at weight 0.1.',
             reset_method='Time-zero frozen-prefix physics replay for every episode; no cached-state restoration',
             optimizer_initialization='Exact seed checkpoint policy and optimizer, including its original uncalibrated critic; no new critic fitting.',
             scope='Short 26-action PPO feasibility run with real-prefix resets, four disjoint two-case workers. Known development cases, unchanged task and physics gates. Not equal-budget comparison to the prior 98304-step calibrated-critic run. Full-cycle evaluation follows training; no automatic promotion.')
    m.pop('new_training_steps',None);(a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    env=SubprocVecEnv([partial(make_env,str(a.parent.resolve()),m,str(a.output.resolve()),rank) for rank in range(4)],start_method='spawn')
    model=PPO.load(a.seed/'policy_initial.zip',env=env,device='cpu',force_reset=True);initial=PPO.load(a.seed/'policy_initial.zip',device='cpu')
    assert state_difference(model.policy.state_dict(),initial.policy.state_dict())==0
    assert state_difference(model.policy.optimizer.state_dict(),initial.policy.optimizer.state_dict())==0
    for key in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','target_kl'):assert getattr(model,key)==m['config'][key]
    parent_steps=int(model.num_timesteps);model.seed=20261024;model.set_random_seed(model.seed)
    model.set_logger(configure(str(a.output),['csv']));model.ep_info_buffer=deque(maxlen=100);model.ep_success_buffer=deque(maxlen=100)
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
    for row in rows:shutil.copy2(a.output/'parallel_after'/f"after_{row['case']}.npz",a.output/f"after_{row['case']}.npz")
    summary.update(after=rows,evaluation_pending=False);(a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(a.output);audit_prefix(a.output);audit_fixed(a.output);audit_added(a.output)
    print(json.dumps(dict(completed=True,additional_steps=summary['additional_steps'],successes=sum(r['success'] for r in rows))),flush=True)


if __name__=='__main__':main()
