"""PPO suffix training of the passing search seed with fixed waist/body commands."""
import argparse,json,shutil,sys,time
from collections import deque
from functools import partial
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from training.full_cycle_residual_env import ROOT
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from training.balanced_case_reset import BalancedCaseReset
from tools.continue_vector_full_cycle_training import sha,state_difference
from tools.train_closure_curriculum_critic import BonusProgress
from tools.evaluate_closure_waist_body import evaluate_parallel
from tools.audit_closure_curriculum_training import audit
from tools.audit_closure_composite_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_controller


def make_env(manifest,output,rank):
    torch.set_num_threads(1)
    env=ClosureCombinedCurriculumEnv(manifest['prefix_cache'],controller_modifier=manifest['controller_modifier'],weight=.1,
        record_dir=Path(output)/f'suffix_records_worker_{rank}')
    return Monitor(BalancedCaseReset(env),str(Path(output)/f'episodes_worker_{rank}'))


def main():
    p=argparse.ArgumentParser()
    for name in ('parent','critic','sampling-seed','env-gate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=98304);a=p.parse_args();assert a.steps>0 and a.steps%1024==0
    torch.set_num_threads(1)
    old=json.loads((a.parent/'manifest.json').read_text());before=json.loads((a.parent/'summary.json').read_text());verified=json.loads((a.parent/'independent_training_audit.json').read_text())
    assert verified['passed'] and verified['experiment_manifest_sha256']==sha(a.parent/'manifest.json') and verified['experiment_summary_sha256']==sha(a.parent/'summary.json') and verified['checkpoint_sha256']==sha(a.parent/'policy.zip')
    assert sum(r['independently_accepted'] for r in verified['trajectories'] if r['label']=='after')==5
    critic=json.loads((a.critic/'summary.json').read_text());sampling=json.loads((a.sampling_seed/'summary.json').read_text());gate=json.loads((a.env_gate/'summary.json').read_text())
    assert critic['passed'] and sha(a.critic/'policy_initial.zip')==critic['checkpoint_sha256']
    assert sampling['passed'] and sampling['parent_checkpoint_sha256']==sha(a.parent/'policy.zip')
    assert critic['seed_checkpoint_sha256']==sha(a.sampling_seed/'policy_initial.zip')==sampling['checkpoint_sha256']
    assert gate['passed'] and gate['environment_source_sha256']==sha('training/closure_combined_curriculum_env.py')
    for evidence in (critic,sampling):
        for path,value in evidence['sources'].items():assert sha(path)==value
    for path,value in old['sources'].items():
        assert sha(a.parent/'source_snapshot'/path)==value
        if path.replace('\\','/').startswith('training/'):assert sha(path)==value
    cache=Path('outputs/closure_prefix_cache_20261024_v4');cp=json.loads((cache/'protocol.json').read_text());cs=json.loads((cache/'summary.json').read_text())
    assert cs['passed'] and cs['protocol_sha256']==sha(cache/'protocol.json') and cp['parent_checkpoint_sha256']==old['prefix_checkpoint_sha256'] and cp['cases']==old['evaluation_cases']
    assert sha(old['prefix_checkpoint'])==old['prefix_checkpoint_sha256']
    a.output.mkdir(exist_ok=False);shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot');shutil.copy2(old['prefix_checkpoint'],a.output/'prefix_policy.zip')
    sources=dict(old['sources'])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        file=Path(file).resolve()
        if file.suffix!='.py' or not file.is_relative_to(ROOT):continue
        rel=file.relative_to(ROOT)
        if rel.parts[0] not in ('tools','training'):continue
        dest=a.output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes());sources[str(rel)]=sha(file)
    manifest={k:old[k] for k in ('contract','config','observation_contract','geometry_mode','evaluation_cases','normal_bias_y_deg','force_history_memory','reference','reference_sha256','bundle_manifest_sha256','prefix_checkpoint_sha256','suffix_start_seconds','horizon_seconds','controller_modifier','zero_baseline_scope','seed_bias','seed_ramp_seconds') if k in old};manifest.update(sources=sources,parent_experiment=str(a.parent.resolve()),parent_manifest_sha256=sha(a.parent/'manifest.json'),parent_checkpoint_sha256=sha(a.parent/'policy.zip'),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),prefix_cache=str(cache.resolve()),prefix_cache_summary_sha256=sha(cache/'summary.json'),
        training_cases=cp['cases'],evaluation_only=False,training_additional_steps=a.steps,additional_steps_requested=a.steps,environments=4,seed=20261024,
        require_inactive_case_exact_parent=False,training_segment='closure_suffix_only',complete_cycle_training=False,training_episode_seconds=14.,
        training_reward_contract='strict-v2-plus-terminal-quality-v1',quality_bonus_weight=.1,exploration_scale=sampling['scale'],
        initial_checkpoint_sha256=sha(a.critic/'policy_initial.zip'),critic_initialization=str(a.critic.resolve()),critic_gate_sha256=sha(a.critic/'summary.json'),
        sampling_seed_gate_sha256=sha(a.sampling_seed/'summary.json'),environment_gate_sha256=sha(a.env_gate/'summary.json'),
        scope='PPO residual learning initialized from a passing search seed, with explicit fixed waist/body commands. Cached 14 s training suffixes; time-zero full-cycle evaluation mandatory. Five development cases; no hardware or broad generalization claim.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    zero=before['zero']
    for row in zero:
        name=f"zero_{row['case']}.npz";assert sha(a.parent/name)==row['trajectory_sha256'];shutil.copy2(a.parent/name,a.output/name)
    (a.output/'zero.json').write_text(json.dumps(zero,indent=2))
    for row in before['after']:
        name=f"after_{row['case']}.npz";assert sha(a.parent/name)==row['trajectory_sha256'];shutil.copy2(a.parent/name,a.output/f"before_{row['case']}.npz")
    (a.output/'before.json').write_text(json.dumps(before['after'],indent=2))
    env=SubprocVecEnv([partial(make_env,manifest,str(a.output.resolve()),rank) for rank in range(4)],start_method='spawn')
    model=PPO.load(a.critic/'policy_initial.zip',env=env,device='cpu',force_reset=True);initial=PPO.load(a.critic/'policy_initial.zip',device='cpu')
    assert state_difference(model.policy.state_dict(),initial.policy.state_dict())==0. and state_difference(model.policy.optimizer.state_dict(),initial.policy.optimizer.state_dict())==0.
    for key in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','target_kl'):assert getattr(model,key)==manifest['config'][key]
    parent_steps=int(model.num_timesteps);model.set_random_seed(20261024);model.set_logger(configure(str(a.output),['csv']))
    model.ep_info_buffer=deque(maxlen=100);model.ep_success_buffer=deque(maxlen=100)
    manifest.update(parent_steps=parent_steps,initial_policy_and_optimizer_restored_exactly=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model.save(a.output/'policy_before');callback=BonusProgress(a.output,parent_steps,4);started=time.perf_counter()
    try:model.learn(total_timesteps=a.steps,reset_num_timesteps=False,callback=callback)
    except Exception as exc:
        model.save(a.output/'policy_interrupted')
        (a.output/'interrupted_status.json').write_text(json.dumps(dict(completed=False,reason=str(exc),steps=int(model.num_timesteps-parent_steps)),indent=2));raise
    finally:env.close()
    model.save(a.output/'policy.zip')
    summary=dict(parent_steps=parent_steps,actual_steps=int(model.num_timesteps),additional_steps=int(model.num_timesteps-parent_steps),
        training_wall_seconds=time.perf_counter()-started,zero=zero,before=before['after'],checkpoint_sha256=sha(a.output/'policy.zip'),case_samples=callback.case_samples,
        completed_training_episodes=len(callback.episodes),successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),evaluation_pending=True)
    if (a.output/'STOP_REQUESTED').exists():
        summary['stopped_by_request']=True;(a.output/'stopped_summary.json').write_text(json.dumps(summary,indent=2));return
    assert summary['additional_steps']==a.steps
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    after=evaluate_parallel(a.output,a.output/'parallel_after',None,workers=3)['trajectories']
    for row in after:
        name=f"after_{row['case']}.npz";shutil.copy2(a.output/'parallel_after'/name,a.output/name)
    (a.output/'after.json').write_text(json.dumps(after,indent=2));summary.update(after=after,evaluation_pending=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));audit(a.output);audit_prefix(a.output);audit_controller(a.output)
    print(json.dumps(dict(completed=True,additional_steps=summary['additional_steps'],successes=sum(r['success'] for r in after))),flush=True)

if __name__=='__main__':main()
