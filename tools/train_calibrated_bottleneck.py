"""Finite matched bottleneck arm with the audited calibration held fixed."""
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
from training.calibrated_landing_env import CalibratedLandingEnv as BoundaryLandingEnv
from training.balanced_case_reset import BalancedCaseReset
from tools.compare_audited_closure_candidates import load_run,sha
from tools.continue_vector_full_cycle_training import state_difference
from tools.train_closure_curriculum_critic import BonusProgress
from tools.evaluate_closure_action_bias_v2 import evaluate_parallel
from tools.audit_action_bias_candidate_v2 import audit
from tools.audit_closure_flexible_prefix_v2 import audit as audit_prefix
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def make_env(protocol,mode,output,rank):
    torch.set_num_threads(1)
    cases={k:protocol['training_cases'][k] for k in protocol['partitions'][rank]}
    env=BoundaryLandingEnv(protocol['parent'],cases=cases,weight=protocol['quality_bonus_weight'],mode=mode)
    return Monitor(BalancedCaseReset(env),str(Path(output)/f'episodes_worker_{rank}'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--mode',choices=('bottleneck',),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    protocol=json.loads((a.protocol/'protocol.json').read_text())
    gate=json.loads((a.protocol/'summary.json').read_text())
    assert gate['passed'] and gate['protocol_sha256']==sha(a.protocol/'protocol.json')
    assert sha(protocol['validation_report'])==protocol['validation_report_sha256']
    for source,digest in protocol['sources'].items(): assert sha(source)==digest
    assert len(gate['rows'])==4
    for row in gate['rows']:
        f=Path(row['directory'])/'summary.json';assert sha(f)==row['summary_sha256']
        r=json.loads(f.read_text());assert r['passed'] and r['reward_reconstruction_passed']
    parent=Path(protocol['parent']);verified=load_run(parent)
    assert verified['audit_sha256']==protocol['parent_audit_sha256']
    assert verified['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
    m=json.loads((parent/'manifest.json').read_text());before=json.loads((parent/'summary.json').read_text())
    steps=protocol['additional_steps_per_arm'];assert steps>0 and steps%1024==0
    assert m['evaluation_cases']==protocol['evaluation_cases']
    root=Path(__file__).resolve().parents[1]
    a.output.mkdir(exist_ok=False);shutil.copytree(parent/'source_snapshot',a.output/'source_snapshot')
    shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
    shutil.copy2(a.protocol/'protocol.json',a.output/'matched_protocol.json')
    sources=dict(m['sources'])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        file=Path(file).resolve()
        if file.suffix!='.py' or not file.is_relative_to(root):continue
        relative=file.relative_to(root)
        if relative.parts[0] not in ('training','tools'):continue
        dest=a.output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(file.read_bytes());sources[str(relative)]=sha(file)
    m.update(sources=sources,parent_experiment=str(parent.resolve()),parent_manifest_sha256=sha(parent/'manifest.json'),
        parent_checkpoint_sha256=verified['checkpoint_sha256'],prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),
        training_cases=protocol['training_cases'],evaluation_only=False,require_inactive_case_exact_parent=False,
        training_additional_steps=steps,additional_steps_requested=steps,seed=protocol['seed'],environments=4,
        initial_checkpoint_sha256=verified['checkpoint_sha256'],initialization=str(parent.resolve()),
        training_reward_contract='strict-v2-calibrated-plus-boundary-'+a.mode+'-quality-v1',
        training_reward_description='Base strict task reward plus nonnegative '+a.mode+' joint/height quality bonus at weight 0.1.',
        quality_bonus_weight=.1,boundary_reward_mode=a.mode,matched_protocol_sha256=sha(a.protocol/'protocol.json'),
        boundary_gate_sha256=sha(a.protocol/'summary.json'),
        optimizer_initialization='Exact audited calibrated-parent raw PPO weights and optimizer; fixed action calibration remains in the environment. No new critic calibration.',
        scope=protocol['scope'])
    m.pop('new_training_steps',None)
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    env=SubprocVecEnv([partial(make_env,protocol,a.mode,str(a.output.resolve()),rank) for rank in range(4)],start_method='spawn')
    model=PPO.load(parent/'policy.zip',env=env,device='cpu',force_reset=True)
    initial=PPO.load(parent/'policy.zip',device='cpu')
    assert state_difference(model.policy.state_dict(),initial.policy.state_dict())==0
    assert state_difference(model.policy.optimizer.state_dict(),initial.policy.optimizer.state_dict())==0
    for key in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','target_kl'):assert getattr(model,key)==m['config'][key]
    parent_steps=int(model.num_timesteps);model.seed=protocol['seed'];model.set_random_seed(model.seed)
    model.set_logger(configure(str(a.output),['csv']));model.ep_info_buffer=deque(maxlen=100);model.ep_success_buffer=deque(maxlen=100)
    model.save(a.output/'policy_before');callback=BonusProgress(a.output,parent_steps,4);started=time.perf_counter()
    try:model.learn(total_timesteps=steps,reset_num_timesteps=False,callback=callback)
    except Exception as exc:
        model.save(a.output/'policy_interrupted')
        (a.output/'interrupted_status.json').write_text(json.dumps(dict(completed=False,reason=str(exc),steps=int(model.num_timesteps-parent_steps)),indent=2));raise
    finally:env.close()
    model.save(a.output/'policy.zip')
    summary=dict(parent_steps=parent_steps,actual_steps=int(model.num_timesteps),additional_steps=int(model.num_timesteps-parent_steps),
        training_wall_seconds=time.perf_counter()-started,before=before['after'],checkpoint_sha256=sha(a.output/'policy.zip'),
        case_samples=callback.case_samples,completed_training_episodes=len(callback.episodes),
        successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),evaluation_pending=True)
    if (a.output/'STOP_REQUESTED').exists():
        summary['stopped_by_request']=True;(a.output/'stopped_summary.json').write_text(json.dumps(summary,indent=2));return
    assert summary['additional_steps']==steps
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    rows=evaluate_parallel(a.output,a.output/'parallel_after',None,workers=4)['trajectories']
    for row in rows:shutil.copy2(a.output/'parallel_after'/f"after_{row['case']}.npz",a.output/f"after_{row['case']}.npz")
    summary.update(after=rows,evaluation_pending=False)
    (a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(a.output);audit_prefix(a.output);audit_fixed(a.output);audit_added(a.output)
    print(json.dumps(dict(completed=True,mode=a.mode,additional_steps=steps,successes=sum(r['success'] for r in rows))),flush=True)


if __name__=='__main__':main()
