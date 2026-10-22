"""Matched blind/oracle training from a verified equal-action initialization."""
import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from training.full_cycle_residual_env import ROOT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation, OBSERVATION_CONTRACT
from training.balanced_case_reset import BalancedCaseReset
from tools.continue_vector_full_cycle_training import VectorProgress, state_difference, sha
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_bar_geometry_policy import evaluate_parallel


def make_env(manifest,mode,output,rank):
    torch.set_num_threads(1)
    env=FullCycleResidualEnv(plan=manifest['reference'],cases=manifest['training_cases'],
        normal_bias_y_deg=manifest['normal_bias_y_deg'],memory_enabled=manifest['force_history_memory'])
    return Monitor(BalancedCaseReset(BarGeometryObservation(env,mode=mode)),str(Path(output)/f'episodes_worker_{rank}'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--mode',choices=['blind','oracle'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    for source,digest in protocol['sources'].items():
        if sha(source)!=digest:raise ValueError('Protocol source changed: '+source)
    parent=Path(protocol['parent']);migration=Path(protocol['migration']);gate=Path(protocol['full_cycle_gate'])
    for folder,filename,expected in [(parent,'policy.zip',protocol['parent_checkpoint_sha256']),
        (migration,'summary.json',protocol['migration_summary_sha256']),(gate,'summary.json',protocol['gate_summary_sha256'])]:
        if sha(folder/filename)!=expected:raise ValueError('Frozen evidence changed')
    for folder in (migration,gate):
        record=json.loads((folder/'summary.json').read_text())
        if not record['passed']:raise ValueError('Required gate failed')
        for source,digest in record['sources'].items():
            if sha(source)!=digest:raise ValueError('Gate source changed')
    migrated=json.loads((migration/'summary.json').read_text())
    if sha(migration/'policy_initial.zip')!=migrated['initial_checkpoint_sha256']:raise ValueError('Migrated policy changed')
    manifest=json.loads((parent/'manifest.json').read_text())
    audit=json.loads((parent/'independent_training_audit.json').read_text())
    if not audit['passed'] or sha(parent/'manifest.json')!=audit['experiment_manifest_sha256'] or sha(parent/'summary.json')!=audit['experiment_summary_sha256']:
        raise ValueError('Parent audit mismatch')
    for source,digest in manifest['sources'].items():
        if source.replace('\\','/').startswith('training/') and sha(ROOT/source)!=digest:
            raise ValueError('Parent environment changed')
    manifest.update(training_cases=protocol['training_cases'],evaluation_cases=protocol['evaluation_cases'])
    steps=protocol['steps'];n_envs=protocol['environments']
    if n_envs!=4 or steps<=0 or steps%1024:raise ValueError('Expected4 environments and complete1024-transition rollouts')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.protocol,args.output/'matched_protocol.json')
    torch.set_num_threads(1)
    env=SubprocVecEnv([partial(make_env,manifest,args.mode,str(args.output.resolve()),rank) for rank in range(n_envs)],start_method='spawn')
    model=PPO.load(migration/'policy_initial.zip',env=env,device='cpu',force_reset=True)
    initial=PPO.load(migration/'policy_initial.zip',device='cpu')
    if model.num_timesteps!=0 or model.policy.optimizer.state or state_difference(model.policy.state_dict(),initial.policy.state_dict())!=0:
        raise ValueError('Expected identical initial policy with fresh optimizer')
    model.set_random_seed(protocol['seed']);model.set_logger(configure(str(args.output),['csv']))
    sources={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and path.suffix=='.py' and '.venv' not in path.parts:
            relative=path.relative_to(ROOT);target=args.output/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[str(relative)]=sha(target)
    manifest.update(sources=sources,geometry_mode=args.mode,observation_contract=OBSERVATION_CONTRACT,
        seed=protocol['seed'],requested_steps=steps,initial_checkpoint_sha256=sha(migration/'policy_initial.zip'),
        parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
        pretrained_steps=90112,initial_training_steps=0,environments=n_envs,protocol_sha256=sha(args.protocol),
        optimizer_initialization='Empty optimizer in both arms; exact migrated actor/critic initialization',
        scope='Known-case information ablation. Privileged oracle offsets, not a vision sensor. Original strict reward and acceptance.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));model.save(args.output/'policy_before')
    zero=reuse_zero_evaluations(parent,args.output,manifest,ROOT)
    callback=VectorProgress(args.output,0);start=time.perf_counter()
    try:model.learn(total_timesteps=steps,reset_num_timesteps=True,callback=callback)
    finally:env.close()
    model.save(args.output/'policy')
    summary=dict(actual_steps=int(model.num_timesteps),additional_steps=int(model.num_timesteps),parent_steps=0,
        training_wall_seconds=time.perf_counter()-start,zero=zero,checkpoint_sha256=sha(args.output/'policy.zip'),
        case_samples=callback.case_samples,completed_training_episodes=len(callback.episodes),
        successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),evaluation_pending=True)
    if (args.output/'STOP_REQUESTED').exists():
        summary['stopped_by_request']=True
        (args.output/'stopped_summary.json').write_text(json.dumps(summary,indent=2));return
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    evaluated=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=3)
    for row in evaluated['trajectories']:
        filename=f"after_{row['case']}.npz";shutil.copy2(args.output/'parallel_after'/filename,args.output/filename)
    (args.output/'after.json').write_text(json.dumps(evaluated['trajectories'],indent=2))
    summary.update(after=evaluated['trajectories'],evaluation_pending=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(completed=True,mode=args.mode,steps=model.num_timesteps)),flush=True)


if __name__=='__main__':main()
