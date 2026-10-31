"""Suffix curriculum training; full-cycle composite evaluation remains mandatory."""
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
from training.bar_geometry_observation import BarGeometryObservation
from training.closure_quality_reward import ClosureQualityReward
from training.closure_exploration_policy import ClosureExplorationPolicy
from training.balanced_case_reset import BalancedCaseReset
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3 as ClosureCurriculumEnv
from tools.continue_vector_full_cycle_training import VectorProgress,state_difference,sha
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_closure_curriculum_policy import evaluate_parallel


def make_env(m,weight,output,rank):
    torch.set_num_threads(1)
    env=ClosureCurriculumEnv(m['prefix_cache'],weight=weight,record_dir=Path(output)/f'suffix_records_worker_{rank}')
    return Monitor(BalancedCaseReset(env),str(Path(output)/f'episodes_worker_{rank}'))


class BonusProgress(VectorProgress):
    def __init__(self,output,parent_steps,n_envs):
        super().__init__(output,parent_steps);self.bonuses=[0.]*n_envs;self.base_returns=[0.]*n_envs
        self.total_bonus=0.;self.maximum_bonus=0.;self.bonus_steps=0

    def _on_step(self):
        for rank,info in enumerate(self.locals['infos']):
            bonus=info['closure_quality_bonus'];self.bonuses[rank]+=bonus;self.base_returns[rank]+=info['base_reward']
            self.total_bonus+=bonus;self.maximum_bonus=max(self.maximum_bonus,bonus);self.bonus_steps+=1
            if 'episode' in info:
                info['closure_episode_bonus']=self.bonuses[rank];info['base_episode_return']=self.base_returns[rank]
                self.bonuses[rank]=0.;self.base_returns[rank]=0.
        result=super()._on_step()
        if (self.num_timesteps-self.parent_steps)%1024==0:
            (self.output/'reward_bonus_status.json').write_text(json.dumps(dict(steps=self.bonus_steps,total_bonus=self.total_bonus,maximum_step_bonus=self.maximum_bonus),indent=2))
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--weight',type=float,choices=[.1],required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());parent=Path(protocol['parent']);gate=Path(protocol['gate'])
    exploration_gate=Path(protocol['exploration_gate'])
    if sha(exploration_gate/'summary.json')!=protocol['exploration_gate_sha256']:raise ValueError('Exploration gate changed')
    if not json.loads((exploration_gate/'summary.json').read_text())['passed']:raise ValueError('Exploration gate failed')
    for file,digest in protocol['control_reuse_hashes'].items():
        if sha(Path(protocol['control_reuse'])/file)!=digest:raise ValueError('Reused control evidence changed')
    for path,digest in protocol['sources'].items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    if sha(gate/'summary.json')!=protocol['gate_summary_sha256']:raise ValueError('Gate changed')
    g=json.loads((gate/'summary.json').read_text())
    if not g['passed']:raise ValueError('Reward gate failed')
    for path,digest in g['sources'].items():
        if sha(path)!=digest:raise ValueError('Gate source changed')
    m=json.loads((parent/'manifest.json').read_text());s=json.loads((parent/'summary.json').read_text());audit=json.loads((parent/'independent_training_audit.json').read_text())
    if not audit['passed']:raise ValueError('Audited parent required')
    for filename,key in [('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256'),('policy.zip','checkpoint_sha256')]:
        if sha(parent/filename)!=audit[key]:raise ValueError('Parent evidence changed')
    if sha(parent/'policy.zip')!=protocol['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    if sha(parent/'independent_training_audit.json')!=protocol['parent_audit_sha256']:raise ValueError('Parent audit changed')
    for path,digest in m['sources'].items():
        if path.replace('\\','/').startswith('training/') and sha(path)!=digest:raise ValueError('Parent environment changed')
    m['training_cases']=protocol['training_cases']
    m['prefix_cache']=protocol['prefix_cache']
    if sha(Path(m['prefix_cache'])/'summary.json')!=protocol['prefix_cache_summary_sha256']:raise ValueError('Prefix cache gate changed')
    if m['evaluation_cases']!=protocol['evaluation_cases'] or m['reference_sha256']!=protocol['reference_sha256']:raise ValueError('Protocol task differs')
    for gate_path, gate_hash in protocol['reset_validation_gates'].items():
        if sha(gate_path) != gate_hash or not json.loads(Path(gate_path).read_text())['passed']:
            raise ValueError('Reset validation gate changed or failed')
    steps=protocol['steps'];n_envs=protocol['environments']
    if n_envs!=4 or steps<=0 or steps%1024:raise ValueError('Complete four-environment PPO rollouts required')
    args.output.mkdir(exist_ok=False);shutil.copy2(args.protocol,args.output/'matched_protocol.json')
    shutil.copy2(parent/'policy.zip',args.output/'prefix_policy.zip')
    torch.set_num_threads(1)
    env=SubprocVecEnv([partial(make_env,m,args.weight,str(args.output.resolve()),rank) for rank in range(n_envs)],start_method='spawn')
    model=PPO.load(Path(protocol['critic_initialization'])/'policy_initial.zip',env=env,device='cpu',force_reset=True,custom_objects={'policy_class':ClosureExplorationPolicy});original=PPO.load(parent/'policy.zip',device='cpu')
    policy_diff=state_difference(model.policy.state_dict(),original.policy.state_dict());optimizer_diff=None  # Critic Adam moments deliberately reset; dictionary structures differ.
    initial_path=Path(protocol['critic_initialization'])/'policy_initial.zip'
    if sha(initial_path)!=protocol['critic_initialization_sha256']:raise ValueError('Critic initialization changed')
    initial=PPO.load(initial_path,device='cpu')
    if state_difference(model.policy.state_dict(),initial.policy.state_dict()) or state_difference(model.policy.optimizer.state_dict(),initial.policy.optimizer.state_dict()):raise ValueError('Warm checkpoint restoration mismatch')
    actor_keys=[k for k in original.policy.state_dict() if not k.startswith(('mlp_extractor.value_net.','value_net.'))]
    actor_difference=state_difference({k:model.policy.state_dict()[k] for k in actor_keys},{k:original.policy.state_dict()[k] for k in actor_keys})
    if actor_difference:raise ValueError('Actor or feature initialization changed')
    parent_steps=int(model.num_timesteps);model.set_random_seed(protocol['seed']);model.set_logger(configure(str(args.output),['csv']))
    sources={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        relative=path.relative_to(ROOT)
        if relative.parts[0] not in ('training','tools'):continue
        target=args.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());sources[str(relative)]=sha(target)
    m['config']=protocol['config']
    for key in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','target_kl'):
        if getattr(model,key)!=m['config'][key]:raise ValueError('Optimizer configuration differs: '+key)
    m={key:m[key] for key in ('contract','config','observation_contract','geometry_mode','training_cases','evaluation_cases',
        'normal_bias_y_deg','force_history_memory','reference','reference_sha256','bundle_manifest_sha256')}
    m.update(sources=sources,evaluation_only=False,
        prefix_checkpoint=str((args.output/'prefix_policy.zip').resolve()),prefix_checkpoint_sha256=sha(parent/'policy.zip'),
        suffix_start_seconds=99.66,horizon_seconds=113.66,training_episode_seconds=14.,
        training_segment='closure_suffix_only',complete_cycle_training=False,
        prefix_cache=protocol['prefix_cache'],prefix_cache_summary_sha256=protocol['prefix_cache_summary_sha256'],training_sampling_contract='closure-only-leg-gaussian-std-times4-v1',
        initial_distribution_changed=True,exploration_gate_sha256=protocol['exploration_gate_sha256'],training_reward_contract='strict-v2-plus-terminal-quality-v1',
        quality_bonus_weight=args.weight,quality_bonus_parameters=protocol['quality_bonus_parameters'],
        seed=protocol['seed'],parent_steps=parent_steps,additional_steps_requested=steps,environments=n_envs,
        parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),initial_checkpoint_sha256=sha(initial_path),critic_initialization=str(initial_path.resolve()),initial_actor_max_difference=actor_difference,protocol_sha256=sha(args.protocol),
        reward_gate_sha256=sha(gate/'summary.json'),initial_policy_max_difference=policy_diff,initial_optimizer_max_difference=optimizer_diff,
        optimizer_initialization='Actor and feature parameters restored exactly from selected parent; critic heads calibrated on fixed-policy exploratory suffix returns, only critic optimizer moments cleared. Terminal leg sampling standard deviation scaled by four. Initial warm checkpoint restored exactly.',
        scope='Suffix-only training from audited cached parent prefixes on five known cases with positive quality bonus; evaluation uses a frozen prefix and learned suffix from time zero. Terminal-only leg exploration multiplier four. Oracle geometry and original controller. No held-out claim; no automatic release replacement.')
    (args.output/'manifest.json').write_text(json.dumps(m,indent=2));model.save(args.output/'policy_before')
    zero=reuse_zero_evaluations(parent,args.output,m,ROOT)
    for r in s['after']:
        name=f"after_{r['case']}.npz"
        if sha(parent/name)!=r['trajectory_sha256']:raise ValueError('Before trajectory changed')
        shutil.copy2(parent/name,args.output/f"before_{r['case']}.npz")
    (args.output/'before.json').write_text(json.dumps(s['after'],indent=2))
    callback=BonusProgress(args.output,parent_steps,n_envs);start=time.perf_counter()
    try:model.learn(total_timesteps=steps,reset_num_timesteps=False,callback=callback)
    finally:env.close()
    model.save(args.output/'policy')
    summary=dict(actual_steps=int(model.num_timesteps),additional_steps=int(model.num_timesteps-parent_steps),parent_steps=parent_steps,
        training_wall_seconds=time.perf_counter()-start,zero=zero,before=s['after'],checkpoint_sha256=sha(args.output/'policy.zip'),
        case_samples=callback.case_samples,completed_training_episodes=len(callback.episodes),successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),
        total_training_quality_bonus=callback.total_bonus,maximum_step_quality_bonus=callback.maximum_bonus,bonus_steps=callback.bonus_steps,evaluation_pending=True)
    if (args.output/'STOP_REQUESTED').exists():
        summary['stopped_by_request']=True;(args.output/'stopped_summary.json').write_text(json.dumps(summary,indent=2));return
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    after=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=3)['trajectories']
    for r in after:
        filename=f"after_{r['case']}.npz";shutil.copy2(args.output/'parallel_after'/filename,args.output/filename)
    (args.output/'after.json').write_text(json.dumps(after,indent=2));summary.update(after=after,evaluation_pending=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    from tools.audit_closure_curriculum_training import audit
    audit(args.output)
    from tools.audit_closure_composite_prefix import audit as audit_prefix
    audit_prefix(args.output)
    print(json.dumps(dict(completed=True,weight=args.weight,additional_steps=summary['additional_steps'])),flush=True)


if __name__=='__main__':main()
