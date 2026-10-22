"""Matched continuation with/without terminal-closure cost on a verified reference."""
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
from training.closure_margin_reward import ClosureMarginReward
from training.balanced_case_reset import BalancedCaseReset
from tools.continue_vector_full_cycle_training import VectorProgress,state_difference,sha
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_bar_geometry_policy import evaluate_parallel


def make_env(m,weight,output,rank):
    torch.set_num_threads(1)
    base=FullCycleResidualEnv(plan=m['reference'],cases=m['training_cases'],normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    wrapped=ClosureMarginReward(BarGeometryObservation(base,mode='oracle'),weight=weight)
    return Monitor(BalancedCaseReset(wrapped),str(Path(output)/f'episodes_worker_{rank}'))


class CostProgress(VectorProgress):
    def __init__(self,output,parent_steps,n_envs):
        super().__init__(output,parent_steps);self.costs=[0.]*n_envs;self.base_returns=[0.]*n_envs
        self.total_cost=0.;self.maximum_cost=0.;self.cost_steps=0

    def _on_step(self):
        for rank,info in enumerate(self.locals['infos']):
            cost=info['closure_reward_cost'];self.costs[rank]+=cost;self.base_returns[rank]+=info['base_reward']
            self.total_cost+=cost;self.maximum_cost=max(self.maximum_cost,cost);self.cost_steps+=1
            if 'episode' in info:
                info['closure_episode_cost']=self.costs[rank];info['base_episode_return']=self.base_returns[rank]
                self.costs[rank]=0.;self.base_returns[rank]=0.
        result=super()._on_step()
        if (self.num_timesteps-self.parent_steps)%1024==0:
            (self.output/'reward_cost_status.json').write_text(json.dumps(dict(steps=self.cost_steps,total_cost=self.total_cost,maximum_step_cost=self.maximum_cost),indent=2))
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--weight',type=float,choices=[0.,.02],required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());parent=Path(protocol['parent']);gate=Path(protocol['gate'])
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
    if sha(parent/'policy_lineage_audit.json')!=protocol['policy_lineage_audit_sha256']:raise ValueError('Actor source provenance changed')
    for path,digest in m['sources'].items():
        if path.replace('\\','/').startswith('training/') and sha(path)!=digest:raise ValueError('Parent environment changed')
    m['training_cases']=protocol['training_cases']
    if m['evaluation_cases']!=protocol['evaluation_cases'] or m['reference_sha256']!=protocol['reference_sha256']:raise ValueError('Protocol task differs')
    steps=protocol['steps'];n_envs=protocol['environments']
    if n_envs!=4 or steps<=0 or steps%1024:raise ValueError('Complete four-environment PPO rollouts required')
    args.output.mkdir(exist_ok=False);shutil.copy2(args.protocol,args.output/'matched_protocol.json')
    torch.set_num_threads(1)
    env=SubprocVecEnv([partial(make_env,m,args.weight,str(args.output.resolve()),rank) for rank in range(n_envs)],start_method='spawn')
    model=PPO.load(parent/'policy.zip',env=env,device='cpu',force_reset=True);original=PPO.load(parent/'policy.zip',device='cpu')
    policy_diff=state_difference(model.policy.state_dict(),original.policy.state_dict());optimizer_diff=state_difference(model.policy.optimizer.state_dict(),original.policy.optimizer.state_dict())
    if policy_diff or optimizer_diff:raise ValueError('Parent actor/critic/optimizer restoration mismatch')
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
    m.update(sources=sources,evaluation_only=False,training_reward_contract='strict-v2-minus-terminal-closure-cost-v1',
        closure_cost_weight=args.weight,closure_cost_parameters=protocol['closure_cost_parameters'],
        seed=protocol['seed'],parent_steps=parent_steps,additional_steps_requested=steps,environments=n_envs,
        parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),initial_checkpoint_sha256=sha(parent/'policy.zip'),protocol_sha256=sha(args.protocol),
        reward_gate_sha256=sha(gate/'summary.json'),initial_policy_max_difference=policy_diff,initial_optimizer_max_difference=optimizer_diff,
        optimizer_initialization='Exact same parent policy, critic and optimizer restored in both arms; fresh physical resets.',
        scope='Known fixed-reference height cases. Oracle geometry, unchanged strict evaluation. Closure cost changes training objective and is not potential-based shaping. All five cases used in training, no held-out claim.')
    (args.output/'manifest.json').write_text(json.dumps(m,indent=2));model.save(args.output/'policy_before')
    zero=reuse_zero_evaluations(parent,args.output,m,ROOT)
    for r in s['after']:
        name=f"after_{r['case']}.npz"
        if sha(parent/name)!=r['trajectory_sha256']:raise ValueError('Before trajectory changed')
        shutil.copy2(parent/name,args.output/f"before_{r['case']}.npz")
    (args.output/'before.json').write_text(json.dumps(s['after'],indent=2))
    callback=CostProgress(args.output,parent_steps,n_envs);start=time.perf_counter()
    try:model.learn(total_timesteps=steps,reset_num_timesteps=False,callback=callback)
    finally:env.close()
    model.save(args.output/'policy')
    summary=dict(actual_steps=int(model.num_timesteps),additional_steps=int(model.num_timesteps-parent_steps),parent_steps=parent_steps,
        training_wall_seconds=time.perf_counter()-start,zero=zero,before=s['after'],checkpoint_sha256=sha(args.output/'policy.zip'),
        case_samples=callback.case_samples,completed_training_episodes=len(callback.episodes),successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),
        total_training_closure_cost=callback.total_cost,maximum_step_closure_cost=callback.maximum_cost,cost_steps=callback.cost_steps,evaluation_pending=True)
    if (args.output/'STOP_REQUESTED').exists():
        summary['stopped_by_request']=True;(args.output/'stopped_summary.json').write_text(json.dumps(summary,indent=2));return
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    after=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=3)['trajectories']
    for r in after:
        filename=f"after_{r['case']}.npz";shutil.copy2(args.output/'parallel_after'/filename,args.output/filename)
    (args.output/'after.json').write_text(json.dumps(after,indent=2));summary.update(after=after,evaluation_pending=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(completed=True,weight=args.weight,additional_steps=summary['additional_steps'])),flush=True)


if __name__=='__main__':main()
