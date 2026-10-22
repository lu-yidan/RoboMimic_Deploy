"""Finite-reset PPO smoke experiment for one right-foot transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from training.foot_residual_env import FootResidualEnv


def evaluate(model,suite,out,label,cases):
    e=FootResidualEnv(suite,cases);results=[]
    for name in e.cases:
        obs,_=e.reset(seed=20261017,options={'case':name})
        states=[e.env.data.qpos.copy()];velocities=[e.env.data.qvel.copy()];times=[float(e.env.data.time)]
        actions=[];observations=[];torques=[];rewards=[]
        for _ in range(e.max_steps):
            observations.append(obs.copy())
            action=np.zeros(6) if model is None else model.predict(obs,deterministic=True)[0]
            obs,reward,term,trunc,info=e.step(action)
            actions.append(action);rewards.append(reward);torques.append(e.last_torque.copy())
            states.append(e.env.data.qpos.copy());velocities.append(e.env.data.qvel.copy());times.append(float(e.env.data.time))
            if term or trunc:break
        np.savez_compressed(out/f'{label}_{name}.npz',qpos=states,qvel=velocities,time_s=times,
                            actions=actions,observations=observations,torques=torques,rewards=rewards)
        results.append(dict(**info,return_sum=float(sum(rewards))))
        (out/(label+'.json')).write_text(json.dumps(results,indent=2))
        print(json.dumps(dict(evaluation=label,**results[-1])),flush=True)
    e.close();return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--seed',type=int,default=20261017);p.add_argument('--cases',nargs='+')
    a=p.parse_args()
    if a.steps<256 or a.steps%256:raise ValueError('steps must be a positive multiple of 256')
    base=FootResidualEnv(a.suite,a.cases);cases=base.cases
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    sources={}
    for rel in ['training/foot_residual_env.py','training/rung_residual_env.py','training/climb_cycle.py',
                'training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py',
                'tools/train_foot_residual.py']:
        target=a.out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,target);sources[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
    config=dict(n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,gamma=.995,gae_lambda=.95,
                policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-2.))
    manifest=dict(seed=a.seed,steps=a.steps,config=config,sources=sources,cases=cases,
                  suite=str(a.suite.resolve()),torch=torch.__version__,sb3=stable_baselines3.__version__,
                  prefixes={name:hashlib.sha256((a.suite/name/'trajectory.npz').read_bytes()).hexdigest() for name in cases},
                  contract='76 observations / 6 right-leg residuals capped at 0.015 rad, filtered at 200 Hz',
                  randomization='Uniform sampling of declared finite prefix cases, no continuous randomization',
                  actor='Proprioception, nominal tracking error, phase, ideal contact force, previous action; no true rung coordinates',
                  scope='One right-foot transfer. Evaluation reuses finite reset cases: pipeline test only, not held-out generalization.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=Monitor(base,str(a.out/'episodes'))
    model=PPO('MlpPolicy',env,seed=a.seed,device='cpu',verbose=0,**config)
    model.set_logger(configure(str(a.out),['csv']))
    zero=evaluate(None,a.suite,a.out,'zero',cases)
    before=evaluate(model,a.suite,a.out,'before',cases)
    start=time.perf_counter();model.learn(total_timesteps=a.steps);elapsed=time.perf_counter()-start
    model.save(a.out/'policy');after=evaluate(model,a.suite,a.out,'after',cases)
    loaded=PPO.load(a.out/'policy.zip',device='cpu');difference=0.
    for name in cases:
        with np.load(a.out/f'after_{name}.npz') as z:
            obs=z['observations'];x=model.predict(obs,deterministic=True)[0];y=loaded.predict(obs,deterministic=True)[0]
            difference=max(difference,float(abs(x-y).max()))
    if difference!=0:raise AssertionError('Checkpoint reload changed actions')
    result=dict(training_completed=True,actual_steps=model.num_timesteps,training_wall_s=elapsed,
                zero=zero,before=before,after=after,checkpoint_reload_max_difference=difference,
                checkpoint_sha256=hashlib.sha256((a.out/'policy.zip').read_bytes()).hexdigest(),
                learned_full_cycle=False,held_out_generalization_validated=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));env.close()
    print(json.dumps({k:result[k] for k in ['training_completed','actual_steps','training_wall_s']}),flush=True)


if __name__=='__main__':main()
