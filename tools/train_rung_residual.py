"""CPU PPO smoke training with zero-policy and held-out grid comparisons."""
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
from training.rung_residual_env import RungResidualEnv


def evaluate(model,out,label):
    e=RungResidualEnv();results=[]
    for k,offset in enumerate([[0,0,0],[-.005,0,0],[.005,0,0],[0,0,-.005],[0,0,.005]]):
        obs,_=e.reset(seed=1000+k,options={'offset':offset})
        states=[e.env.data.qpos.copy()];velocities=[e.env.data.qvel.copy()];times=[0.];actions=[];rewards=[];torques=[];observations=[]
        for _ in range(e.max_steps):
            observations.append(obs.copy())
            act=np.zeros(7) if model is None else model.predict(obs,deterministic=True)[0]
            obs,r,term,trunc,info=e.step(act)
            states.append(e.env.data.qpos.copy());velocities.append(e.env.data.qvel.copy());times.append(float(e.env.data.time))
            actions.append(act);rewards.append(r);torques.append(e.last_torque.copy())
            if term or trunc:break
        np.savez_compressed(out/f'{label}_{k}.npz',qpos=states,qvel=velocities,time_s=times,actions=actions,
                            observations=observations,torques=torques,rewards=rewards)
        results.append(dict(offset_m=offset,reward=float(sum(rewards)),**info))
        print(json.dumps(dict(evaluation=label,**results[-1])),flush=True)
    e.close();return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2048);p.add_argument('--seed',type=int,default=20261010)
    a=p.parse_args()
    if a.steps<256 or a.steps%256:raise ValueError('steps must be a positive multiple of 256')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    sources=['training/rung_residual_env.py','training/climb_robustness.py','training/climb_cycle.py',
             'training/wall_support_env.py','training/climb_env.py','tools/train_rung_residual.py']
    hashes={}
    for rel in sources:
        dest=a.out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        hashes[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()
    config=dict(n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,gamma=.995,gae_lambda=.95,
                policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-2.))
    manifest=dict(seed=a.seed,steps=a.steps,config=config,sources=hashes,torch=torch.__version__,sb3=stable_baselines3.__version__,
                  contract='77 observations / 7 bounded right-arm joint residuals, maximum 0.01 rad, filtered at 200 Hz',
                  randomization='rung_05 X/Z independently uniform +/-5 mm, unseen by actor',
                  reference_sha256=hashlib.sha256((ROOT/'outputs/compact_cycle_plan_v13/plan.npz').read_bytes()).hexdigest(),
                  bundle_manifest_sha256=hashlib.sha256((ROOT/'outputs/wall_support_bundle_pd15_v2/manifest.json').read_bytes()).hexdigest(),
                  evaluation='Same 5 deterministic grid cases before/after and zero residual; not a statistical claim',
                  deployment='New research contract, incompatible with old 100D/29D actor. No real hardware deployment.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=Monitor(RungResidualEnv(),str(a.out/'episodes'))
    model=PPO('MlpPolicy',env,seed=a.seed,device='cpu',verbose=0,**config)
    model.set_logger(configure(str(a.out),['csv']))
    zero=evaluate(None,a.out,'zero');before=evaluate(model,a.out,'before')
    start=time.perf_counter();model.learn(total_timesteps=a.steps);elapsed=time.perf_counter()-start
    model.save(a.out/'policy');after=evaluate(model,a.out,'after')
    loaded=PPO.load(a.out/'policy.zip',device='cpu');difference=0.
    for k in range(5):
        observations=np.load(a.out/f'after_{k}.npz')['observations']
        original=model.predict(observations,deterministic=True)[0];restored=loaded.predict(observations,deterministic=True)[0]
        difference=max(difference,float(np.max(abs(original-restored))))
    if difference!=0:raise AssertionError('Reload changed policy actions')
    result=dict(training_completed=True,actual_steps=model.num_timesteps,training_wall_s=elapsed,
                zero=zero,before=before,after=after,checkpoint_reload_max_difference=difference,
                checkpoint_sha256=hashlib.sha256((a.out/'policy.zip').read_bytes()).hexdigest(),
                learned_climbing_claim=False,full_cycle_policy_validated=False,
                note='Training pipeline smoke run. Improvement requires held-out evaluation and more seeds.')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));env.close()
    print(json.dumps({k:result[k] for k in ('training_completed','actual_steps','training_wall_s','checkpoint_reload_max_difference')}),flush=True)


if __name__=='__main__':main()
