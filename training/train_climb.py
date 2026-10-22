"""Small CPU PPO baseline with the exact deployment observation/action contract."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from training.climb_env import ClimbSupportEnv, DEFAULT_BUNDLE

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evaluate(model,bundle,out,label):
    env=ClimbSupportEnv(bundle);obs,_=env.reset(seed=20261009)
    qpos=[env.data.qpos.copy()];observations=[obs.copy()];actions=[];rewards=[];times=[0.]
    for _ in range(env.max_steps):
        action,_=model.predict(obs,deterministic=True)
        obs,reward,terminated,truncated,info=env.step(action)
        qpos.append(env.data.qpos.copy());observations.append(obs.copy());actions.append(action)
        rewards.append(reward);times.append(float(env.data.time))
        if terminated or truncated:break
    np.savez_compressed(out/(label+'.npz'),qpos=qpos,observations=observations,actions=actions,rewards=rewards,time_s=times)
    env.close()
    return dict(episode_reward=float(np.sum(rewards)),steps=len(actions),terminated=terminated,
                truncated=truncated,final=info,randomization='none; one deterministic development episode')

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/climb_ppo_smoke_v1')
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--seed',type=int,default=20261009)
    a=p.parse_args()
    if a.steps<256 or a.steps%256:raise ValueError('Use a positive multiple of the 256-step rollout')
    if a.out.exists():raise FileExistsError('Choose another output directory; preserve prior training')
    a.out.mkdir(parents=True)
    torch.set_num_threads(1)
    sources=['training/train_climb.py','training/climb_env.py','policy/climb/contract.py']
    (a.out/'frozen_source').mkdir()
    for name in sources:
        target=a.out/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    env=Monitor(ClimbSupportEnv(a.bundle),str(a.out/'episodes'))
    config=dict(n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,gamma=.99,gae_lambda=.95,
                clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
                policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]),log_std_init=-2.))
    manifest=dict(seed=a.seed,requested_steps=a.steps,device='cpu',threads=1,config=config,
        torch_version=torch.__version__,stable_baselines3=stable_baselines3.__version__,numpy=np.__version__,
        bundle_manifest_sha256=sha(a.bundle/'manifest.json'),contract_sha256=env.unwrapped.contract.fingerprint,
        sources={s:sha(ROOT/s) for s in sources},observation_normalization=False,
        task='initial_support_pose',evaluation_scope='single deterministic development initial state; not a robustness test')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=PPO('MlpPolicy',env,seed=a.seed,device='cpu',verbose=0,**config)
    model.set_logger(configure(str(a.out),['csv']))
    before=evaluate(model,a.bundle,a.out,'before')
    start=time.perf_counter()
    model.learn(total_timesteps=a.steps,progress_bar=False)
    elapsed=time.perf_counter()-start
    model.save(a.out/'policy')
    after=evaluate(model,a.bundle,a.out,'after')
    loaded=PPO.load(a.out/'policy.zip',device='cpu')
    saved=np.load(a.out/'after.npz')
    maximum_difference=0.
    for observation in saved['observations']:
        aa,_=model.predict(observation,deterministic=True)
        bb,_=loaded.predict(observation,deterministic=True)
        np.testing.assert_array_equal(aa,bb)
        maximum_difference=max(maximum_difference,float(np.max(abs(aa-bb))))
    for s,h in manifest['sources'].items():assert sha(ROOT/s)==h
    result=dict(training_completed=True,actual_steps=model.num_timesteps,wall_seconds=elapsed,
        training_steps_per_second=model.num_timesteps/elapsed,before=before,after=after,
        checkpoint_sha256=sha(a.out/'policy.zip'),checkpoint_reload_action_max_difference=maximum_difference,
        learned_climbing_claim=False,onnx_exported=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    env.close();print(json.dumps(result,indent=2))

if __name__=='__main__':main()
