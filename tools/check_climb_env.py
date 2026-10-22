"""Gym API validation and recorded zero/noisy-action dynamics baselines."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
import numpy as np
import mujoco
import gymnasium
from gymnasium.utils.env_checker import check_env
from training.climb_env import ClimbSupportEnv, DEFAULT_BUNDLE

ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/climb_env_baseline_v1')
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError('Choose a new output directory')
    a.out.mkdir(parents=True)
    sources=['training/climb_env.py','policy/climb/contract.py','tools/check_climb_env.py']
    manifest=dict(sources={s:sha(ROOT/s) for s in sources},bundle_manifest_sha256=sha(a.bundle/'manifest.json'),
                  mujoco=mujoco.__version__,gymnasium=gymnasium.__version__,numpy=np.__version__,
                  cases=['zero','small_noise_seed20261009'],duration_s=4.,noise_std=.03,trained_actor=False)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=ClimbSupportEnv(a.bundle)
    check_env(env,skip_render_check=True)
    rows=[]
    for name in manifest['cases']:
        obs,info=env.reset(seed=20261009);rng=np.random.default_rng(20261009)
        records=[]; qpos=[env.data.qpos.copy()]; actions=[]; observations=[obs.copy()]
        wall=time.perf_counter(); total_reward=0.
        for i in range(env.max_steps):
            action=np.zeros(29,np.float32) if name=='zero' else np.clip(rng.normal(0,.03,29),-1,1).astype(np.float32)
            obs,reward,terminated,truncated,info=env.step(action)
            actions.append(action);qpos.append(env.data.qpos.copy());observations.append(obs.copy())
            records.append(info);total_reward+=reward
            if terminated or truncated: break
        elapsed=time.perf_counter()-wall
        np.savez_compressed(a.out/(name+'.npz'),qpos=qpos,actions=actions,observations=observations,
                            time_s=[0.]+[r['simulation_time_s'] for r in records])
        (a.out/(name+'_metrics.json')).write_text(json.dumps(records,indent=2,allow_nan=False))
        rows.append(dict(case=name,policy_steps=len(records),wall_seconds=elapsed,
                         policy_steps_per_second=len(records)/elapsed,total_reward=total_reward,
                         terminated=terminated,truncated=truncated,final=info))
    env.close()
    for s,digest in manifest['sources'].items(): assert sha(ROOT/s)==digest
    result=dict(api_check_passed=True,cases=rows,training_started=False,
                success_scope='pose hold only; no hang/load-transfer/climbing claim')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
