"""Record zero-residual foot-transfer episodes and quantify restart differences."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from training.foot_residual_env import FootResidualEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+')
    a=p.parse_args();e=FootResidualEnv(a.suite,a.cases)
    a.out.mkdir(parents=True,exist_ok=False)
    sources={}
    for rel in ['training/foot_residual_env.py','training/rung_residual_env.py','training/climb_cycle.py',
                'training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py',
                'tools/check_foot_residual.py']:
        target=a.out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,target);sources[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest=dict(suite=str(a.suite.resolve()),cases=e.cases,sources=sources,
                  prefixes={name:hashlib.sha256((a.suite/name/'trajectory.npz').read_bytes()).hexdigest() for name in e.cases},
                  contract='76 obs / 6 right-leg residuals, max 0.015 rad, 50 Hz actor / 200 Hz control / 2000 Hz physics',
                  scope='Zero residual restart validation; not a trained policy')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[]
    for name in e.cases:
        obs,reset=e.reset(seed=20261017,options={'case':name})
        states=[e.env.data.qpos.copy()];velocities=[e.env.data.qvel.copy()];times=[float(e.env.data.time)]
        torques=[];observations=[]
        for _ in range(e.max_steps):
            observations.append(obs)
            obs,reward,term,trunc,info=e.step(np.zeros(6))
            states.append(e.env.data.qpos.copy());velocities.append(e.env.data.qvel.copy());times.append(float(e.env.data.time))
            torques.append(e.last_torque.copy())
            if term or trunc:break
        np.savez_compressed(a.out/(name+'.npz'),qpos=states,qvel=velocities,time_s=times,torques=torques,observations=observations)
        with np.load(a.suite/name/'trajectory.npz') as z:
            original=np.stack([np.interp(times,z['time_s'],z['qpos'][:,i]) for i in range(e.env.model.nq)],axis=1)
        difference=abs(np.asarray(states)-original)
        result=dict(**info,reset=reset,restart_max_qpos_difference=float(difference.max()),
                    restart_max_base_difference_m=float(np.linalg.norm(difference[:,:3],axis=1).max()),
                    restart_max_joint_difference_rad=float(difference[:,e.env.qadr].max()))
        results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)
    e.close()


if __name__=='__main__':main()
