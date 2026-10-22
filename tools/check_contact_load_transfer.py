"""Zero-residual baseline for the harder load-transfer training task."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from training.contact_load_transfer_residual_env import LoadTransferResidualEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['nominal','row01_z_+5mm','row01_x_-10mm'])
    a=p.parse_args();e=LoadTransferResidualEnv(a.suite,a.cases)
    a.out.mkdir(parents=True,exist_ok=False)
    sources={}
    for rel in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/contact_frame_controller.py','training/rung_residual_env.py',
                'training/climb_cycle.py','training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py',
                'tools/check_contact_load_transfer.py']:
        dest=a.out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        sources[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(dict(suite=str(a.suite.resolve()),cases=a.cases,sources=sources,
        contract='Load transfer 49..61.56 s; 76 obs / 6 right-leg residuals; zero residual baseline; exact contact geometry controller; per-physics-step native joint bounds',
        prefixes={name:hashlib.sha256((a.suite/name/'trajectory.npz').read_bytes()).hexdigest() for name in a.cases}),indent=2))
    results=[]
    for name in a.cases:
        obs,reset=e.reset(seed=20261017,options={'case':name})
        qs=[e.env.data.qpos.copy()];vs=[e.env.data.qvel.copy()];times=[float(e.env.data.time)]
        torques=[];observations=[];rewards=[]
        for _ in range(e.max_steps):
            observations.append(obs.copy());obs,reward,term,trunc,info=e.step(np.zeros(6))
            qs.append(e.env.data.qpos.copy());vs.append(e.env.data.qvel.copy());times.append(float(e.env.data.time))
            torques.append(e.last_torque.copy());rewards.append(reward)
            if term or trunc:break
        np.savez_compressed(a.out/(name+'.npz'),qpos=qs,qvel=vs,time_s=times,torques=torques,observations=observations)
        results.append(dict(**info,reset=reset,return_sum=float(sum(rewards))))
        (a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True)
    e.close()


if __name__=='__main__':main()
