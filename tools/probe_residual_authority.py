"""Finite constant-action probe, not training or held-out policy validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run_trial(item):
    import numpy as np
    from training.biased_load_transfer_residual_env import BiasedLoadTransferResidualEnv
    out,suite,name,action=item;trial=out/name;trial.mkdir()
    manifest=json.loads((out/'manifest.json').read_text());manifest['constant_action']=action
    shutil.copytree(out/'source_snapshot',trial/'source_snapshot')
    (trial/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=BiasedLoadTransferResidualEnv(suite);case='row01_z_+5mm'
    obs,_=env.reset(seed=20261017,options={'case':case,'normal_bias_y_deg':-3.})
    q=[env.env.data.qpos.copy()];v=[env.env.data.qvel.copy()];times=[float(env.env.data.time)]
    observations=[];actions=[];torques=[];rewards=[]
    for _ in range(env.max_steps):
        observations.append(obs.copy());obs,reward,term,trunc,info=env.step(action)
        actions.append(action);torques.append(env.last_torque.copy());rewards.append(reward)
        q.append(env.env.data.qpos.copy());v.append(env.env.data.qvel.copy());times.append(float(env.env.data.time))
        if term or trunc:break
    np.savez_compressed(trial/(case+'.npz'),qpos=q,qvel=v,time_s=times,actions=actions,torques=torques,observations=observations,rewards=rewards)
    result=dict(**info,trial=name,constant_action=action,return_sum=float(sum(rewards)))
    (trial/'results.json').write_text(json.dumps([result],indent=2));env.close()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.out=a.out.resolve();a.suite=a.suite.resolve();a.out.mkdir(parents=True,exist_ok=False)
    actions=[('zero',[0.]*6)]
    for joint in range(6):
        for amplitude in [-.5,-.25,.25,.5]:
            action=[0.]*6;action[joint]=amplitude;actions.append((f'joint{joint}_{amplitude:+g}',action))
    source_paths=['training/biased_load_transfer_residual_env.py','training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/contact_frame_controller.py','training/biased_contact_frame_controller.py','training/rung_residual_env.py','training/climb_cycle.py','training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py','tools/probe_residual_authority.py']
    sources={}
    for rel in source_paths:
        target=a.out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target);sources[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest=dict(suite=str(a.suite),cases=['row01_z_+5mm'],sources=sources,prefixes={'row01_z_+5mm':hashlib.sha256((a.suite/'row01_z_+5mm/trajectory.npz').read_bytes()).hexdigest()},normal_bias_y_deg=-3.,actions=actions,contract='76obs/6right-leg bounded constant residual actions; original lowpass and0.015radscale; no learned policy',scope='25 predeclared single-axis actions includingzero on one known failure. Feasibility/authority diagnostic only; not generalization or success probability.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_trial,[(a.out,a.suite,name,action) for name,action in actions]):
            results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:result[k] for k in ['trial','success','reason','time_s']}),flush=True)


if __name__=='__main__':main()
