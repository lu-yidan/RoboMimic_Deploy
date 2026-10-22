"""Predeclared fixed contact-estimation biases on three load-transfer cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run_trial(item):
    import numpy as np
    from training.load_transfer_residual_env import LoadTransferResidualEnv
    from training.robust_tracking_controller import RobustTrackingController
    out,suite,case,label,point,angle=item
    trial=out/(label+'__'+case);trial.mkdir()
    shutil.copytree(out/'source_snapshot',trial/'source_snapshot')
    manifest=json.loads((out/'manifest.json').read_text());manifest['cases']=[case]
    manifest['prefixes']={case:manifest['prefixes'][case]}
    manifest.update(point_bias_x_m=point,normal_bias_y_deg=angle)
    (trial/'manifest.json').write_text(json.dumps(manifest,indent=2))
    task=LoadTransferResidualEnv(suite,[case]);obs,reset=task.reset(seed=20261017,options={'case':case})
    task.controller=RobustTrackingController(task.env,torque_weight=10000.,point_bias_x_m=point,normal_bias_y_deg=angle)
    q=[task.env.data.qpos.copy()];v=[task.env.data.qvel.copy()];times=[float(task.env.data.time)]
    torques=[];observations=[];actions=[]
    for _ in range(task.max_steps):
        observations.append(obs.copy());action=np.zeros(6)
        obs,reward,term,trunc,info=task.step(action)
        actions.append(action);q.append(task.env.data.qpos.copy());v.append(task.env.data.qvel.copy());times.append(float(task.env.data.time));torques.append(task.last_torque.copy())
        if term or trunc:break
    np.savez_compressed(trial/(case+'.npz'),qpos=q,qvel=v,time_s=times,torques=torques,observations=observations,actions=actions)
    result=dict(**info,relaxation_steps=task.controller.relaxation_steps,minimum_tracking_scale=task.controller.minimum_tracking_scale,trial=trial.name,point_bias_x_m=point,normal_bias_y_deg=angle,reset=reset)
    (trial/'results.json').write_text(json.dumps([result],indent=2));task.close()
    audit=subprocess.run([sys.executable,str(ROOT/'tools/audit_load_transfer.py'),'--run',str(trial)],capture_output=True,text=True)
    (trial/'audit_console.log').write_text(audit.stdout+audit.stderr)
    result['integrity_passed']=audit.returncode==0 and json.loads((trial/'independent_audit.json').read_text())['integrity_passed']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    a.out=a.out.resolve();a.suite=a.suite.resolve();a.out.mkdir(parents=True,exist_ok=False)
    cases=['nominal','row01_z_+5mm','row01_x_-10mm']
    biases=[('zero',0.,0.),('normal_minus3deg',0.,-3.),('normal_plus3deg',0.,3.)]
    sources={}
    for rel in ['training/robust_contact_frame_controller.py','training/robust_tracking_controller.py','training/tracking_relaxation.py','training/load_transfer_residual_env.py',
                'training/foot_residual_env.py','training/rung_residual_env.py','training/climb_cycle.py',
                'training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py',
                'tools/run_robust_tracking_suite.py','tools/audit_load_transfer.py']:
        dest=a.out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        sources[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()
    manifest=dict(suite=str(a.suite),cases=cases,biases=biases,sources=sources,
        prefixes={name:hashlib.sha256((a.suite/name/'trajectory.npz').read_bytes()).hexdigest() for name in cases},
        normal_uncertainty_deg=3.,contract='49..61.56 s load transfer; zero residual; conservative cone for 3 degree normal uncertainty plus 1/.75/.5/.25/0 tracking-demand reduction',
        scope='Exact contact existence/identity and ideal force feedback retained. Bias constant across contacts and episode. No white noise, latency, topology error, full-cycle or real-sensor validation.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    tasks=[(a.out,a.suite,case,label,point,angle) for label,point,angle in biases for case in cases]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_trial,tasks):
            results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:result[k] for k in ['trial','success','integrity_passed','reason','time_s']}),flush=True)
    print(f'Complete: {len(results)} trials; {sum(r["success"] for r in results)} successful',flush=True)


if __name__=='__main__':main()
