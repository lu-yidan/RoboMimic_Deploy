"""Frozen pure-memory comparison on five geometries excluded from PPO training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
CASES=['nominal','row01_z_+5mm','row01_x_-10mm']
ANGLES=[0.,-3.,3.,-1.5,1.5]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def trial(item):
    import torch
    from stable_baselines3 import PPO
    from training.contact_load_transfer_residual_env import LoadTransferResidualEnv
    from training.force_history_contact_controller import ForceHistoryContactController
    out,method,angle,parent=item
    torch.set_num_threads(1);old=json.loads((parent/'manifest.json').read_text())
    run=out/f'{method}__normal_{angle:+g}deg';run.mkdir()
    suite=Path(old['suite']);env=LoadTransferResidualEnv(suite,CASES)
    policy=PPO.load(parent/'policy.zip',device='cpu') if method=='ppo' else None
    action=np.array([-.25,0.,0.,0.,0.,0.]) if method=='constant' else np.zeros(6)
    sources={}
    for rel in set(old['sources'])|{'tools/evaluate_known_force_history.py','training/contact_memory_controller.py','training/force_history_contact_controller.py'}:
        p=run/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p);sources[rel]=sha(p)
    manifest=dict(suite=str(suite),sources=sources,prefixes={case:sha(suite/case/'trajectory.npz') for case in CASES},method=method,constant_action=action.tolist(),normal_bias_y_deg=angle,contract='Frozen76obs/6act local residual; no newtraining',policy_source=str(parent/'policy.zip'),policy_sha256=sha(parent/'policy.zip'))
    if policy is not None:shutil.copy2(parent/'policy.zip',run/'policy.zip')
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
    for case in CASES:
        obs,_=env.reset(seed=old['seed'],options={'case':case})
        env.controller=ForceHistoryContactController(env.env,torque_weight=10000.,normal_bias_y_deg=angle,normal_uncertainty_deg=0.)
        qs=[env.env.data.qpos.copy()];vs=[env.env.data.qvel.copy()];times=[float(env.env.data.time)]
        actions=[];observations=[];torques=[];rewards=[]
        for _ in range(env.max_steps):
            observations.append(obs.copy());used=policy.predict(obs,deterministic=True)[0] if policy is not None else action
            obs,reward,term,trunc,info=env.step(used)
            actions.append(used.copy());rewards.append(reward);torques.append(env.last_torque.copy())
            qs.append(env.env.data.qpos.copy());vs.append(env.env.data.qvel.copy());times.append(float(env.env.data.time))
            if term or trunc:break
        np.savez_compressed(run/(case+'.npz'),qpos=qs,qvel=vs,time_s=times,actions=actions,observations=observations,torques=torques,rewards=rewards)
        (run/(case+'_memory_events.json')).write_text(json.dumps(env.controller.memory_events,indent=2))
        rows.append(dict(**info,normal_bias_y_deg=angle,method=method,memory_uses=len(env.controller.memory_events)))
        (run/'results.json').write_text(json.dumps(rows,indent=2))
    env.close()
    tool='audit_untrained_policy.py' if method=='ppo' else 'audit_constant_residual.py'
    audit=subprocess.run([sys.executable,str(ROOT/'tools'/tool),'--run',str(run)],capture_output=True,text=True)
    (run/'audit.log').write_text(audit.stdout+audit.stderr)
    valid=audit.returncode==0 and json.loads((run/'independent_audit.json').read_text())['integrity_passed']
    return dict(run=run.name,method=method,angle=angle,successes=sum(r['success'] for r in rows),count=len(rows),integrity_passed=valid)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    parent=ROOT/'outputs/biased_load_transfer_ppo_32768_v1';old=json.loads((parent/'manifest.json').read_text())
    assert CASES==old['cases']
    assert json.loads((parent/'continuation_audit.json').read_text())['passed']
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(cases=CASES,angles=ANGLES,methods=['persistence'],policy_sha256=sha(parent/'policy.zip'),selection_sha256=sha(ROOT/'outputs/constant_residual_selection_v1.json'),scope='Same three PPOtraining geometries and five evaluation angles; no unseen-geometry claim. Force-confirmed20ms persistence; no current gap query, no conetightening. Exact contact points/topology and idealforce remain like baseline. Fifteen new evaluations, compare against earlier45. All geometry failures retained; no statistical success-rate claim.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(trial,[(a.out,m,angle,parent) for angle in ANGLES for m in protocol['methods']]):
            results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
