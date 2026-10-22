"""Physical contacts intact; mask only left-hook contact localization for10/20/30ms."""
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

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def trial(item):
    from training.contact_load_transfer_residual_env import LoadTransferResidualEnv
    from training.visibility_dropout_controller import VisibilityDropoutController
    out,duration,persistence=item;name=f"{'persistence' if persistence else 'raw'}_{round(duration*1000)}ms";run=out/name;run.mkdir()
    parent=json.loads((ROOT/'outputs/biased_load_transfer_ppo_32768_v1/manifest.json').read_text());suite=Path(parent['suite']);case='nominal'
    env=LoadTransferResidualEnv(suite,[case]);obs,_=env.reset(seed=parent['seed'],options={'case':case})
    env.controller=VisibilityDropoutController(env.env,torque_weight=10000.,normal_bias_y_deg=0.,normal_uncertainty_deg=0.,dropout_duration_s=duration,persistence_enabled=persistence)
    sources={}
    for rel in set(parent['sources'])|{'training/visibility_dropout_controller.py','training/force_history_contact_controller.py','training/contact_memory_controller.py','tools/run_visibility_dropout_study.py'}:
        path=run/'source_snapshot'/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,path);sources[rel]=sha(path)
    manifest=dict(suite=str(suite),sources=sources,prefixes={case:sha(suite/case/'trajectory.npz')},constant_action=[0.]*6,dropout_duration_s=duration,persistence_enabled=persistence,contract='Zero residual; left-hook contact localization unavailable while true physics and force sensing remain intact.')
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    q=[env.env.data.qpos.copy()];v=[env.env.data.qvel.copy()];times=[float(env.env.data.time)];actions=[];observations=[];torques=[];rewards=[]
    for _ in range(env.max_steps):
        observations.append(obs.copy());action=np.zeros(6);obs,reward,term,trunc,info=env.step(action)
        actions.append(action);rewards.append(reward);torques.append(env.last_torque.copy());q.append(env.env.data.qpos.copy());v.append(env.env.data.qvel.copy());times.append(float(env.env.data.time))
        if term or trunc:break
    np.savez_compressed(run/(case+'.npz'),qpos=q,qvel=v,time_s=times,actions=actions,observations=observations,torques=torques,rewards=rewards)
    (run/'results.json').write_text(json.dumps([info],indent=2));(run/'memory_events.json').write_text(json.dumps(env.controller.memory_events,indent=2));(run/'mask_events.json').write_text(json.dumps(env.controller.mask_events,indent=2));env.close()
    proc=subprocess.run([sys.executable,str(ROOT/'tools/audit_constant_residual.py'),'--run',str(run)],capture_output=True,text=True);(run/'audit.log').write_text(proc.stdout+proc.stderr)
    valid=proc.returncode==0 and json.loads((run/'independent_audit.json').read_text())['integrity_passed']
    return dict(run=name,duration_s=duration,persistence=persistence,success=info['success'],reason=info['reason'],time_s=info['time_s'],integrity_passed=valid)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(durations_s=[0.,.01,.02,.03],persistence=[False,True],start_s=53.65,case='nominal',normal_bias_deg=0.,scope='Contact localization/dropout only. Physical contacts and ideal force feedback unchanged. Expected bounded diagnostic termination for too-long blackout; not loss-of-physical-support or safe-stop validation.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(trial,[(a.out,duration,persistence) for duration in protocol['durations_s'] for persistence in protocol['persistence']]):
            results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
