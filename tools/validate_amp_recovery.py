"""Evaluate the actual AMP deployment policy against the paired recovery poses.
No DDS or hardware actions. Same CPU MuJoCo/metrics as validate_ft12k_sim.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.validate_ft12k_sim import make_model, run_case

class AmpAuditAdapter:
    """Only expose audit interfaces and synchronize the deployed gyro alias.
    Action history, mapping, gains, command filter and enter/run remain native AMP.
    """
    def __init__(self,state,output):
        from policy.amp.Amp import Amp
        self.native=Amp(state,output);self.state=state
        self.kps=self.native.kps;self.kds=self.native.kds
        self.tau_limit=np.array(yaml.safe_load((ROOT/'deploy_mujoco/config/mujoco.yaml').read_text())['tau_limit'])
    def enter(self):
        self.state.ang_vel=self.state.root_ang_vel_b.copy()
        self.native.enter()
    def run(self):
        self.state.ang_vel=self.state.root_ang_vel_b.copy()
        self.native.run()
    def _build_obs(self):
        return np.clip(self.native._obs_buf,-100,100).copy()

def worker(args):
    from unittest.mock import patch
    import onnxruntime as ort
    from common.ctrlcomp import StateAndCmd,PolicyOutput
    cases,out=args;out=Path(out)
    state=StateAndCmd(29);output=PolicyOutput(29)
    state.vel_cmd[:]=0
    original=ort.SessionOptions
    def single_thread():
        options=original();options.intra_op_num_threads=1;options.inter_op_num_threads=1;return options
    with patch.object(ort,'SessionOptions',single_thread):policy=AmpAuditAdapter(state,output)
    m=make_model();results=[]
    for case in cases:
        r,trace,obs=run_case(case,m,policy,state,output,seconds=20,record=True)
        np.savez_compressed(out/(case['name']+'.npz'),state=trace,obs=obs,nq=m.nq,nv=m.nv)
        z=trace[:,m.nq+m.nv];u=trace[:,m.nq+m.nv+1];hold=trace[:,m.nq+m.nv+2]
        def persistent(mask):return bool(np.any(np.convolve(mask.astype(int),np.ones(10,dtype=int),'valid')>=10))
        fall=(z<.65)|(u<.5)
        r['refall_after_first_upright']=persistent(fall&np.maximum.accumulate((z>=1.15)&(u>=.93)))
        r['refall_after_stable_1s']=persistent(fall&np.maximum.accumulate(hold>=1-1e-6))
        r['last10s_base_xy_path_m']=float(np.linalg.norm(np.diff(trace[-500:,:2],axis=0),axis=1).sum())
        (out/(case['name']+'.json')).write_text(json.dumps(r,indent=2)+'\n');results.append(r)
        print(case['name'],r['success_10s'],flush=True)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    cases=json.loads(a.cases.read_text());(a.out/'cases.json').write_text(json.dumps(cases))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        results=sum(list(pool.map(worker,[(cases[i::a.workers],str(a.out)) for i in range(a.workers)])),[])
    import mujoco
    cfgpath=ROOT/'policy/amp/config/amp.yaml';cfg=yaml.safe_load(cfgpath.read_text());model=ROOT/'policy/amp/model'/cfg['model_path']
    summary={'profile':'amp','model':cfg['model_path'],'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'cases_sha256':hashlib.sha256(a.cases.read_bytes()).hexdigest(),'mujoco':mujoco.__version__,'n':len(results),'success':sum(r['success_10s'] for r in results),'seconds':20,'physics_dt':.002,'control_dt':.02,'command':[0,0,0],'per_direction':{k:{'n':sum(r['direction']==k for r in results),'success':sum(r['success_10s'] for r in results if r['direction']==k)} for k in sorted(set(r['direction'] for r in results))},'refall_after_first_upright':sum(r['refall_after_first_upright'] for r in results),'refall_after_stable_1s':sum(r['refall_after_stable_1s'] for r in results),'numerical_instability':sum(r['unstable'] is not None for r in results)}
    for field,label in [('peak_tau','tau'),('peak_dq','speed'),('peak_power','power')]:
        values=np.array([r[field] for r in results]);summary[f'peak_{label}_p95']=float(np.percentile(values.max(axis=1),95));summary[f'per_joint_{label}_p95']=dict(zip(results[0]['joint_names'],np.percentile(values,95,axis=0).tolist()))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
