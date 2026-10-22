"""Replay frozen measured states to screen torque objectives, not physical success."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--weights',type=float,nargs='+',default=[0,100,1000,10000]);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    manifest=json.loads((a.run/'manifest.json').read_text())
    e=WallSupportEnv(manifest['bundle']);e.reset();z=np.load(a.run/'trajectory.npz');plan=np.load(a.run/'reference.npz')
    rows=[];summary=[]
    for weight in a.weights:
        c=CycleTorqueController(e,torque_weight=weight);ratios=[];failed=0
        for i in range(0,len(z['time_s'])-1,25):
            t=z['time_s'][i];j=min(max(int(np.searchsorted(plan['time'],t,side='right')-1),0),len(plan['time'])-2)
            dt=plan['time'][j+1]-plan['time'][j];u=np.clip((t-plan['time'][j])/dt,0,1)
            q=(1-u)*plan['qpos'][j]+u*plan['qpos'][j+1];v=np.zeros(e.model.nv)
            v[:3]=(plan['qpos'][j+1,:3]-plan['qpos'][j,:3])/dt
            v[e.vadr]=(plan['qpos'][j+1,e.qadr]-plan['qpos'][j,e.qadr])/dt
            e.data.qpos[:]=z['qpos'][i];e.data.qvel[:]=z['qvel'][i]
            try:
                tau,info=c.command(q,v,plan['active'][j+1]);ratio=abs(tau)/e.limits[:,1];ratios.append(ratio)
                rows.append(dict(weight=weight,time_s=float(t),phase=str(plan['phase'][j+1]),peak=float(max(ratio))))
            except RuntimeError as exc:
                failed+=1;rows.append(dict(weight=weight,time_s=float(t),error=str(exc)))
        r=np.asarray(ratios)
        result=dict(weight=weight,failed=failed,samples=len(r),peak=float(r.max()),mean_peak=float(r.max(axis=1).mean()),near_limit_fraction=float(np.mean(r.max(axis=1)>.99)),solver_retries=c.solver_retries)
        summary.append(result);print(result,flush=True)
    (a.out/'screen.json').write_text(json.dumps(dict(scope='Frozen-state allocation only, not a dynamics rollout',summary=summary,rows=rows),indent=2));e.close()


if __name__=='__main__':main()
