"""Compare planned force with interval-mean physical force during load transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.climb_robustness import GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    task=LoadTransferResidualEnv(a.suite,[a.case]);task.reset(options={'case':a.case})
    a.out.mkdir(parents=True,exist_ok=False);e=task.env;m=e.model;d=e.data;c=task.controller
    records=[];reason='complete';states=[]
    while d.time<61.56-1e-8:
        t=float(d.time);q,v,active=task.reference(t)
        try:torque,info=c.command(q,v,active)
        except RuntimeError as exc:reason=str(exc);break
        planned=np.zeros((4,3));planned[np.asarray(active,dtype=bool)]=np.asarray(info['force_plan']).reshape(-1,2,3).sum(axis=1)
        desired=np.zeros(m.nv)
        desired[:3]=80*(q[:3]-d.qpos[:3])+18*(v[:3]-d.qvel[:3])
        rotation=np.zeros(3);mujoco.mju_quat2Vel(rotation,d.qpos[3:7],1.)
        desired[3:6]=-80*rotation-18*d.qvel[3:6]
        desired[e.vadr]=100*(q[e.qadr]-d.qpos[e.qadr])+20*(v[e.vadr]-d.qvel[e.vadr])
        actual=[];accelerations=[];d.ctrl[e.motors]=torque
        for _ in range(10):
            assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
            forces=np.zeros((4,3))
            for contact in e.contact_metrics()['contacts']:
                if contact['group'] in GROUPS:forces[GROUPS.index(contact['group'])]+=contact['force_on_robot_world_n']
            actual.append(forces);accelerations.append(d.qacc.copy())
        mean=np.mean(actual,axis=0);acc=np.mean(accelerations,axis=0)
        records.append(dict(time_s=t,end_s=float(d.time),active=active.tolist(),planned_force_n=planned.tolist(),
            actual_interval_mean_force_n=mean.tolist(),force_error_n=(mean-planned).tolist(),
            desired_acceleration=desired.tolist(),actual_interval_mean_acceleration=acc.tolist(),
            torque_ratio=info['torque_ratio']))
        states.append(d.qpos.copy())
    force=np.array([r['force_error_n'] for r in records]);acc=np.array([np.array(r['actual_interval_mean_acceleration'])-r['desired_acceleration'] for r in records])
    result=dict(case=a.case,reason=reason,time_s=float(d.time),intervals=len(records),
        group_force_rmse_n={g:float(np.sqrt(np.mean(force[:,i]**2))) for i,g in enumerate(GROUPS)},
        base_acceleration_rmse_m_s2=float(np.sqrt(np.mean(acc[:,:3]**2))),
        joint_acceleration_rmse_rad_s2=float(np.sqrt(np.mean(acc[:,e.vadr]**2))),
        scope='Restart at recorded 49 s qpos/qvel. Compare next-command planned force with following 5 ms mean physical force; no force-feedback correction.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'records.json').write_text(json.dumps(records));(a.out/'summary.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(a.out/'states.npz',qpos=states)
    print(json.dumps(result,indent=2));task.close()


if __name__=='__main__':main()
