"""Static posture/force feasibility scan around a failure state; no recovery claim."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.clearance_cycle import LimbIK
from training.climb_robustness import apply_rung_offsets


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--torque-fraction',type=float,default=.9)
    a=p.parse_args()
    if not .5<=a.torque_fraction<=1:raise ValueError('Conservative torque fraction must be 0.5..1')
    a.out.mkdir(parents=True,exist_ok=False)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    apply_rung_offsets(e,json.loads((a.run/'scene_variant.json').read_text())['offsets'])
    f=np.load(a.run/'failure_state.npz');q0=f['qpos'].copy();active=f['active'].astype(bool)
    if active.tolist()!=[False,True,True,True]:raise ValueError('Scan expects right foot and two hooks')
    e.data.qpos[:]=q0;e.data.qvel[:]=0;mujoco.mj_forward(e.model,e.data)
    ik=LimbIK(e,leg_clearance=.014)
    hands={k:e.data.xpos[ik.bodies[k]].copy() for k in ['lh','rh']}
    orientations={k:e.data.xquat[ik.bodies[k]].copy() for k in hands}
    bid=ik.bodies['rf'];R=e.data.xmat[bid].reshape(3,3)
    foot=e.data.xpos[bid]+R@np.array([.09,0,-.035])-.0148*R[:,2]
    yaw=float(np.arctan2(R[1,0],R[0,0]));native=e.limits.copy()
    # Restrict optimizer bounds only; the model's native actuator limits stay intact.
    e.limits=native*a.torque_fraction
    sources={}
    for rel in ['tools/scan_support_posture.py','training/climb_cycle.py','training/clearance_cycle.py',
                'training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py']:
        dest=a.out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        sources[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(dict(source=str(a.run.resolve()),sources=sources,
        failure_state_sha256=hashlib.sha256((a.run/'failure_state.npz').read_bytes()).hexdigest(),
        torque_fraction=a.torque_fraction,grid_m=[-.01,0,.01],
        scope='27 static root translations. Measured support poses fixed; qvel set to zero for gravity analysis. No dynamic path or recovery validation.'),indent=2))
    results=[];states=[]
    for delta in itertools.product([-.01,0,.01],repeat=3):
        q=q0.copy();q[:3]+=delta
        q,fe=ik.contact_leg(q,'rf',foot,yaw)
        q,he=ik.pose(q,hands,regularization=.001,orientations=orientations)
        error=max(fe,*he.values());e.data.qpos[:]=q;e.data.qvel[:]=0;mujoco.mj_forward(e.model,e.data)
        penetration=0.
        for c in e.data.contact:
            ba,bb=int(e.model.geom_bodyid[c.geom1]),int(e.model.geom_bodyid[c.geom2])
            ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
            if not (ia or ib):continue
            body=ba if ia else bb
            if ia==ib or body not in [ik.bodies[k] for k in ('rf','lh','rh')]:penetration=max(penetration,float(-c.dist))
        record=dict(delta_m=list(delta),ik_error_m=error,unexpected_penetration_m=penetration,
                    max_joint_change_rad=float(abs(q[e.qadr]-q0[e.qadr]).max()),feasible=False)
        if error<.002 and penetration<1e-5:
            controller=CycleTorqueController(e,torque_weight=10000.)
            try:torque,info=controller.command(q,np.zeros(e.model.nv),active)
            except RuntimeError as exc:record['reason']=str(exc)
            else:
                record.update(feasible=True,native_torque_peak_ratio=float(np.max(abs(torque)/np.max(abs(native),axis=1))),
                    torque=torque.tolist(),force_plan=info['force_plan'])
        else:record['reason']='IK or collision rejection'
        results.append(record);states.append(q)
    np.savez_compressed(a.out/'candidates.npz',qpos=states)
    (a.out/'results.json').write_text(json.dumps(results,indent=2))
    feasible=[r for r in results if r['feasible']]
    print(json.dumps(dict(feasible_count=len(feasible),total=len(results),
        best=min(feasible,key=lambda r:r['native_torque_peak_ratio']) if feasible else None),indent=2));e.close()


if __name__=='__main__':main()
