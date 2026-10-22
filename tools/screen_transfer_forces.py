"""Offline static feasibility near the compact left-foot transfer failure."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import LimbIK,CycleTorqueController,rung

ROOT=Path(__file__).resolve().parents[1]
e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset();ik=LimbIK(e)
p=np.load(ROOT/'outputs/compact_cycle_plan_v6/plan.npz');rows=[];states=[]
for z in [1.04,1.10,1.16,1.23]:
    idx=np.flatnonzero(p['phase']=='lf_unload');seed=p['qpos'][idx[np.argmin(abs(p['qpos'][idx,2]-z))]].copy()
    u=(z-.99)/.24;point=rung(.3)+[0,.2,.2*u]
    for x in [-.10,-.06,-.02,.02,.06]:
      for y in [-.04,-.08,-.12]:
       for w in [.3,.4,.5]:
        q=seed.copy();q[:3]=[x,y,z];q[e.qadr[14]]=w
        q,le=ik.contact_leg(q,'lf',point,0.)
        q,re=ik.contact_leg(q,'rf',rung(.55)+[0,-.2,0],0.)
        q,he=ik.pose(q,{k:rung(1.55)+[0,yy,-.0068] for k,yy in [('lh',.2),('rh',-.2)]})
        error=max(le,re,*he.values());bad=False
        e.data.qpos[:]=q;e.data.qvel[:]=0;mujoco.mj_forward(e.model,e.data)
        for c in e.data.contact:
            ba,bb=[int(e.model.geom_bodyid[g]) for g in [c.geom1,c.geom2]]
            ina,inb=ba in e.robot_bodies,bb in e.robot_bodies
            if not (ina or inb):continue
            expected=ina!=inb and (ba if ina else bb) in e.contact_groups
            if c.dist<(-.0007 if expected else -.00001):bad=True
        status='geometry';ratio=None
        if error<.002 and not bad:
            try:
                _,info=CycleTorqueController(e).command(q,np.zeros(e.model.nv),np.array([0,1,1,1],bool))
                status='feasible';ratio=info['torque_ratio']
            except RuntimeError as exc:status=str(exc)
        row=dict(x=x,y=y,z=z,waist=w,error=error,collision=bad,status=status,ratio=ratio)
        rows.append(row);states.append(q)
        if status=='feasible':print(row,flush=True)
out=ROOT/'outputs/compact_step_screen_v2'
(out/'transfer_forces.json').write_text(json.dumps(rows,indent=2))
np.savez_compressed(out/'transfer_forces.npz',qpos=states)
e.close()
