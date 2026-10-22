"""Screen body positions and smaller foot yaw without changing the wall model."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import LimbIK, rung

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--fine',action='store_true');a=p.parse_args()
out=ROOT/('outputs/compact_step_screen_v2' if a.fine else 'outputs/compact_step_screen_v1')
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset();ik=LimbIK(e)
baseline=np.load(ROOT/'outputs/cycle_contact_plan_v17/plan.npz')
seed=baseline['qpos'][np.searchsorted(baseline['time'],29.)].copy()
records=[];states=[]
for yaw in ([30,40,45,50,60] if a.fine else [0,30,60,90]):
    for x in ([-.025,0,.025] if a.fine else [-.06,0,.06,.12,.17]):
        for z in ([1.03,1.05,1.07,1.09] if a.fine else [1.07,1.115,1.16]):
            q=seed.copy();q[:3]=[x,0,z]
            q,le=ik.contact_leg(q,'lf',rung(.30)+[0,.2,0],0.)
            q,re=ik.contact_leg(q,'rf',rung(.55)+[0,-.2,.015],-np.deg2rad(yaw))
            q,he=ik.pose(q,{k:rung(1.55)+[0,y,-.0068] for k,y in [('lh',.2),('rh',-.2)]})
            ik.d.qpos[:]=q;mujoco.mj_forward(e.model,ik.d)
            bad=[]
            for c in ik.d.contact:
                ba,bb=[int(e.model.geom_bodyid[g]) for g in [c.geom1,c.geom2]]
                if not (ba in e.robot_bodies or bb in e.robot_bodies):continue
                contact=(ba in e.robot_bodies)!=(bb in e.robot_bodies) and (ba if ba in e.robot_bodies else bb) in e.contact_groups
                if c.dist<(-.0007 if contact else -.00001):bad.append([e.model.body(ba).name,e.model.body(bb).name,float(c.dist)])
            knee_y=[float(ik.d.xpos[e.model.body(side+'_knee_link').id,1]) for side in ['left','right']]
            error=max(le,re,*he.values())
            row=dict(yaw=yaw,x=x,z=z,error=error,collisions=bad,knee_width=knee_y[0]-knee_y[1],feasible=error<.002 and not bad)
            records.append(row);states.append(q)
            if row['feasible']:print(row,flush=True)
(out/'screen.json').write_text(json.dumps(records,indent=2));np.savez_compressed(out/'states.npz',qpos=states)
print('Feasible',sum(r['feasible'] for r in records),'/',len(records));e.close()
