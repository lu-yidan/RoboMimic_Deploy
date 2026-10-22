"""Quantify foot yaw and knee spread on reference or recorded cycle states."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv

ROOT=Path(__file__).resolve().parents[1]


def measure(e,states):
    d=mujoco.MjData(e.model);rows=[]
    knees=[e.model.body(s+'_knee_link').id for s in ['left','right']]
    feet=[e.model.body(s+'_ankle_roll_link').id for s in ['left','right']]
    for q in states:
        d.qpos[:]=q;mujoco.mj_kinematics(e.model,d)
        yaw=[np.rad2deg(np.arctan2(d.xmat[b,3],d.xmat[b,0])) for b in feet]
        rows.append([*yaw,abs(d.xpos[knees[0],1]-d.xpos[knees[1],1]),
                     max(abs(d.xpos[k,1])-abs(d.xpos[f,1]) for k,f in zip(knees,feet))])
    values=np.array(rows)
    return dict(foot_yaw_peak_deg=float(abs(values[:,:2]).max()),
                knee_width_peak_m=float(values[:,2].max()),knee_outside_foot_peak_m=float(values[:,3].max()),
                hip_yaw_peak_deg=float(np.rad2deg(abs(states[:,e.qadr[[2,8]]]).max())),
                leg_joint_total_travel_rad=float(abs(np.diff(states[:,e.qadr[:12]],axis=0)).sum()),
                sample_count=len(states))


def main():
    p=argparse.ArgumentParser();p.add_argument('--states',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    data=np.load(a.states);result=measure(e,data['qpos']);result['source']=str(a.states)
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));e.close()


if __name__=='__main__':main()
