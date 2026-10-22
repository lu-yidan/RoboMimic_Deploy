"""Scratch-data bounded IK diagnosis; no physical run and no feasibility guarantee."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/early_swing14_lift26_plan_v1/plan.npz';plan=dict(np.load(source))
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');m=e.model;d=mujoco.MjData(m)
    q0=plan['qpos'][-1].copy();d.qpos[:]=q0;mujoco.mj_kinematics(m,d)
    bodies=[m.body(name).id for name in ['left_ankle_roll_link','right_ankle_roll_link','hook','right_hook']]
    target_pos=d.xpos[bodies].copy();target_pos[:2,2]+=.01
    target_quat=d.xquat[bodies].copy();apply_rung_offsets(e,{'rung_01':[0.,0.,.01]})
    low=np.r_[np.full(3,-.03/np.sqrt(3)),np.maximum(-.039,m.jnt_range[e.joints,0]-q0[e.qadr]+1e-8)]
    high=np.r_[np.full(3,.03/np.sqrt(3)),np.minimum(.039,m.jnt_range[e.joints,1]-q0[e.qadr]-1e-8)]
    def set_pose(x):
        d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:];mujoco.mj_kinematics(m,d)
    def errors(x):
        set_pose(x);pos=d.xpos[bodies]-target_pos;rot=[]
        for body,quat in zip(bodies,target_quat):
            delta=np.zeros(4);mujoco.mju_mulQuat(delta,d.xquat[body],quat*np.array([1,-1,-1,-1]));vel=np.zeros(3);mujoco.mju_quat2Vel(vel,delta,1.);rot.append(vel)
        return pos,np.asarray(rot)
    def objective(x):
        pos,rot=errors(x);return np.r_[pos.ravel(),.2*rot.ravel(),1e-5*x]
    actual=dict(np.load(ROOT/'outputs/early_swing14_lift26_dynamic_v1/row01_z_+10mm/trajectory.npz'))['qpos'][-1]
    seeds=[np.zeros(32),np.r_[actual[:3]-q0[:3],actual[e.qadr]-q0[e.qadr]],np.r_[[0.,0.,.01],np.zeros(29)]]
    results=[]
    for i,seed in enumerate(seeds):
        fit=least_squares(objective,np.clip(seed,low+1e-10,high-1e-10),bounds=(low,high),max_nfev=600,ftol=1e-11,gtol=1e-11,xtol=1e-11)
        pos,rot=errors(fit.x);mujoco.mj_collision(m,d);unexpected=[]
        for c in d.contact:
            ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
            if not(ia or ib):continue
            if (ia and ib) or (ba if ia else bb) not in e.contact_groups:
                if c.dist < -1e-5:unexpected.append(dict(geom1=m.geom(c.geom1).name,geom2=m.geom(c.geom2).name,depth_m=float(-c.dist)))
        row=dict(seed=i,optimizer_success=bool(fit.success),nfev=fit.nfev,max_position_error_m=float(np.max(np.linalg.norm(pos,axis=1))),max_orientation_error_rad=float(np.max(np.linalg.norm(rot,axis=1))),max_joint_deviation_rad=float(np.max(abs(fit.x[3:]))),base_translation_m=fit.x[:3].tolist(),minimum_native_joint_margin_rad=float(np.min(np.minimum(d.qpos[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-d.qpos[e.qadr]))),unexpected_penetrations=unexpected)
        row['kinematic_candidate_found']=row['max_position_error_m']<.001 and row['max_orientation_error_rad']<.01 and not unexpected
        np.savez_compressed(a.out/f'candidate_{i}.npz',qpos=d.qpos.copy(),offset=fit.x,target_positions=target_pos,target_quaternions=target_quat)
        results.append(row)
    result=dict(results=results,reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Offline scratch-data IK, feet positions shifted10mm with original orientations, hooks unchanged. Joints within0.039rad ofnominal and native bounds, base translation norm<=0.03m, base orientation fixed. Local optimizer failure does not prove infeasibility; candidate does not prove contact-force balance, transition safety, or dynamic closure. No live simulation state modified.')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));e.close()


if __name__=='__main__':main()
