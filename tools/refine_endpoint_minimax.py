"""Offline minimax joint-deviation refinement with exact limb pose constraints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import minimize
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');m=e.model;d=mujoco.MjData(m)
    source=ROOT/'outputs/highrod_endpoint_pose_probe_v1/candidate_2.npz';seed=dict(np.load(source));plan=dict(np.load(ROOT/'outputs/early_swing14_lift26_plan_v1/plan.npz'));q0=plan['qpos'][-1]
    bodies=[m.body(n).id for n in ['left_ankle_roll_link','right_ankle_roll_link','hook','right_hook']]
    positions=seed['target_positions'];quaternions=seed['target_quaternions']
    def pose_error(x):
        d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:32];mujoco.mj_kinematics(m,d);rotation=[]
        for body,quat in zip(bodies,quaternions):
            delta=np.zeros(4);mujoco.mju_mulQuat(delta,d.xquat[body],quat*np.array([1,-1,-1,-1]));angle=np.zeros(3);mujoco.mju_quat2Vel(angle,delta,1.);rotation.append(angle)
        return np.r_[(d.xpos[bodies]-positions).ravel(),np.asarray(rotation).ravel()]
    low=np.r_[np.full(3,-.03/np.sqrt(3)),np.maximum(-.039,m.jnt_range[e.joints,0]-q0[e.qadr]+1e-8),0.]
    high=np.r_[np.full(3,.03/np.sqrt(3)),np.minimum(.039,m.jnt_range[e.joints,1]-q0[e.qadr]-1e-8),.039]
    x0=np.r_[seed['offset'],np.max(abs(seed['offset'][3:]))]
    fit=minimize(lambda x:x[-1]+1e-4*np.dot(x[:3],x[:3]),x0,method='SLSQP',bounds=list(zip(low,high)),constraints=[dict(type='eq',fun=pose_error),dict(type='ineq',fun=lambda x:np.r_[x[-1]-x[3:32],x[-1]+x[3:32]])],options=dict(maxiter=300,ftol=1e-11))
    error=pose_error(fit.x);q=d.qpos.copy();apply_rung_offsets(e,{'rung_01':[0.,0.,.01]});mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);unexpected=[]
    for c in d.contact:
        ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
        if (ia or ib) and ((ia and ib) or (ba if ia else bb) not in e.contact_groups) and c.dist < -1e-5:unexpected.append(float(-c.dist))
    native=float(np.min(np.minimum(q[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-q[e.qadr])))
    verified=bool(np.max(abs(error))<1e-6 and np.max(abs(fit.x[3:32]))<=fit.x[-1]+1e-7 and native>=0 and not unexpected)
    result=dict(optimizer_success=bool(fit.success),optimizer_message=fit.message,iterations=fit.nit,candidate_verified=verified,max_pose_component_error=float(np.max(abs(error))),maximum_joint_deviation_rad=float(np.max(abs(fit.x[3:32]))),epigraph_value_rad=float(fit.x[-1]),base_translation_m=fit.x[:3].tolist(),native_joint_margin_rad=native,unexpected_penetrations_m=unexpected,seed_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Offline exactpose minimax refinement; local solver, no global optimum proof. True+10mm geometry supplied. No dynamic, force feasibility, or repeatability claim.')
    np.savez_compressed(a.out/'candidate.npz',qpos=q,offset=fit.x[:32],target_positions=positions,target_quaternions=quaternions)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));e.close()
    if not verified:raise SystemExit(1)

if __name__=='__main__':main()
