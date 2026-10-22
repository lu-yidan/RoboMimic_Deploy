"""Use the planner's sole-to-rod tangency model instead of fixing foot pitch."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/terminal_sole_tangency_probe_v1'))
    parser.add_argument('--root-z-cap',type=float,choices=(.006,.008),default=.006);parser.add_argument('--heights',type=float,nargs='+',default=[10.,15.])
    args=parser.parse_args();out=args.output;out.mkdir(exist_ok=False)
    path=Path('outputs/early_swing14_lift26_plan_v1/plan.npz');env=FullCycleResidualEnv(plan=path);e=env.env;m=e.model;d=mujoco.MjData(m)
    q0=env.plan['qpos'][-1].copy();bodies=env.endpoint_bodies;local=np.array([.09,0.,-.035])
    d.qpos[:]=q0;mujoco.mj_kinematics(m,d);positions=d.xpos[bodies].copy();quats=d.xquat[bodies].copy();rotations=d.xmat[bodies].reshape(-1,3,3).copy()
    nominal=np.array([positions[i]+rotations[i]@local-.0148*rotations[i,:,2] for i in (2,3)])
    yaw=np.array([np.arctan2(rotations[i,1,0],rotations[i,0,0]) for i in (2,3)])
    rows=[]
    try:
        for height_mm in args.heights:
            if height_mm not in (10.,15.):raise ValueError('Known10/15mm probe only')
            height=height_mm/1000
            target=nominal+np.array([0.,0.,height])
            env.cases={'probe':{'rung_01':[0.,0.,height]}};env.reset(options={'case':'probe'})
            lo=np.r_[[-.03/np.sqrt(3),-.03/np.sqrt(3),-.008],np.maximum(-.039,m.jnt_range[e.joints,0]+1e-7-q0[e.qadr])]
            hi=np.r_[[.03/np.sqrt(3),.03/np.sqrt(3),args.root_z_cap],np.minimum(.039,m.jnt_range[e.joints,1]-1e-7-q0[e.qadr])]
            def errors(x):
                d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:];mujoco.mj_kinematics(m,d)
                values=[];foot_errors=[];angle_errors=[]
                for i in (0,1):
                    diff=np.zeros(4);mujoco.mju_mulQuat(diff,d.xquat[bodies[i]],quats[i]*np.array([1,-1,-1,-1]));angle=np.zeros(3);mujoco.mju_quat2Vel(angle,diff,1.)
                    values.extend(d.xpos[bodies[i]]-positions[i]);values.extend(.2*angle)
                for j,i in enumerate((2,3)):
                    R=d.xmat[bodies[i]].reshape(3,3);point=d.xpos[bodies[i]]+R@local-.0148*R[:,2]
                    error=point-target[j];heading=np.arctan2(R[1,0],R[0,0]);dyaw=np.arctan2(np.sin(heading-yaw[j]),np.cos(heading-yaw[j]));roll=R[1,2]-rotations[i,1,2]
                    values.extend(error);values.extend([.2*roll,.08*dyaw]);foot_errors.append(error);angle_errors.extend([roll,dyaw])
                return np.asarray(values),np.asarray(foot_errors),np.asarray(angle_errors)
            def objective(x):return np.r_[errors(x)[0],1e-5*x]
            with np.load(f'outputs/bounded_terminal_geometry_probe_v1/height{height*1000:g}_seed2.npz') as z:seed_q=z['qpos']-q0
            seeds=[np.zeros(32),np.r_[seed_q[:3],seed_q[e.qadr]],np.r_[[0.,0.,.006],np.zeros(29)]]
            for seed_id,seed in enumerate(seeds):
                fit=least_squares(objective,np.clip(seed,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11)
                residual,foot,angles=errors(fit.x);mujoco.mj_collision(m,d);bad=[]
                for contact in d.contact:
                    a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);ia,ib=a in e.robot_bodies,b in e.robot_bodies
                    if not (ia or ib):continue
                    expected=ia!=ib and (a if ia else b) in e.contact_groups
                    tolerance=.0007 if expected else .00001
                    if contact.dist < -tolerance:bad.append(dict(geom1=m.geom(contact.geom1).name,geom2=m.geom(contact.geom2).name,penetration_m=float(-contact.dist)))
                file=out/f'height{height*1000:g}_seed{seed_id}.npz';np.savez_compressed(file,qpos=d.qpos.copy(),original_qpos=q0,target_rod_points=target,hook_positions=positions[:2],hook_quaternions=quats[:2],foot_yaws=yaw)
                rows.append(dict(height_mm=height*1000,seed=seed_id,candidate=bool(abs(residual).max()<1e-5 and abs(angles).max()<1e-4 and not bad),
                    maximum_scaled_pose_error=float(abs(residual).max()),sole_point_error_m=float(np.linalg.norm(foot,axis=1).max()),
                    foot_roll_yaw_error=float(abs(angles).max()),max_joint_delta_rad=float(abs(fit.x[3:]).max()),base_delta_m=fit.x[:3].tolist(),collisions=bad,candidate_sha256=sha(file)))
        result=dict(rows=rows,root_z_cap_m=args.root_z_cap,reference_sha256=sha(path),source_sha256=sha(__file__),
            scope='Offline tangency model reused from existing planner: sole local[.09,0,-.035], rod radius.0148m. Foot pitch is free, hooks remain fixed; root rise capped at0.25+root_z_cap_m and joint changes<=.039rad. Physical acceptance unchanged. No dynamic guarantee or global infeasibility proof.')
        (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'probe_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(rows,indent=2))
    finally:env.close()


if __name__=='__main__':main()
