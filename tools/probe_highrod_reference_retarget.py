"""Probe support-foot reference retargeting in scratch kinematics only."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/highrod_reference_retarget_probe_v1');out.mkdir(exist_ok=False)
    plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz';env=FullCycleResidualEnv(plan=plan,cases={'estimated10':{'rung_01':[0.,0.,.01]}})
    env.reset(options={'case':'estimated10'});e=env.env;m=e.model;d=mujoco.MjData(m)
    bodies=[m.body(name).id for name in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
    ankle=list(e.joints).index(m.joint('right_ankle_pitch_joint').id)
    rows=[]
    try:
        for t in (58.,59.4,60.,60.5,61.,62.):
            q0=env.reference(t)[0];d.qpos[:]=q0;mujoco.mj_kinematics(m,d)
            target=d.xpos[bodies].copy();target[3,2]+=.01;quat=d.xquat[bodies].copy()
            lower=np.r_[np.full(3,-.03/np.sqrt(3)),np.maximum(-.15,m.jnt_range[e.joints,0]+.005-q0[e.qadr])]
            upper=np.r_[np.full(3,.03/np.sqrt(3)),np.minimum(.15,m.jnt_range[e.joints,1]-.005-q0[e.qadr])]
            lower[3+ankle]=max(lower[3+ankle],m.jnt_range[e.joints[ankle],0]+.08-q0[e.qadr[ankle]])
            def errors(x):
                d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:];mujoco.mj_kinematics(m,d)
                rot=[]
                for body,orientation in zip(bodies,quat):
                    difference=np.zeros(4);mujoco.mju_mulQuat(difference,d.xquat[body],orientation*np.array([1,-1,-1,-1]))
                    vector=np.zeros(3);mujoco.mju_quat2Vel(vector,difference,1.);rot.append(vector)
                return d.xpos[bodies]-target,np.asarray(rot)
            def objective(x):
                pos,rot=errors(x);return np.r_[pos.ravel(),.2*rot.ravel(),1e-5*x]
            fit=least_squares(objective,np.clip(np.zeros(32),lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            pos,rot=errors(fit.x);mujoco.mj_collision(m,d);unexpected=[]
            for contact in d.contact:
                a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);ia,ib=a in e.robot_bodies,b in e.robot_bodies
                if not (ia or ib):continue
                expected=ia!=ib and (a if ia else b) in e.contact_groups
                if not expected and contact.dist < -1e-5:unexpected.append(float(contact.dist))
            position=float(np.linalg.norm(pos,axis=1).max());rotation=float(np.linalg.norm(rot,axis=1).max())
            file=out/f'time{t:g}.npz';np.savez_compressed(file,qpos=d.qpos.copy(),original_qpos=q0,target_positions=target,target_quaternions=quat)
            rows.append(dict(time_s=t,kinematic_candidate=bool(position<1e-5 and rotation<1e-4 and not unexpected),optimizer_success=bool(fit.success),
                max_position_error_m=position,max_rotation_error_rad=rotation,base_delta_m=fit.x[:3].tolist(),max_joint_delta_rad=float(abs(fit.x[3:]).max()),
                original_ankle_rad=float(q0[e.qadr[ankle]]),candidate_ankle_rad=float(d.qpos[e.qadr[ankle]]),unexpected_penetrations=unexpected,candidate_sha256=sha(file)))
        result=dict(rows=rows,reference_sha256=sha(plan),source_sha256=sha(__file__),
            scope='Offline local IK: right support foot target raised10mm, hooks and airborne left foot unchanged, all orientations preserved. Base shift<=30mm, joint changes<=0.15rad, native margins>=5mrad and right ankle>=80mrad. No dynamics or residual-policy authority claim.')
        (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'probe_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(rows,indent=2))
    finally:env.close()


if __name__=='__main__':main()
