"""Independently validate saved terminal poses and sole contact locations."""
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/terminal_contact_minimax_v1');summary=json.loads((p/'summary.json').read_text());rows=[]
    env=FullCycleResidualEnv(plan='outputs/early_swing14_lift26_plan_v1/plan.npz');e=env.env;m=e.model;d=mujoco.MjData(m);bodies=env.endpoint_bodies
    try:
        for row in summary['rows']:
            h=row['height_mm'];file=p/f'height{h:g}.npz';assert sha(file)==row['candidate_sha256']
            with np.load(file) as z:record={k:z[k].copy() for k in z.files}
            q=record['qpos'];original=record['original_qpos'];delta=q-original
            np.testing.assert_array_equal(original,env.plan['qpos'][-1]);np.testing.assert_array_equal(q[3:7],original[3:7])
            env.cases={'probe':{'rung_01':[0.,0.,h/1000]}};env.reset(options={'case':'probe'})
            d.qpos[:]=original;mujoco.mj_kinematics(m,d);original_rot=d.xmat[bodies].reshape(-1,3,3).copy()
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);hook_error=float(np.linalg.norm(d.xpos[bodies[:2]]-record['hook_positions'],axis=1).max())
            hook_angle=0.
            for j in (0,1):
                diff=np.zeros(4);mujoco.mju_mulQuat(diff,d.xquat[bodies[j]],record['hook_quaternions'][j]*np.array([1,-1,-1,-1]));rotation=np.zeros(3);mujoco.mju_quat2Vel(rotation,diff,1.);hook_angle=max(hook_angle,float(np.linalg.norm(rotation)))
            foot_error=angle_error=0.
            for j,i in enumerate((2,3)):
                x=float(record['local_contact_x'][j]);geom=m.geom(('left','right')[j]+'_sole_surface')
                assert .07-1e-12<=x<=.11+1e-12 and float(geom.pos[0]-geom.size[0])<x<float(geom.pos[0]+geom.size[0])
                R=d.xmat[bodies[i]].reshape(3,3);point=d.xpos[bodies[i]]+R@np.array([x,0.,-.035])-.0148*R[:,2]
                foot_error=max(foot_error,float(np.linalg.norm(point-record['target_rod_points'][j])))
                yaw=np.arctan2(R[1,0],R[0,0])-record['foot_yaws'][j]
                angle_error=max(angle_error,abs(float(R[1,2]-original_rot[i,1,2])),abs(float(np.arctan2(np.sin(yaw),np.cos(yaw)))))
            native=float(np.minimum(q[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-q[e.qadr]).min())
            mujoco.mj_collision(m,d);bad=[]
            for contact in d.contact:
                a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);ia,ib=a in e.robot_bodies,b in e.robot_bodies
                if not (ia or ib):continue
                expected=ia!=ib and (a if ia else b) in e.contact_groups
                if contact.dist < -(.0007 if expected else .00001):bad.append(float(-contact.dist))
            checks=dict(hook_pose=hook_error<1e-6 and hook_angle<1e-5,sole_tangency=foot_error<1e-6 and angle_error<1e-5,
                joint_closure=float(abs(delta[e.qadr]).max())<=.039+1e-10,root_height=-.008-1e-10<=delta[2]<=.006+1e-10,
                base_translation=float(np.linalg.norm(delta[:3]))<=.03,native_bounds=native>=0,collision=not bad)
            checks={key:bool(value) for key,value in checks.items()}
            rows.append(dict(height_mm=h,passed=all(checks.values()),checks=checks,hook_error_m=hook_error,sole_error_m=foot_error,
                joint_closure_rad=float(abs(delta[e.qadr]).max()),root_delta_z_m=float(delta[2]),candidate_sha256=sha(file)))
        result=dict(passed=all(r['passed'] for r in rows),rows=rows,summary_sha256=sha(p/'summary.json'),source_sha256=sha(__file__),scope='Independent static saved-pose checks. No force balance or transition execution claim.')
        (p/'independent_audit.json').write_text(json.dumps(result,indent=2));(p/'independent_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
        if not result['passed']:raise SystemExit(1)
    finally:env.close()


if __name__=='__main__':main()
