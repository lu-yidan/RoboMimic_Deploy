"""Temporary right-support target retargeting with a bounded offline IK solve."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.temporal_base_reference_env import envelope_and_rate
from tools.plan_clearance_cycle import audit_plan
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/support_retarget_plan_v1');out.mkdir(exist_ok=False)
    source=Path('outputs/highrod_endpoint_adapt_plan_v1/plan.npz');env=FullCycleResidualEnv(plan=source);e=env.env;m=e.model;d=mujoco.MjData(m)
    plan={k:v.copy() for k,v in env.plan.items()}
    # Insert the exact exit time without changing any previous knot or segment.
    if np.min(abs(plan['time']-65.))>1e-8:
        index=int(np.searchsorted(plan['time'],65.));q,v,active,phase=env.reference(65.)
        for key,value in [('time',65.),('qpos',q),('active',active),('phase',phase),('errors',float(max(plan['errors'])))]:
            plan[key]=np.insert(plan[key],index,value,axis=0)
    bodies=[m.body(name).id for name in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
    ankle=list(e.joints).index(m.joint('right_ankle_pitch_joint').id);records=[]
    try:
        for i,t in enumerate(plan['time']):
            gain,_=envelope_and_rate(float(t))
            if gain<=1e-12:continue
            q0=plan['qpos'][i].copy();d.qpos[:]=q0;mujoco.mj_kinematics(m,d)
            target=d.xpos[bodies].copy();target[3,2]+=.01*gain;quat=d.xquat[bodies].copy()
            lo=np.r_[np.full(3,-.03/np.sqrt(3)),np.maximum(-.15,m.jnt_range[e.joints,0]+.005-q0[e.qadr])]
            hi=np.r_[np.full(3,.03/np.sqrt(3)),np.minimum(.15,m.jnt_range[e.joints,1]-.005-q0[e.qadr])]
            lo[3+ankle]=max(lo[3+ankle],m.jnt_range[e.joints[ankle],0]+.005+.075*gain-q0[e.qadr[ankle]])
            def errors(x):
                d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:];mujoco.mj_kinematics(m,d)
                rot=[]
                for body,orientation in zip(bodies,quat):
                    diff=np.zeros(4);mujoco.mju_mulQuat(diff,d.xquat[body],orientation*np.array([1,-1,-1,-1]));value=np.zeros(3);mujoco.mju_quat2Vel(value,diff,1.);rot.append(value)
                return d.xpos[bodies]-target,np.asarray(rot)
            def objective(x):
                pos,rot=errors(x);return np.r_[pos.ravel(),.2*rot.ravel(),1e-5*x]
            fit=least_squares(objective,np.clip(np.zeros(32),lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            pos,rot=errors(fit.x);error=float(np.linalg.norm(pos,axis=1).max());rotation=float(np.linalg.norm(rot,axis=1).max())
            records.append(dict(time_s=float(t),gain=gain,position_error_m=error,rotation_error_rad=rotation,max_joint_delta_rad=float(abs(fit.x[3:]).max()),base_delta_m=fit.x[:3].tolist()))
            if error>=1e-5 or rotation>=1e-4:
                (out/'failed_node.json').write_text(json.dumps(records[-1],indent=2));raise RuntimeError('Retarget IK rejected')
            plan['qpos'][i]=d.qpos.copy()
        np.savez_compressed(out/'plan.npz',**plan)
        audit=audit_plan(e,plan,dt=.005);audit['parent_ik_residual']=audit.pop('ik_residual')
        audit['scope']='Nominal dense5ms collision/native limits; retarget node pose errors checked separately. No dynamic guarantee.'
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
        manifest=dict(parent=str(source),parent_sha256=sha(source),plan_sha256=sha(out/'plan.npz'),source_sha256=sha(__file__),
            window_s=[57.,65.],ramp_s=2.,right_foot_target_peak_z_m=.01,peak_ankle_lower_margin_rad=.08,
            pose_bounds=dict(base_translation_norm_m=.03,joint_delta_rad=.15),node_checks=records,
            scope='Known10mm geometry used to retarget right support foot only in the near-limit window. Four endpoint orientations and other three positions preserved at each solved knot. Base/joint reference changes; not existing26D residual policy action. Initial/final poses and duration unchanged.')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'builder_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(dict(passed=audit['passed'],modified_nodes=len(records),peak_reference_joint_speed_rad_s=audit['peak_reference_joint_speed_rad_s'],failures=len(audit['failures']))))
        if not audit['passed']:raise SystemExit(1)
    finally:env.close()


if __name__=='__main__':main()
