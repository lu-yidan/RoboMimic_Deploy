"""Terminal IK with explicit root-height and joint-closure reference budgets."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/bounded_terminal_geometry_probe_v1');out.mkdir(exist_ok=False)
    path=Path('outputs/early_swing14_lift26_plan_v1/plan.npz');env=FullCycleResidualEnv(plan=path);e=env.env;m=e.model;d=mujoco.MjData(m)
    q0=env.plan['qpos'][-1].copy();bodies=env.endpoint_bodies
    d.qpos[:]=q0;mujoco.mj_kinematics(m,d);positions=d.xpos[bodies].copy();quats=d.xquat[bodies].copy()
    with np.load('outputs/highrod_endpoint_adapt_plan_v1/plan.npz') as z:old=z['qpos'][-1]-q0
    seeds=[np.zeros(32),np.r_[old[:3],old[e.qadr]],np.r_[[0.,0.,.006],np.zeros(29)]]
    rows=[]
    try:
        for height in (.01,.015):
            target=positions.copy();target[2:,2]+=height
            lo=np.r_[[-.03/np.sqrt(3),-.03/np.sqrt(3),-.008],np.maximum(-.039,m.jnt_range[e.joints,0]+1e-7-q0[e.qadr])]
            hi=np.r_[[.03/np.sqrt(3),.03/np.sqrt(3),.006],np.minimum(.039,m.jnt_range[e.joints,1]-1e-7-q0[e.qadr])]
            def errors(x):
                d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:];mujoco.mj_kinematics(m,d)
                rotations=[]
                for body,quat in zip(bodies,quats):
                    delta=np.zeros(4);mujoco.mju_mulQuat(delta,d.xquat[body],quat*np.array([1,-1,-1,-1]));v=np.zeros(3);mujoco.mju_quat2Vel(v,delta,1.);rotations.append(v)
                return d.xpos[bodies]-target,np.asarray(rotations)
            def objective(x):
                pos,rot=errors(x);return np.r_[pos.ravel(),.2*rot.ravel(),1e-5*x]
            for seed_id,seed in enumerate(seeds):
                fit=least_squares(objective,np.clip(seed,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11)
                pos,rot=errors(fit.x);error=float(np.linalg.norm(pos,axis=1).max());rotation=float(np.linalg.norm(rot,axis=1).max())
                filename=out/f'height{height*1000:g}_seed{seed_id}.npz';np.savez_compressed(filename,qpos=d.qpos.copy(),original_qpos=q0,target_positions=target,target_quaternions=quats)
                rows.append(dict(height_mm=height*1000,seed=seed_id,optimizer_success=bool(fit.success),pose_candidate=bool(error<1e-5 and rotation<1e-4),
                    position_error_m=error,rotation_error_rad=rotation,max_joint_delta_rad=float(abs(fit.x[3:]).max()),base_delta_m=fit.x[:3].tolist(),candidate_sha256=sha(filename)))
        result=dict(rows=rows,reference_sha256=sha(path),source_sha256=sha(__file__),
            scope='Scratch IK only. Feet raised10/15mm, original hook targets and four orientations. Reference root rise constrained to0.242..0.256m and joint closure<=0.039rad. These are reference budgets, not dynamic guarantees. No collision or force feasibility claim at this probe stage.')
        (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'probe_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(rows,indent=2))
    finally:env.close()


if __name__=='__main__':main()
