"""Minimize terminal joint change while optimizing valid sole contact locations."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/terminal_contact_minimax_v1');out.mkdir(exist_ok=False);rows=[]
    env=FullCycleResidualEnv(plan='outputs/early_swing14_lift26_plan_v1/plan.npz');e=env.env;m=e.model;d=mujoco.MjData(m);bodies=env.endpoint_bodies
    try:
        for height,folder in [(10.,'terminal_contact_location_height10_probe_v1'),(15.,'terminal_contact_location_probe_v1')]:
            p=Path('outputs')/folder;records=json.loads((p/'summary.json').read_text())
            selected=min((r for r in records['rows'] if r['candidate']),key=lambda r:r['max_joint_delta_rad'])
            source=p/f"height{height:g}_seed{selected['seed']}.npz";assert sha(source)==selected['candidate_sha256']
            with np.load(source) as z:seed={k:z[k].copy() for k in z.files}
            q0=seed['original_qpos'];delta=seed['qpos']-q0;env.cases={'probe':{'rung_01':[0.,0.,height/1000]}};env.reset(options={'case':'probe'})
            d.qpos[:]=q0;mujoco.mj_kinematics(m,d);roll0=d.xmat[bodies[2:]].reshape(-1,3,3)[:,1,2].copy()
            def pose_error(x):
                d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[e.qadr]+=x[3:32];mujoco.mj_kinematics(m,d);values=[]
                for i in (0,1):
                    diff=np.zeros(4);mujoco.mju_mulQuat(diff,d.xquat[bodies[i]],seed['hook_quaternions'][i]*np.array([1,-1,-1,-1]));rot=np.zeros(3);mujoco.mju_quat2Vel(rot,diff,1.)
                    values.extend(d.xpos[bodies[i]]-seed['hook_positions'][i]);values.extend(.2*rot)
                for j,i in enumerate((2,3)):
                    R=d.xmat[bodies[i]].reshape(3,3);point=d.xpos[bodies[i]]+R@np.array([.09+x[32+j],0.,-.035])-.0148*R[:,2]
                    heading=np.arctan2(R[1,0],R[0,0]);diff=heading-seed['foot_yaws'][j]
                    values.extend(point-seed['target_rod_points'][j]);values.extend([.2*(R[1,2]-roll0[j]),.08*np.arctan2(np.sin(diff),np.cos(diff))])
                return np.asarray(values)
            lo=np.r_[[-.03/np.sqrt(3),-.03/np.sqrt(3),-.008],np.maximum(-.039,m.jnt_range[e.joints,0]+1e-7-q0[e.qadr]),[-.02,-.02],0.]
            hi=np.r_[[.03/np.sqrt(3),.03/np.sqrt(3),.006],np.minimum(.039,m.jnt_range[e.joints,1]-1e-7-q0[e.qadr]),[.02,.02],.039]
            initial=np.r_[delta[:3],delta[e.qadr],seed['local_contact_x']-.09,.039]
            fit=minimize(lambda x:x[-1]+1e-4*np.dot(x[:3],x[:3])+1e-4*np.dot(x[32:34],x[32:34]),np.clip(initial,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),
                constraints=[dict(type='eq',fun=pose_error),dict(type='ineq',fun=lambda x:np.r_[x[-1]-x[3:32],x[-1]+x[3:32]])],options=dict(maxiter=400,ftol=1e-11))
            residual=pose_error(fit.x);q=d.qpos.copy();mujoco.mj_collision(m,d);bad=[]
            for contact in d.contact:
                a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);ia,ib=a in e.robot_bodies,b in e.robot_bodies
                if not (ia or ib):continue
                expected=ia!=ib and (a if ia else b) in e.contact_groups
                if contact.dist < -(.0007 if expected else .00001):bad.append(float(-contact.dist))
            maximum=float(abs(fit.x[3:32]).max());native=float(np.minimum(q[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-q[e.qadr]).min())
            verified=bool(abs(residual).max()<1e-6 and maximum<=fit.x[-1]+1e-7 and maximum<=.039+1e-10 and native>=0 and not bad)
            file=out/f'height{height:g}.npz';np.savez_compressed(file,**dict(seed,qpos=q,local_contact_x=.09+fit.x[32:34]))
            rows.append(dict(height_mm=height,candidate_verified=verified,optimizer_success=bool(fit.success),optimizer_message=fit.message,
                pose_component_error=float(abs(residual).max()),max_joint_delta_rad=maximum,epigraph_rad=float(fit.x[-1]),
                base_delta_m=fit.x[:3].tolist(),local_contact_x=(.09+fit.x[32:34]).tolist(),native_margin_rad=native,collisions=bad,
                seed_sha256=sha(source),candidate_sha256=sha(file)))
        result=dict(rows=rows,source_sha256=sha(__file__),scope='Local static minimax, not global optimality or dynamic feasibility. Actual sole dimensions unchanged; contact localx stays0.07..0.11m, root rise<=0.256m and joint changes<=0.039rad. Target heights supplied offline.')
        (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'optimizer_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(rows,indent=2))
    finally:env.close()


if __name__=='__main__':main()
