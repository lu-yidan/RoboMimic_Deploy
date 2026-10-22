"""Quantify wrist conflict and acceleration recourse at saved failure states.

Only offline LP problems; native limits remain intact in all dynamic runs.
"""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import linprog
from training.load_transfer_residual_env import LoadTransferResidualEnv


def rows(A,lo,hi):
    eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-9)
    l=np.isfinite(lo)&~eq;h=np.isfinite(hi)&~eq
    return np.vstack([A[h],-A[l]]),np.r_[hi[h],-lo[l]],A[eq],hi[eq]


def main():
    out=ROOT/'outputs/axis_acceleration_recourse_v1';out.mkdir(exist_ok=False);results=[]
    for name in ['row01_x_-10mm']:
        source=ROOT/'outputs/axis_allocation_diagnostic_v1'/(name+'.npz');z=dict(np.load(source))
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',[name]);task.reset(options={'case':name})
        e=task.env;m=e.model;d=e.data;d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.time=float(z['time_s']);mujoco.mj_forward(m,d)
        A=z['A'];lo=z['lower'];hi=z['upper'];n=A.shape[1];nv=m.nv
        idx=np.r_[np.arange(6),e.vadr];assert np.array_equal(idx,np.arange(nv))
        names=[m.joint(j).name for j in e.joints];wrist_row=6+names.index('right_wrist_pitch_joint')
        # Minimum symmetric relaxation of this single motor bound; never applied.
        ub,b,eq,beq=rows(A,lo,hi)
        augmented=np.c_[ub,np.zeros(len(ub))]
        for i,row in enumerate(ub):
            if np.array_equal(row,A[wrist_row]) or np.array_equal(row,-A[wrist_row]):augmented[i,-1]=-1.
        solved=linprog(np.r_[np.zeros(n),1.],A_ub=augmented,b_ub=b,A_eq=np.c_[eq,np.zeros(len(eq))],b_eq=beq,
            bounds=[(None,None)]*n+[(0,None)],method='highs')
        wrist=dict(feasible=bool(solved.success))
        if solved.success:wrist.update(minimum_extra_symmetric_motor_bound_nm=float(solved.x[-1]),original_bound_nm=5.)
        M=np.zeros((nv,nv));mujoco.mj_fullM(m,M,d.qM)
        physical=np.zeros((len(A),n+nv));physical[:,:n]=A;physical[:nv,n:]=-M
        shift=np.zeros(len(A));shift[:nv]=M@z['desired_acceleration']
        p_ub,p_b,p_eq,p_beq=rows(physical,lo-shift,hi-shift)
        # L1 deviation from desired acceleration with declared task weighting.
        avar=np.zeros((nv,n+nv));avar[:,n:]=np.eye(nv)
        objective=np.r_[np.zeros(n+nv),np.full(6,10.),np.ones(nv-6)]
        p_ub=np.vstack([np.c_[p_ub,np.zeros((len(p_ub),nv))],np.c_[avar,-np.eye(nv)],np.c_[-avar,-np.eye(nv)]])
        p_b=np.r_[p_b,z['desired_acceleration'],-z['desired_acceleration']]
        p_eq=np.c_[p_eq,np.zeros((len(p_eq),nv))]
        limits=np.r_[np.full(3,5.),np.full(3,10.),np.full(nv-6,50.)]
        bounds=[(None,None)]*n+[(-v,v) for v in limits]+[(0,None)]*nv
        recourse={}
        # Independent contact acceleration equations, using attached-point Jdot.
        _,_,active=task.reference(d.time);jacobians=[];derivatives=[]
        for enabled,body_name in zip(active,['left_ankle_roll_link','right_ankle_roll_link','hook','right_hook']):
            if not enabled:continue
            body=m.body(body_name).id
            for contact in d.contact:
                a,bg=int(contact.geom1),int(contact.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[bg])
                if ba==body and bb==0:external=bg
                elif bb==body and ba==0:external=a
                else:continue
                geom=m.geom(external).name
                if not(geom.startswith('rung_') and len(geom)==7 and geom[-2:].isdigit()):continue
                J=np.zeros((3,nv));Jdot=np.zeros((3,nv))
                mujoco.mj_jac(m,d,J,None,contact.pos,body);mujoco.mj_jacDot(m,d,Jdot,None,contact.pos,body)
                jacobians.append(J);derivatives.append(Jdot@d.qvel)
        J=np.vstack(jacobians);jdot=np.concatenate(derivatives)
        np.testing.assert_allclose(J,A[:nv].T,atol=1e-10,rtol=0)
        for mode in ['free','contact_acceleration_consistent']:
            equal=p_eq;right=p_beq
            if mode!='free':
                equal=np.vstack([equal,np.c_[np.zeros((len(J),n)),J,np.zeros((len(J),nv))]])
                right=np.r_[right,-jdot-20*(J@d.qvel)]
            solution=linprog(objective,A_ub=p_ub,b_ub=p_b,A_eq=equal,b_eq=right,bounds=bounds,method='highs')
            entry=dict(feasible=bool(solution.success),message=solution.message)
            if solution.success:
                acceleration=solution.x[n:n+nv];delta=acceleration-z['desired_acceleration']
                entry.update(weighted_l1_deviation=float(solution.fun),peak_acceleration_change=float(max(abs(delta))),
                    base_translation_change=delta[:3].tolist(),base_rotation_change=delta[3:6].tolist(),
                    joint_acceleration_changes={name:float(delta[6+i]) for i,name in enumerate(names) if abs(delta[6+i])>1e-5},
                    contact_acceleration_residual=float(max(abs(J@acceleration+jdot+20*(J@d.qvel)))))
            recourse[mode]=entry
        results.append(dict(case=name,wrist_relaxation=wrist,acceleration_recourse=recourse,
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest()));task.close()
    report=dict(results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Single-state LP analysis only. No torque limits changed in a dynamic run. Contact-consistent variant adds J*a+Jdot*v=-20*J*v on current scheduled contact points; no future-contact or stability guarantee.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
