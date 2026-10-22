"""Read-only single-state feasibility ablations; never a dynamic success claim."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import linprog
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.axis_robust_contact_controller import AxisRobustContactController


def feasibility(A,lower,upper,bounds=None):
    equal=np.isfinite(lower)&np.isfinite(upper)&(abs(upper-lower)<1e-9)
    lo=np.isfinite(lower)&~equal;hi=np.isfinite(upper)&~equal
    result=linprog(np.zeros(A.shape[1]),A_ub=np.vstack([A[hi],-A[lo]]),b_ub=np.r_[upper[hi],-lower[lo]],
        A_eq=A[equal],b_eq=upper[equal],bounds=bounds or [(None,None)]*A.shape[1],method='highs')
    report=dict(feasible=bool(result.success),status=int(result.status),message=result.message)
    if result.success:report['maximum_violation']=float(max(0.,np.max(lower-A@result.x),np.max(A@result.x-upper)))
    return report,result.x


def main():
    out=ROOT/'outputs/axis_allocation_diagnostic_v1';out.mkdir(exist_ok=False)
    results=[]
    for name in ['row01_x_-10mm']:
        run=ROOT/'outputs/axis_contact_segments_v1'/('normal_plus3deg__'+name)
        z=dict(np.load(run/(name+'.npz')))
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',[name]);task.reset(options={'case':name})
        e=task.env;m=e.model;d=e.data;d.qpos[:]=z['qpos'][-1];d.qvel[:]=z['qvel'][-1];d.time=float(z['time_s'][-1]);mujoco.mj_forward(m,d)
        q,v,active=task.reference(d.time);captured={}
        class Capture:
            def setup(self,**kwargs):
                captured.update(kwargs);raise RuntimeError('Captured without solving')
        c=AxisRobustContactController(e,torque_weight=10000.,normal_bias_y_deg=3.)
        with patch('training.robust_contact_frame_controller.osqp.OSQP',Capture):
            try:c.command(q,v,active)
            except RuntimeError as error:assert str(error)=='Captured without solving'
        A=captured['A'].toarray();lo=captured['l'];hi=captured['u'];n=A.shape[1]
        rows={};rows['all_constraints'],_=feasibility(A,lo,hi)
        keep=np.r_[np.arange(6),np.arange(6+len(e.vadr),len(A))]
        rows['remove_motor_bounds'],_=feasibility(A[keep],lo[keep],hi[keep])
        # Friction rows follow base equations, motor bounds and per-point normals.
        keep=np.arange(6+len(e.vadr)+n//3)
        rows['remove_friction_cones'],_=feasibility(A[keep],lo[keep],hi[keep])
        names=[m.joint(j).name for j in e.joints]
        selected=[6+i for i,s in enumerate(names) if 'wrist_pitch' in s or 'wrist_yaw' in s]
        keep=np.setdiff1d(np.arange(len(A)),selected)
        rows['remove_wrist_pitch_yaw_bounds'],_=feasibility(A[keep],lo[keep],hi[keep])
        for i,joint in enumerate(names):
            keep=np.delete(np.arange(len(A)),6+i)
            check,_=feasibility(A[keep],lo[keep],hi[keep])
            if check['feasible']:rows['single_bound_removal_'+joint]=check
        # Same force geometry and native motor bounds, with generalized acceleration free.
        desired=np.zeros(m.nv);desired[:3]=80*(q[:3]-d.qpos[:3])+18*(v[:3]-d.qvel[:3])
        rot=np.zeros(3);mujoco.mju_quat2Vel(rot,d.qpos[3:7],1.)
        desired[3:6]=-80*rot-18*d.qvel[3:6]
        desired[e.vadr]=100*(q[e.qadr]-d.qpos[e.qadr])+20*(v[e.vadr]-d.qvel[e.vadr])
        indices=np.r_[np.arange(6),e.vadr]
        augmented=np.zeros((len(A),n+m.nv));augmented[:,:n]=A;augmented[:len(indices),n:]=-c.M[indices]
        offset=np.zeros(len(A));offset[:len(indices)]=c.M[indices]@desired
        acceleration_limits=np.r_[np.full(3,5.),np.full(3,10.),np.full(m.nv-6,50.)]
        bounds=[(None,None)]*n+[(-x,x) for x in acceleration_limits]
        check,x=feasibility(augmented,lo-offset,hi-offset,bounds)
        if x is not None:check['acceleration_deviation_peak']=float(max(abs(x[n:]-desired)))
        rows['free_acceleration_with_declared_bounds']=check
        np.savez_compressed(out/(name+'.npz'),A=A,lower=lo,upper=hi,desired_acceleration=desired,qpos=d.qpos,qvel=d.qvel,time_s=d.time)
        results.append(dict(case=name,time_s=float(d.time),active=active.tolist(),checks=rows,
            trajectory_sha256=hashlib.sha256((run/(name+'.npz')).read_bytes()).hexdigest()))
        task.close()
    report=dict(results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Static LP feasibility at saved failure state; removed constraints are diagnostics only and never applied to the simulator. Free acceleration bounds 5 m/s^2 translation, 10 rad/s^2 base rotation, 50 rad/s^2 joints. No contact-acceleration consistency or trajectory stability proof.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
