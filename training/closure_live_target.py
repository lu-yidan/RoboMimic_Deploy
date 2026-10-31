"""Terminal-only joint acceleration/contact force allocation experiment.

Uses actual simulated contacts. Preserves the original controller's contact
memory update and all motor bounds. No actor or endpoint reference changes.
"""
import numpy as np
import mujoco
from training.closure_constrained_acceleration import solve_probe


def install_acceleration_target(base, maximum_gain):
    if not np.isfinite(maximum_gain) or not 0. <= maximum_gain <= 1.:
        raise ValueError('QP blend must be within [0, 1]')
    base.live_trace=[]
    base.live_prediction=None
    original_check=base.physics_check
    def checked(qref,phase,controller):
        reason=original_check(qref,phase,controller)
        if base.live_prediction is not None:
            e=base.env;d=e.data
            friction=np.zeros(e.model.nv)
            mask=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
            mujoco.mj_mulJacTVec(e.model,d,friction,np.where(mask,d.efc_force,0.))
            forces=[]
            for i in range(d.ncon):
                f=np.zeros(6);mujoco.mj_contactForce(e.model,d,i,f)
                forces.append(f[:3]@d.contact[i].frame.reshape(3,3))
            base.live_trace.append((float(d.time),d.qvel.copy(),d.qacc.copy(),d.qfrc_constraint.copy(),friction.copy(),
                np.sum(forces,axis=0) if forces else np.zeros(3),base.live_prediction.copy(),d.qfrc_actuator.copy()))
        return reason
    base.physics_check=checked
    base.closure_qp_stats=dict(commands=0,max_constraint_residual=0.,max_roundoff_clip_nm=0.)
    for controller in (base.baseline,base.contact):
        original=controller.command

        def command(qref,vref,active,_original=original,_controller=controller):
            torque,info=_original(qref,vref,active)
            u=float(np.clip((base.env.data.time-99.66)/3.,0.,1.))
            gain=maximum_gain*u*u*(3.-2.*u)
            if gain==0.:return torque,info
            bodies=[_controller.ik.bodies[k] for k,on in zip(('lf','rf','lh','rh'),active) if on]
            try:result=solve_probe(base.env,base.env.data,qref,vref,bodies)
            except ValueError as exc:raise RuntimeError('Terminal acceleration QP: '+str(exc)) from exc
            if not result['passed']:raise RuntimeError('Terminal acceleration QP: '+result['status'])
            e=base.env
            error=float(max(0.,np.max(e.limits[:,0]-result['torque']),np.max(result['torque']-e.limits[:,1])))
            if error>1e-3:raise RuntimeError('Terminal QP motor feasibility')
            base.live_prediction=result['acceleration'].copy()
            bounded=np.clip(result['torque'],e.limits[:,0],e.limits[:,1])
            combined=(1.-gain)*torque+gain*bounded
            if not np.isfinite(combined).all():raise RuntimeError('Nonfinite terminal QP torque')
            stats=base.closure_qp_stats;stats['commands']+=1
            stats['max_constraint_residual']=max(stats['max_constraint_residual'],result['max_constraint_residual'])
            stats['max_roundoff_clip_nm']=max(stats['max_roundoff_clip_nm'],error)
            info=dict(info,torque_ratio=float(np.max(abs(combined)/e.limits[:,1])))
            return combined,info

        controller.command=command
