"""Oracle acceleration/force QP intervention in a fixed 57..65s window."""
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.temporal_base_reference_env import envelope_and_rate
from training.contact_acceleration_soft_probe import solve_probe


class WindowAccelerationController:
    def __init__(self,controller,owner):self.controller=controller;self.owner=owner
    def __getattr__(self,key):return getattr(self.controller,key)
    def command(self,qref,vref,active):
        # Update the original controller and its contact-memory history as usual.
        torque,info=self.controller.command(qref,vref,active);o=self.owner;e=o.env
        gain,_=envelope_and_rate(float(e.data.time))
        if gain<=0:return torque,info
        if abs(self.controller.leg_kp-100)>1e-10:raise RuntimeError('Unexpected local-window tracking gain')
        bodies=[self.controller.ik.bodies[k] for k,on in zip(('lf','rf','lh','rh'),active) if on]
        try:r=solve_probe(e,e.data,qref,vref,bodies)
        except ValueError as exc:raise RuntimeError('Oracle acceleration QP: '+str(exc)) from exc
        if not r['passed']:raise RuntimeError('Oracle acceleration QP: '+r['status'])
        # Match the existing controller's verified 1e-3 solver-roundoff gate.
        bound_error=float(max(0.,np.max(e.limits[:,0]-r['torque']),np.max(r['torque']-e.limits[:,1])))
        if bound_error>1e-3:raise RuntimeError('Oracle QP motor feasibility')
        bounded=np.clip(r['torque'],e.limits[:,0],e.limits[:,1])
        result=(1-gain)*torque+gain*bounded
        if not np.isfinite(result).all() or np.any(result<e.limits[:,0]-1e-6) or np.any(result>e.limits[:,1]+1e-6):raise RuntimeError('Blended torque bounds')
        o.qp_records.append(dict(time_s=float(e.data.time),gain=gain,contact_count=r['contact_count'],constraint_residual=r['max_constraint_residual'],sticking_residual=r['sticking_residual_max'],acceleration_change_norm=r['acceleration_change_norm'],torque_difference_max_nm=float(abs(bounded-torque).max()),solver_roundoff_clip_nm=bound_error))
        info=dict(info,torque_ratio=float(np.max(abs(result)/e.limits[:,1])),acceleration_qp_gain=gain,acceleration_qp_scope='Actual MuJoCo contacts; local oracle intervention, distinct from biased contact estimator')
        return result,info


class WindowAccelerationQPEnv(FullCycleResidualEnv):
    def reset(self,**kwargs):
        self.qp_records=[]
        return super().reset(**kwargs)
    def select_controller(self,active,phase):
        return WindowAccelerationController(super().select_controller(active,phase),self)
