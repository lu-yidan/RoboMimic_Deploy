"""Relax tracking demand only; preserve force, dynamics and native torque bounds."""
import numpy as np
from training.climb_cycle import CycleTorqueController


def scaled_reference(env,qref,vref,scale):
    if not np.isfinite(scale) or not 0<=scale<=1:raise ValueError('Tracking scale must be 0..1')
    if scale==1:return qref.copy(),vref.copy()
    q=qref.copy();v=vref.copy();d=env.data
    q[:3]=d.qpos[:3]+scale*(qref[:3]-d.qpos[:3])
    v[:3]=d.qvel[:3]+scale*(vref[:3]-d.qvel[:3])
    q[env.qadr]=d.qpos[env.qadr]+scale*(qref[env.qadr]-d.qpos[env.qadr])
    v[env.vadr]=d.qvel[env.vadr]+scale*(vref[env.vadr]-d.qvel[env.vadr])
    # Original controller's upright rotational stabilization remains unchanged.
    return q,v


class TrackingRelaxationController(CycleTorqueController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.relaxation_steps=0;self.minimum_tracking_scale=1.

    def command(self,qref,vref,active):
        reasons=[]
        for scale in (1.,.75,.5,.25,0.):
            q,v=scaled_reference(self.e,qref,vref,scale)
            try:torque,info=super().command(q,v,active)
            except RuntimeError as exc:reasons.append(str(exc));continue
            if scale<1:self.relaxation_steps+=1
            self.minimum_tracking_scale=min(self.minimum_tracking_scale,scale)
            info.update(tracking_scale=scale,relaxation_steps=self.relaxation_steps,
                        minimum_tracking_scale=self.minimum_tracking_scale)
            return torque,info
        raise RuntimeError('No feasible tracking scale: '+reasons[-1])
