"""Bounded temporary base-reference change through the existing torque controller.

This changes a tracking target, never qpos, qvel, applied force, or constraints.
It is a separate reference ablation, not an action of the existing26D policy.
"""
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


def envelope_and_rate(time_s,start_s=57.,end_s=65.,ramp_s=2.):
    if not np.isfinite([time_s,start_s,end_s,ramp_s]).all() or ramp_s<=0 or end_s-start_s<2*ramp_s:
        raise ValueError('Finite window with two positive ramps required')
    if time_s<=start_s or time_s>=end_s:return 0.,0.
    if time_s<start_s+ramp_s:
        u=(time_s-start_s)/ramp_s;return u*u*(3-2*u),6*u*(1-u)/ramp_s
    if time_s>end_s-ramp_s:
        u=(end_s-time_s)/ramp_s;return u*u*(3-2*u),-6*u*(1-u)/ramp_s
    return 1.,0.


class TemporalBaseReferenceEnv(FullCycleResidualEnv):
    def __init__(self,*args,base_bias,**kwargs):
        self.base_reference_bias=np.asarray(base_bias,dtype=float)
        if self.base_reference_bias.shape!=(3,) or not np.isfinite(self.base_reference_bias).all() or np.linalg.norm(self.base_reference_bias)>.005:
            raise ValueError('Finite3D base-reference bias with norm<=5mm required')
        super().__init__(*args,**kwargs)
        self.reference_modifier_contract='bounded-base-reference-57to65s-cubic2s-v1'

    def reference(self,time_s):
        q,v,active,phase=super().reference(time_s)
        scale,rate=envelope_and_rate(time_s)
        q[:3]+=scale*self.base_reference_bias
        v[:3]+=rate*self.base_reference_bias
        return q,v,active,phase
