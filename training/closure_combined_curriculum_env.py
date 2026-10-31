"""Suffix training with explicit, fixed waist/body reference authority.

The actor still has 26 joint actions. Added controller commands are fixed and
geometry gated, and are not claimed as learned actions. Cache restore occurs
only at episode reset; complete-cycle evaluation remains mandatory.
"""
import numpy as np
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target


class ClosureCombinedCurriculumEnv(ClosureCurriculumEnvV3):
    def __init__(self,*args,controller_modifier,**kwargs):
        self.controller_modifier=dict(controller_modifier)
        bias=np.asarray(self.controller_modifier['waist_bias'],dtype=float)
        if bias.shape!=(3,) or not np.isfinite(bias).all() or abs(bias).max()>1.:raise ValueError('Invalid waist residual')
        if not 0.<=self.controller_modifier['body_drop_m']<=.015:raise ValueError('Invalid body reference drop')
        if not 0.<self.controller_modifier['waist_ramp_seconds']<=10.:raise ValueError('Invalid waist ramp')
        self.controller_geometry_gain=None
        super().__init__(*args,**kwargs)

    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        self.controller_geometry_gain=float(np.clip((float(obs[138])-.625)/.125,0.,1.))
        base=self.env.unwrapped;c=self.controller_modifier
        self.waist_state=install_waist_target(base,c['waist_bias'],geometry_gain=self.controller_geometry_gain,
            ramp_seconds=c['waist_ramp_seconds'],record=self.trace is not None)
        install_body_target(base,self.controller_geometry_gain*c['body_drop_m'])
        if self.trace is not None:self.trace['waist_reference_trace']=self.waist_state['trace']
        return obs,self.annotate(info)

    def annotate(self,info):
        result=super().annotate(info)
        result.update(controller_modifier=self.controller_modifier,controller_geometry_gain=self.controller_geometry_gain,
            control_scope='26D actor with additional fixed waist/body reference commands. Not a pure 26D-controller training comparison.')
        return result
