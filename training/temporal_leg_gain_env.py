"""Isolated leg tracking gain calibration with original motor and safety limits."""
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.temporal_base_reference_env import envelope_and_rate


class TemporalLegGainEnv(FullCycleResidualEnv):
    def __init__(self,*args,peak_leg_kp,**kwargs):
        self.peak_leg_kp=float(peak_leg_kp)
        if not np.isfinite(self.peak_leg_kp) or not 100.<=self.peak_leg_kp<=300.:
            raise ValueError('Declared leg gain must be between100 and300')
        super().__init__(*args,**kwargs)

    def select_controller(self,active,phase):
        controller=super().select_controller(active,phase)
        gain,_=envelope_and_rate(float(self.env.data.time))
        # The window lies strictly inside lf_unload for this audited reference.
        if gain and phase!='lf_unload':
            raise ValueError('Gain intervention requires the declared lf_unload phase')
        controller.leg_kp += gain*(self.peak_leg_kp-100.)
        return controller
