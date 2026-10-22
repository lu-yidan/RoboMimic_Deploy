"""Time-scale only lf_unload, keeping the sensing-bias window in reference time."""
import json
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.periodic_force_history_controller import PeriodicForceHistoryController


def reference_clock(t,factor,start=52.2,end=70.2):
    if not np.isfinite([t,factor]).all() or factor<1.:
        raise ValueError('Finite time and factor>=1 required')
    new_end=start+factor*(end-start)
    return t if t<=start else start+(t-start)/factor if t<=new_end else t-(factor-1)*(end-start)


class ReferenceClockContactController(PeriodicForceHistoryController):
    def __init__(self,*args,unload_factor,**kwargs):
        self.unload_factor=unload_factor
        super().__init__(*args,**kwargs)

    def bias_window_active(self):
        original=reference_clock(float(self.d.time),self.unload_factor)
        return bool(49.<=original<61.56)


class RetimedUnloadEnv(FullCycleResidualEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        metadata=json.loads((self.plan_path.parent/'manifest.json').read_text())
        self.unload_factor=float(metadata['unload_time_factor'])
        if self.unload_factor not in (1.5,2.) or len(self.ends)!=1:
            raise ValueError('Single-cycle declared factor required')

    def reset(self,**kwargs):
        result=super().reset(**kwargs)
        self.contact=ReferenceClockContactController(self.env,joint_kp=100.,leg_kp=100.,torque_weight=10000.,
            normal_bias_y_deg=self.normal_bias,memory_enabled=self.memory_enabled,cycle_ends=self.ends,
            unload_factor=self.unload_factor)
        return result
