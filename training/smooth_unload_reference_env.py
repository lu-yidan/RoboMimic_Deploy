"""C1 joint/base reference interpolation only during the original unload phase."""
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicHermiteSpline
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


class SmoothUnloadReferenceEnv(FullCycleResidualEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        indices=np.flatnonzero(self.plan['phase']=='lf_unload');first=int(indices[0]-1);last=int(indices[-1])
        times=self.plan['time'];qs=self.plan['qpos'];local_t=times[first:last+1];local_q=qs[first:last+1]
        if not np.array_equal(qs[:,3:7],np.broadcast_to(qs[0,3:7],qs[:,3:7].shape)):
            raise ValueError('This experiment requires a constant base quaternion')
        slopes=PchipInterpolator(local_t,local_q,axis=0).derivative()(local_t)
        slopes[0]=(qs[first]-qs[first-1])/(times[first]-times[first-1])
        slopes[-1]=(qs[last+1]-qs[last])/(times[last+1]-times[last])
        self.smooth_window=(float(local_t[0]),float(local_t[-1]))
        self.spline=CubicHermiteSpline(local_t,local_q,slopes,axis=0)
        self.spline_velocity=self.spline.derivative()

    def reference(self,time_s):
        q,v,active,phase=super().reference(time_s)
        if hasattr(self,'spline') and self.smooth_window[0]<=time_s<=self.smooth_window[1]:
            q=self.spline(time_s);derivative=self.spline_velocity(time_s)
            v[:3]=derivative[:3];v[self.env.vadr]=derivative[self.env.qadr]
        return q,v,active,phase
