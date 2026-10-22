"""Conservative contact cones plus bounded tracking-demand reduction."""
from training.robust_contact_frame_controller import RobustContactFrameController
from training.tracking_relaxation import scaled_reference


class RobustTrackingController(RobustContactFrameController):
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
            info.update(tracking_scale=scale,relaxation_steps=self.relaxation_steps,minimum_tracking_scale=self.minimum_tracking_scale)
            return torque,info
        raise RuntimeError('No feasible conservative tracking scale: '+reasons[-1])
