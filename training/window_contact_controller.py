"""Controlled full-cycle intervention for a previously isolated load-transfer window."""
from training.contact_frame_controller import ContactFrameController
from training.biased_contact_frame_controller import BiasedContactFrameController
from training.contact_memory_controller import ContactMemoryController


class WindowContactController(ContactFrameController):
    def __init__(self, env, joint_kp=100., torque_weight=0., leg_kp=None, normal_bias_y_deg=-3., memory_enabled=False):
        super().__init__(env,joint_kp=joint_kp,torque_weight=torque_weight,leg_kp=leg_kp)
        cls=ContactMemoryController if memory_enabled else BiasedContactFrameController
        extra={'normal_uncertainty_deg':0.} if memory_enabled else {}
        self.intervention=cls(env,joint_kp=joint_kp,torque_weight=torque_weight,leg_kp=leg_kp,normal_bias_y_deg=normal_bias_y_deg,**extra)
        self.memory_enabled=memory_enabled

    def bias_window_active(self):
        return 49.<=self.d.time<61.56

    def command(self,qref,vref,active):
        if self.bias_window_active():
            self.intervention.leg_kp=self.leg_kp
            retries=self.intervention.solver_retries;fallbacks=self.intervention.solver_fallbacks
            try:return self.intervention.command(qref,vref,active)
            finally:
                self.solver_retries+=self.intervention.solver_retries-retries
                self.solver_fallbacks+=self.intervention.solver_fallbacks-fallbacks
        return super().command(qref,vref,active)

    @property
    def memory_events(self):
        return getattr(self.intervention,'memory_events',[])
