"""Landing-window training with the declared fixed controller candidate.

Optional five-dimensional reference commands are constants, not actor outputs.
The 148-input, 26-action learning interface and native task limits are retained.
"""
import numpy as np
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from training.closure_foreaft_reference import install_adaptive_reference


class LandingControllerCurriculumEnv(ClosureCombinedCurriculumEnv):
    def __init__(self,*args,reference_command=None,**kwargs):
        self.reference_command=None if reference_command is None else np.asarray(reference_command,dtype=float)
        if self.reference_command is not None:
            if self.reference_command.shape!=(5,) or not np.isfinite(self.reference_command).all() or abs(self.reference_command).max()>1.:
                raise ValueError('Five bounded constant reference commands required')
        self.constant_reference_state=None
        super().__init__(*args,**kwargs)
        if abs(self.start-70.2)>1e-8:
            raise ValueError('This environment requires the declared 70.2-second landing window')

    def reset(self,**kwargs):
        self.constant_reference_state=None
        obs,info=super().reset(**kwargs)
        if self.reference_command is not None:
            self.constant_reference_state=install_adaptive_reference(self.env.unwrapped,self.controller_modifier,
                geometry_gain=self.controller_geometry_gain,record=self.trace is not None)
            self.constant_reference_state['set_action'](self.reference_command)
            if self.trace is not None:self.trace['adaptive_reference_trace']=self.constant_reference_state['trace']
        return obs,self.annotate(info)

    def annotate(self,info):
        result=super().annotate(info)
        result.update(training_segment='left_foot_landing_and_closure',training_episode_seconds=43.46,
            fixed_additional_reference_command=None if self.reference_command is None else self.reference_command.tolist(),
            control_scope='Trainable 26D actor with declared fixed waist/body and optional fixed fore-aft/waist reference commands. Added authority is not learned by this actor.')
        return result
