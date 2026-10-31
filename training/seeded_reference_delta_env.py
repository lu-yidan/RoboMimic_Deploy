"""Learn small reference corrections around the verified constant controller.

The 26-action teacher is frozen. Zero delta reproduces the existing five
constant reference commands exactly, including their float64 values.
"""
import numpy as np
from training.closure_foreaft_curriculum_env import ClosureForeaftCurriculumEnv


class SeededReferenceDeltaEnv(ClosureForeaftCurriculumEnv):
    def __init__(self, *args, seed_command, delta_scale=.25, **kwargs):
        self.seed_command = np.asarray(seed_command, dtype=np.float64)
        if (self.seed_command.shape != (5,) or not np.isfinite(self.seed_command).all()
                or abs(self.seed_command).max() > 1.):
            raise ValueError('Five finite bounded seed commands required')
        if not np.isfinite(delta_scale) or not 0. < delta_scale <= 1.:
            raise ValueError('Delta scale must be in (0, 1]')
        self.delta_scale = float(delta_scale)
        super().__init__(*args, **kwargs)
        if abs(self.start - 99.66) > 1e-7:
            raise ValueError('This experiment uses only the 14-second closure window')

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.feedback_state['set_action'](self.seed_command)
        if self.trace is not None:
            self.trace['reference_delta_actions'] = []
        return obs, self.annotate(info)

    def step(self, action):
        delta = np.asarray(action, dtype=np.float64)
        if delta.shape != (5,) or not np.isfinite(delta).all():
            raise ValueError('Five finite delta actions required')
        delta = np.clip(delta, -1., 1.)
        command = np.clip(self.seed_command + self.delta_scale * delta, -1., 1.)
        if self.trace is not None:
            self.trace['reference_delta_actions'].append(delta.copy())
        return super().step(command)

    def annotate(self, info):
        result = super().annotate(info)
        result.update(reference_delta_scale=self.delta_scale,
                      reference_seed_command=self.seed_command.tolist(),
                      control_scope='Frozen 26-action teacher with five trainable reference deltas around the fixed seed. Total reference bounds unchanged. Not a 26-action PPO comparison.')
        return result
