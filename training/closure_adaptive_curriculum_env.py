"""Four-action reference feedback over a frozen 26-action closure teacher.

The 152 inputs are the existing 148 observations plus four normalized filter
states. Cached state restoration is confined to episode reset. Evaluation must
start at time zero and reproduce the frozen prefix without state replacement.
"""
import hashlib
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from training.closure_adaptive_reference import install_adaptive_reference, SCALES

OBSERVATION_CONTRACT='closure-adaptive-reference-152obs-4act-v1'


class ClosureAdaptiveCurriculumEnv(ClosureCombinedCurriculumEnv):
    def __init__(self,*args,teacher_checkpoint,teacher_sha256,**kwargs):
        path=Path(teacher_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=teacher_sha256:
            raise ValueError('Frozen teacher checkpoint changed')
        self.teacher=PPO.load(path,device='cpu')
        if self.teacher.observation_space.shape!=(148,) or self.teacher.action_space.shape!=(26,):
            raise ValueError('148-input, 26-action teacher required')
        self.teacher_sha256=teacher_sha256
        self.feedback_state=None
        super().__init__(*args,**kwargs)
        self.observation_space=gym.spaces.Box(
            low=np.r_[self.observation_space.low,np.full(4,-1.)].astype(np.float32),
            high=np.r_[self.observation_space.high,np.full(4,1.)].astype(np.float32),dtype=np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,(4,),dtype=np.float32)

    def feedback_observation(self,observation):
        return np.r_[observation,self.feedback_state['residual']/SCALES].astype(np.float32)

    def reset(self,**kwargs):
        self.feedback_state=None
        obs,info=super().reset(**kwargs)
        self.feedback_state=install_adaptive_reference(self.env.unwrapped,self.controller_modifier,
            geometry_gain=self.controller_geometry_gain,record=self.trace is not None)
        if self.trace is not None:
            self.trace.update(adaptive_actions=[],adaptive_observations=[],adaptive_reference_trace=self.feedback_state['trace'])
        return self.feedback_observation(obs),self.annotate(info)

    def step(self,action):
        if self.feedback_state is None:
            raise RuntimeError('Reset required')
        self.feedback_state['set_action'](action)
        if self.trace is not None:
            self.trace['adaptive_actions'].append(self.feedback_state['command'].copy())
            self.trace['adaptive_observations'].append(self.feedback_observation(self.last_obs))
        teacher_action=self.teacher.predict(self.last_obs,deterministic=True)[0]
        obs,reward,done,truncated,info=super().step(teacher_action)
        return self.feedback_observation(obs),reward,done,truncated,self.annotate(info)

    def annotate(self,info):
        result=super().annotate(info)
        result.update(adaptive_observation_contract=OBSERVATION_CONTRACT,teacher_sha256=self.teacher_sha256,
            control_scope='Frozen 26D teacher plus four learned reference commands; new action authority, not a pure 26D PPO comparison.',
            recorded_action_arrays='actions/observations are the frozen teacher; adaptive_actions/adaptive_observations are the four-action policy.')
        return result
