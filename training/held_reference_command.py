"""Hold a feedback command while preserving the 50 Hz teacher and physics cadence.

The macro reward discounts each underlying reward at its original time. PPO's
macro discount must be physical_gamma**repeat. This changes feedback decision
frequency only, not the teacher/controller/physics time steps or task gates.
"""
import gymnasium as gym
import numpy as np


class HeldReferenceCommand(gym.Wrapper):
    def __init__(self,env,*,repeat=5,physical_gamma=.9995):
        super().__init__(env)
        if not isinstance(repeat,int) or repeat<1 or repeat>10:
            raise ValueError('Integer action hold of one to ten teacher steps required')
        if not 0.<physical_gamma<=1.:
            raise ValueError('Invalid physical-step discount')
        self.repeat=repeat;self.physical_gamma=float(physical_gamma)
        self.macro_gamma=self.physical_gamma**repeat
        self.teacher_steps=0

    def reset(self,**kwargs):
        self.teacher_steps=0
        obs,info=self.env.reset(**kwargs)
        return obs,dict(info,feedback_hold_steps=self.repeat,feedback_period_seconds=.02*self.repeat)

    def step(self,action):
        value=np.asarray(action,dtype=float)
        if value.shape!=self.action_space.shape or not np.isfinite(value).all():
            raise ValueError('Finite feedback action matching declared shape required')
        discounted=0.;undiscounted=0.
        for index in range(self.repeat):
            obs,reward,done,truncated,info=self.env.step(value)
            discounted+=self.physical_gamma**index*float(reward)
            undiscounted+=float(reward);self.teacher_steps+=1
            if done or truncated:break
        return obs,discounted,done,truncated,dict(info,feedback_hold_steps=self.repeat,
            macro_executed_teacher_steps=index+1,macro_teacher_steps_total=self.teacher_steps,
            macro_undiscounted_reward=undiscounted,macro_discounted_reward=discounted,
            macro_gamma=self.macro_gamma)
