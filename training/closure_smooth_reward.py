"""Prospective bounded closure cost without a hard plateau near observed failures.

This is a separate candidate, not used by the frozen closure-margin experiment.
It still changes the objective and does not change physical acceptance.
"""
import math
import gymnasium as gym
import numpy as np
from training.strict_full_cycle_residual_env import CONTRACT


def bounded_square(value):
    """s²/(1+s²), evaluated without overflow for nonnegative scalar s."""
    if value<1.:
        square=value*value
        return square/(1.+square)
    inverse=1./value
    return 1./(1.+inverse*inverse)


def closure_cost(closure_rad,rise_m,ramp,*,weight=.02):
    if not all(math.isfinite(v) for v in (closure_rad,rise_m,ramp,weight)) or closure_rad<0 or not 0<=ramp<=1 or weight<0:
        raise ValueError('Finite nonnegative closure/weight and ramp in [0,1] required')
    joint=bounded_square(max(0.,closure_rad-.032)/.008)
    height=bounded_square(max(0.,abs(rise_m-.25)-.006)/.004)
    return float(.5*weight*ramp*(joint+height))


class ClosureSmoothReward(gym.Wrapper):
    def __init__(self,env,*,weight=.02):
        super().__init__(env);base=env.unwrapped
        if base.contract!=CONTRACT or len(base.ends)!=1:raise ValueError('Single-cycle strict task required')
        closure_cost(0.,.25,0.,weight=weight);self.weight=float(weight)
        indices=np.flatnonzero(base.plan['phase']=='posture_closure')
        if not len(indices) or indices[0]==0:raise ValueError('Posture closure phase required')
        self.start=float(base.plan['time'][indices[0]-1])

    def step(self,action):
        observation,reward,terminated,truncated,info=self.env.step(action);physical=self.env.unwrapped.env
        q=physical.data.qpos;initial=physical.initial
        closure=float(abs(q[physical.qadr]-initial[physical.qadr]).max());rise=float(q[2]-initial[2])
        u=float(np.clip((physical.data.time-self.start)/3.,0.,1.));ramp=u*u*(3-2*u)
        cost=closure_cost(closure,rise,ramp,weight=self.weight)
        return observation,reward-cost,terminated,truncated,dict(info,base_reward=float(reward),closure_reward_cost=cost,current_closure_rad=closure,current_rise_m=rise,closure_cost_ramp=ramp)
