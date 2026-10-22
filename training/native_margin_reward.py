"""Optional training-only bounded cost near native joint limits.

This intentionally changes the training objective; it is NOT potential-based
reward shaping. Strict task acceptance and the physical joint bounds stay fixed.
"""
import gymnasium as gym
import numpy as np
from training.strict_full_cycle_residual_env import CONTRACT


def margin_cost(margin_rad, *, width_rad=.02, weight=.01):
    if not np.isfinite([margin_rad,width_rad,weight]).all() or width_rad<=0 or weight<0:
        raise ValueError('Finite margin, positive width and nonnegative weight required')
    return float(weight*np.clip((width_rad-margin_rad)/width_rad,0.,1.)**2)


class NativeMarginReward(gym.Wrapper):
    def __init__(self,env,*,width_rad=.02,weight=.01):
        super().__init__(env)
        if env.unwrapped.contract!=CONTRACT:raise ValueError('Strict endpoint task required')
        margin_cost(0.,width_rad=width_rad,weight=weight)
        self.width_rad,self.weight=width_rad,weight

    def step(self,action):
        observation,base,terminated,truncated,info=self.env.step(action)
        physical=self.env.unwrapped.env
        q=physical.data.qpos[physical.qadr]
        bounds=physical.model.jnt_range[physical.joints]
        # Current state, not the episode's cumulative minimum margin.
        margin=float(np.minimum(q-bounds[:,0],bounds[:,1]-q).min())
        cost=margin_cost(margin,width_rad=self.width_rad,weight=self.weight)
        return observation,base-cost,terminated,truncated,dict(info,
            base_reward=float(base),native_margin_reward_cost=cost,current_native_margin_rad=margin)
