"""Positive terminal quality shaping; physical task and acceptance unchanged.

The nonnegative bonus avoids adding an incentive to terminate early to escape
an accumulated closure penalty. This changes the objective; it is not claimed
to be policy-invariant potential shaping.
"""
import math
import gymnasium as gym
import numpy as np
from training.strict_full_cycle_residual_env import CONTRACT


def quality_bonus(closure_rad, rise_m, ramp, *, weight=.1):
    if not all(math.isfinite(x) for x in (closure_rad, rise_m, ramp, weight)):
        raise ValueError('Finite reward inputs required')
    if closure_rad < 0 or not 0 <= ramp <= 1 or weight < 0:
        raise ValueError('Invalid closure, ramp or weight')
    joint = max(0., closure_rad - .032) / .008
    height = max(0., abs(rise_m - .25) - .006) / .004
    return float(.5 * weight * ramp * (1. / (1. + joint * joint) +
                                       1. / (1. + height * height)))


class ClosureQualityReward(gym.Wrapper):
    def __init__(self, env, *, weight=.1):
        super().__init__(env)
        base = env.unwrapped
        if base.contract != CONTRACT or len(base.ends) != 1:
            raise ValueError('Single-cycle strict task required')
        quality_bonus(0., .25, 0., weight=weight)
        self.weight = float(weight)
        indices = np.flatnonzero(base.plan['phase'] == 'posture_closure')
        if not len(indices) or indices[0] == 0:
            raise ValueError('Posture closure phase required')
        self.start = float(base.plan['time'][indices[0] - 1])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        physical = self.env.unwrapped.env
        q, initial = physical.data.qpos, physical.initial
        closure = float(abs(q[physical.qadr] - initial[physical.qadr]).max())
        rise = float(q[2] - initial[2])
        u = float(np.clip((physical.data.time - self.start) / 3., 0., 1.))
        ramp = u * u * (3 - 2 * u)
        bonus = quality_bonus(closure, rise, ramp, weight=self.weight)
        return obs, reward + bonus, terminated, truncated, dict(info,
            base_reward=float(reward), closure_quality_bonus=bonus,
            current_closure_rad=closure, current_rise_m=rise, closure_cost_ramp=ramp)
