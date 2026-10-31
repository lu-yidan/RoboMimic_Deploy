"""Matched boundary curricula; reward choice never changes task acceptance."""
import math
from training.live_prefix_landing_env import LivePrefixLandingEnv
from training.closure_quality_reward import quality_bonus


def boundary_bonus(closure_rad, rise_m, ramp, *, mode, weight=.1):
    original = quality_bonus(closure_rad, rise_m, ramp, weight=weight)
    if mode == 'average':
        return original
    if mode != 'bottleneck':
        raise ValueError('Unknown boundary reward mode')
    joint = max(0., closure_rad - .032) / .008
    height = max(0., abs(rise_m - .25) - .006) / .004
    return float(weight * ramp / (1. + max(joint * joint, height * height)))


class BoundaryLandingEnv(LivePrefixLandingEnv):
    def __init__(self, parent, *, cases=None, weight=.1, record=False, mode='average'):
        boundary_bonus(0., .25, 0., mode=mode, weight=weight)
        super().__init__(parent, cases=cases, weight=weight, record=record)
        self.reward_mode = mode

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        original = info['closure_quality_bonus']
        bonus = boundary_bonus(info['current_closure_rad'], info['current_rise_m'],
                               info['closure_cost_ramp'], mode=self.reward_mode, weight=self.weight)
        if self.reward_mode != 'average':
            reward = info['base_reward'] + bonus
        return obs, reward, done, truncated, dict(info,
            closure_quality_bonus=bonus, average_quality_bonus=original,
            boundary_reward_mode=self.reward_mode)
