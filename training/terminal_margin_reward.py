"""Training-only terminal margin reward; acceptance, dynamics and observations unchanged.

Motivation (measured on the 2026-10-27 geometry run): the per-step closure quality bonus is worth
only 0.29 reward for a 0.3 mrad improvement, i.e. 0.7% of the +-20 terminal step, while the failing
geometries need 0.2-0.45 mrad. This wrapper adds one bounded term at the terminal step:

    r_terminal += weight * clip(margin / scale, -1, 1)
    margin = min(40 - closure_mrad, (10 - |rise_mm - 250|) * 10)        # the audited scaled margin

With weight=6 and scale=1.5 mrad a 0.3 mrad improvement is worth 1.20 (3% of the terminal step); the
term is bounded to +-6 and does not accumulate over steps, so the episode return scale stays ~90.

Early termination (swing contact, joint violation) is NOT scored: the +-20 terminal step already
carries that failure and there is no endpoint to measure.
"""
import numpy as np

import gymnasium as gym

from training.strict_full_cycle_residual_env import CONTRACT

CLOSURE_LIMIT_MRAD = 40.
RISE_CENTRE_MM = 250.
RISE_LIMIT_MM = 260.


def margin_mrad(closure_rad, rise_m):
    """Audited acceptance margin in scaled units: 1 unit = 1 mrad of closure or 0.1 mm of rise."""
    if not np.isfinite([closure_rad, rise_m]).all() or closure_rad < 0:
        raise ValueError('Finite nonnegative closure and finite rise required')
    return float(min(CLOSURE_LIMIT_MRAD - 1000. * closure_rad,
                     (10. - abs(1000. * rise_m - RISE_CENTRE_MM)) * 10.))


def terminal_bonus(closure_rad, rise_m, *, weight=6., scale_mrad=1.5, completed=True):
    if not np.isfinite([weight, scale_mrad]).all() or weight < 0 or scale_mrad <= 0:
        raise ValueError('Nonnegative weight and positive scale required')
    if not completed:
        return 0.
    return float(weight * np.clip(margin_mrad(closure_rad, rise_m) / scale_mrad, -1., 1.))


def _strict_env(env):
    """Resolve the strict endpoint environment by walking the wrapper chain."""
    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if hasattr(node, 'cycle_results') and hasattr(node, 'ends'):
            return node
        node = getattr(node, 'env', None)
    raise ValueError('Strict endpoint task required')


class TerminalMarginReward(gym.Wrapper):
    def __init__(self, env, *, weight=6., scale_mrad=1.5):
        super().__init__(env)
        # The strict endpoint env is only reachable after reset(), so nothing to resolve here.
        terminal_bonus(.03, .25, weight=weight, scale_mrad=scale_mrad)
        self.weight = float(weight)
        self.scale_mrad = float(scale_mrad)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            base = _strict_env(self.env)
            complete = len(base.cycle_results) == len(base.ends)
            physical = base.env
            qpos, initial = physical.data.qpos, physical.initial
            closure = float(np.abs(qpos[physical.qadr] - initial[physical.qadr]).max())
            rise = float(qpos[2] - initial[2])
            bonus = terminal_bonus(closure, rise, weight=self.weight,
                                   scale_mrad=self.scale_mrad, completed=complete)
            reward = reward + bonus
            info = dict(info, terminal_margin_bonus=bonus, cycle_complete=complete)
        return observation, reward, terminated, truncated, info
