"""Finite-case sampling without replacement within each shuffled case block."""
import gymnasium as gym
import numpy as np


class BalancedCaseReset(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.case_names = list(env.unwrapped.cases)
        if not self.case_names:
            raise ValueError('A nonempty declared case set is required')
        self.order_rng = None
        self.pending_cases = []

    def reset(self, *, seed=None, options=None):
        if seed is not None or self.order_rng is None:
            self.order_rng = np.random.default_rng(seed)
            self.pending_cases = []
        options = dict(options or {})
        if 'case' not in options:
            if not self.pending_cases:
                self.pending_cases = self.order_rng.permutation(self.case_names).tolist()
            options['case'] = self.pending_cases.pop(0)
        obs, info = self.env.reset(seed=seed, options=options)
        info['reset_sampling'] = 'One visit per declared case in each independently shuffled block'
        return obs, info
