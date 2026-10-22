"""Equal-size blind/oracle inputs for a future information ablation.

Oracle offsets are privileged simulation geometry, not a camera measurement.
The strict task and motor residual bounds are unchanged. No trained 136-input
checkpoint can be loaded directly as a 148-input policy.
"""
import gymnasium as gym
import numpy as np
from training.strict_full_cycle_residual_env import CONTRACT

ROWS = ('rung_01', 'rung_02', 'rung_05', 'rung_06')
OBSERVATION_CONTRACT = 'strict-v2-148obs-bar-offset-oracle-or-blind-v1'


class BarGeometryObservation(gym.ObservationWrapper):
    def __init__(self, env, *, mode):
        super().__init__(env)
        if env.contract != CONTRACT or mode not in ('blind', 'oracle'):
            raise ValueError('Strict task with blind or oracle mode required')
        if env.observation_space.shape != (136,):
            raise ValueError('Expected original136-input interface')
        self.mode = mode
        self.observation_contract = OBSERVATION_CONTRACT
        model = env.env.model
        self.sites = np.array([model.site(row+'_target').id for row in ROWS])
        if np.any(model.site_bodyid[self.sites] != 0):
            raise ValueError('Expected world-fixed target sites')
        self.nominal_positions = env.nominal_geometry[1][self.sites].copy()
        self.observation_space = gym.spaces.Box(
            low=np.r_[env.observation_space.low, np.full(12, -1.)].astype(np.float32),
            high=np.r_[env.observation_space.high, np.full(12, 1.)].astype(np.float32),
            dtype=np.float32)

    def observation(self, observation):
        offsets = (self.env.env.model.site_pos[self.sites]-self.nominal_positions)/.02
        if not np.isfinite(offsets).all() or np.max(abs(offsets)) > 1.+1e-10:
            raise ValueError('Bar offset outside declared20mm geometry domain')
        features = offsets.reshape(-1) if self.mode == 'oracle' else np.zeros(12)
        return np.r_[observation, features].astype(np.float32)
