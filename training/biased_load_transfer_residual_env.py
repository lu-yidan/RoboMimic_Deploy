"""Residual learning with fixed-per-episode contact-normal calibration errors.

The actor never receives the calibration error. Contact identity and force
feedback remain ideal, so this is not a real perception deployment claim.
"""
import gymnasium as gym
import numpy as np
from training.contact_load_transfer_residual_env import LoadTransferResidualEnv
from training.biased_contact_frame_controller import BiasedContactFrameController

CASES = ['nominal', 'row01_z_+5mm', 'row01_x_-10mm']
TRAIN_ANGLES = [0., -3., 3.]
EVAL_ANGLES = [0., -3., 3., -1.5, 1.5]


def scenario_name(case, angle):
    return f'{case}__normal_{float(angle):+g}deg'


class BiasedLoadTransferResidualEnv(LoadTransferResidualEnv):
    def __init__(self, suite):
        super().__init__(suite, CASES)
        self.scenarios = [(case, angle) for angle in TRAIN_ANGLES for case in CASES]
        self.probabilities = np.full(len(self.scenarios), .5 / (len(self.scenarios) - 1))
        self.probabilities[self.scenarios.index(('row01_z_+5mm', -3.))] = .5

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if options is not None:
            case = options['case']; angle = float(options['normal_bias_y_deg'])
            if case not in CASES or not np.isfinite(angle) or abs(angle) > 3.:
                raise ValueError('Scenario outside declared geometry/normal-bias range')
        else:
            case, angle = self.scenarios[int(self.np_random.choice(len(self.scenarios), p=self.probabilities))]
        obs, info = super().reset(seed=None, options={'case': case})
        self.normal_bias_y_deg = angle
        self.scenario = scenario_name(case, angle)
        self.controller = BiasedContactFrameController(self.env, torque_weight=10000., normal_bias_y_deg=angle)
        info.update(scenario=self.scenario, normal_bias_y_deg=angle, contract='biased-load-transfer-v1-76obs-6act')
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(scenario=self.scenario, normal_bias_y_deg=self.normal_bias_y_deg,
                    controller_assumption='Fixed biased normals; exact contact points/topology and ideal force feedback',
                    scope='49..61.56s right-leg residual; finite prefix/angle distribution; no full-cycle learning claim')
        return obs, reward, terminated, truncated, info
