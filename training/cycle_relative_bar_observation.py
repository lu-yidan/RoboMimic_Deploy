"""Shift privileged rod-offset roles with the active cycle; no camera claim."""
import numpy as np
from training.bar_geometry_observation import BarGeometryObservation, ROWS
from training.cycle_clock import cycle_index


class CycleRelativeBarObservation(BarGeometryObservation):
    def __init__(self, env, *, mode):
        super().__init__(env, mode=mode)
        model = env.env.model
        self.role_rows = [[f'rung_{int(row.split("_")[1])+cycle:02d}' for row in ROWS]
                          for cycle in range(len(env.ends))]
        self.role_sites = np.array([[model.site(row+'_target').id for row in rows] for rows in self.role_rows])
        if np.any(model.site_bodyid[self.role_sites] != 0):
            raise ValueError('World-fixed cycle-relative target sites required')
        self.role_nominal = env.nominal_geometry[1][self.role_sites].copy()

    def observation(self, observation):
        index = min(cycle_index(float(self.env.env.data.time), self.env.ends), len(self.env.ends)-1)
        offsets = (self.env.env.model.site_pos[self.role_sites[index]]-self.role_nominal[index])/.02
        if not np.isfinite(offsets).all() or abs(offsets).max() > 1.+1e-10:
            raise ValueError('Cycle-relative offset outside the declared 20 mm domain')
        features = offsets.reshape(-1) if self.mode == 'oracle' else np.zeros(12)
        return np.r_[observation, features].astype(np.float32)
