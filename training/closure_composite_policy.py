"""Deterministic full-cycle deployment of a frozen prefix and learned suffix."""
import hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO


class ClosureCompositePolicy:
    def __init__(self, suffix, manifest):
        path = Path(manifest['prefix_checkpoint'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['prefix_checkpoint_sha256']:
            raise ValueError('Frozen prefix policy changed')
        self.prefix = PPO.load(path, device='cpu')
        self.suffix = suffix
        self.boundary = np.float32(1. - manifest['suffix_start_seconds'] / manifest['horizon_seconds'])

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError('Composite is a deterministic evaluation/deployment adapter')
        obs = np.asarray(observation)
        if obs.ndim == 1:
            if obs.shape != (148,):
                raise ValueError('Expected 148-dimensional observation')
            actor = self.suffix if obs[105] <= self.boundary else self.prefix
            return actor.predict(obs, deterministic=True)
        if obs.ndim != 2 or obs.shape[1] != 148:
            raise ValueError('Expected observation batch')
        return np.stack([self.predict(row)[0] for row in obs]), None
