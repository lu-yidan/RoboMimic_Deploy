"""Calibrated knee/ankle action offsets inside the existing 26D action bounds.

This is a fixed policy adapter, not a learned PPO improvement. The environment's
0.015-rad residual limit, filtering, physical limits and acceptance stay intact.
"""
import numpy as np
from training.closure_composite_policy import ClosureCompositePolicy


class ClosureActionBiasPolicy:
    def __init__(self, suffix, manifest):
        self.base=ClosureCompositePolicy(suffix,manifest)
        if manifest.get('policy_adapter')!='closure-action-bias-v1':
            raise ValueError('Explicit action-bias policy adapter declaration required')
        spec=manifest['closure_action_bias']
        if spec['geometry_gate']!='existing_high_rod':
            raise ValueError('Use the existing declared high-rod geometry gate')
        self.bias=np.asarray(spec['normalized_bias'],dtype=np.float64)
        self.start=float(spec['start_seconds']);self.ramp=float(spec['ramp_seconds'])
        self.horizon=float(manifest['horizon_seconds'])
        if self.bias.shape!=(26,) or not np.isfinite(self.bias).all() or np.max(abs(self.bias))>.2:
            raise ValueError('Expected 26 finite offsets bounded by 0.2')
        if np.any(self.bias[[i for i in range(26) if i not in (3,4)]]):
            raise ValueError('Only left knee and ankle-pitch action offsets are declared')
        if self.start not in (70.2,99.66) or self.ramp!=3. or self.horizon!=113.66:
            raise ValueError('Unsupported bias timing')

    def predict(self, observation, deterministic=True):
        obs=np.asarray(observation)
        if obs.ndim==2:
            return np.stack([self.predict(row,deterministic)[0] for row in obs]),None
        action,state=self.base.predict(obs,deterministic)
        if not np.any(self.bias):return action,state
        time=self.horizon*(1.-float(obs[105]))
        u=float(np.clip((time-self.start)/self.ramp,0.,1.))
        gain=float(np.clip((float(obs[138])-.625)/.125,0.,1.))
        if u==0. or gain==0.:return action,state
        weight=u*u*(3.-2.*u)*gain
        return np.clip(np.asarray(action,dtype=np.float64)+weight*self.bias,-1.,1.).astype(np.float32),state
