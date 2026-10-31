"""Deterministic bounded action-space diagnostic, not a trained policy."""
import numpy as np


class ClosureGeometryBiasPolicy:
    def __init__(self, policy, manifest):
        self.policy=policy
        self.bias=np.asarray(manifest['closure_action_bias'],dtype=np.float64)
        if self.bias.shape!=(26,) or not np.isfinite(self.bias).all() or abs(self.bias).max()>1.:
            raise ValueError('Bounded 26D diagnostic bias required')
        activation=manifest['geometry_bias_activation']
        if manifest['geometry_mode']!='oracle' or activation!={'lower_m':.0125,'upper_m':.015,'feature_index':138,'scale_m':.02}:
            raise ValueError('Declared existing oracle geometry feature required')
        self.start=float(manifest['suffix_start_seconds'])
        self.horizon=float(manifest['horizon_seconds'])

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic:raise ValueError('Diagnostic is deterministic only')
        obs=np.asarray(observation)
        action,state=self.policy.predict(obs,state=state,episode_start=episode_start,deterministic=True)
        t=self.horizon*(1.-obs[...,105].astype(np.float64))
        u=np.clip((t-self.start)/3.,0.,1.)
        ramp=u*u*(3.-2.*u)
        gain=np.clip((obs[...,138].astype(np.float64)-.625)/.125,0.,1.)
        output=np.clip(action+np.expand_dims(ramp*gain,-1)*self.bias,-1.,1.).astype(np.float32)
        return output,state
