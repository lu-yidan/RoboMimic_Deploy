"""Train raw PPO actions through the same fixed calibration used in evaluation."""
import numpy as np
from training.boundary_landing_env import BoundaryLandingEnv


class CalibratedLandingEnv(BoundaryLandingEnv):
    def __init__(self,parent,**kwargs):
        super().__init__(parent,**kwargs)
        m=self.manifest;spec=m['closure_action_bias']
        if m.get('policy_adapter')!='closure-action-bias-v2':raise ValueError('Audited v2 calibration required')
        self.calibration_bias=np.asarray(spec['normalized_bias'],dtype=float)
        if self.calibration_bias.shape!=(26,) or not np.isfinite(self.calibration_bias).all() or abs(self.calibration_bias).max()>.5:raise ValueError('Invalid fixed calibration')
        if any(self.calibration_bias[i]!=0 for i in range(26) if i not in (3,4)):raise ValueError('Only knee/ankle calibration supported')
        if spec['start_seconds']!=70.2 or spec['ramp_seconds']!=3. or spec['geometry_gate']!='existing_high_rod' or m['horizon_seconds']!=113.66:raise ValueError('Unsupported calibration timing')
        self.last_executed_action=None

    def step(self,action):
        raw=np.asarray(action)
        if raw.shape!=(26,) or not np.isfinite(raw).all() or abs(raw).max()>1.:raise ValueError('Expected bounded raw PPO action')
        obs=self.last_obs
        elapsed=113.66*(1.-float(obs[105]));u=float(np.clip((elapsed-70.2)/3.,0.,1.))
        gain=float(np.clip((float(obs[138])-.625)/.125,0.,1.))
        executed=raw if u==0. or gain==0. or not np.any(self.calibration_bias) else np.clip(np.asarray(raw,dtype=float)+u*u*(3.-2.*u)*gain*self.calibration_bias,-1.,1.).astype(np.float32)
        self.last_executed_action=executed.copy()
        return super().step(executed)
