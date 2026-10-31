"""Search-initialized Gaussian residual mean with original observation/action bounds.

Fixed, declared geometric feedforward is added to the trainable actor mean.
This is a combined search initialization and PPO policy, not pure PPO from the
published actor. Sampling and likelihood use the identical shifted Gaussian.
"""
import torch
from training.closure_exploration_policy import ClosureExplorationPolicy


class ClosureSeededPolicy(ClosureExplorationPolicy):
    def __init__(self,*args,closure_seed_bias=None,seed_ramp_seconds=3.,**kwargs):
        super().__init__(*args,**kwargs)
        vector=torch.as_tensor(closure_seed_bias if closure_seed_bias is not None else [0.]*26,dtype=torch.float64)
        if vector.shape!=(26,) or not torch.isfinite(vector).all() or vector.abs().max()>1.:
            raise ValueError('Bounded 26D seed bias required')
        if not 0.<seed_ramp_seconds<=10.:raise ValueError('Ramp duration outside declared interval')
        self.register_buffer('closure_seed_bias',vector)
        self.seed_ramp_seconds=float(seed_ramp_seconds)

    def closure_distribution(self,obs,latent_pi):
        distribution=super().closure_distribution(obs,latent_pi)
        time=113.66*(1.-obs[:,105].to(torch.float64))
        u=torch.clamp((time-99.66)/self.seed_ramp_seconds,0.,1.)
        ramp=u*u*(3.-2.*u)
        gain=torch.clamp((obs[:,138].to(torch.float64)-.625)/.125,0.,1.)
        mean=distribution.distribution.mean
        # Match the frozen parent's bounded command before adding feedforward.
        shifted=(torch.clamp(mean,-1.,1.).to(torch.float64)+(ramp*gain)[:,None]*self.closure_seed_bias).to(mean.dtype)
        distribution.distribution=torch.distributions.Normal(shifted,distribution.distribution.stddev)
        return distribution

    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters()
        data.update(closure_seed_bias=self.closure_seed_bias.detach().cpu().tolist(),seed_ramp_seconds=self.seed_ramp_seconds)
        return data
