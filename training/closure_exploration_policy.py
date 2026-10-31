"""On-policy Gaussian exploration scaling only during terminal closure.

The raw remaining-time observation (index 105) determines the scale. Sampling,
PPO likelihood evaluation and entropy use the same distribution. Deterministic
actions and value predictions retain the inherited network exactly.
"""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class ClosureExplorationPolicy(ActorCriticPolicy):
    horizon_seconds = 113.66
    closure_start_seconds = 99.66
    ramp_seconds = 3.
    leg_std_multiplier = 4.

    def closure_distribution(self, obs, latent_pi):
        if obs.shape[-1] != 148 or self.action_space.shape != (26,) or self.use_sde:
            raise ValueError('Expected 148-observation, 26-action diagonal Gaussian policy')
        time = self.horizon_seconds * (1. - obs[:, 105])
        u = torch.clamp((time - self.closure_start_seconds) / self.ramp_seconds, 0., 1.)
        ramp = u * u * (3. - 2. * u)
        mask = torch.zeros(26, device=obs.device, dtype=obs.dtype)
        mask[:12] = 1.
        scale = 1. + (self.leg_std_multiplier - 1.) * ramp[:, None] * mask
        return DiagGaussianDistribution(26).proba_distribution(
            self.action_net(latent_pi), self.log_std + torch.log(scale))

    def latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi, vf = features
        return self.mlp_extractor.forward_actor(pi), self.mlp_extractor.forward_critic(vf)

    def forward(self, obs, deterministic=False):
        pi, vf = self.latents(obs)
        distribution = self.closure_distribution(obs, pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(vf), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self.latents(obs)
        distribution = self.closure_distribution(obs, pi)
        return self.value_net(vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        features = BasePolicy.extract_features(self, obs, self.pi_features_extractor)
        return self.closure_distribution(obs, self.mlp_extractor.forward_actor(features))
