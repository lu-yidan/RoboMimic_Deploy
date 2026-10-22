"""Training-only reward shaping; observation, dynamics and acceptance unchanged."""
import gymnasium as gym
import numpy as np
from training.endpoint_potential import endpoint_potential, potential_reward
from training.strict_full_cycle_residual_env import CONTRACT


class EndpointPotentialWrapper(gym.Wrapper):
    def __init__(self, env, *, gamma, weight=4., ramp_seconds=3.):
        super().__init__(env)
        if env.contract != CONTRACT:
            raise ValueError('Strict endpoint environment required')
        potential_reward(0., 0., gamma)
        endpoint_potential(0., 0., weight=weight)
        if not np.isfinite(ramp_seconds) or ramp_seconds <= 0:
            raise ValueError('Positive finite ramp duration required')
        self.gamma, self.weight, self.ramp_seconds = gamma, weight, ramp_seconds
        self.closure_starts = []
        for k, end in enumerate(env.ends):
            start = 0. if k == 0 else env.ends[k-1]
            indices = np.flatnonzero((env.plan['phase'] == 'posture_closure') &
                                    (env.plan['time'] > start) & (env.plan['time'] <= end))
            if not len(indices) or indices[0] == 0:
                raise ValueError('Reference lacks closure phase')
            self.closure_starts.append(float(env.plan['time'][indices[0]-1]))
        self.previous_potential = None

    def potential(self):
        env = self.env
        t = float(env.env.data.time)
        cycle = min(int(np.searchsorted(env.ends, t, side='right')), len(env.ends)-1)
        u = float(np.clip((t-self.closure_starts[cycle])/self.ramp_seconds, 0., 1.))
        if not u:
            return 0.
        error = float(np.linalg.norm(env.env.data.xpos[env.endpoint_bodies]-env.desired_poses[cycle], axis=1).max())
        return endpoint_potential(error, u*u*(3.-2.*u), weight=self.weight)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential()
        return observation, dict(info, reward_potential=self.previous_potential)

    def step(self, action):
        if self.previous_potential is None:
            raise RuntimeError('Reset required')
        observation, base, terminated, truncated, info = self.env.step(action)
        # This strict task uses true finite-horizon termination. A rollout batch
        # boundary must never reset the potential; external truncation needs a
        # separately specified bootstrap contract.
        if truncated:
            raise RuntimeError('Unexpected truncation: reward contract undefined')
        current = 0. if terminated else self.potential()
        shaping = potential_reward(self.previous_potential, current, self.gamma)
        self.previous_potential = None if terminated else current
        return observation, base+shaping, terminated, truncated, dict(
            info, base_reward=base, shaping_reward=shaping, reward_potential=current)
