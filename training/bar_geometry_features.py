"""Preserve pretrained136-input network through a zero-initialized geometry branch."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BarGeometryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        if observation_space.shape != (148,):
            raise ValueError('Expected148 observation dimensions')
        super().__init__(observation_space, features_dim=136)
        self.geometry_projection=nn.Linear(12,136,bias=False)
        nn.init.zeros_(self.geometry_projection.weight)

    def forward(self, observations):
        return observations[:,:136]+self.geometry_projection(observations[:,136:])
