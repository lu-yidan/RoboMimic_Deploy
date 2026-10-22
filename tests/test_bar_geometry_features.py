import unittest
import gymnasium as gym
import numpy as np
import torch
from training.bar_geometry_features import BarGeometryFeatures


class GeometryFeatureTest(unittest.TestCase):
    def test_zero_branch_preserves_features_and_oracle_can_learn(self):
        module=BarGeometryFeatures(gym.spaces.Box(-np.inf,np.inf,(148,),dtype=np.float32))
        x=torch.randn(4,148)
        self.assertTrue(torch.equal(module(x),x[:,:136]))
        module(x).sum().backward()
        self.assertGreater(float(module.geometry_projection.weight.grad.abs().max()),0.)

    def test_blind_extra_parameters_do_not_receive_geometry_signal(self):
        module=BarGeometryFeatures(gym.spaces.Box(-np.inf,np.inf,(148,),dtype=np.float32))
        x=torch.randn(4,148)
        x[:,136:]=0.
        module(x).sum().backward()
        self.assertEqual(float(module.geometry_projection.weight.grad.abs().max()),0.)


if __name__=='__main__':unittest.main()
