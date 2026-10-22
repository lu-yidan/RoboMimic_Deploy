import unittest
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from training.strict_full_cycle_residual_env import CONTRACT
from training.endpoint_potential_wrapper import EndpointPotentialWrapper


class ScriptedEnv(gym.Env):
    """Exercise reward lifecycle without attributing these tests to physics."""
    contract = CONTRACT
    observation_space = gym.spaces.Box(-1., 1., (1,))
    action_space = gym.spaces.Box(-1., 1., (1,))
    ends = np.array([5.])
    plan = {'time': np.array([0., 1., 5.]),
            'phase': np.array(['start', 'posture_closure', 'closed_hold'])}
    endpoint_bodies = [0]
    desired_poses = [np.zeros((1, 3))]

    def __init__(self, *, success=False, truncate=False):
        self.success, self.truncate = success, truncate
        self.env = SimpleNamespace(data=SimpleNamespace(time=0., xpos=np.array([[.02, 0., 0.]])))

    def reset(self, **kwargs):
        self.env.data.time = 0.
        return np.zeros(1), {}

    def step(self, action):
        self.env.data.time += 1.
        done = self.env.data.time == 5.
        return np.zeros(1), -20. if done else .02, done and not self.truncate, done and self.truncate, {'success': done and self.success}


class WrapperTest(unittest.TestCase):
    def test_true_terminal_success_and_failure_and_batch_boundary(self):
        for success in (False, True):
            env = EndpointPotentialWrapper(ScriptedEnv(success=success), gamma=.9995)
            env.reset()
            shaped = []
            for _ in range(5):
                _, _, done, _, info = env.step(np.zeros(1))
                shaped.append(info['shaping_reward'])
                if not done:
                    self.assertEqual(env.previous_potential, info['reward_potential'])
                    self.assertLess(env.previous_potential, 0.)
            self.assertEqual(info['reward_potential'], 0.)
            self.assertAlmostEqual(float(.9995**np.arange(5)@shaped), 0., places=12)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(1))
            env.reset()
            self.assertEqual(env.previous_potential, 0.)

    def test_unexpected_truncation_rejected(self):
        env = EndpointPotentialWrapper(ScriptedEnv(truncate=True), gamma=.9995)
        env.reset()
        for _ in range(4):
            env.step(np.zeros(1))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected truncation'):
            env.step(np.zeros(1))


if __name__ == '__main__':
    unittest.main()
