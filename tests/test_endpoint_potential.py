import unittest
import numpy as np
from training.endpoint_potential import endpoint_potential, potential_reward


class EndpointPotentialTest(unittest.TestCase):
    def test_bound_terminal_and_invalid_inputs(self):
        self.assertEqual(endpoint_potential(.5, 1.), -36.)
        self.assertEqual(endpoint_potential(.02, 0.), 0.)
        self.assertEqual(endpoint_potential(float('nan'), 1., terminal=True), 0.)
        for error, ramp in [(float('nan'), 0.), (.01, 1.1), (-.01, .5)]:
            with self.assertRaises(ValueError):
                endpoint_potential(error, ramp)

    def test_discounted_return_telescopes_for_success_and_early_failure(self):
        rng = np.random.default_rng(9)
        for length in (1, 37, 5536):
            gamma = .9995
            phi = -rng.uniform(0., 36., length+1)
            phi[-1] = 0.
            rewards = np.array([potential_reward(a, b, gamma) for a, b in zip(phi[:-1], phi[1:])])
            self.assertAlmostEqual(float(gamma**np.arange(length)@rewards), -phi[0], places=10)

    def test_missing_terminal_zero_would_leave_a_policy_dependent_term(self):
        gamma = .9995
        phi = [0., -3., -4.]
        total = potential_reward(phi[0], phi[1], gamma)+gamma*potential_reward(phi[1], phi[2], gamma)
        self.assertAlmostEqual(total, gamma**2*phi[-1])
        self.assertNotEqual(total, 0.)


if __name__ == '__main__':
    unittest.main()
