import unittest
import numpy as np
from tools.evaluate_repeated_full_cycle_policy import cycle_local_observation


class CycleLocalInputTest(unittest.TestCase):
    def test_first_cycle_matches_original_and_preserves_other_features(self):
        obs = np.arange(136, dtype=np.float32)
        for t in (0., .02, 49., 110.70):
            expected = obs.copy()
            expected[105] = max(0., 1.-t/110.72)
            np.testing.assert_array_equal(cycle_local_observation(obs, t, [110.72, 221.44]), expected)
        np.testing.assert_array_equal(obs, np.arange(136, dtype=np.float32))

    def test_boundary_starts_new_cycle_and_final_horizon_is_zero(self):
        obs = np.zeros(136, dtype=np.float32)
        self.assertEqual(cycle_local_observation(obs, 110.72, [110.72, 221.44])[105], 1.)
        self.assertEqual(cycle_local_observation(obs, 166.08, [110.72, 221.44])[105], .5)
        self.assertEqual(cycle_local_observation(obs, 221.44, [110.72, 221.44])[105], 0.)

    def test_eight_cycle_clock_uses_current_cycle(self):
        ends = np.arange(1, 9)*110.72
        obs = np.zeros(136, dtype=np.float32)
        for cycle in range(8):
            t = cycle*110.72+55.36
            self.assertAlmostEqual(float(cycle_local_observation(obs, t, ends)[105]), .5)


if __name__ == '__main__':
    unittest.main()
