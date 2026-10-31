"""Geometric and state-preservation checks for offline target displacement."""
from pathlib import Path
import unittest
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.hand_target_reference import retarget_hand_reference

ROOT = Path(__file__).resolve().parents[1]


class HandTargetReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WallSupportEnv(ROOT / 'outputs/wall_support_bundle_pd15_v2')
        cls.env.reset()
        with np.load(ROOT / 'outputs/v2_swing_two_reference_20261029_v1/plan.npz') as z:
            cls.plan = {k: z[k] for k in z.files}

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def run_shift(self, **overrides):
        args = dict(key='rh', offset_m=[.002, 0., .002], start_s=123.66, end_s=125.66)
        args.update(overrides)
        return retarget_hand_reference(self.env, self.plan, **args)

    def test_zero_is_exact(self):
        result, _ = self.run_shift(offset_m=[0., 0., 0.])
        for key in self.plan:
            np.testing.assert_array_equal(result[key], self.plan[key])

    def test_actual_cartesian_shift_and_live_state_preserved(self):
        live = (self.env.data.qpos.copy(), self.env.data.qvel.copy(), self.env.data.time)
        original = self.plan['qpos'].copy()
        result, _ = self.run_shift()
        d = mujoco.MjData(self.env.model)
        body = self.env.model.body('right_hook').id
        # Sample the interior ramp and sustained support, independently of IK residuals.
        indices = np.flatnonzero(self.plan['time'] > 123.66)[::19]
        for index in indices:
            time = self.plan['time'][index]
            u = np.clip((time - 123.66) / 2., 0., 1.)
            weight = 6*u**5 - 15*u**4 + 10*u**3
            d.qpos[:] = original[index]
            mujoco.mj_kinematics(self.env.model, d)
            before = d.xpos[body].copy()
            d.qpos[:] = result['qpos'][index]
            mujoco.mj_kinematics(self.env.model, d)
            np.testing.assert_allclose(d.xpos[body] - before, weight * np.array([.002, 0., .002]), atol=2e-5, rtol=0)
        np.testing.assert_array_equal(self.plan['qpos'], original)
        np.testing.assert_array_equal(self.env.data.qpos, live[0])
        np.testing.assert_array_equal(self.env.data.qvel, live[1])
        self.assertEqual(self.env.data.time, live[2])
        for key in ('time', 'phase', 'active', 'initial', 'cycle_ends_s'):
            np.testing.assert_array_equal(result[key], self.plan[key])

    def test_invalid_or_loaded_ramp_rejected(self):
        for override in ({'key': 'lf'}, {'offset_m': [0., 0., .003]}, {'offset_m': [np.nan, 0., 0.]},
                         {'start_s': 123.661}, {'start_s': 130., 'end_s': 132.}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.run_shift(**override)


if __name__ == '__main__':
    unittest.main()
