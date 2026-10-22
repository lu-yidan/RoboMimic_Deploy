"""Training contract, reset isolation, safety termination and cycle boundary tests."""
import unittest
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv, ROOT


class FullCycleTrainingTest(unittest.TestCase):
    def setUp(self):
        self.env = FullCycleResidualEnv(cases={'nominal': {}, 'raised': {'rung_01': [0., 0., .005]}})
        self.env.reset(seed=7, options={'case': 'nominal'})

    def tearDown(self):
        self.env.close()

    def test_contract_and_repeated_reset(self):
        e = self.env
        first = e.env.data.qpos.copy()
        obs, reward, done, truncated, info = e.step(np.zeros(26))
        self.assertTrue(e.observation_space.contains(obs))
        self.assertFalse(done or truncated)
        self.assertAlmostEqual(info['time_s'], .02)
        e.reset(options={'case': 'raised'})
        self.assertEqual(e.env.data.time, 0.)
        e.reset(options={'case': 'nominal'})
        np.testing.assert_array_equal(first, e.env.data.qpos)
        np.testing.assert_array_equal(e.nominal_geometry[0], e.env.model.geom_pos)
        np.testing.assert_array_equal(e.nominal_geometry[2], e.env.model.bvh_aabb)
        self.assertTrue(np.all(e.residual == 0.))

    def test_bad_action_does_not_advance(self):
        for action in (np.zeros(6), np.full(26, np.nan), np.full(26, np.inf)):
            with self.assertRaises(ValueError):
                self.env.step(action)
            self.assertEqual(self.env.env.data.time, 0.)

    def test_native_limit_fails_inside_control_tick(self):
        e = self.env.env
        e.data.qpos[e.qadr[0]] = e.model.jnt_range[e.joints[0], 1]+.02
        mujoco.mj_forward(e.model, e.data)
        self.env.baseline.command = lambda *args: (np.zeros(29), {'torque_ratio': 0.})
        _, reward, done, truncated, info = self.env.step(np.zeros(26))
        self.assertTrue(done)
        self.assertFalse(truncated or info['success'])
        self.assertEqual(info['reason'], 'native_joint_position_violation')
        self.assertLess(info['time_s'], .005)
        self.assertLess(reward, 0.)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(26))

    def test_external_force_rejected(self):
        self.env.env.data.qfrc_applied[0] = 1.
        with self.assertRaisesRegex(AssertionError, 'External applied force'):
            self.env.step(np.zeros(26))

    def test_time_limit_cannot_report_success_without_endpoints(self):
        # A time jump is not a valid reset and cannot satisfy a 2-second tail.
        self.env.env.data.time = self.env.horizon-.02
        self.env.tail.clear()
        self.env.env.data.time = self.env.horizon
        self.assertFalse(self.env.check_endpoint())
        self.assertFalse(self.env.cycle_results[-1]['passed'])

    def test_cross_cycle_controller_reset_preserves_physics(self):
        e = FullCycleResidualEnv(plan=ROOT/'outputs/clearance_two_cycle_plan_v1/plan.npz')
        try:
            e.reset()
            e.contact.leg_kp = 300.
            e.transfer_entered = True
            e.controller_cycle = 0
            e.env.data.time = e.ends[0]
            before_q, before_v = e.env.data.qpos.copy(), e.env.data.qvel.copy()
            _, _, active, phase = e.reference(float(e.env.data.time))
            controller = e.select_controller(active, phase)
            self.assertIs(controller, e.baseline)
            self.assertEqual(e.contact.leg_kp, 100.)
            self.assertFalse(e.transfer_entered)
            np.testing.assert_array_equal(before_q, e.env.data.qpos)
            np.testing.assert_array_equal(before_v, e.env.data.qvel)
        finally:
            e.close()


if __name__ == '__main__':
    unittest.main()
