"""Verify the isolated actual-contact residual contract and native-limit stop."""
import unittest
from pathlib import Path
import numpy as np
import mujoco
from training.contact_load_transfer_residual_env import LoadTransferResidualEnv
from training.contact_frame_controller import ContactFrameController

ROOT = Path(__file__).resolve().parents[1]


class ContactLoadTransferTest(unittest.TestCase):
    def setUp(self):
        self.env = LoadTransferResidualEnv(ROOT / 'outputs/early_contact_reset_grid_v1', ['nominal'])
        self.obs, self.info = self.env.reset(seed=7, options={'case': 'nominal'})

    def tearDown(self):
        self.env.close()

    def test_contract_and_zero_step(self):
        self.assertEqual(self.obs.shape, (76,))
        self.assertEqual(self.env.action_space.shape, (6,))
        self.assertIsInstance(self.env.controller, ContactFrameController)
        obs, reward, terminal, timeout, info = self.env.step(np.zeros(6))
        self.assertFalse(terminal)
        self.assertFalse(timeout)
        self.assertTrue(np.isfinite(obs).all())
        self.assertGreater(info['minimum_native_joint_margin_rad'], 0)
        self.assertAlmostEqual(info['time_s'], 49.02, places=8)

    def test_position_violation_stops_inside_policy_step(self):
        e = self.env.env
        e.data.qpos[e.qadr[0]] = e.model.jnt_range[e.joints[0], 1] + .02
        mujoco.mj_forward(e.model, e.data)
        self.env.controller.command = lambda *args: (np.zeros(29), {'torque_ratio': 0.})
        _, reward, terminal, timeout, info = self.env.step(np.zeros(6))
        self.assertTrue(terminal)
        self.assertFalse(timeout)
        self.assertFalse(info['success'])
        self.assertEqual(info['reason'], 'native_joint_position_violation')
        self.assertLess(info['time_s'], 49.005)
        self.assertLess(info['minimum_native_joint_margin_rad'], 0)
        self.assertLess(reward, 0)

    def test_rejects_collision_before_reset(self):
        with self.assertRaisesRegex(ValueError, 'flight_contact_peak_n'):
            LoadTransferResidualEnv(ROOT / 'outputs/early_contact_reset_grid_v1', ['row01_z_+10mm'])


if __name__ == '__main__':
    unittest.main()
