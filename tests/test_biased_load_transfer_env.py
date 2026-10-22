import unittest
from pathlib import Path
import numpy as np
from training.biased_load_transfer_residual_env import BiasedLoadTransferResidualEnv

ROOT = Path(__file__).resolve().parents[1]


class BiasedResidualTest(unittest.TestCase):
    def test_bias_is_not_actor_label_or_physics_mutation(self):
        env = BiasedLoadTransferResidualEnv(ROOT / 'outputs/early_contact_reset_grid_v1')
        try:
            obs0, _ = env.reset(seed=9, options={'case': 'nominal', 'normal_bias_y_deg': 0.})
            q0 = env.env.data.qpos.copy(); geometry = env.env.model.geom_pos.copy()
            obs1, info = env.reset(seed=9, options={'case': 'nominal', 'normal_bias_y_deg': -3.})
            np.testing.assert_array_equal(obs0, obs1)
            np.testing.assert_array_equal(q0, env.env.data.qpos)
            np.testing.assert_array_equal(geometry, env.env.model.geom_pos)
            self.assertEqual(info['normal_bias_y_deg'], -3.)
            self.assertEqual(obs1.shape, (76,))
            _, _, terminal, timeout, result = env.step(np.zeros(6))
            self.assertFalse(terminal or timeout)
            self.assertGreater(result['minimum_native_joint_margin_rad'], 0.)
            with self.assertRaises(ValueError):
                env.reset(options={'case': 'nominal', 'normal_bias_y_deg': 4.})
        finally:
            env.close()

    def test_seeded_scenario_choice_reproducible(self):
        env = BiasedLoadTransferResidualEnv(ROOT / 'outputs/early_contact_reset_grid_v1')
        try:
            first, info1 = env.reset(seed=123)
            second, info2 = env.reset(seed=123)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(info1['scenario'], info2['scenario'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
