from pathlib import Path
import unittest
import numpy as np
from training.foot_residual_env import FootResidualEnv, ROOT


class FootResidualTests(unittest.TestCase):
    def test_reset_restores_nominal_bvh_and_recorded_velocity(self):
        e=FootResidualEnv(ROOT/'outputs/clearance_back20_leg14_dynamics_v1', ['nominal','row01_x_-5mm'])
        try:
            e.reset(seed=1,options={'case':'row01_x_-5mm'})
            self.assertFalse(np.array_equal(e.env.model.bvh_aabb,e.nominal_bvh))
            obs,_=e.reset(seed=1,options={'case':'nominal'})
            np.testing.assert_array_equal(e.env.model.bvh_aabb,e.nominal_bvh)
            np.testing.assert_array_equal(e.env.data.qpos,e.prefixes['nominal'][0])
            np.testing.assert_array_equal(e.env.data.qvel,e.prefixes['nominal'][1])
            self.assertEqual(obs.shape,(76,));self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(e.action_space.shape,(6,))
            self.assertEqual(e.guard.expected,['rung_00','rung_00','rung_05','rung_05'])
            self.assertFalse(np.any(e.env.data.qfrc_applied));self.assertFalse(np.any(e.env.data.xfrc_applied))
        finally:e.close()


if __name__=='__main__':unittest.main()
