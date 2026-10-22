"""Recorded-state endpoint classification tests, not new dynamic simulations."""
import unittest
import mujoco
import numpy as np
from training.full_cycle_residual_env import ROOT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints


class StrictEndpointTest(unittest.TestCase):
    def check_case(self, path, expected):
        env = FullCycleResidualEnv()
        try:
            env.reset()
            with np.load(path) as data:
                z = {key: data[key].copy() for key in data.files}
            independent = audit_endpoints(env.env, z, env.plan)[0]
            valid = []
            for i in np.flatnonzero(z['time_s'] > env.ends[0]-2.):
                env.env.data.qpos[:] = z['qpos'][i]
                env.env.data.qvel[:] = z['qvel'][i]
                mujoco.mj_kinematics(env.env.model, env.env.data)
                valid.append(env.strict_sample(0)[0])
            self.assertEqual(all(valid), expected)
            # These fixtures isolate the extra checks; the old closure test is
            # evaluated separately by the parent and independent audit.
            if expected:
                self.assertTrue(independent['passed'])
            else:
                self.assertFalse(independent['passed'])
        finally:
            env.close()

    def test_nominal_accepted(self):
        self.check_case(ROOT/'outputs/full_cycle_ppo8192_v2/zero_nominal.npz', True)

    def test_low5_false_positive_rejected(self):
        self.check_case(ROOT/'outputs/full_cycle_curriculum_zero_v1/foot_z_minus5/zero_foot_z_minus5.npz', False)

    def test_low10_rejected(self):
        self.check_case(ROOT/'outputs/full_cycle_curriculum_zero_v1/foot_z_minus10/zero_foot_z_minus10.npz', False)


if __name__ == '__main__':
    unittest.main()
