import unittest
from pathlib import Path
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.probe_hook_loading import capped_command

ROOT=Path(__file__).resolve().parents[1]


class ProbeQPTest(unittest.TestCase):
    def setUp(self):
        self.env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');self.env.reset()
        self.q=np.load(ROOT/'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz')['qpos'][0].copy()
        self.env.data.qpos[:]=self.q;mujoco.mj_forward(self.env.model,self.env.data)
        self.v=np.zeros(self.env.model.nv)

    def tearDown(self):self.env.close()

    def controller(self):return CycleTorqueController(self.env,joint_kp=100.,leg_kp=100.,torque_weight=10000.)

    def test_default_caps_are_bitwise_original_qp(self):
        for active in ([True,True,True,True],[True,True,False,True]):
            original=self.controller().command(self.q,self.v,active)
            capped=capped_command(self.controller(),self.q,self.v,active,np.full(4,350.))
            np.testing.assert_array_equal(original[0],capped[0])
            np.testing.assert_array_equal(original[1]['force_plan'],capped[1]['force_plan'])

    def test_probe_caps_and_native_torque_limits(self):
        torque,details=capped_command(self.controller(),self.q,self.v,np.ones(4,bool),[350,350,7.5,350])
        force=np.array(details['force_plan']).reshape(-1,3)
        self.assertLessEqual(float(force[4:6,2].max()),7.5+1e-3)
        self.assertTrue(np.all(torque<=self.env.limits[:,1]) and np.all(torque>=self.env.limits[:,0]))
        self.assertLessEqual(details['base_equilibrium_residual'],1e-3)

    def test_invalid_cap_rejected(self):
        with self.assertRaises(ValueError):capped_command(self.controller(),self.q,self.v,np.ones(4,bool),[350,350,-1,350])


if __name__=='__main__':unittest.main()
