import unittest
import mujoco
import numpy as np
from training.contact_frame_controller import ContactFrameController
from training.probing_contact_frame_controller import ProbingContactFrameController
from training.contact_landing import ContactLanding
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.rung_residual_env import ROOT


class LandingTests(unittest.TestCase):
    def test_default_equivalence_and_zero_caps_cannot_support_weight(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',['nominal'])
        try:
            task.reset(options={'case':'nominal'});e=task.env;q,v,active=task.reference(e.data.time)
            expected,_=ContactFrameController(e,torque_weight=10000.).command(q,v,active)
            c=ProbingContactFrameController(e,torque_weight=10000.);actual,_=c.command(q,v,active)
            np.testing.assert_allclose(expected,actual,rtol=0,atol=1e-8)
            q0=e.data.qpos.copy();v0=e.data.qvel.copy()
            c.normal_caps=dict.fromkeys(['lf','rf','lh','rh'],0.)
            with self.assertRaises(RuntimeError):c.command(q,v,active)
            np.testing.assert_array_equal(e.data.qpos,q0);np.testing.assert_array_equal(e.data.qvel,v0)
            c.normal_caps={'lf':351.}
            with self.assertRaises(ValueError):c.command(q,v,active)
        finally:task.close()

    def test_missing_foot_contact_searches_without_state_write_and_blocks_body_rise(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/contact_frame_continued_remaining_v1',['row01_z_-10mm'])
        try:
            task.reset(options={'case':'row01_z_-10mm'});e=task.env
            z=np.load(ROOT/'outputs/contact_frame_continued_remaining_v1/row01_z_-10mm/trajectory.npz')
            i=int(np.argmin(abs(z['time_s']-79.72)))
            e.data.qpos[:]=z['qpos'][i];e.data.qvel[:]=z['qvel'][i];e.data.time=79.72;mujoco.mj_forward(e.model,e.data)
            q,v,active=task.reference(79.720001);q0=e.data.qpos.copy();v0=e.data.qvel.copy()
            c=ProbingContactFrameController(e,torque_weight=10000.);landing=ContactLanding(e)
            changed,velocity,enabled=landing.update(q,v,active,'lf_load',c)
            self.assertFalse(enabled[0]);self.assertEqual(landing.mode,'searching')
            self.assertAlmostEqual(landing.depth,.0001)
            np.testing.assert_array_equal(e.data.qpos,q0);np.testing.assert_array_equal(e.data.qvel,v0)
            with self.assertRaisesRegex(RuntimeError,'before body rise'):
                landing.update(q,v,active,'body_rise',c)
        finally:task.close()


if __name__=='__main__':unittest.main()
