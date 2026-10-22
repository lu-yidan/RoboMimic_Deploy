import unittest
import numpy as np
from training.contact_frame_controller import ContactFrameController
from training.biased_contact_frame_controller import BiasedContactFrameController
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.rung_residual_env import ROOT


class ContactBiasTests(unittest.TestCase):
    def test_zero_bias_matches_oracle_and_bias_does_not_mutate_geometry(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',['nominal'])
        try:
            task.reset(options={'case':'nominal'});e=task.env;q,v,active=task.reference(e.data.time)
            expected,_=ContactFrameController(e,torque_weight=10000.).command(q,v,active)
            actual,_=BiasedContactFrameController(e,torque_weight=10000.).command(q,v,active)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
            geom=e.model.geom_pos.copy();q0=e.data.qpos.copy();v0=e.data.qvel.copy()
            positions=np.array([c.pos.copy() for c in e.data.contact]);frames=np.array([c.frame.copy() for c in e.data.contact])
            biased=BiasedContactFrameController(e,torque_weight=10000.,point_bias_x_m=.001,normal_bias_y_deg=1.)
            biased.command(q,v,active)
            np.testing.assert_array_equal(e.model.geom_pos,geom)
            np.testing.assert_array_equal(e.data.qpos,q0);np.testing.assert_array_equal(e.data.qvel,v0)
            np.testing.assert_array_equal(np.array([c.pos for c in e.data.contact]),positions)
            np.testing.assert_array_equal(np.array([c.frame for c in e.data.contact]),frames)
            np.testing.assert_allclose(biased.normal_rotation.T@biased.normal_rotation,np.eye(3),atol=1e-14)
            with self.assertRaises(ValueError):BiasedContactFrameController(e,point_bias_x_m=np.nan)
        finally:task.close()


if __name__=='__main__':unittest.main()
