import unittest
from pathlib import Path
import numpy as np
from training.contact_load_transfer_residual_env import LoadTransferResidualEnv
from training.window_contact_controller import WindowContactController
from training.contact_frame_controller import ContactFrameController
from training.biased_contact_frame_controller import BiasedContactFrameController

ROOT=Path(__file__).resolve().parents[1]


class WindowControllerTest(unittest.TestCase):
    def test_delegation_matches_original_controllers_without_state_writes(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/early_contact_reset_grid_v1',['nominal'])
        try:
            task.reset(options={'case':'nominal'});e=task.env;q,v,active=task.reference(e.data.time)
            before=e.data.qpos.copy()
            wrapper=WindowContactController(e,torque_weight=10000.,normal_bias_y_deg=-3.)
            biased=BiasedContactFrameController(e,torque_weight=10000.,normal_bias_y_deg=-3.)
            actual,_=wrapper.command(q,v,active);expected,_=biased.command(q,v,active)
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
            e.data.time=np.nextafter(49.,-np.inf)
            self.assertFalse(wrapper.bias_window_active())
            normal=ContactFrameController(e,torque_weight=10000.)
            actual,_=wrapper.command(q,v,active);expected,_=normal.command(q,v,active)
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
            e.data.time=61.56;self.assertFalse(wrapper.bias_window_active())
            e.data.time=np.nextafter(61.56,-np.inf);self.assertTrue(wrapper.bias_window_active())
            np.testing.assert_array_equal(before,e.data.qpos)
        finally:task.close()


if __name__=='__main__':unittest.main()
