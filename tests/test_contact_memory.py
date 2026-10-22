import unittest
import mujoco
import numpy as np
from training.contact_memory_controller import ContactMemoryController
from training.axis_robust_contact_controller import AxisRobustContactController
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.rung_residual_env import ROOT


class MemoryTests(unittest.TestCase):
    def test_actual_contact_equivalence_and_memory_is_bounded(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',['nominal'])
        try:
            task.reset(options={'case':'nominal'});e=task.env;q,v,active=task.reference(e.data.time)
            c=ContactMemoryController(e,torque_weight=10000.);expected,_=AxisRobustContactController(e,torque_weight=10000.).command(q,v,active)
            actual,_=c.command(q,v,active);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
            self.assertIn('lh',c.support_memory)
            body=c.ik.bodies['lh'];stamp=c.support_memory['lh']['time_s']
            q0=e.data.qpos.copy();v0=e.data.qvel.copy()
            self.assertTrue(c.memory_points('lh',body,stamp+.005))
            np.testing.assert_array_equal(e.data.qpos,q0);np.testing.assert_array_equal(e.data.qvel,v0)
            with self.assertRaisesRegex(RuntimeError,'expired'):c.memory_points('lh',body,stamp+.025)
            with self.assertRaisesRegex(RuntimeError,'No confirmed'):c.memory_points('unknown',body,stamp)
            # A separated real geometry must fail despite a recently cached point.
            e.data.qpos[0]-=.01;mujoco.mj_forward(e.model,e.data)
            with self.assertRaisesRegex(RuntimeError,'gap exceeded'):c.memory_points('lh',body,stamp+.005)
            e.data.qpos[:]=q0;mujoco.mj_forward(e.model,e.data)
            released=active.copy();released[2]=False
            try:c.command(q,v,released)
            except RuntimeError:pass
            self.assertNotIn('lh',c.support_memory)
        finally:task.close()


if __name__=='__main__':unittest.main()
