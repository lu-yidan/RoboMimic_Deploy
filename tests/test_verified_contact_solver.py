import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from training.verified_contact_memory_controller import VerifiedContactMemoryController
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.climb_cycle import solve_force_qp_fallback
from training.rung_residual_env import ROOT


class VerificationTests(unittest.TestCase):
    def test_false_infeasible_status_is_rechecked_with_identical_constraints(self):
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',['nominal'])
        class IncorrectStatus:
            def setup(self,**kwargs):pass
            def solve(self,**kwargs):return SimpleNamespace(info=SimpleNamespace(status_val=4,status='primal infeasible inaccurate'))
            def warm_start(self,**kwargs):pass
        try:
            task.reset(options={'case':'nominal'});e=task.env;q,v,active=task.reference(e.data.time)
            c=VerifiedContactMemoryController(e,torque_weight=10000.);q0=e.data.qpos.copy();v0=e.data.qvel.copy()
            with patch('training.verified_contact_memory_controller.osqp.OSQP',IncorrectStatus):torque,info=c.command(q,v,active)
            problem=c.solver_problems[0];force=problem['solution'];product=problem['A']@force
            self.assertTrue(c.solver_status_events[0]['verified'])
            self.assertLessEqual(max(0.,np.max(problem['lower']-product),np.max(product-problem['upper'])),1e-5)
            self.assertTrue(np.all(torque>=e.limits[:,0]-1e-6) and np.all(torque<=e.limits[:,1]+1e-6))
            np.testing.assert_array_equal(e.data.qpos,q0);np.testing.assert_array_equal(e.data.qvel,v0)
        finally:task.close()

    def test_genuinely_infeasible_constraints_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'no feasible LP seed'):
            solve_force_qp_fallback(np.eye(1),np.zeros(1),np.array([[1.],[1.]]),np.array([1.,-np.inf]),np.array([np.inf,0.]))


if __name__=='__main__':unittest.main()
