import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from training.window_acceleration_qp_roundoff_env import WindowAccelerationController


class RoundoffTests(unittest.TestCase):
    def setup_controller(self,time):
        e=SimpleNamespace(data=SimpleNamespace(time=time),limits=np.array([[-2.,2.],[-3.,3.]]))
        owner=SimpleNamespace(env=e,qp_records=[])
        controller=SimpleNamespace(leg_kp=100.,ik=SimpleNamespace(bodies=dict(lf=1,rf=2,lh=3,rh=4)),command=lambda *args:(np.array([.2,.3]),dict(torque_ratio=.1)))
        return WindowAccelerationController(controller,owner),owner

    def solution(self,extra):
        return dict(passed=True,torque=np.array([2.+extra,-3.-extra]),contact_count=3,max_constraint_residual=extra,sticking_residual_max=.02,acceleration_change_norm=1.)

    def test_outside_window_is_exact(self):
        for time in (0.,57.,65.,70.):
            c,o=self.setup_controller(time)
            with patch('training.window_acceleration_qp_roundoff_env.solve_probe') as solve:
                torque,_=c.command(None,None,[1,1,1,1]);solve.assert_not_called()
                np.testing.assert_array_equal(torque,[.2,.3]);self.assertEqual(o.qp_records,[])

    def test_small_solver_roundoff_bounded_and_reported(self):
        c,o=self.setup_controller(59.)
        with patch('training.window_acceleration_qp_roundoff_env.solve_probe',return_value=self.solution(1e-5)):
            torque,info=c.command(None,None,[1,1,1,1])
        np.testing.assert_array_equal(torque,[2.,-3.]);self.assertEqual(info['torque_ratio'],1.)
        self.assertAlmostEqual(o.qp_records[0]['solver_roundoff_clip_nm'],1e-5)

    def test_material_violation_is_rejected(self):
        c,_=self.setup_controller(59.)
        with patch('training.window_acceleration_qp_roundoff_env.solve_probe',return_value=self.solution(.002)):
            with self.assertRaisesRegex(RuntimeError,'motor feasibility'):c.command(None,None,[1,1,1,1])


if __name__=='__main__':unittest.main()
