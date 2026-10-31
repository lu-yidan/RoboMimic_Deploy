import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from training.closure_acceleration_target import install_acceleration_target


class TerminalQPTests(unittest.TestCase):
    def base(self,time=103.):
        def controller():
            return SimpleNamespace(command=lambda q,v,a:(np.ones(29),{}),ik=SimpleNamespace(bodies={}))
        return SimpleNamespace(baseline=controller(),contact=controller(),
            env=SimpleNamespace(data=SimpleNamespace(time=time),limits=np.tile([-10.,10.],(29,1))))

    def test_before_window_uses_original(self):
        b=self.base(99.66);install_acceleration_target(b,1.)
        with patch('training.closure_acceleration_target.solve_probe') as solve:
            torque,_=b.contact.command(None,None,[False]*4)
            solve.assert_not_called();np.testing.assert_array_equal(torque,np.ones(29))

    def test_roundoff_bounded_and_recorded(self):
        b=self.base();install_acceleration_target(b,1.)
        answer=dict(passed=True,torque=np.full(29,10.+1e-5),max_constraint_residual=1e-5)
        with patch('training.closure_acceleration_target.solve_probe',return_value=answer):
            torque,_=b.contact.command(None,None,[False]*4)
        np.testing.assert_array_equal(torque,np.full(29,10.))
        self.assertEqual(b.closure_qp_stats['commands'],1)
        self.assertGreater(b.closure_qp_stats['max_roundoff_clip_nm'],0.)

    def test_real_bound_violation_rejected(self):
        b=self.base();install_acceleration_target(b,1.)
        answer=dict(passed=True,torque=np.full(29,11.),max_constraint_residual=1.)
        with patch('training.closure_acceleration_target.solve_probe',return_value=answer):
            with self.assertRaises(RuntimeError):b.contact.command(None,None,[False]*4)


if __name__=='__main__':unittest.main()
