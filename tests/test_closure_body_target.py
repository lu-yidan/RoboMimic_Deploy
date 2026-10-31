import unittest
from types import SimpleNamespace
import numpy as np
from training.closure_body_target import body_target_offset,install_body_target


class BodyTargetTests(unittest.TestCase):
    def test_smooth_target_and_derivative(self):
        self.assertEqual(body_target_offset(99.66,.015),(0.,0.))
        self.assertEqual(body_target_offset(103.,.015),(-.015,0.))
        for t in (100.,101.,102.):
            h=1e-5
            fd=(body_target_offset(t+h,.01)[0]-body_target_offset(t-h,.01)[0])/(2*h)
            self.assertAlmostEqual(fd,body_target_offset(t,.01)[1],places=9)

    def test_only_command_copies_change(self):
        calls=[]
        def original(q,v,a):calls.append((q.copy(),v.copy()));return 'torque',{}
        base=SimpleNamespace(baseline=SimpleNamespace(command=original),contact=SimpleNamespace(command=original),
            env=SimpleNamespace(data=SimpleNamespace(time=103.,qpos=np.ones(10))))
        install_body_target(base,.005)
        q=np.ones(10);v=np.zeros(9)
        base.contact.command(q,v,None)
        self.assertAlmostEqual(calls[-1][0][2],.995)
        np.testing.assert_array_equal(q,np.ones(10))
        np.testing.assert_array_equal(base.env.data.qpos,np.ones(10))
        base.env.data.time=90.;base.baseline.command(q,v,None)
        np.testing.assert_array_equal(calls[-1][0],q)

    def test_reject_invalid_authority(self):
        for drop in (-.001,.016,float('nan')):
            with self.assertRaises(ValueError):body_target_offset(100.,drop)


if __name__=='__main__':unittest.main()
