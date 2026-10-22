import unittest
from types import SimpleNamespace
import numpy as np
from training.unload_protection import current_pose_hold


class UnloadProtectionTests(unittest.TestCase):
    def test_never_adds_a_support_or_changes_state(self):
        q=np.arange(36,dtype=float);calls=[]
        def command(qref,vref,active):
            calls.append((qref.copy(),vref.copy(),active.copy()))
            return np.zeros(29),{'torque_ratio':0.}
        controller=SimpleNamespace(d=SimpleNamespace(qpos=q.copy()),m=SimpleNamespace(nv=35),command=command)
        _,_,hold,retained=current_pose_hold(controller,[False,True,True,True],[True,True,True,True])
        np.testing.assert_array_equal(controller.d.qpos,q)
        np.testing.assert_array_equal(hold,q)
        np.testing.assert_array_equal(retained,[False,True,True,True])
        np.testing.assert_array_equal(calls[0][1],np.zeros(35))
        hold[0]=-1;self.assertEqual(controller.d.qpos[0],0.)
        with self.assertRaises(RuntimeError):
            current_pose_hold(controller,[False,True,True,True],[True,False,True,True])
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
