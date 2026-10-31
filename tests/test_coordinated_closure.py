import unittest
from types import SimpleNamespace
import numpy as np
from tools.evaluate_coordinated_closure import correction


class CoordinatedClosureTests(unittest.TestCase):
    def base(self, time):
        data=SimpleNamespace(time=time,qpos=np.full(29,.05))
        return SimpleNamespace(env=SimpleNamespace(data=data,qadr=np.arange(29)),
            indices=np.r_[np.arange(12),np.arange(15,29)],plan={'qpos':np.zeros((1,29))})

    def test_prefix_and_nonleg_joints_unchanged(self):
        b=self.base(99.66)
        np.testing.assert_array_equal(correction(b,1.),np.zeros(26))
        b.env.data.time=103.
        np.testing.assert_array_equal(correction(b,1.)[12:],np.zeros(14))

    def test_bounded_direction_and_no_state_mutation(self):
        b=self.base(103.);before=b.env.data.qpos.copy()
        np.testing.assert_array_equal(correction(b,2.)[:12],-np.ones(12))
        np.testing.assert_array_equal(before,b.env.data.qpos)
        b.env.data.qpos[:12]=-.05
        np.testing.assert_array_equal(correction(b,2.)[:12],np.ones(12))

    def test_ramp_and_zero_gain(self):
        b=self.base(101.16)
        np.testing.assert_allclose(correction(b,1.)[:12],-.5)
        np.testing.assert_array_equal(correction(b,0.),np.zeros(26))


if __name__=='__main__':unittest.main()
