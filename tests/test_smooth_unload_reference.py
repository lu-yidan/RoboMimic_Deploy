import unittest
import numpy as np
from training.smooth_unload_reference_env import SmoothUnloadReferenceEnv
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


class SmoothReferenceTests(unittest.TestCase):
    def test_knot_continuity_and_unchanged_outside_phase(self):
        path='outputs/highrod_endpoint_adapt_plan_v1/plan.npz'
        smooth=SmoothUnloadReferenceEnv(plan=path);original=FullCycleResidualEnv(plan=path)
        try:
            low,high=smooth.smooth_window
            for t in smooth.plan['time'][(smooth.plan['time']>=low)&(smooth.plan['time']<=high)]:
                left=smooth.reference(t-1e-7);right=smooth.reference(t+1e-7)
                self.assertLess(float(abs(left[0]-right[0]).max()),1e-6)
                self.assertLess(float(abs(left[1]-right[1]).max()),1e-4)
            for t in (0.,49.,low-.001,high+.001,smooth.horizon):
                a=original.reference(t);b=smooth.reference(t)
                for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
                self.assertEqual(a[3],b[3])
            for x,y in zip(smooth.desired_poses,original.desired_poses):np.testing.assert_array_equal(x,y)
        finally:smooth.close();original.close()


if __name__=='__main__':unittest.main()
