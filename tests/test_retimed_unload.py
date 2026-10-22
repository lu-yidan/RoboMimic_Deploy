import unittest
import numpy as np
from training.retimed_unload_env import RetimedUnloadEnv,reference_clock
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


class RetimingTests(unittest.TestCase):
    def test_same_geometric_path_and_scaled_velocity(self):
        original=FullCycleResidualEnv(plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz')
        try:
            for factor in (1.5,2.):
                env=RetimedUnloadEnv(plan=f'outputs/retimed_unload_factor{factor:g}_plan_v1/plan.npz')
                try:
                    for t in (0.,49.,52.,53.,59.,60.,66.,70.3,100.,113.66):
                        actual=t if t<=52.2 else 52.2+factor*(t-52.2) if t<=70.2 else t+18*(factor-1)
                        self.assertAlmostEqual(reference_clock(actual,factor),t)
                        q,v,a,phase=original.reference(t);qq,vv,aa,pp=env.reference(actual)
                        np.testing.assert_allclose(q,qq,atol=1e-12,rtol=0)
                        np.testing.assert_allclose(v/(factor if 52.2<t<70.2 else 1.),vv,atol=1e-10,rtol=0)
                        np.testing.assert_array_equal(a,aa);self.assertEqual(phase,pp)
                    for x,y in zip(original.desired_poses,env.desired_poses):np.testing.assert_array_equal(x,y)
                    env.reset()
                    for actual,expected in ((48.9,False),(49.,True),(52.2+factor*(61.55-52.2),True),(52.2+factor*(61.57-52.2),False)):
                        env.env.data.time=actual;self.assertEqual(env.contact.bias_window_active(),expected)
                finally:env.close()
        finally:original.close()


if __name__=='__main__':unittest.main()
