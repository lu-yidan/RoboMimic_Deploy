import math
import unittest
from training.closure_smooth_reward import closure_cost


class SmoothCostTests(unittest.TestCase):
    def test_observed_failure_has_local_improvement_signal(self):
        cost=closure_cost(.05,.2605,1.)
        self.assertLess(closure_cost(.049,.2605,1.),cost)
        self.assertLess(closure_cost(.05,.26025,1.),cost)
        self.assertGreater(cost,0.);self.assertLess(cost,.02)

    def test_zero_gate_and_extreme_bound(self):
        self.assertEqual(closure_cost(.03,.25,1.),0.)
        self.assertEqual(closure_cost(.05,.27,0.),0.)
        self.assertEqual(closure_cost(.05,.27,1.,weight=0.),0.)
        cost=closure_cost(1e308,1e308,1.)
        self.assertTrue(math.isfinite(cost));self.assertLessEqual(cost,.02)

    def test_monotonicity_across_original_clipping_threshold(self):
        values=[closure_cost(c,.2605,1.) for c in (.035,.039,.04,.041,.05,.06)]
        self.assertTrue(all(a<b for a,b in zip(values,values[1:])))


if __name__=='__main__':unittest.main()
