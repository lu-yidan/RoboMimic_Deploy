import unittest
from training.closure_margin_reward import closure_cost


class ClosureRewardTests(unittest.TestCase):
    def test_safe_region_and_inactive_window(self):
        self.assertAlmostEqual(closure_cost(.032,.256,1.),0.,places=15)
        self.assertEqual(closure_cost(.2,.5,0.),0.)
        self.assertEqual(closure_cost(.2,.5,1.,weight=0.),0.)

    def test_bounds_and_monotonic_penalty(self):
        values=[closure_cost(c,.25,1.) for c in (.01,.032,.036,.04,.1)]
        self.assertEqual(values,sorted(values));self.assertAlmostEqual(values[2],.0025)
        self.assertEqual(closure_cost(.2,.5,1.),.02)
        self.assertAlmostEqual(closure_cost(.2,.5,.3),.006)

    def test_invalid_inputs(self):
        for args in ((-.1,.25,1.),(.1,float('nan'),1.),(.1,.25,1.1)):
            with self.assertRaises(ValueError):closure_cost(*args)


if __name__=='__main__':unittest.main()
