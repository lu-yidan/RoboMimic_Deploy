import numpy as np
import unittest
from training.native_margin_reward import margin_cost


class NativeMarginTests(unittest.TestCase):
    def test_bounded_cost_and_safe_region(self):
        self.assertEqual(margin_cost(.02),0.)
        self.assertEqual(margin_cost(.03),0.)
        self.assertAlmostEqual(margin_cost(.01),.0025)
        self.assertEqual(margin_cost(0.),.01)
        self.assertEqual(margin_cost(-.1),.01)
        values=[margin_cost(x) for x in np.linspace(-.1,.1,101)]
        self.assertTrue(all(0<=v<=.01 for v in values))
        self.assertTrue(np.all(np.diff(values)<=0))


    def test_invalid_configuration(self):
        for kwargs in [dict(margin_rad=np.nan),dict(margin_rad=0.,width_rad=0.),dict(margin_rad=0.,weight=-1.)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):margin_cost(**kwargs)


    def test_zero_weight_exactly_preserves_reward_contribution(self):
        self.assertTrue(all(margin_cost(x,weight=0.)==0 for x in [-1.,0.,.01,1.]))


if __name__=='__main__':unittest.main()
