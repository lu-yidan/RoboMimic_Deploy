import unittest
import numpy as np
from training.temporal_leg_gain_env import TemporalLegGainEnv


class TemporalLegGainTests(unittest.TestCase):
    def test_gain_does_not_accumulate_or_leak(self):
        env=TemporalLegGainEnv(peak_leg_kp=300.)
        try:
            env.reset();env.env.data.time=52.2
            env.select_controller(np.array([False,True,True,True]),'lf_unload')
            for t,expected in ((57.,100.),(58.,200.),(59.,300.),(60.,300.),(60.,300.),(64.,200.),(65.,100.),(66.,100.)):
                env.env.data.time=t
                controller=env.select_controller(np.array([False,True,True,True]),'lf_unload')
                self.assertAlmostEqual(controller.leg_kp,expected)
            env.env.data.time=70.2
            self.assertEqual(env.select_controller(np.array([False,True,True,True]),'lf_swing').leg_kp,100.)
        finally:env.close()

    def test_invalid_gain(self):
        for gain in (99.,301.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):TemporalLegGainEnv(peak_leg_kp=gain)


if __name__=='__main__':unittest.main()
