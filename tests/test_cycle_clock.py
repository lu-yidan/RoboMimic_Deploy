import unittest
import numpy as np
from training.cycle_clock import cycle_index


class CycleClockTests(unittest.TestCase):
    def test_roundoff_below_boundary_must_not_switch_early(self):
        ends=np.array([110.72,221.44,332.16])
        for i,end in enumerate(ends):
            self.assertEqual(cycle_index(np.nextafter(end,-np.inf),ends),i)
            self.assertEqual(cycle_index(end-5e-9,ends),i)
            self.assertEqual(cycle_index(end,ends),i+1)
            self.assertEqual(cycle_index(np.nextafter(end,np.inf),ends),i+1)

    def test_controller_cycle_and_reference_phase_share_boundary(self):
        times=np.array([0.,110.72,110.77,221.44]);phases=['settle','closed_hold','settle','closed_hold']
        ends=np.array([110.72,221.44])
        for time,expected_cycle,expected_phase in [(110.72-5e-9,0,'closed_hold'),(110.72,1,'settle'),(110.72+5e-9,1,'settle')]:
            knot=np.searchsorted(times,time,side='right')-1
            self.assertEqual(phases[knot+1],expected_phase)
            self.assertEqual(cycle_index(time,ends),expected_cycle)


if __name__=='__main__':unittest.main()
