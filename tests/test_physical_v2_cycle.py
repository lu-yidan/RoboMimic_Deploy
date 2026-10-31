import unittest
from training.physical_v2_cycle_env import endpoint_checks


class PhysicalV2AcceptanceTest(unittest.TestCase):
    def sample(self):
        return dict(rise_m=.26005,normal_n={k:20. for k in ('lf','rf','lh','rh')},
            vertical_n={k:15. for k in ('lf','rf','lh','rh')},intended_contacts=True,
            effector_error_m=.006,tilt_rad=.01,linear_speed_m_s=.001,
            angular_speed_rad_s=.01,joint_speed_rad_s=.01,closure_rad=.05)

    def test_closure_is_quality_only_and_deadband_is_explicit(self):
        s=self.sample();self.assertTrue(all(endpoint_checks(s,1).values()))
        s['rise_m']=.2611;self.assertFalse(endpoint_checks(s,1)['rise'])

    def test_four_supports_and_intended_rod_required(self):
        s=self.sample();s['normal_n']['lf']=9.9
        self.assertFalse(endpoint_checks(s,1)['normal_support'])
        s=self.sample();s['intended_contacts']=False
        self.assertFalse(endpoint_checks(s,1)['intended_contacts'])

    def test_cumulative_height_and_posture(self):
        s=self.sample();s['rise_m']=2.0
        self.assertTrue(all(endpoint_checks(s,8).values()))
        self.assertFalse(endpoint_checks(s,7)['rise'])
        s['joint_speed_rad_s']=.11
        self.assertFalse(endpoint_checks(s,8)['joint_speed'])


if __name__=='__main__':unittest.main()
