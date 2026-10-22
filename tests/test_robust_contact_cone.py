import unittest
import numpy as np
from training.robust_contact_frame_controller import conservative_mu


class RobustConeTests(unittest.TestCase):
    def test_candidate_corners_remain_inside_true_cones_for_bounded_normal_error(self):
        mu=.7;angle=np.deg2rad(3.);coefficient=conservative_mu(mu,3.)
        self.assertLess(coefficient,mu);self.assertEqual(conservative_mu(mu,0),mu)
        # Extremal candidate rays: normal along Z; arbitrary tangent rotations.
        for theta in np.linspace(0,2*np.pi,65):
            force=np.array([coefficient*np.cos(theta),coefficient*np.sin(theta),1.])
            for azimuth in np.linspace(0,2*np.pi,33):
                normal=np.array([np.sin(angle)*np.cos(azimuth),np.sin(angle)*np.sin(azimuth),np.cos(angle)])
                normal_force=force@normal;tangent=force-normal_force*normal
                # The inscribed circle lies within every orientation of the actual polygon.
                self.assertLessEqual(np.linalg.norm(tangent)-mu*np.cos(np.pi/8)*normal_force,1e-12)


if __name__=='__main__':unittest.main()
