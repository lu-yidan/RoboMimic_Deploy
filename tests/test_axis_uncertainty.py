import unittest
import numpy as np
from scipy.optimize import linprog
from training.axis_robust_contact_controller import axis_cone_rows


class AxisUncertaintyTests(unittest.TestCase):
    def test_sampled_constraints_with_margin_cover_intermediate_angles(self):
        rng=np.random.default_rng(1801);mu=.7
        for _ in range(8):
            frame,_=np.linalg.qr(rng.normal(size=(3,3)))
            cone=axis_cone_rows(frame,mu,3.)
            for _ in range(6):
                solved=linprog(rng.normal(size=3),A_ub=cone,b_ub=np.zeros(len(cone)),
                    A_eq=frame[:1],b_eq=[1.],bounds=[(None,None)]*3,method='highs')
                self.assertTrue(solved.success)
                for theta in np.linspace(-np.deg2rad(3),np.deg2rad(3),131):
                    c,s=np.cos(theta),np.sin(theta);R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
                    actual=frame@R.T
                    force=actual@solved.x
                    for phi in np.arange(8)*np.pi/4+np.pi/8:
                        self.assertLessEqual(np.cos(phi)*force[1]+np.sin(phi)*force[2]-mu*np.cos(np.pi/8)*force[0],1e-7)

    def test_zero_uncertainty_preserves_original_faces(self):
        frame=np.eye(3);mu=.7
        expected=np.array([np.cos(phi)*frame[1]+np.sin(phi)*frame[2]-mu*np.cos(np.pi/8)*frame[0] for phi in np.arange(8)*np.pi/4+np.pi/8])
        np.testing.assert_array_equal(axis_cone_rows(frame,mu,0.),expected)


if __name__=='__main__':unittest.main()
