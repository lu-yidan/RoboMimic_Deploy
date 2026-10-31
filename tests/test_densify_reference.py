"""Reference resampling must preserve interpolation, masks and exact old knots."""
import unittest
import numpy as np
from training.densify_reference import densify_after


class DensifyReferenceTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(time=np.array([0., 1., 1.15, 1.55]),
            qpos=np.array([[0., 1.], [3., 4.], [2., 5.], [7., -1.]]),
            active=np.array([[1,1,1,1], [1,1,0,1], [1,0,1,1], [0,1,1,1]], dtype=bool),
            phase=np.array(['start','raise','seat','load']), errors=np.array([0.,.001,.002,.001]),
            initial=np.array([0.,1.]), cycle_ends_s=np.array([1.55]))

    @staticmethod
    def sample(plan, t):
        times = plan['time']
        j = min(max(int(np.searchsorted(times,t,side='right')-1),0), len(times)-2)
        u = np.clip((t-times[j])/(times[j+1]-times[j]),0,1)
        return (1-u)*plan['qpos'][j]+u*plan['qpos'][j+1], plan['active'][j+1], plan['phase'][j+1]

    def test_old_knots_prefix_and_metadata_exact(self):
        new = densify_after(self.plan, 1., .02)
        indices = np.searchsorted(new['time'],self.plan['time'])
        for key in ('time','qpos','active','phase','errors'):
            np.testing.assert_array_equal(new[key][indices],self.plan[key])
        for key in ('initial','cycle_ends_s'):
            np.testing.assert_array_equal(new[key],self.plan[key])
        np.testing.assert_array_equal(new['time'][new['time']<=1],self.plan['time'][:2])
        self.assertLessEqual(np.diff(new['time'][new['time']>=1]).max(), .02+1e-12)

    def test_linear_function_and_interval_masks_preserved(self):
        new = densify_after(self.plan, 1., .02)
        for time in np.r_[np.linspace(0,1.55,203),self.plan['time'],new['time']]:
            a,b=self.sample(self.plan,time),self.sample(new,time)
            # Re-association of floating-point interpolation adds roundoff;
            # original stored knots themselves remain bit-for-bit exact.
            tolerance=32*np.finfo(float).eps*np.max(abs(self.plan['qpos']))
            np.testing.assert_allclose(a[0],b[0],atol=tolerance,rtol=0)
            np.testing.assert_array_equal(a[1],b[1])
            self.assertEqual(a[2],b[2])

    def test_invalid_sampling_rejected(self):
        for start,dt in ((.5,.02),(1.,0),(1.,np.nan)):
            with self.assertRaises(ValueError):
                densify_after(self.plan,start,dt)


if __name__=='__main__':
    unittest.main()
