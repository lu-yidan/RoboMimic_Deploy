import unittest
import numpy as np
from training.temporal_base_reference_env import TemporalBaseReferenceEnv,envelope_and_rate
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


class BaseReferenceTests(unittest.TestCase):
    def test_envelope_derivative_and_bounds(self):
        for t in np.linspace(56,66,41):
            g,v=envelope_and_rate(t);self.assertTrue(0<=g<=1)
            numerical=(envelope_and_rate(t+1e-5)[0]-envelope_and_rate(t-1e-5)[0])/2e-5
            self.assertAlmostEqual(v,numerical,places=5)
        for t in (0.,57.,65.,113.):self.assertEqual(envelope_and_rate(t),(0.,0.))

    def test_only_base_reference_changes_and_endpoints_preserved(self):
        base=FullCycleResidualEnv();bias=np.array([-.001,.0015,.0005]);shifted=TemporalBaseReferenceEnv(base_bias=bias)
        try:
            original_plan=base.plan['qpos'].copy()
            for t in (0.,57.,58.,59.,60.,63.,64.,65.,base.horizon):
                q,v,a,phase=base.reference(t);qq,vv,aa,pp=shifted.reference(t);g,dg=envelope_and_rate(t)
                np.testing.assert_allclose(qq[:3]-q[:3],g*bias,rtol=0,atol=1e-15)
                np.testing.assert_allclose(vv[:3]-v[:3],dg*bias,rtol=0,atol=1e-15)
                np.testing.assert_array_equal(qq[3:],q[3:]);np.testing.assert_array_equal(vv[3:],v[3:]);np.testing.assert_array_equal(a,aa);self.assertEqual(phase,pp)
            np.testing.assert_array_equal(original_plan,shifted.plan['qpos'])
            for a,b in zip(base.desired_poses,shifted.desired_poses):np.testing.assert_array_equal(a,b)
        finally:base.close();shifted.close()

    def test_invalid_bias(self):
        for bias in ([.006,0,0],[float('nan'),0,0],[0,0]):
            with self.assertRaises(ValueError):TemporalBaseReferenceEnv(base_bias=bias)


if __name__=='__main__':unittest.main()
