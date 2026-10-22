import unittest
from pathlib import Path
import numpy as np

from training.hand_transfer import RightHookReference, smooth
from training.wall_support_env import WallSupportEnv

BUNDLE=Path(__file__).resolve().parents[1]/'outputs/wall_support_bundle_pd15_v2'


@unittest.skipUnless((BUNDLE/'manifest.json').exists(),'Prepare the wall bundle first')
class HandReferenceTests(unittest.TestCase):
    def setUp(self):
        self.env=WallSupportEnv(BUNDLE)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_ik_does_not_overwrite_live_physics(self):
        e=self.env;r=RightHookReference(e)
        q=e.data.qpos.copy();v=e.data.qvel.copy();ctrl=e.data.ctrl.copy()
        target=r.start+np.array([0,0,.05])
        arm,pe,re=r.solve(q,target,iterations=200)
        self.assertLess(pe,1e-4);self.assertLess(re,1e-3)
        self.assertTrue(np.all(arm>=r.bounds[:,0]));self.assertTrue(np.all(arm<=r.bounds[:,1]))
        np.testing.assert_array_equal(e.data.qpos,q)
        np.testing.assert_array_equal(e.data.qvel,v)
        np.testing.assert_array_equal(e.data.ctrl,ctrl)
        self.assertEqual(e.data.time,0.)

    def test_same_rung_cycle_really_returns_to_same_rung(self):
        r=RightHookReference(self.env,rise=0.,clearance=.05,initial_clearance=.05,retreat=.06)
        np.testing.assert_allclose(r.target(18)[0],r.start)
        np.testing.assert_allclose(r.target(8)[0]-r.start,[-.06,0,.05])
        self.assertEqual(r.target(8)[1],'free_hold')

    def test_transfer_respects_wall_tilt_and_smooth_phase_boundaries(self):
        r=RightHookReference(self.env)
        np.testing.assert_allclose(r.target(18)[0]-r.start,[.25*np.tan(np.deg2rad(5)),0,.25])
        for t in r.times[1:-1]:
            p=r.target(t)[0]
            self.assertLess(np.linalg.norm(r.target(t+1e-4)[0]-p),1e-9)
            self.assertLess(np.linalg.norm(r.target(t-1e-4)[0]-p),1e-9)


if __name__=='__main__':unittest.main()
