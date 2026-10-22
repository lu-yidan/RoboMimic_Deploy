import unittest
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.unload_lookahead import UnloadLookahead
from training.rung_residual_env import ROOT


class UnloadLookaheadTests(unittest.TestCase):
    def test_prediction_never_changes_live_integration_state(self):
        e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
        with np.load(ROOT/'outputs/clearance_back20_leg14_dynamics_v1/nominal/trajectory.npz') as z:
            i=np.argmin(abs(z['time_s']-54.))
            e.data.qpos[:]=z['qpos'][i];e.data.qvel[:]=z['qvel'][i];e.data.time=float(z['time_s'][i])
        mujoco.mj_forward(e.model,e.data)
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        before=np.empty(mujoco.mj_stateSize(e.model,spec));after=before.copy()
        mujoco.mj_getState(e.model,e.data,before,spec)
        shadow=UnloadLookahead(e,{},.005)
        try:
            q=e.data.qpos.copy()
            result=shadow.predict(e,lambda t:(q.copy(),np.zeros(e.model.nv),np.array([False,True,True,True]),'lf_unload'),100.)
            self.assertFalse(result['failed'])
            self.assertAlmostEqual(result['lead_s'],.005,places=8)
            mujoco.mj_getState(e.model,e.data,after,spec)
            np.testing.assert_array_equal(before,after)
        finally:shadow.close();e.close()


if __name__=='__main__':unittest.main()
