"""Check actual endpoint geometry, bounded changes and no live-state edits."""
from pathlib import Path
import unittest
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.wholebody_hand_reference import refine_wholebody_hand_targets

ROOT=Path(__file__).resolve().parents[1]


class WholebodyHandReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');cls.env.reset()
        with np.load(ROOT/'outputs/v2_swing_two_reference_20261029_v1/plan.npz') as z:
            cls.plan={k:z[k] for k in z.files}
        cls.windows={'rh':[123.66,125.66],'lh':[136.66,138.66]}

    @classmethod
    def tearDownClass(cls):cls.env.close()

    def test_zero_offset_exact(self):
        out,_=refine_wholebody_hand_targets(self.env,self.plan,offset_m=[0,0,0],windows=self.windows,continuation=True)
        for key in self.plan:np.testing.assert_array_equal(out[key],self.plan[key])

    def test_four_endpoint_geometry_and_bounds(self):
        indices=[int(np.argmin(abs(self.plan['time']-t))) for t in (123.66,125.66,136.66,138.66,165.66,227.32)]
        plan={k:(v[indices].copy() if k in ('time','qpos','phase','active','errors') else v.copy()) for k,v in self.plan.items()}
        original={k:v.copy() for k,v in plan.items()}
        live=self.env.data.qpos.copy()
        result,_=refine_wholebody_hand_targets(self.env,plan,offset_m=[.002,0,.002],windows=self.windows,continuation=True)
        m=self.env.model;d=mujoco.MjData(m)
        bodies=[m.body(n).id for n in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
        for i,time in enumerate(plan['time']):
            d.qpos[:]=plan['qpos'][i];mujoco.mj_kinematics(m,d)
            target=d.xpos[bodies].copy();rotations=d.xmat[bodies].copy()
            for b,key in enumerate(('lh','rh')):
                start,end=self.windows[key];u=np.clip((time-start)/(end-start),0,1)
                target[b]+=(6*u**5-15*u**4+10*u**3)*np.array([.002,0,.002])
            d.qpos[:]=result['qpos'][i];mujoco.mj_kinematics(m,d)
            np.testing.assert_allclose(d.xpos[bodies],target,atol=2e-5,rtol=0)
            np.testing.assert_allclose(d.xmat[bodies],rotations,atol=1e-4,rtol=0)
        self.assertLessEqual(np.linalg.norm(result['qpos'][:,:3]-plan['qpos'][:,:3],axis=1).max(),.003+1e-12)
        self.assertLessEqual(abs(result['qpos'][:,self.env.qadr]-plan['qpos'][:,self.env.qadr]).max(),.15+1e-12)
        np.testing.assert_array_equal(result['qpos'][:,3:7],plan['qpos'][:,3:7])
        np.testing.assert_array_equal(self.env.data.qpos,live)
        for key in original:np.testing.assert_array_equal(plan[key],original[key])

    def test_loaded_interval_and_invalid_offset_rejected(self):
        bad={k:v.copy() for k,v in self.plan.items()};bad['active'][:,3]=True
        with self.assertRaises(ValueError):
            refine_wholebody_hand_targets(self.env,bad,offset_m=[.002,0,.002],windows=self.windows)
        for offset in ([.003,0,0],[np.nan,0,0],[0,0]):
            with self.assertRaises(ValueError):
                refine_wholebody_hand_targets(self.env,self.plan,offset_m=offset,windows=self.windows)


if __name__=='__main__':unittest.main()
