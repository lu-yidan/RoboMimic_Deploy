"""Safety invariants for the offline lift planner on the actual reference model."""
from pathlib import Path
import unittest
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.swing_foot_lift import refine_swing_foot_lift

ROOT=Path(__file__).resolve().parents[1]


class SwingLiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');cls.env.reset()
        cls.plan=dict(np.load(ROOT/'outputs/support_retarget_plan_v1/plan.npz'))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_zero_lift_is_exact_identity(self):
        new,_=refine_swing_foot_lift(self.env,self.plan,foot_lift=0)
        for key in self.plan:
            np.testing.assert_array_equal(new[key],self.plan[key])

    def test_landing_taper_preserves_contacts_and_critical_corridor(self):
        sine,_=refine_swing_foot_lift(self.env,self.plan)
        taper,info=refine_swing_foot_lift(self.env,self.plan,envelope='landing_taper')
        t=self.plan['time']
        np.testing.assert_array_equal(taper['qpos'][t<=77.56],sine['qpos'][t<=77.56])
        np.testing.assert_array_equal(taper['qpos'][t>=80.66],self.plan['qpos'][t>=80.66])
        self.assertTrue(all(info['checks'].values()))
        # Check the Cartesian target independently of the IK implementation.
        d=mujoco.MjData(self.env.model);foot=self.env.model.body('left_ankle_roll_link').id
        for index in np.flatnonzero((t>54)&(t<77.5))[::17]:
            d.qpos[:]=self.plan['qpos'][index];mujoco.mj_kinematics(self.env.model,d);before=d.xpos[foot].copy()
            d.qpos[:]=taper['qpos'][index];mujoco.mj_kinematics(self.env.model,d);delta=d.xpos[foot]-before
            expected=np.array([0,0,.008*np.sin(np.pi*(t[index]-53.7)/(82.26-53.7))])
            self.assertLess(np.linalg.norm(delta-expected),2e-5)

    def test_invalid_intervals_and_unbounded_lift_rejected(self):
        for args in ({'foot_lift':-.001},{'foot_lift':.1},{'swing_start':53.75},
                     {'envelope':'landing_taper','taper_end':90},{'envelope':'unknown'}):
            with self.assertRaises(ValueError):refine_swing_foot_lift(self.env,self.plan,**args)


if __name__=='__main__':unittest.main()
