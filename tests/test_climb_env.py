"""Requires a separately imported, validated local mounted-wall bundle."""
import unittest
import numpy as np
import mujoco
from training.climb_env import ClimbSupportEnv, DEFAULT_BUNDLE


@unittest.skipUnless((DEFAULT_BUNDLE/'manifest.json').exists(), 'Import scene bundle first')
class EnvironmentTests(unittest.TestCase):
    def test_seeded_reset_restores_physics_history_and_target(self):
        env = ClimbSupportEnv()
        first, _ = env.reset(seed=7)
        env.step(np.zeros(29))
        restored, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, restored)
        self.assertEqual(env.data.time, 0.)
        np.testing.assert_array_equal(env.data.qvel, np.zeros(35))
        np.testing.assert_array_equal(restored[64:93], np.zeros(29))
        self.assertTrue(env.observation_space.contains(restored))
        env.close()

    def test_motor_pd_matches_independent_physics_loop(self):
        env = ClimbSupportEnv(); env.reset()
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_BUNDLE/'model.xml'))
        data = mujoco.MjData(model); data.qpos[:] = env.initial
        mujoco.mj_forward(model, data)
        action = np.linspace(-.01,.01,29,dtype=np.float32)
        target, _ = env.contract.decode_action(action)
        for _ in range(env.substeps):
            torque = env.contract.kps*(target-data.qpos[env.qadr])-env.contract.kds*data.qvel[env.vadr]
            data.ctrl[env.motors] = np.clip(torque,env.limits[:,0],env.limits[:,1])
            mujoco.mj_step(model,data)
        _,_,terminated,_,info = env.step(action)
        self.assertFalse(terminated)
        self.assertEqual(info['physics_steps_this_action'],40)
        np.testing.assert_array_equal(env.data.qpos,data.qpos)
        np.testing.assert_array_equal(env.data.qvel,data.qvel)
        self.assertAlmostEqual(env.data.time,.02,places=12)
        np.testing.assert_array_equal(env.data.qfrc_applied,np.zeros(35))
        np.testing.assert_array_equal(env.data.xfrc_applied,np.zeros((model.nbody,6)))
        env.close()

    def test_time_limit_invalid_action_and_fall_are_distinct(self):
        env = ClimbSupportEnv(horizon_s=.02); env.reset()
        with self.assertRaises(ValueError): env.step(np.full(29,np.nan))
        self.assertEqual(env.data.time,0.)
        _,reward,terminated,truncated,_ = env.step(np.zeros(29))
        self.assertFalse(terminated); self.assertTrue(truncated); self.assertTrue(np.isfinite(reward))
        with self.assertRaises(RuntimeError): env.step(np.zeros(29))
        env.reset(); env.data.qpos[2] = env.initial[2]-.16
        mujoco.mj_forward(env.model,env.data)
        _,_,terminated,truncated,info = env.step(np.zeros(29))
        self.assertTrue(terminated); self.assertFalse(truncated)
        self.assertEqual(info['reason'],'fall')
        env.close()

if __name__ == '__main__': unittest.main()
