"""Physics-only reset must not reset controller history, phase or pending state."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import mujoco
import numpy as np
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand
from deploy_mujoco.reset_sim import reset_to_fixed_pose
from deploy_mujoco.viewer_camera import configure_camera

ROOT = Path(__file__).resolve().parents[1]


class SimResetCameraTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
        self.data = mujoco.MjData(self.model)

    def test_backspace_preserves_controller_and_refreshes_observations(self):
        active = Mock()
        active.history = np.arange(29, dtype=np.float32)
        active.phase = 42
        fixed = Mock()
        fixed._default_out = np.zeros(29)
        mode = object()
        fsm = SimpleNamespace(cur_policy=active, fixed_pose_1=fixed, FSMmode=mode, sim_counter=123)
        state, output = StateAndCmd(29), PolicyOutput(29)
        state.skill_cmd = FSMCommand.CMD_AMP
        state.php_high_speed = True
        state.php_depth = np.ones((58,87), np.float32)
        state.ang_vel[:] = 1
        state.root_lin_vel_b[:] = 2
        output.actions = np.arange(29, dtype=np.float32) / 10
        output.kps = np.ones(29) * 20
        before = output.actions.copy()
        self.data.time = 7.5
        self.data.qpos[0] = 9
        self.data.qvel[:] = 3
        reset_to_fixed_pose(self.model,self.data,fsm,state,output,preserve_controller=True)
        self.assertIs(fsm.cur_policy,active)
        self.assertIs(fsm.FSMmode,mode)
        self.assertEqual(fsm.sim_counter,123)
        active.enter.assert_not_called()
        active.exit.assert_not_called()
        active.run.assert_not_called()
        fixed.enter.assert_not_called()
        self.assertEqual(active.phase,42)
        np.testing.assert_array_equal(active.history,np.arange(29))
        np.testing.assert_array_equal(output.actions,before)
        np.testing.assert_array_equal(output.kps,20)
        self.assertEqual(state.skill_cmd,FSMCommand.CMD_AMP)
        self.assertTrue(state.php_high_speed)
        self.assertIsNone(state.php_depth)
        self.assertEqual(self.data.time,7.5)
        np.testing.assert_array_equal(self.data.qvel,0)
        np.testing.assert_array_equal(state.ang_vel,0)
        np.testing.assert_array_equal(state.root_lin_vel_b,0)
        np.testing.assert_allclose(state.torso_quat_w,self.data.body('torso_link').xquat)
        self.assertNotEqual(self.data.qpos[0],9)

    def test_camera_defaults_to_free_and_tracking_is_opt_in(self):
        camera = mujoco.MjvCamera()
        camera.distance = 7
        camera.azimuth = -140
        configure_camera(camera,self.model,{})
        self.assertEqual(camera.distance,7)
        self.assertEqual(camera.azimuth,-140)
        self.assertEqual(camera.type,mujoco.mjtCamera.mjCAMERA_FREE)
        configure_camera(camera,self.model,{"follow":True})
        self.assertEqual(camera.type,mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(camera.trackbodyid,self.model.body('torso_link').id)
        configure_camera(camera,self.model,{'follow':False,'lookat':[1,2,3],'distance':4})
        self.assertEqual(camera.type,mujoco.mjtCamera.mjCAMERA_FREE)
        np.testing.assert_array_equal(camera.lookat,[1,2,3])
        self.assertEqual(camera.distance,4)
        with self.assertRaisesRegex(ValueError,'not found'):
            configure_camera(camera,self.model,{'follow':True,'body':'missing'})

    def test_php_preset_starts_passive(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT/'deploy_mujoco/config'), version_base=None):
            cfg = compose(config_name='mujoco_php')
        self.assertEqual(cfg.startup_policy, 'passive')
        self.assertFalse(cfg.camera.follow)
        self.assertTrue(cfg.php_depth_preview)
