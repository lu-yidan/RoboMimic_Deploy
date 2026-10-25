"""PHP contracts and FSM transitions, with actual released ONNX models and EGL depth."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import time
import unittest
from pathlib import Path
import numpy as np
import mujoco
import torch
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.deployment_config import load_php_config
from common.utils import FSMStateName, FSMCommand
from policy.php_parkour.PHPParkour import PHPParkour, direction_command
from policy.php_parkour.depth import DepthCamera, preprocess_depth
from deploy_mujoco.scene_builder import load_model
from FSM.FSM import FSM, FSMMode

ROOT = Path(__file__).resolve().parents[1]

class PHPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models = Path(load_php_config()['model_dir'])
        if not all((models / name).is_file() for name in ('student.onnx', 'depth_backbone.onnx')):
            raise unittest.SkipTest('Download external PHP student assets before integration tests')
        torch.set_num_threads(1)
        cls.model = load_model(ROOT/'g1_description/scene.xml', {'name': 'php_steps'})
        cls.names = [cls.model.joint(int(cls.model.actuator_trnid[i, 0])).name for i in range(29)]

    def setUp(self):
        self.state, self.output = StateAndCmd(29), PolicyOutput(29)
        self.policy = PHPParkour(self.state, self.output, load_php_config(), self.names)
        self.state.php_depth = np.zeros((58, 87), np.float32)
        self.state.php_depth_time = time.monotonic()

    def test_observation_wire_order_and_mapping(self):
        s, p = self.state, self.policy
        s.q = p.default + np.arange(29) * .01
        s.dq = np.arange(29) * .02
        s.ang_vel = np.array([1., 2., 3.])
        p.last_action[:] = np.arange(29) * .03
        latent = np.arange(32, dtype=np.float32)
        obs = p.build_observation(latent)[0]
        np.testing.assert_allclose(obs[:29], p.last_action)
        np.testing.assert_allclose(obs[29:32], s.ang_vel)
        np.testing.assert_allclose(obs[32:61], (s.q - p.default)[p.to_model])
        np.testing.assert_allclose(obs[61:90], s.dq[p.to_model])
        np.testing.assert_allclose(obs[90:93], [0, 0, -1])
        self.assertEqual(obs[93:108].argmax(), 0)
        np.testing.assert_allclose(obs[108:], latent)
        p.enter(); p.run()
        self.assertFalse(p.failed)
        np.testing.assert_allclose(self.output.actions, p.default + (p.last_action*p.scale)[p.to_runtime])

    def test_commands_and_stale_depth_damp(self):
        for vel, low, high in [([0,0],0,0),([1,0],1,6),([1,1],2,7),([0,1],3,8),([1,-1],4,9),([0,-1],5,10),([-1,0],11,11)]:
            self.assertEqual(direction_command(vel).argmax(),low)
            self.assertEqual(direction_command(vel,True).argmax(),high)
        self.state.php_depth_time -= 1
        self.policy.run()
        self.assertTrue(self.policy.failed)
        self.assertEqual(self.policy.checkChange(), FSMStateName.PASSIVE)
        np.testing.assert_array_equal(self.output.kps, 0)
        self.assertTrue((self.output.kds > 0).all())

    def test_camera_projection_and_depth_endpoints(self):
        for meters, normalized in [(0.3,-0.5),(3.,.5)]:
            np.testing.assert_allclose(preprocess_depth(np.full((60,106),meters,np.float32)),normalized,atol=1e-6)
        d = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,d)
        camera = DepthCamera(self.model)
        self.assertAlmostEqual(self.model.vis.map.znear * self.model.stat.extent, .1, places=6)
        self.addCleanup(camera.close)
        camera.capture(d,self.state)
        self.assertTrue(self.policy.depth_ready())
        self.assertGreater(np.std(self.state.php_depth),.01)
        # A second pose must produce a new observation, rather than frozen zeros.
        first=self.state.php_depth.copy()
        d.qpos[0] += 2
        mujoco.mj_forward(self.model,d)
        camera.capture(d,self.state)
        self.assertGreater(np.max(np.abs(first-self.state.php_depth)),.01)

    def test_fsm_switches_and_preemption(self):
        from unittest.mock import patch, MagicMock
        # Replace unrelated heavy policies; exercise the actual FSM and PHP inference.
        with patch.multiple('FSM.FSM', **{name: MagicMock() for name in
                ('PassiveMode','FixedPose','LocoMode','BeyondMimic','BeyondMimicMJ','Amp','SmpRecovery')}):
            f = FSM(self.state,self.output,joint_names=self.names,enable_freekick=False,
                    php_config=load_php_config())
        states = [(f.passive_mode,FSMStateName.PASSIVE),(f.fixed_pose_1,FSMStateName.FIXEDPOSE),
                  (f.loco_policy,FSMStateName.LOCOMODE),(f.amp_policy,FSMStateName.SKILL_AMP),
                  (f.smp_recovery_policy,FSMStateName.SKILL_SMP_RECOVERY)]
        for policy,name in states:
            policy.name=name
            policy.checkChange.return_value=name
        for source in (f.passive_mode,f.fixed_pose_1,f.loco_policy,f.amp_policy):
            f.cur_policy=source; f.FSMmode=FSMMode.NORMAL
            self.state.php_depth_time=time.monotonic()
            self.state.skill_cmd=FSMCommand.CMD_PHP_PARKOUR
            f.run()
            self.assertIs(f.cur_policy,f.php_policy)
            self.assertFalse(f.php_policy.failed)
            self.state.skill_cmd=FSMCommand.CMD_AMP
            f.run(); f.run()
            self.assertIs(f.cur_policy,f.amp_policy)
        for cmd, expected in [(FSMCommand.CMD_SMP_RECOVERY,f.smp_recovery_policy),
                              (FSMCommand.PASSIVE,f.passive_mode)]:
            f.cur_policy=f.php_policy;f.FSMmode=FSMMode.NORMAL
            self.state.skill_cmd=cmd
            f.run()
            self.assertIs(f.cur_policy,expected)
        # No depth: refuse entry rather than synthesizing zero features.
        f.cur_policy=f.fixed_pose_1;f.FSMmode=FSMMode.NORMAL
        self.state.php_depth=None;self.state.skill_cmd=FSMCommand.CMD_PHP_PARKOUR
        f.run()
        self.assertIs(f.cur_policy,f.fixed_pose_1)

    def test_modifier_release_does_not_switch_back_to_amp(self):
        from unittest.mock import MagicMock
        from common.joystick import JoystickButton
        from policy.php_parkour.input import PHPButtons
        buttons, joy = PHPButtons(), MagicMock()
        joy.is_button_just_pressed.side_effect = lambda b: b == JoystickButton.A
        joy.is_button_pressed.side_effect = lambda b: b in (JoystickButton.A, JoystickButton.L1)
        joy.is_button_released.return_value = False
        self.assertEqual(buttons.update(joy), FSMCommand.CMD_PHP_PARKOUR)
        joy.is_button_just_pressed.return_value = False
        joy.is_button_just_pressed.side_effect = None
        joy.is_button_pressed.side_effect = None
        joy.is_button_pressed.return_value = False
        joy.is_button_released.side_effect = lambda b: b == JoystickButton.A
        self.assertIsNone(buttons.update(joy))
        self.assertEqual(buttons.update(joy), FSMCommand.CMD_AMP)

    def test_chord_accepts_a_first_and_does_not_repeat_while_held(self):
        from unittest.mock import Mock
        from common.joystick import JoystickButton
        from policy.php_parkour.input import PHPButtons
        buttons, joy = PHPButtons(), Mock()
        held = {JoystickButton.A}
        joy.is_button_pressed.side_effect = lambda b: b in held
        joy.is_button_released.return_value = False
        self.assertIsNone(buttons.update(joy))
        held.add(JoystickButton.L1)
        self.assertEqual(buttons.update(joy), FSMCommand.CMD_PHP_PARKOUR)
        self.assertIsNone(buttons.update(joy))
        held.clear()
        joy.is_button_released.side_effect = lambda b: b == JoystickButton.A
        self.assertIsNone(buttons.update(joy))

    def test_rejected_amp_in_passive_is_silent(self):
        import contextlib
        import io
        from unittest.mock import patch, MagicMock
        with patch.multiple('FSM.FSM', **{name: MagicMock() for name in
                ('FixedPose','LocoMode','BeyondMimic','BeyondMimicMJ','Amp','SmpRecovery')}):
            fsm = FSM(self.state,self.output,enable_freekick=False)
        self.state.skill_cmd = FSMCommand.CMD_AMP
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            fsm.run()
        self.assertIs(fsm.cur_policy,fsm.passive_mode)
        self.assertEqual(text.getvalue(),'')
