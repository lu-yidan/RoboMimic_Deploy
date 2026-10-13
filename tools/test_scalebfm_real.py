"""Offline Unitree input/config checks. Does not import SDK or create DDS channels."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.remote_controller import KeyMap
from common.utils import FSMCommand, FSMStateName
from deploy_real.scalebfm_input import RealPolicyInput
from deploy_real.config import Config
from policy.scalebfm.motion_catalog import MotionCatalog


class RealInputTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(skill_cmd=FSMCommand.INVALID)
        self.catalog = MotionCatalog('/tmp/one.npz', [], '/')
        self.catalog.paths.append(Path('/tmp/two.npz'))
        self.loaded = []
        self.tracker = SimpleNamespace(motion_path=Path('/tmp/one.npz'), load_motion=self.loaded.append)
        self.fsm = SimpleNamespace(cur_policy=SimpleNamespace(name=FSMStateName.SKILL_AMP),
                                   scalebfm_policy=self.tracker)
        self.router = RealPolicyInput(self.catalog)

    def press(self, *keys):
        self.state.skill_cmd = FSMCommand.INVALID
        buttons = [False] * 16
        for key in keys: buttons[key] = True
        self.router.update(buttons, self.fsm, self.state, None)
        return self.state.skill_cmd

    def test_selection_entry_and_hold(self):
        self.press(KeyMap.right)
        self.press(KeyMap.right)
        self.assertEqual(self.catalog.index, 1)
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.press(KeyMap.R1), FSMCommand.CMD_SCALEBFM)
        self.assertEqual(self.loaded, [Path('/tmp/two.npz')])
        self.assertEqual(self.press(KeyMap.R1), FSMCommand.INVALID)
        self.press()
        self.fsm.cur_policy.name = FSMStateName.LOCOMODE
        self.assertEqual(self.press(KeyMap.R1), FSMCommand.CMD_SCALEBFM)

    def test_damping_priority_and_bias_modifier(self):
        self.press(KeyMap.L1, KeyMap.right)
        self.assertEqual(self.catalog.index, 0)
        for key in (KeyMap.start, KeyMap.A, KeyMap.B, KeyMap.R1, KeyMap.R2):
            self.assertEqual(self.press(KeyMap.F1, key), FSMCommand.PASSIVE)
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.press(KeyMap.select, KeyMap.R1), FSMCommand.PASSIVE)

    def test_passive_rejected(self):
        self.fsm.cur_policy.name = FSMStateName.PASSIVE
        self.assertEqual(self.press(KeyMap.R1), FSMCommand.INVALID)
        self.assertEqual(self.loaded, [])

    def test_damping_preempts_pending_enter(self):
        from FSM.FSM import FSM, FSMMode
        from unittest.mock import Mock
        fsm = FSM.__new__(FSM)
        fsm.policy_output = SimpleNamespace()
        fsm.state_cmd = SimpleNamespace(skill_cmd=FSMCommand.PASSIVE)
        fsm.cur_policy = active = Mock()
        fsm.passive_mode = Mock()
        fsm.FSMmode = FSMMode.CHANGE
        fsm.run()
        active.run.assert_not_called()
        active.enter.assert_not_called()
        active.exit.assert_called_once()
        fsm.passive_mode.run.assert_called_once()
        self.assertEqual(fsm.FSMmode, FSMMode.NORMAL)

    def test_recovery_key_and_unique_ids(self):
        self.assertEqual(self.press(KeyMap.Y), FSMCommand.CMD_SMP_RECOVERY)
        self.assertEqual(self.press(KeyMap.Y), FSMCommand.INVALID)
        self.assertEqual(self.press(KeyMap.F1, KeyMap.Y), FSMCommand.PASSIVE)
        self.assertNotEqual(FSMStateName.SKILL_SMP_RECOVERY.value, FSMStateName.SKILL_SCALEBFM.value)
        self.assertNotEqual(FSMCommand.CMD_SMP_RECOVERY.value, FSMCommand.CMD_SCALEBFM.value)

    def test_real_config_joint_order(self):
        import mujoco
        root = Path(__file__).resolve().parents[1]
        cfg = Config()
        model = mujoco.MjModel.from_xml_path(str(root/'g1_description/scene.xml'))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i,0])) for i in range(29)]
        self.assertEqual(cfg.joint_names, names)
        self.assertEqual(cfg.control_dt, .02)
        self.assertTrue(cfg.scalebfm.reference_forcing)

if __name__ == '__main__': unittest.main()
