"""Bridge policy integration without opening DDS or commanding motors."""
import struct
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from common.utils import FSMCommand, FSMStateName
from common.remote_controller import KeyMap
from bridge.python.policy_runtime import PolicyRuntime


class ScaleBFMBridgeTest(unittest.TestCase):
    def make_runtime(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        module = 'bridge.python.policy_runtime.'
        for name in ('BridgeStateSubscriber', 'DiagnosticsSubscriber', 'BallStateSubscriber',
                     'TargetStateSubscriber', 'TargetStatePublisher', 'BallStatePublisher'):
            stack.enter_context(patch(module + name))
        stack.enter_context(patch.object(PolicyRuntime, 'wait_for_bridge_state'))
        fsm = stack.enter_context(patch(module + 'FSM'))
        fsm.return_value.scalebfm_policy.motion_path = Path('/tmp/reference.npz')
        cfg = SimpleNamespace(num_joints=29, control_dt=.02, torch_threads=1,
            scalebfm=SimpleNamespace(enabled=True, motion_globs=[]), joint_names=['j']*29,
            bridge_domain_id=0, bridge_state_topic='test/state', bridge_state_stale_ms=200,
            log_enabled=False)
        runtime = PolicyRuntime(cfg)
        self.addCleanup(runtime.close)
        return runtime, fsm

    def test_preload_and_bridge_state_to_r1(self):
        runtime, fsm = self.make_runtime()
        self.assertFalse(fsm.call_args.kwargs['enable_freekick'])
        fsm.return_value.scalebfm_policy.preload_motions.assert_called_once_with(runtime.catalog.paths)
        raw = bytearray(24)
        struct.pack_into('H', raw, 2, 1 << KeyMap.R1)
        state = SimpleNamespace(q=[.1]*29, dq=[.2]*29, imu_quat_wxyz=[1,0,0,0],
                                imu_gyro=[.3,.4,.5], remote_raw=raw)
        runtime._apply_robot_state(state)
        runtime._set_remote_from_bridge(state)
        runtime.FSM_controller.cur_policy.name = FSMStateName.SKILL_AMP
        runtime._apply_remote_commands()
        self.assertEqual(runtime.state_cmd.skill_cmd, FSMCommand.CMD_SCALEBFM)
        np.testing.assert_allclose(runtime.state_cmd.q, state.q)
        np.testing.assert_allclose(runtime.state_cmd.root_ang_vel_b, state.imu_gyro)
        runtime.state_cmd.skill_cmd = FSMCommand.INVALID
        runtime._apply_remote_commands()
        self.assertEqual(runtime.state_cmd.skill_cmd, FSMCommand.INVALID)

    def test_stale_state_damps_without_running_tracker(self):
        runtime, _ = self.make_runtime()
        runtime.bridge_state_sub.latest.return_value = SimpleNamespace(tick=0)
        frame = runtime.step()
        self.assertTrue(frame.request_damping)
        np.testing.assert_array_equal(frame.kp, np.zeros(29))
        runtime.FSM_controller.run.assert_not_called()

    def test_bias_modifier_does_not_select_motion_and_f1_wins(self):
        runtime, _ = self.make_runtime()
        runtime.catalog = MagicMock()
        runtime.policy_input.catalog = runtime.catalog
        buttons = runtime.remote_controller.button_states
        buttons[KeyMap.L2] = buttons[KeyMap.right] = True
        runtime._apply_remote_commands()
        runtime.catalog.move.assert_not_called()
        self.assertAlmostEqual(runtime.state_cmd.ball_y_bias, -.05)
        buttons[KeyMap.F1] = buttons[KeyMap.R1] = True
        runtime._apply_remote_commands()
        self.assertEqual(runtime.state_cmd.skill_cmd, FSMCommand.PASSIVE)

if __name__ == '__main__':
    unittest.main()
