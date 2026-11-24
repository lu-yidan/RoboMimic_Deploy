"""Offline bridge safety checks with synthetic state and inference outputs."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bridge.python.policy_runtime import PolicyRuntime, PolicyCommandFrame
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.remote_controller import KeyMap
from common.utils import FSMCommand, FSMStateName
from deploy_real.scalebfm_input import RealPolicyInput
from policy.acp.state_ipc import StateReader, StateWriter
from policy.acp.state_ipc import StateSample
from policy.acp.goal_ipc import GoalReader, write_goal
from onboard.localization.dlio.core import atomic_json
from policy.acp.state_ipc import boot_id


class ACPBridgeTest(unittest.TestCase):
    def test_live_goal_missing_blocks_acp_and_new_goal_uses_same_policy(self):
        runtime, state = self.fixture(), self.state()
        runtime.acp_config['real']['frame_id'] = 'acp_odom'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            stamp = state._local_received_monotonic
            StateWriter(path, 'acp_odom').write(pelvis_pos_w=[0, 0, .78], pelvis_quat_w=[1, 0, 0, 0],
                                          captured_at=stamp, sequence=1, ground_height=0.)
            runtime.acp_source = StateReader(runtime.acp_config, path)
            runtime.acp_source.poll()
            runtime._acp_goal_reader = GoalReader(folder)
            status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                          pose_valid=True, pose_captured_at=stamp, sampled_at=stamp)
            atomic_json(Path(folder)/'acp-dlio-status.json', status)
            runtime._apply_acp_state(state)
            self.assertFalse(runtime.state_cmd.acp_pose_valid)
            policy = runtime.FSM_controller.acp_policy
            write_goal(folder, status, [1, .2], .3)
            runtime._apply_acp_state(state)
            self.assertTrue(runtime.state_cmd.acp_pose_valid)
            np.testing.assert_allclose(policy.target, [1, .2])
            write_goal(folder, status, [.5, -.4], -.2)
            runtime._apply_acp_state(state)
            self.assertTrue(runtime.state_cmd.acp_pose_valid)
            np.testing.assert_allclose(policy.target, [.5, -.4])
            policy.enter.assert_not_called()
            status['session'] = 'restarted'
            atomic_json(Path(folder)/'acp-dlio-status.json', status)
            runtime._apply_acp_state(state)
            self.assertFalse(runtime.state_cmd.acp_pose_valid)

    def fixture(self):
        runtime = PolicyRuntime.__new__(PolicyRuntime)
        runtime.state_cmd, runtime.policy_output = StateAndCmd(29), PolicyOutput(29)
        runtime.acp_enabled, runtime.connect = True, True
        runtime.acp_config = {'max_age_s': .15, 'max_sync_skew_s': .04, 'real': {'frame_id': 'map'}}
        runtime._acp_last_tick, runtime._acp_tick_time = None, time.monotonic()
        runtime._logger, runtime._acp_error, runtime._cmd_seq = None, None, 0
        runtime._acp_effort_limits = np.arange(1, 30, dtype=np.float32)
        runtime.FSM_controller = MagicMock()
        runtime.FSM_controller.cur_policy.name = FSMStateName.SKILL_ACP
        policy = runtime.FSM_controller.acp_policy
        policy.failed, policy.last_error = False, None
        policy.error.return_value = None
        return runtime

    def state(self, tick=1):
        return SimpleNamespace(tick=tick, q=[0.]*29, dq=[.2]*29,
            imu_quat_wxyz=[1., 0., 0., 0.], imu_gyro=[.1, .2, .3], remote_raw=bytes(24),
            _local_received_monotonic=time.monotonic())

    def test_localization_file_to_robot_inputs_and_torque_bounds(self):
        runtime, state = self.fixture(), self.state()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            writer = StateWriter(path)
            writer.write(pelvis_pos_w=[.2, -.1, .8], pelvis_quat_w=[1., 0., 0., 0.],
                captured_at=state._local_received_monotonic, sequence=1, ground_height=.1)
            runtime.acp_source = StateReader(runtime.acp_config, path)
            self.assertFalse(runtime.acp_source.poll().error)
            runtime._apply_robot_state(state)
            runtime._apply_acp_state(state)
            self.assertTrue(runtime.state_cmd.acp_pose_valid)
            np.testing.assert_array_equal(runtime.state_cmd.acp_ang_vel, np.float32(state.imu_gyro))
            np.testing.assert_allclose(runtime.state_cmd.pelvis_pos_w, [.2, -.1, .8])
            runtime.policy_output.actions = np.linspace(-20, 20, 29)
            runtime.policy_output.kps, runtime.policy_output.kds = np.full(29, 50.), np.full(29, 2.)
            self.assertFalse(runtime._guard_acp_output())
            s, o = runtime.state_cmd, runtime.policy_output
            torque = o.kps*(o.actions-s.q)-o.kds*s.dq
            self.assertTrue(np.all(np.abs(torque) <= runtime._acp_effort_limits+1e-4))
            runtime._acp_tick_time -= 1
            runtime._apply_acp_state(state)
            self.assertFalse(runtime.state_cmd.acp_pose_valid)
            runtime._acp_tick_time += 1
            state._local_received_monotonic += .1
            runtime._apply_acp_state(state)
            self.assertFalse(runtime.state_cmd.acp_pose_valid)
            self.assertIsNone(runtime.state_cmd.acp_terrain)

    def test_delayed_pose_uses_matching_history_and_pd_keeps_latest_state(self):
        runtime = self.fixture()
        sample = StateSample()
        runtime.acp_source = SimpleNamespace(latest=lambda: sample)
        for tick, now, joint in [(1, 100., 1.), (2, 100.02, 2.), (3, 100.08, 3.)]:
            with patch('bridge.python.policy_runtime.time.monotonic', return_value=now):
                state = self.state(tick)
                state.q = [joint]*29
                state.imu_gyro = [joint]*3
                runtime._acp_tick_time = now
                runtime._apply_robot_state(state)
                if tick == 3:
                    sample = StateSample(np.array([0., 0., .78]), np.array([1., 0., 0., 0.]),
                                         np.zeros((187, 3)), 100., 100., 100., 1, 'map', '')
                runtime._apply_acp_state(state)
        s = runtime.state_cmd
        self.assertTrue(s.acp_pose_valid)
        np.testing.assert_array_equal(s.q, np.full(29, 3.))
        np.testing.assert_array_equal(s.acp_joint_pos, np.ones(29))
        np.testing.assert_array_equal(s.acp_ang_vel, np.ones(3))
        self.assertEqual(s.acp_robot_time, 100.)
        self.assertEqual(s.acp_robot_tick, 1)
        runtime.policy_output.kps, runtime.policy_output.kds = np.full(29, 50.), np.full(29, 2.)
        runtime.policy_output.actions = np.zeros(29)
        self.assertFalse(runtime._guard_acp_output())
        o = runtime.policy_output
        self.assertTrue(np.all(np.abs(o.kps*(o.actions-s.q)-o.kds*s.dq) <= runtime._acp_effort_limits+1e-4))
        self.assertTrue(np.all(o.actions > 2.))

    def test_invalid_gyro_old_receipt_and_stalled_tick(self):
        runtime, state = self.fixture(), self.state()
        self.assertIsNone(runtime._acp_state_error(state))
        state.imu_gyro[1] = float('nan')
        self.assertIn('imu_gyro', runtime._acp_state_error(state))
        state.imu_gyro[1] = .2
        state._local_received_monotonic -= 1
        self.assertIn('old', runtime._acp_state_error(state))
        state._local_received_monotonic = time.monotonic()
        runtime._acp_tick_time -= 1
        self.assertIn('stalled', runtime._acp_state_error(state))

    def test_invalid_output_requests_same_tick_damping(self):
        runtime = self.fixture()
        runtime.policy_output.kps, runtime.policy_output.kds = np.full(29, 50.), np.full(29, 2.)
        runtime.policy_output.actions[4] = float('nan')
        self.assertTrue(runtime._guard_acp_output())
        self.assertEqual(runtime.state_cmd.skill_cmd, FSMCommand.PASSIVE)
        runtime.FSM_controller.run.assert_called_once()
        runtime.FSM_controller.cur_policy.name = FSMStateName.PASSIVE
        runtime.policy_output.executed_fsm_state = FSMStateName.PASSIVE.value
        self.assertTrue(runtime._guard_acp_output(was_active=True))

    def test_fixed_pose_ramp_is_bounded_without_requiring_localization(self):
        runtime = self.fixture()
        runtime.FSM_controller.cur_policy.name = FSMStateName.FIXEDPOSE
        runtime.policy_output.executed_fsm_state = FSMStateName.FIXEDPOSE.value
        runtime.FSM_controller.acp_policy.error.side_effect = AssertionError('map checked during FixedPose')
        runtime.FSM_controller.acp_policy.failed = True  # a previous ACP failure does not block standing
        runtime.state_cmd.dq[:] = 4.
        runtime.policy_output.kps, runtime.policy_output.kds = np.full(29, 100.), np.full(29, 3.)
        runtime.policy_output.actions[:] = -2.
        self.assertFalse(runtime._guard_acp_output())
        s, o = runtime.state_cmd, runtime.policy_output
        torque = o.kps*(o.actions-s.q)-o.kds*s.dq
        self.assertTrue(np.all(np.abs(torque) <= runtime._acp_effort_limits+1e-4))
        runtime.policy_output.actions[3] = float('nan')
        self.assertTrue(runtime._guard_acp_output())
        self.assertEqual(runtime.state_cmd.skill_cmd, FSMCommand.PASSIVE)

    def test_y_selects_acp_once_and_damping_preempts(self):
        inputs = RealPolicyInput(acp_enabled=True)
        buttons, state = [False]*16, StateAndCmd(29)
        fsm, output = MagicMock(), PolicyOutput(29)
        buttons[KeyMap.Y] = True
        inputs.update(buttons, fsm, state, output)
        self.assertEqual(state.skill_cmd, FSMCommand.CMD_ACP)
        state.skill_cmd = FSMCommand.INVALID
        inputs.update(buttons, fsm, state, output)
        self.assertEqual(state.skill_cmd, FSMCommand.INVALID)
        buttons[KeyMap.F1] = True
        inputs.update(buttons, fsm, state, output)
        self.assertEqual(state.skill_cmd, FSMCommand.PASSIVE)

    def test_y_retains_smp_without_acp_and_old_chord_does_not_enter(self):
        state, output, fsm = StateAndCmd(29), PolicyOutput(29), MagicMock()
        buttons = [False]*16
        buttons[KeyMap.Y] = True
        RealPolicyInput(acp_enabled=False).update(buttons,fsm,state,output)
        self.assertEqual(state.skill_cmd,FSMCommand.CMD_SMP_RECOVERY)
        state.skill_cmd = FSMCommand.INVALID
        buttons = [False]*16
        buttons[KeyMap.L1] = buttons[KeyMap.X] = True
        RealPolicyInput(acp_enabled=True).update(buttons,fsm,state,output)
        self.assertEqual(state.skill_cmd,FSMCommand.INVALID)

    def test_trace_keeps_exact_graph_io_and_is_bounded(self):
        runtime, state = self.fixture(), self.state()
        runtime._acp_trace_enabled, runtime._acp_trace_limit, runtime._acp_trace = True, 1, []
        runtime.config = SimpleNamespace(bridge_read_only=True)
        runtime._logger = MagicMock()
        policy = runtime.FSM_controller.acp_policy
        policy.manifest = {'policy_sha256': 'fixture'}
        policy.last_observation = np.arange(939, dtype=np.float32)[None]
        policy.last_raw_action = np.arange(23, dtype=np.float32)
        policy.last_action = np.ones(23, np.float32)
        policy.contacts, policy.occupancy = np.zeros((1, 9, 3)), np.ones((1, 9, 1))
        runtime._record_acp_trace(state)
        runtime._record_acp_trace(state)
        self.assertEqual(len(runtime._acp_trace), 1)
        self.assertFalse(runtime._acp_trace_enabled)
        with tempfile.TemporaryDirectory() as folder:
            runtime._logger._bin_path = str(Path(folder)/'log.bin')
            runtime._save_acp_trace()
            with np.load(Path(folder)/'log.acp_trace.npz', allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['planner_obs'], policy.last_observation)
                np.testing.assert_array_equal(trace['raw_actions'][0], policy.last_raw_action)
                self.assertEqual(trace['state_tick'][0], state.tick)
                self.assertTrue(trace['read_only'].item())


class ReadOnlyEntrypointTest(unittest.TestCase):
    def test_read_only_never_constructs_publisher_even_at_exit(self):
        from bridge.python import deploy_policy
        runtime = MagicMock()
        runtime.step.return_value = PolicyCommandFrame(1, np.zeros(29), np.zeros(29), np.zeros(29),
                                                       exit_requested=True)
        with patch.object(deploy_policy, 'Config', return_value=SimpleNamespace(num_joints=29, control_dt=.02)), \
             patch.object(deploy_policy, 'PolicyRuntime', return_value=runtime), \
             patch.object(deploy_policy, 'BridgeCmdPublisher', side_effect=AssertionError('publisher opened')):
            deploy_policy.main(['--read-only'])
        runtime.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
