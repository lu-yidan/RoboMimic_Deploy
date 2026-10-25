"""PHP camera and C++ bridge contract tests: no USB, DDS participant or motors."""
import copy
from pathlib import Path
import struct
import time
import unittest
from unittest.mock import patch
import numpy as np

from common.utils import FSMStateName
from common.remote_controller import KeyMap
from policy.php_parkour.real_depth import (DepthProjector, DepthBuffer, Intrinsics,
                                         load_camera_config)
ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT/'policy/php_parkour/config/real_depth.yaml'


def synthetic_frame():
    f = 53 / np.tan(np.deg2rad(89.5/2))
    return np.full((60,106),1.5,np.float32), Intrinsics(106,60,f,f,53,30)


def make_state(tick=1, keys=()):
    from bridge.python.bridge_state_dds import BridgeState
    import yaml
    q = yaml.safe_load((ROOT/'policy/php_parkour/config/php.yaml').read_text())['default_joint_pos']
    raw = bytearray(24)
    struct.pack_into('<H',raw,2,sum(1<<k for k in keys))
    return BridgeState(tick=tick,q=q,dq=[0.]*29,imu_quat_wxyz=[1.,0,0,0],
                       imu_gyro=[0.,0,0],remote_raw=list(raw))


def offline_config():
    from deploy_real.config import Config
    from common.deployment_config import load_php_config
    cfg = Config()
    cfg.log_enabled = False
    cfg.scalebfm.enabled = False
    cfg.php = load_php_config()
    cfg.php['enabled'] = True
    cfg.php['camera'] = load_camera_config(CAMERA)
    return cfg


class RealDepthTest(unittest.TestCase):
    def setUp(self):
        self.config = load_camera_config(CAMERA)

    def test_identity_projection_preserves_training_depth(self):
        from policy.php_parkour.depth_processing import preprocess_depth
        depth,k = synthetic_frame()
        depth[:, :53] = .8
        result,coverage = DepthProjector(self.config).project(depth,k)
        np.testing.assert_allclose(result,preprocess_depth(depth),atol=1e-6)
        self.assertEqual(coverage,1.)

    def test_mount_rotation_and_translation_change_projection(self):
        depth,k = synthetic_frame()
        reference,_ = DepthProjector(self.config).project(depth,k)
        moved = copy.deepcopy(self.config)
        moved['mount']['rpy_deg'][1] += 2
        result,coverage = DepthProjector(moved).project(depth,k)
        self.assertFalse(np.allclose(result,reference))
        self.assertGreater(coverage,.9)
        moved = copy.deepcopy(self.config)
        moved['mount']['position_m'][0] += .02
        result,_ = DepthProjector(moved).project(depth,k)
        self.assertFalse(np.allclose(result,reference))

    def test_intrinsics_depth_units_and_occlusion(self):
        depth,k = synthetic_frame()
        buffer = DepthBuffer(self.config)
        depth[:, :] = 0 # all-zero sensor frames must NOT look like clear terrain
        buffer.submit(depth,k,time.monotonic(),1)
        self.assertIsNone(buffer.latest().image)
        self.assertIn('coverage',buffer.latest().error)
        with self.assertRaises(ValueError):
            DepthProjector(self.config).project(depth,Intrinsics(106,60,0,1,53,30))

    def test_frame_freshness_duplicate_and_disconnect(self):
        depth,k = synthetic_frame()
        buffer = DepthBuffer(self.config)
        now=time.monotonic()
        buffer.submit(depth,k,now,1)
        self.assertIsNotNone(buffer.latest().image)
        buffer.submit(depth,k,time.monotonic(),1)
        self.assertIsNone(buffer.latest().image)
        buffer.submit(depth,k,time.monotonic()-.3,2)
        self.assertIsNone(buffer.latest().image)
        buffer.submit(depth,k,time.monotonic()+1,3)
        self.assertIsNone(buffer.latest().image)
        buffer.submit(depth,k,time.monotonic(),4)
        with patch('policy.php_parkour.real_depth.time.monotonic',return_value=time.monotonic()+1):
            self.assertIsNone(buffer.latest().image)
        buffer.invalidate('USB disconnected')
        self.assertEqual(buffer.latest().error,'USB disconnected')


    def test_sdk_capture_scale_and_clock_without_usb(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from policy.php_parkour.real_depth import RealSenseDepth
        import sys
        depth,k=synthetic_frame()
        for domain,accepted in [('global',True),('hardware',False)]:
            source=RealSenseDepth(self.config)
            rs=MagicMock()
            rs.timestamp_domain.global_time='global'
            rs.timestamp_domain.system_time='system'
            rs.distortion.none='none'
            device=MagicMock();device.get_info.return_value='mock-camera'
            rs.context.return_value.query_devices.return_value=[device]
            pipe=rs.pipeline.return_value
            sensor=pipe.start.return_value.get_device.return_value.first_depth_sensor.return_value
            sensor.supports.return_value=True
            sensor.get_depth_scale.return_value=.001
            frame=MagicMock()
            frame.get_frame_timestamp_domain.return_value=domain
            frame.get_frame_number.return_value=1
            frame.get_data.return_value=np.full(depth.shape,1500,np.uint16)
            frame.profile.as_video_stream_profile.return_value.get_intrinsics.return_value=SimpleNamespace(
                width=k.width,height=k.height,fx=k.fx,fy=k.fy,ppx=k.cx,ppy=k.cy,
                model='rectified-brown',coeffs=[0.]*5)
            def wait(**kwargs):
                frame.get_timestamp.return_value=(time.time()-.005)*1000
                source._stop.set()
                return SimpleNamespace(get_depth_frame=lambda:frame)
            pipe.wait_for_frames.side_effect=wait
            with patch.dict(sys.modules,pyrealsense2=rs):
                source._capture()
            pipe.stop.assert_called_once()
            if accepted:
                self.assertIsNotNone(source.latest().image)
                raw,_,_=source.recording()
                np.testing.assert_allclose(raw,1.5)
            else:
                self.assertIsNone(source.latest().image)
                self.assertIn('timestamp',source.latest().error)


class PHPBridgeOfflineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail loudly if any offline path constructs transport (including the finalizer).
        cls.dds = patch('cyclonedds.domain.DomainParticipant.__init__',side_effect=AssertionError('DDS opened'))
        cls.dds.start()

    @classmethod
    def tearDownClass(cls):
        cls.dds.stop()

    def setUp(self):
        from bridge.python.policy_runtime import PolicyRuntime
        cfg = offline_config()
        self.source = DepthBuffer(cfg.php['camera'])
        self.runtime = PolicyRuntime(cfg,connect=False,depth_source=self.source)
        self.addCleanup(self.runtime.close)
        self.tick = 0

    def step(self,keys=(),depth=True):
        self.tick += 1
        if depth:
            frame,k = synthetic_frame()
            self.source.submit(frame,k,time.monotonic(),self.tick)
        return self.runtime.step(make_state(self.tick,keys))

    def enter_php(self):
        self.step([KeyMap.A])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)
        frame=self.step([KeyMap.A,KeyMap.L1])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_PHP_PARKOUR)
        self.assertFalse(frame.request_damping)
        return frame

    def test_entry_buttons_pd_limits_and_wire_roundtrip(self):
        from bridge.python.bridge_cmd_dds import BridgeCmd
        frame=self.enter_php()
        self.assertTrue(np.isfinite(frame.q_des).all())
        self.assertTrue((frame.kp>0).all())
        state=self.runtime.state_cmd
        torque=frame.kp*(frame.q_des-state.q)-frame.kd*state.dq
        self.assertTrue((np.abs(torque)<=self.runtime._php_effort_limits+1e-4).all())
        decoded=BridgeCmd.deserialize(BridgeCmd(seq=frame.seq,q_des=frame.q_des.tolist(),
            kp=frame.kp.tolist(),kd=frame.kd.tolist()).serialize())
        np.testing.assert_allclose(decoded.q_des,frame.q_des)
        self.step([KeyMap.A]) # modifier released before A: no AMP switch
        self.step([])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_PHP_PARKOUR)
        self.step([KeyMap.F2]);self.assertTrue(state.php_high_speed)
        self.step([KeyMap.F2]);self.assertTrue(state.php_high_speed)
        self.step([KeyMap.Y])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_SMP_RECOVERY)
        self.step([KeyMap.L1,KeyMap.A])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_PHP_PARKOUR)
        self.step([KeyMap.F1,KeyMap.L1,KeyMap.A])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)

    def test_loss_damps_same_tick_and_requires_new_chord(self):
        self.enter_php()
        self.source.invalidate('USB disconnected')
        frame=self.step([KeyMap.L1,KeyMap.A],depth=False)
        self.assertTrue(frame.request_damping)
        self.assertTrue((frame.kp==0).all())
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)
        self.step([KeyMap.L1,KeyMap.A]) # healthy sensor cannot auto-reenter while chord held
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)
        self.step([]);self.step([KeyMap.L1,KeyMap.A])
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_PHP_PARKOUR)

    def test_missing_depth_rejects_and_bridge_loss_clears_php(self):
        self.step([KeyMap.L1,KeyMap.A],depth=False)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)
        self.step([]);self.step([KeyMap.L1,KeyMap.A])
        frame=self.runtime.step(make_state(tick=0))
        self.assertTrue(frame.request_damping)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)

    def test_invalid_state_and_inference_failure_damp(self):
        self.enter_php()
        bad=make_state(20);bad.q[0]=float('nan')
        self.assertTrue(self.runtime.step(bad).request_damping)
        self.step([]);self.step([KeyMap.L1,KeyMap.A])
        policy=self.runtime.FSM_controller.php_policy
        with patch.object(policy.student,'run',side_effect=RuntimeError('injected inference failure')):
            self.assertTrue(self.step([]).request_damping)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)

    def test_no_camera_does_not_damp_other_policies(self):
        self.source.invalidate('camera not started')
        self.step([KeyMap.start], depth=False)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.FIXEDPOSE)
        self.step([KeyMap.A], depth=False)
        frame=self.step([], depth=False)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_AMP)
        self.assertFalse(frame.request_damping)
        frame=self.step([KeyMap.L1,KeyMap.A], depth=False)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_AMP)
        self.assertFalse(frame.request_damping)
        frame=self.step([KeyMap.Y], depth=False)
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_SMP_RECOVERY)
        self.assertFalse(frame.request_damping)

    def test_external_camera_reader_loss_and_restart(self):
        import tempfile
        from policy.php_parkour.depth_ipc import DepthReader, DepthWriter
        from policy.php_parkour.real_depth import DepthSample
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'depth.npz'
            cfg=self.runtime.php_config['camera']
            reader=DepthReader(cfg,path)
            self.runtime.depth_source=reader
            writer=DepthWriter(path,cfg)
            def publish():
                writer.write(DepthSample(np.zeros((58,87),np.float32),time.monotonic(),1,1.,''))
                reader.poll()
            publish()
            frame=self.runtime.step(make_state(1,[KeyMap.L1,KeyMap.A]))
            self.assertFalse(frame.request_damping)
            self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_PHP_PARKOUR)
            writer.write(DepthSample(error='Camera stopped'));reader.poll()
            self.assertTrue(self.runtime.step(make_state(2,[KeyMap.L1,KeyMap.A])).request_damping)
            writer=DepthWriter(path,cfg);publish()
            self.runtime.step(make_state(3,[KeyMap.L1,KeyMap.A]))
            self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)
            self.runtime.step(make_state(4,[]))
            self.runtime.step(make_state(5,[KeyMap.L1,KeyMap.A]))
            self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.SKILL_PHP_PARKOUR)

    def test_depth_expires_during_inference(self):
        self.enter_php()
        self.runtime.state_cmd.php_depth_time=time.monotonic()-.2
        self.assertTrue(self.runtime._guard_php_output())
        self.assertEqual(self.runtime.FSM_controller.cur_policy.name,FSMStateName.PASSIVE)

    def test_stalled_robot_tick_damps(self):
        self.enter_php()
        self.runtime._php_tick_time=time.monotonic()-.2
        self.assertTrue(self.runtime.step(make_state(self.tick)).request_damping)


class PHPTmuxTest(unittest.TestCase):
    def test_layout_only_prefills_onboard_commands(self):
        import json, os, subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            fake=directory/'tmux'
            fake.write_text('#!'+sys.executable+"\nimport json,os,sys\nwith open(os.environ['TMUX_TEST_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nif sys.argv[1]=='has-session': sys.exit(1)\nprint('%1')\n")
            fake.chmod(0o755)
            log=directory/'calls.jsonl'
            env=dict(os.environ,PATH=str(directory)+os.pathsep+os.environ['PATH'],
                TMUX_TEST_LOG=str(log),LYD_DEPLOY_CONFIG='/tmp/php config.yaml',
                REPO_DIR='/tmp/repo space',PYTHON_BIN='/tmp/python env/bin/python')
            subprocess.run(['bash',str(ROOT/'tools/deploy/start_tmux_layout.sh'),'test-php','--detach'],
                           env=env,check=True,capture_output=True)
            calls=[json.loads(line) for line in log.read_text().splitlines()]
            keys=[call for call in calls if call[0]=='send-keys']
            self.assertEqual(len(keys),7)
            labels=[call for call in calls if call[0]=='set-option' and '@purpose' in call]
            self.assertEqual(len(labels),7)
            self.assertTrue(any('电机通信' in call[-1] for call in labels))
            self.assertTrue(any('Camera' in call[-1] for call in labels))
            self.assertTrue(any('Dashboard' in call[-1] for call in labels))
            self.assertEqual(sum('pane-border-status' in call for call in calls),2)
            self.assertEqual(sum(call[0]=='split-window' for call in calls),5)
            self.assertTrue(all('-l' in call and 'Enter' not in call and 'C-m' not in call for call in keys))
            commands='\n'.join(call[-1] for call in keys)
            self.assertIn('bridge/python/deploy_policy.py',commands)
            self.assertIn('LYD_DEPLOY_CONFIG=',commands)
            self.assertIn('tools/diagnostics/check_ball_state.py',commands)
            self.assertIn('sensor_dashboard.py',commands)
            self.assertIn('tools/capture/run_camera.py',commands)
            self.assertIn('run_ball_fuser.sh',commands)
            self.assertEqual(commands.count('LYD_DEPLOY_CONFIG='),3)
            self.assertNotIn('target_ball_detector',commands)
            self.assertNotIn('rm -rf',commands)
