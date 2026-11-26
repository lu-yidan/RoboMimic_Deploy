"""Physical conventions and failure cases that can silently corrupt ACP input."""
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from onboard.localization.dlio.core import (HostClock, PoseGate, WaistHistory, atomic_json,
                                           imu_si, pelvis_pose, quaternion_wxyz)
from onboard.localization.dlio.relay import point_times
from onboard.localization.dlio.check import assess, assess_goal
from onboard.localization.dlio.calibrate import validate_calibration
from onboard.perception.lidar.mid360_to_base import compute_mid360_to_base_transform, Rz, Ry
from policy.acp.observations import rotation
from policy.acp.state_ipc import StateWriter, StateReader, boot_id
from policy.acp.goal_ipc import GoalReader, relative_goal, write_goal
import policy.acp.goal_ipc as goal_ipc


def test_saved_height_reanchors_new_world_without_reusing_old_floor(tmp_path, monkeypatch):
    from onboard.localization.dlio import calibrate
    status = dict(schema=1, boot_id=boot_id(), session='first', frame_id='acp_odom',
                  pose_valid=True, sampled_at=100., pose_captured_at=99.99,
                  imu_age_s=.005, raw_cloud_age_s=.06, registration_age_s=.08,
                  speed_mps=.01, turn_rate_rps=.02, mount_angles_rad=[0., 0.],
                  pelvis_pos_w=[0., 0., -.5])
    profile = tmp_path/'height.json'
    monkeypatch.setattr(calibrate.time, 'monotonic', lambda: 100.01)
    def run(*args):
        atomic_json(tmp_path/'acp-dlio-status.json', status)
        monkeypatch.setattr('sys.argv', ['calibrate', '--ipc-dir', str(tmp_path),
                             '--profile', str(profile), '--confirm-mount', *args])
        calibrate.main()
        return json.loads((tmp_path/'acp-dlio-calibration.json').read_text())
    assert run('--pelvis-height', '.78')['ground_height'] == pytest.approx(-1.28)
    assert 'ground_height' not in json.loads(profile.read_text())
    status.update(session='second', pelvis_pos_w=[0., 0., 2.])
    result = run('--reuse-height')
    assert result['session'] == 'second'
    assert result['ground_height'] == pytest.approx(1.22)
    status['mount_angles_rad'] = [.1, 0.]
    with pytest.raises(SystemExit):
        run('--reuse-height')
    status.update(mount_angles_rad=[0., 0.], speed_mps=.2)
    with pytest.raises(SystemExit):
        run('--reuse-height')
    status.update(speed_mps=.01, localization_fault='ACP_REGISTRATION_REJECT test')
    with pytest.raises(SystemExit):
        run('--reuse-height')
    assert json.loads((tmp_path/'acp-dlio-calibration.json').read_text()) == result


def test_registration_rejection_takes_priority_over_missing_calibration():
    status = dict(boot_id='host', sampled_at=100., calibration_ready=False,
                  localization_fault='ACP_REGISTRATION_REJECT position_step=0.11; restart required')
    assert assess(status, 100.01, 'host')[0] == 'registration_rejected'


def test_relay_registration_log_latches_first_fault_and_withdraws_state(tmp_path):
    # Execute the actual callback without importing ROS or creating a participant.
    import ast
    from pathlib import Path
    from onboard.localization.dlio import relay
    tree = ast.parse(Path(relay.__file__).read_text())
    callback = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == 'estimator_log')
    namespace = dict(directory=tmp_path, args=SimpleNamespace(publish_acp=True))
    exec(compile(ast.Module(body=[callback], type_ignores=[]), relay.__file__, 'exec'), namespace)
    node = SimpleNamespace(started_wall=100., registration_fault='', error='', stage_errors={})
    path = tmp_path/'acp-state.npz'
    path.write_bytes(b'previous valid state')
    msg = SimpleNamespace(name='dlio_odom_node', stamp=SimpleNamespace(sec=99, nanosec=0),
                          msg='ACP_REGISTRATION_REJECT old')
    namespace['estimator_log'](node, msg)
    assert path.exists() and not node.registration_fault
    msg.stamp.sec = 101
    msg.name = 'other_node'
    namespace['estimator_log'](node, msg)
    assert path.exists()
    msg.name, msg.msg = 'dlio_odom_node', 'ACP_REGISTRATION_REJECT first'
    namespace['estimator_log'](node, msg)
    assert not path.exists() and node.registration_fault == msg.msg
    msg.msg = 'ACP_REGISTRATION_REJECT later'
    namespace['estimator_log'](node, msg)
    assert node.stage_errors['registration'] == 'ACP_REGISTRATION_REJECT first'


def test_calibration_requires_fresh_sources_and_explains_motion_limits():
    status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  pose_valid=True, sampled_at=100., pose_captured_at=99.99,
                  imu_age_s=.005, raw_cloud_age_s=.06, registration_age_s=.08,
                  speed_mps=.01, turn_rate_rps=.02)
    validate_calibration(status, .78, 100.01)
    status['speed_mps'] = .08
    with pytest.raises(ValueError, match='speed=0.0800'):
        validate_calibration(status, .78, 100.01)
    status.update(speed_mps=.01, turn_rate_rps=.15)
    with pytest.raises(ValueError, match='turn_rate=0.1500'):
        validate_calibration(status, .78, 100.01)
    status.update(turn_rate_rps=.02, registration_age_s=.245)
    with pytest.raises(ValueError, match='registration not fresh'):
        validate_calibration(status, .78, 100.01)


def test_calibration_rejects_jump_even_if_previous_pose_is_recent():
    status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  pose_valid=True, sampled_at=100., pose_captured_at=99.99,
                  imu_age_s=.005, raw_cloud_age_s=.06, registration_age_s=.08,
                  speed_mps=.01, turn_rate_rps=.02, localization_fault='latched jump')
    with pytest.raises(ValueError, match='latched jump'):
        validate_calibration(status, .78, 100.01)
    status['localization_fault'] = ''
    for height in (.2, 1.3, float('nan')):
        with pytest.raises(ValueError, match='height must be'):
            validate_calibration(status, height, 100.01)
    status['speed_mps'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_calibration(status, .78, 100.01)


def test_bringup_diagnosis_distinguishes_missing_calibration_from_stale_pose():
    status = dict(boot_id='host', sampled_at=100., pose_captured_at=99.99, pose_valid=True,
                  imu_age_s=.01, registration_age_s=.08, acp_output_enabled=True,
                  acp_output_valid=False, calibration_ready=False)
    assert assess(status, 100.01, 'host')[0] == 'calibration'
    assert assess(status, 100.01, 'other')[0] == 'host'
    assert assess(status, 101., 'host')[0] == 'relay'
    status.update(acp_output_valid=True, calibration_ready=True)
    assert assess(status, 100.01, 'host')[0] == 'ready'
    # The checker must account for time since the status file was sampled.
    status.update(pose_valid=False, registration_age_s=.24)
    assert assess(status, 100.02, 'host')[0] == 'registration'
    status.update(imu_age_s=.14)
    assert assess(status, 100.02, 'host')[0] == 'imu'
    # Recovering the point stream must not hide an already latched world reset.
    status['stage_errors'] = dict(odom='DLIO discontinuity/reset; restart and recalibrate world/targets')
    assert assess(status, 100.02, 'host')[0] == 'localization_reset'


def test_articulated_head_is_not_pelvis_odometry():
    world_pelvis = np.eye(4)
    world_pelvis[:3, :3] = Rz(.8) @ Ry(.1)
    world_pelvis[:3, 3] = [1.2, -.4, -.5]  # DLIO's origin is not the ground.
    for waist in ([0., 0., 0.], [.7, -.2, .3], [-.5, .15, -.3]):
        world_lidar = world_pelvis @ compute_mid360_to_base_transform(*waist)
        pos, quat = pelvis_pose(world_lidar[:3, 3], quaternion_wxyz(world_lidar[:3, :3]),
                                waist, .593412, 0.)
        np.testing.assert_allclose(pos, world_pelvis[:3, 3], atol=1e-12)
        np.testing.assert_allclose(rotation(quat), world_pelvis[:3, :3], atol=1e-12)
    for angle in (0., np.pi-1e-10, np.pi, -np.pi):
        np.testing.assert_allclose(rotation(quaternion_wxyz(Rz(angle))), Rz(angle), atol=1e-12)


def test_clock_mapping_preserves_acquisition_and_latches_clock_step():
    clock = HostClock(wall=1800000000., mono=100.)
    assert clock.convert(1800000000.-.03, wall=1800000000., mono=100.) == pytest.approx(99.97)
    with pytest.raises(ValueError, match='stale/future'):
        clock.convert(1800000000.-.2, wall=1800000000., mono=100.)
    with pytest.raises(ValueError, match='clock stepped'):
        clock.convert(1800000001., wall=1800000001., mono=100.)
    with pytest.raises(ValueError, match='clock stepped'):
        clock.convert(1800000000., wall=1800000000., mono=100.)


def test_livox_units_and_duplicate_tick_cannot_renew_waist():
    accel, gyro = imu_si([0, 0, 1], [.2, 0, 0])
    np.testing.assert_allclose(accel, [0, 0, 9.80665])
    np.testing.assert_allclose(gyro, [.2, 0, 0])
    np.testing.assert_allclose(imu_si(accel, gyro, 'm/s2')[0], accel)
    history = WaistHistory()
    assert history.add(10, 100., np.arange(29))
    assert not history.add(10, 100.1, np.arange(29))
    np.testing.assert_allclose(history.nearest(100.02, 100.03)[0], [12, 13, 14])
    with pytest.raises(ValueError, match='stalled'):
        history.nearest(100.02, 100.16)
    with pytest.raises(ValueError, match='reset'):
        history.add(9, 100.2, np.zeros(29))


def test_waist_session_skew_and_pose_reset_are_rejected():
    history = WaistHistory()
    packet = dict(schema=1, boot_id=boot_id(), source_id='first', rows=[[99.98, 1, .1, .2, .3], [100., 2, .2, .3, .4]])
    history.load(packet, 100.01)
    with pytest.raises(ValueError, match='skew'):
        history.nearest(99.9, 100.01)
    packet['source_id'] = 'second'
    with pytest.raises(ValueError, match='restarted'):
        history.load(packet, 100.01)
    gate = PoseGate()
    assert gate.check(100., [0, 0, 0], [1, 0, 0, 0], 'acp_odom', 'acp_lidar_base')
    assert not gate.check(100., [0, 0, 0], [1, 0, 0, 0], 'acp_odom', 'acp_lidar_base')
    with pytest.raises(ValueError, match='discontinuity'):
        gate.check(100.01, [2, 0, 0], [1, 0, 0, 0], 'acp_odom', 'acp_lidar_base')
    fault = gate.failure_details.copy()
    assert fault['translation_m'] == 2.
    assert fault['dt_s'] == pytest.approx(.01)
    assert fault['translation_m'] > fault['translation_limit_m']
    with pytest.raises(ValueError, match='discontinuity'):
        gate.check(100.02, [0, 0, 0], [1, 0, 0, 0], 'acp_odom', 'acp_lidar_base')
    assert gate.failure_details == fault


def test_point_timestamp_ns_seconds_and_row_padding():
    values = np.zeros((2, 80, 4), dtype=np.float64)
    stamp = time.time()-.15
    values[:, :, 2] = (stamp+np.arange(160).reshape(2, 80)*.0005)*1e9
    msg = SimpleNamespace(fields=[SimpleNamespace(name='timestamp', datatype=8, count=1, offset=16)],
                          width=80, height=2, point_step=32, row_step=80*32, is_bigendian=False,
                          header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp), nanosec=int(stamp%1*1e9))),
                          data=values.tobytes())
    first, last = point_times(msg)
    assert first == pytest.approx(stamp)
    assert last-first == pytest.approx(.0795, abs=1e-6)
    values[:, :, 2] *= 1e-9  # Seconds masquerading as ns must not disable deskew silently.
    msg.data = values.tobytes()
    with pytest.raises(ValueError, match='absolute host-epoch ns'):
        point_times(msg)


def test_real_ipc_negative_world_height_and_static_plane(tmp_path):
    path = tmp_path/'state.npz'
    now = time.monotonic()-.01
    StateWriter(path, 'acp_odom').write(pelvis_pos_w=[1, 2, -.5], pelvis_quat_w=[1, 0, 0, 0],
                                     captured_at=now, sequence=0, ground_height=-1.28)
    reader = StateReader(dict(real=dict(frame_id='acp_odom')), path)
    sample = reader.poll()
    assert sample.error == ''
    np.testing.assert_allclose(sample.terrain[:, 2], .28, atol=1e-6)


def test_live_goal_latches_world_point_and_rejects_other_session(tmp_path):
    now = time.monotonic()
    status = dict(schema=1, boot_id=boot_id(), session='session-1', frame_id='acp_odom',
                  pose_valid=True, pose_captured_at=now-.01, sampled_at=now,
                  pelvis_pos_w=[2., 3., -.5], pelvis_quat_w=quaternion_wxyz(Rz(np.pi/2)).tolist())
    target, heading = relative_goal(status, 1., 0., 0.)
    np.testing.assert_allclose(target, [2., 4.], atol=1e-10)
    assert heading == pytest.approx(np.pi/2)
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    write_goal(tmp_path, status, target, heading, now=now)
    reader = GoalReader(tmp_path)
    np.testing.assert_allclose(reader.read(now=now)[0], [2., 4.])
    status['pelvis_pos_w'] = [5., 6., -.5]
    status.update(sampled_at=now+3., pose_captured_at=now+3.)
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    # Accepted world target remains fixed and does not expire after two seconds.
    np.testing.assert_allclose(reader.read(now=now+3.)[0], [2., 4.])
    status['session'] = 'session-2'
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    with pytest.raises(ValueError, match='DLIO restarted'):
        reader.read(now=now+3.)
    with pytest.raises(ValueError, match='different localization session'):
        GoalReader(tmp_path).read(now=now+3.)


def test_live_goal_clock_is_sampled_after_each_ipc_read(tmp_path, monkeypatch):
    status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  pose_valid=True, pose_captured_at=100., sampled_at=100.005)
    packet = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  request_id='goal-1', captured_at=100.015, target_position=[1., 2.], target_heading=.2)
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    atomic_json(tmp_path/'acp-goal.json', packet)
    clock, original_read = [100.], goal_ipc.read_packet

    def concurrent_read(path, **kwargs):
        # Producer writes during a read, after the reader's previous clock sample.
        clock[0] += .01
        return original_read(path, **kwargs)

    monkeypatch.setattr(goal_ipc, 'read_packet', concurrent_read)
    monkeypatch.setattr(goal_ipc.time, 'monotonic', lambda: clock[0])
    reader = GoalReader(tmp_path)
    np.testing.assert_allclose(reader.read()[0], [1., 2.])
    status['localization_fault'] = 'latched jump'
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    with pytest.raises(ValueError, match='latched jump'):
        reader.read()
    status['localization_fault'] = ''
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    clock[0] = 100.3
    with pytest.raises(ValueError, match='fresh DLIO'):
        reader.read()


def test_goal_replacement_caches_the_key_of_the_actual_packet(tmp_path, monkeypatch):
    status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  pose_valid=True, pose_captured_at=100., sampled_at=100.)
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    write_goal(tmp_path, status, [1., 0.], 0., now=100.)
    original_read, replaced = goal_ipc.read_packet, [False]

    def concurrent_read(path, **kwargs):
        if path.name == 'acp-goal.json' and not replaced[0]:
            replaced[0] = True
            write_goal(tmp_path, status, [2., 0.], 0., now=100.01)
        return original_read(path, **kwargs)

    monkeypatch.setattr(goal_ipc, 'read_packet', concurrent_read)
    reader = GoalReader(tmp_path)
    np.testing.assert_allclose(reader.read(now=100.02)[0], [2., 0.])
    np.testing.assert_allclose(reader.read(now=100.03)[0], [2., 0.])


def test_goal_diagnosis_does_not_confuse_session_identity_with_policy_acceptance():
    status = dict(session='new', frame_id='acp_odom')
    goal = dict(schema=1, boot_id='host', session='old', frame_id='acp_odom',
                captured_at=100., request_id='request', target_position=[0., 0.], target_heading=0.)
    assert assess_goal(status, None, 100.1, 'host')[0] == 'missing'
    assert assess_goal(status, goal, 100.1, 'host')[0] == 'wrong_session'
    goal['session'] = 'new'
    assert assess_goal(status, goal, 100.1, 'host')[0] == 'current_session_fresh'
    assert assess_goal(status, goal, 103., 'host')[0] == 'current_session_existing'
    assert assess_goal(status, goal, 99., 'host')[0] == 'invalid'
    goal['target_position'] = [float('nan'), 0.]
    assert assess_goal(status, goal, 100.1, 'host')[0] == 'invalid'


def test_unaccepted_goal_expires_but_fresh_resubmission_recovers(tmp_path):
    status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  pose_valid=True, pose_captured_at=100., sampled_at=100.)
    write_goal(tmp_path, status, [1., 0.], 0., now=100.)
    status.update(pose_captured_at=103., sampled_at=103.)
    atomic_json(tmp_path/'acp-dlio-status.json', status)
    reader = GoalReader(tmp_path)
    with pytest.raises(ValueError, match='expired before policy acceptance'):
        reader.read(now=103.)
    assert reader.target is None
    write_goal(tmp_path, status, [2., 0.], 0., now=103.)
    np.testing.assert_allclose(reader.read(now=103.)[0], [2., 0.])


def test_goal_command_requires_matching_policy_acceptance(tmp_path):
    now = time.monotonic()
    status = dict(schema=1, boot_id=boot_id(), session='live', frame_id='acp_odom',
                  pose_valid=True, pose_captured_at=now, sampled_at=now)
    packet = write_goal(tmp_path, status, [1., 0.], 0., now=now)
    with pytest.raises(ValueError, match='NOT confirmed'):
        goal_ipc.wait_for_acceptance(tmp_path, packet, timeout=0.)
    reader = SimpleNamespace(session='live', request_id='previous-request')
    goal_ipc.acknowledge_goal(tmp_path, reader)
    with pytest.raises(ValueError, match='NOT confirmed'):
        goal_ipc.wait_for_acceptance(tmp_path, packet, timeout=0.)
    reader.request_id = packet['request_id']
    goal_ipc.acknowledge_goal(tmp_path, reader)
    ack = goal_ipc.wait_for_acceptance(tmp_path, packet, timeout=0.)
    assert ack['request_id'] == packet['request_id']
    packet['session'] = 'restarted'
    with pytest.raises(ValueError, match='NOT confirmed'):
        goal_ipc.wait_for_acceptance(tmp_path, packet, timeout=0.)
