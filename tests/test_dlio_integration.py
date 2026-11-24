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
from onboard.perception.lidar.mid360_to_base import compute_mid360_to_base_transform, Rz, Ry
from policy.acp.observations import rotation
from policy.acp.state_ipc import StateWriter, StateReader, boot_id
from policy.acp.goal_ipc import GoalReader, relative_goal, write_goal


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
    with pytest.raises(ValueError, match='discontinuity'):
        gate.check(100.02, [0, 0, 0], [1, 0, 0, 0], 'acp_odom', 'acp_lidar_base')


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
