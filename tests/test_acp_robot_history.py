"""Temporal alignment without restamping, extrapolation or stale-tick renewal."""
from types import SimpleNamespace

import numpy as np
import pytest

from policy.acp.robot_history import RobotHistory


def state(tick, value):
    return SimpleNamespace(tick=tick, q=np.full(29, value), dq=np.full(29, -value),
                           imu_gyro=np.full(3, value/10), imu_quat_wxyz=np.array([1., 0., 0., 0.]))


def test_nearest_capture_preserves_values_and_duplicate_tick_does_not_renew():
    history = RobotHistory()
    first = state(1, 1.)
    history.add(first, 10., 10.)
    first.q[:] = 99
    history.add(state(2, 2.), 10.02, 10.02)
    history.add(state(2, 88.), 10.08, 10.08)
    matched = history.nearest(10.004, 10.08)
    assert matched.tick == 1 and matched.captured_at == 10.
    np.testing.assert_array_equal(matched.q, np.ones(29))
    assert not matched.q.flags.writeable
    assert not matched.imu_quat_wxyz.flags.writeable
    np.testing.assert_array_equal(matched.imu_quat_wxyz, [1., 0., 0., 0.])
    assert history.samples[-1].captured_at == 10.02
    with pytest.raises(ValueError, match='stale'):
        history.nearest(10.16, 10.18)


@pytest.mark.parametrize('failure', ['tick_reset', 'time_reset'])
def test_tick_or_receipt_time_reset_latches(failure):
    history = RobotHistory()
    history.add(state(2, 1.), 10., 10.)
    with pytest.raises(ValueError, match='reset'):
        history.add(state(1 if failure == 'tick_reset' else 3, 1.), 9.99, 10.01)
    with pytest.raises(ValueError, match='reset'):
        history.add(state(4, 1.), 10.02, 10.02)


def test_no_match_and_nonfinite_state_are_rejected():
    history = RobotHistory()
    history.add(state(1, 1.), 10., 10.)
    with pytest.raises(ValueError, match='synchronized'):
        history.nearest(10.08, 10.09)
    with pytest.raises(ValueError, match='future'):
        history.add(state(2, 1.), 10.1, 10.09)
    bad = state(2, 1.)
    bad.imu_gyro[1] = np.nan
    with pytest.raises(ValueError, match='imu_gyro'):
        history.add(bad, 10.02, 10.02)
    with pytest.raises(ValueError, match='stale'):
        history.nearest(9., 10.02)


def test_history_is_bounded_and_missing_robot_stream_expires():
    history = RobotHistory()
    for i in range(100):
        history.add(state(i+1, i), 10.+i*.002, 10.+i*.002)
    assert len(history.samples) == 64
    with pytest.raises(ValueError, match='stale'):
        history.nearest(10.34, 10.36)


@pytest.mark.parametrize('quat', [[0., 0., 0., 0.], [np.nan, 0., 0., 0.], [1., 0., 0.]])
def test_invalid_imu_quaternion_is_not_stored(quat):
    history = RobotHistory()
    bad = state(1, 1.)
    bad.imu_quat_wxyz = quat
    with pytest.raises(ValueError, match='quaternion|imu_quat'):
        history.add(bad, 10., 10.)
    assert not history.samples
