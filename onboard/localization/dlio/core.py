"""DLIO/ACP geometry and acquisition-time checks, independent of ROS and DDS."""
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from onboard.perception.lidar.mid360_to_base import compute_mid360_to_base_transform
from policy.acp.observations import rotation
from policy.acp.state_ipc import boot_id


def atomic_json(path, payload):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    try:
        temporary.write_text(json.dumps(payload, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    with Path(path).expanduser().open() as stream:
        text = stream.read(65537)
    if len(text) > 65536:
        raise ValueError('Oversized localization IPC')
    return json.loads(text)


class HostClock:
    """Map Unix ROS stamps to this boot's monotonic clock; never re-stamp old data."""
    def __init__(self, wall=None, mono=None):
        self.offset = (time.time() if wall is None else wall) - (time.monotonic() if mono is None else mono)
        self.failed = False

    def convert(self, stamp, max_age=.15, wall=None, mono=None):
        wall = time.time() if wall is None else wall
        mono = time.monotonic() if mono is None else mono
        if abs((wall-mono)-self.offset) > .02:
            self.failed = True
        if self.failed:
            raise ValueError('Host clock stepped: restart localization and recalibrate')
        result = float(stamp) - self.offset
        if not math.isfinite(result) or result <= 0 or not 0 <= mono-result <= max_age:
            raise ValueError('Acquisition stamp stale/future or not host Unix time')
        return result


def imu_si(acceleration, angular_velocity, unit='g'):
    accel, gyro = np.asarray(acceleration, dtype=float), np.asarray(angular_velocity, dtype=float)
    if accel.shape != (3,) or gyro.shape != (3,) or not np.isfinite(np.r_[accel, gyro]).all():
        raise ValueError('Nonfinite/invalid IMU vector')
    if unit not in ('g', 'm/s2'):
        raise ValueError('Declare acceleration unit g or m/s2 explicitly')
    return accel * (9.80665 if unit == 'g' else 1.), gyro


def quaternion_wxyz(matrix):
    """Rotation matrix -> unit quaternion (stable at 180 degrees)."""
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or np.linalg.det(r) < .99:
        raise ValueError('Invalid rotation matrix')
    candidates = np.array([1+np.trace(r), 1+r[0, 0]-r[1, 1]-r[2, 2],
                           1-r[0, 0]+r[1, 1]-r[2, 2], 1-r[0, 0]-r[1, 1]+r[2, 2]])
    index = int(np.argmax(candidates))
    w, x, y, z = 0, 1, 2, 3
    q = np.empty(4)
    q[index] = math.sqrt(max(0, candidates[index])) / 2
    d = 4*q[index]
    if index == w:
        q[x], q[y], q[z] = (r[2, 1]-r[1, 2])/d, (r[0, 2]-r[2, 0])/d, (r[1, 0]-r[0, 1])/d
    elif index == x:
        q[w], q[y], q[z] = (r[2, 1]-r[1, 2])/d, (r[0, 1]+r[1, 0])/d, (r[0, 2]+r[2, 0])/d
    elif index == y:
        q[w], q[x], q[z] = (r[0, 2]-r[2, 0])/d, (r[0, 1]+r[1, 0])/d, (r[1, 2]+r[2, 1])/d
    else:
        q[w], q[x], q[y] = (r[1, 0]-r[0, 1])/d, (r[0, 2]+r[2, 0])/d, (r[1, 2]+r[2, 1])/d
    return q / np.linalg.norm(q)


def pelvis_pose(position, quaternion, waist, head_angle, lidar_angle):
    waist = np.asarray(waist, dtype=float)
    position = np.asarray(position, dtype=float)
    if (waist.shape != (3,) or position.shape != (3,) or
            not np.isfinite(np.r_[waist, position, head_angle, lidar_angle]).all()):
        raise ValueError('Invalid mount/joint/position values')
    # Nominal G1 URDF chain, with explicitly reviewed fixed head/mount angles.
    pelvis_lidar = compute_mid360_to_base_transform(*waist, q_head=head_angle, q_mid=lidar_angle)
    # Analytic inverse of a rigid transform; avoid a general LAPACK solve at 100 Hz.
    world_pelvis_rotation = rotation(quaternion) @ pelvis_lidar[:3, :3].T
    world_pelvis_position = position-world_pelvis_rotation @ pelvis_lidar[:3, 3]
    return world_pelvis_position, quaternion_wxyz(world_pelvis_rotation)


class WaistHistory:
    """Only fresh unique bridge ticks; bounded nearest-time lookup without extrapolation."""
    def __init__(self):
        self.rows = []
        self.last_tick = None
        self.last_stamp = -1.
        self.source_id = None

    def add(self, tick, stamp, q):
        if tick == self.last_tick:
            return False
        if tick <= 0 or (self.last_tick is not None and tick < self.last_tick) or stamp <= self.last_stamp:
            raise ValueError('Bridge tick/time reset; restart exporter and localization')
        q = np.asarray(q, dtype=float)
        if q.shape != (29,) or not np.isfinite(q).all() or not math.isfinite(stamp):
            raise ValueError('Invalid bridge joints')
        self.rows.append([float(stamp), int(tick)] + q[12:15].tolist())
        self.rows = self.rows[-96:]
        self.last_tick, self.last_stamp = int(tick), float(stamp)
        return True

    def load(self, packet, now):
        if packet.get('schema') != 1 or packet.get('boot_id') != boot_id():
            raise ValueError('Waist IPC schema/boot mismatch')
        source = packet.get('source_id')
        if not isinstance(source, str) or not source:
            raise ValueError('Waist source missing')
        if self.source_id is not None and source != self.source_id:
            raise ValueError('Waist exporter restarted; restart localization')
        rows = np.asarray(packet['rows'], dtype=float)
        if (rows.ndim != 2 or rows.shape[1] != 5 or not 1 <= len(rows) <= 96 or not np.isfinite(rows).all()
                or np.any(np.diff(rows[:, 0]) <= 0) or np.any(np.diff(rows[:, 1]) <= 0)
                or np.any(rows[:, 1] < 1) or np.any(rows[:, 1] != np.floor(rows[:, 1]))
                or not 0 <= now-rows[-1, 0] <= .15):
            raise ValueError('Waist data stale, reordered, or malformed')
        if self.rows and (rows[-1, 0] < self.rows[-1][0] or rows[-1, 1] < self.rows[-1][1]):
            raise ValueError('Waist history moved backwards')
        self.rows, self.source_id = rows.tolist(), source

    def nearest(self, stamp, now, max_skew=.04):
        if not self.rows or not 0 <= now-self.rows[-1][0] <= .15:
            raise ValueError('Waist stream stalled')
        row = min(self.rows, key=lambda r: abs(r[0]-stamp))
        if abs(row[0]-stamp) > max_skew or not 0 <= now-row[0] <= .15:
            raise ValueError('Waist/odometry time skew')
        return np.array(row[2:]), row[0]


class PoseGate:
    """Reject odometry resets/jumps, retaining the first failure until restart."""
    def __init__(self, frame='acp_odom', child='acp_lidar_base'):
        self.frame, self.child = frame, child
        self.previous = None
        self.failed = ''

    def check(self, stamp, position, quat, frame, child):
        if self.failed:
            raise ValueError(self.failed)
        if frame != self.frame or child != self.child:
            raise ValueError('Unexpected DLIO frames')
        r = rotation(quat)
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid odometry position')
        if self.previous is not None:
            old_t, old_p, old_r = self.previous
            dt = stamp-old_t
            if dt == 0:
                return False  # DLIO timer can re-publish the same IMU stamp.
            angle = math.acos(float(np.clip((np.trace(old_r.T @ r)-1)/2, -1, 1)))
            if dt < 0 or dt > .5 or np.linalg.norm(position-old_p) > .03+3*dt or angle > .05+8*dt:
                self.failed = 'DLIO discontinuity/reset; restart and recalibrate world/targets'
                raise ValueError(self.failed)
        self.previous = (stamp, position.copy(), r)
        return True
