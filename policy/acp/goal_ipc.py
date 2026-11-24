"""Explicit, session-bound world goals. Updating a goal does not reset ACP history."""
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from policy.acp.observations import yaw
from policy.acp.state_ipc import boot_id


def read_packet(path, *, with_key=False):
    with Path(path).expanduser().open() as stream:
        stat = os.fstat(stream.fileno()) if with_key else None
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError('Oversized goal/status IPC')
    packet = json.loads(raw)
    if with_key:
        return packet, (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    return packet


def write_goal(directory, status, target, heading, now=None):
    now = time.monotonic() if now is None else now
    validate_status(status, now)
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all() or not math.isfinite(heading):
        raise ValueError('Finite XY target and heading required')
    payload = dict(schema=1, boot_id=boot_id(), session=status['session'], frame_id=status['frame_id'],
                   request_id=str(uuid.uuid4()), captured_at=now, target_position=target.tolist(),
                   target_heading=float(np.arctan2(np.sin(heading), np.cos(heading))))
    path = Path(directory).expanduser() / 'acp-goal.json'
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    try:
        temp.write_text(json.dumps(payload, allow_nan=False) + '\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return payload


def validate_status(status, now):
    status_age = now-float(status.get('sampled_at', 0.))
    pose_age = now-float(status.get('pose_captured_at', 0.))
    if (status.get('schema') != 1 or status.get('boot_id') != boot_id() or not status.get('pose_valid')
            or status.get('localization_fault')
            or not status.get('session') or status.get('frame_id') != 'acp_odom'
            or not 0 <= status_age <= .25 or not 0 <= pose_age <= .15):
        raise ValueError('Need a fresh DLIO pelvis pose in the current host/session '
                         '(pose_valid=%s, pose_age=%.1f ms, status_age=%.1f ms, fault=%s)' % (
                             status.get('pose_valid'), 1000*pose_age, 1000*status_age,
                             status.get('localization_fault') or 'none'))


def relative_goal(status, forward, left, heading_delta):
    psi = yaw(status['pelvis_quat_w'])
    c, s = np.cos(psi), np.sin(psi)
    target = np.asarray(status['pelvis_pos_w'][:2]) + [c*forward-s*left, s*forward+c*left]
    return target, psi+heading_delta


class GoalReader:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser()
        self.session = self.request_id = self.key = None
        self.target = self.heading = None

    def read(self, now=None):
        status = read_packet(self.directory / 'acp-dlio-status.json')
        # A producer can replace the file during the read. Sample the clock
        # afterwards so that this genuinely fresh packet is not called future.
        observed_at = time.monotonic() if now is None else now
        validate_status(status, observed_at)
        if self.session is None:
            self.session = status['session']
        if self.session != status['session']:
            raise ValueError('DLIO restarted: restart policy and set a new goal')
        path = self.directory / 'acp-goal.json'
        # Match the cache identity to the actual open file, even if the writer
        # replaces the path between operations.
        packet, key = read_packet(path, with_key=True)
        if key != self.key:
            observed_at = time.monotonic() if now is None else now
            validate_status(status, observed_at)
            target = np.asarray(packet['target_position'], dtype=np.float32)
            heading = float(packet['target_heading'])
            if (packet.get('schema') != 1 or packet.get('boot_id') != boot_id()
                    or packet.get('session') != self.session or packet.get('frame_id') != 'acp_odom'
                    or not 0 <= observed_at-float(packet['captured_at']) <= 2.
                    or not isinstance(packet.get('request_id'), str) or not packet['request_id']
                    or target.shape != (2,) or not np.isfinite(target).all() or not math.isfinite(heading)):
                raise ValueError('Invalid/stale goal or goal from a different localization session')
            if packet['request_id'] == self.request_id:
                raise ValueError('Re-stamped duplicate goal request')
            self.key, self.request_id = key, packet['request_id']
            self.target, self.heading = target, heading
        if self.target is None:
            raise ValueError('Set an explicit ACP goal before pressing Y')
        return self.target.copy(), self.heading
