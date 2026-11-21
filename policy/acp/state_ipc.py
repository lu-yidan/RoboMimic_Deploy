"""Timestamped host-local ACP localization and terrain transport; no motor I/O.

Writers publish atomically. Readers preserve capture timestamps and reject
rewritten duplicate packets, stale data, frame changes, and invalid payloads.
"""
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import threading
import time
import zipfile

import numpy as np

from policy.acp.observations import grid_xy
from policy.acp.terrain import flat_grid


def state_path(config):
    return Path(config.get('real', {}).get(
        'state_file', f'/dev/shm/robomimic-{os.getuid()}/acp-state.npz')).expanduser()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


@dataclass(frozen=True)
class StateSample:
    pelvis_pos_w: object = None
    pelvis_quat_w: object = None
    terrain: object = None
    captured_at: float = 0.
    pose_captured_at: float = 0.
    terrain_captured_at: float = 0.
    sequence: int = -1
    frame_id: str = ''
    error: str = 'ACP state producer not started'


class StateWriter:
    """Helper for an independently validated localization/map producer."""
    def __init__(self, path, frame_id='map'):
        self.path, self.frame_id = Path(path).expanduser(), str(frame_id)
        self.boot = boot_id()

    def write(self, *, pelvis_pos_w, pelvis_quat_w, captured_at, sequence,
              terrain=None, ground_height=None, pose_captured_at=None,
              terrain_captured_at=None):
        if (terrain is None) == (ground_height is None):
            raise ValueError('Provide either the complete terrain grid or a measured ground_height')
        stamp = float(captured_at)
        payload = dict(schema_version=1, boot_id=self.boot, frame_id=self.frame_id,
                       sequence=sequence, captured_at=stamp,
                       pose_captured_at=stamp if pose_captured_at is None else pose_captured_at,
                       terrain_captured_at=stamp if terrain_captured_at is None else terrain_captured_at,
                       pelvis_pos_w=np.asarray(pelvis_pos_w, dtype=np.float32),
                       pelvis_quat_w=np.asarray(pelvis_quat_w, dtype=np.float32))
        if terrain is None:
            payload.update(terrain_mode='plane', ground_height=ground_height)
        else:
            payload.update(terrain_mode='grid', terrain=np.asarray(terrain, dtype=np.float32))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
        try:
            with temp.open('wb') as stream:
                np.savez(stream, **payload)
            temp.replace(self.path)
        finally:
            temp.unlink(missing_ok=True)


class StateReader:
    def __init__(self, config, path=None):
        self.path = state_path(config) if path is None else Path(path)
        self.frame_id = str(config.get('real', {}).get('frame_id', 'map'))
        self.max_age = float(config.get('max_age_s', .15))
        self.max_skew = float(config.get('max_sync_skew_s', .04))
        if (not self.frame_id or not np.isfinite([self.max_age, self.max_skew]).all()
                or not 0 < self.max_age <= .5 or not 0 <= self.max_skew <= self.max_age):
            raise ValueError('Invalid ACP frame or timing limits')
        self.boot = boot_id()
        self.sample = StateSample()
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.thread = None
        self.previous = self.file_key = None

    def poll(self):
        try:
            with self.path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if key == self.file_key:
                    return self.latest()
                self.file_key = key
                if stat.st_size > 65536:
                    raise ValueError('Oversized ACP state packet')
                packet = stream.read(65537)
            if len(packet) > 65536:
                raise ValueError('Oversized ACP state packet')
            with zipfile.ZipFile(BytesIO(packet)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 65536:
                    raise ValueError('Oversized expanded ACP state packet')
            with np.load(BytesIO(packet), allow_pickle=False, max_header_size=1024) as data:
                if int(data['schema_version'].item()) != 1 or str(data['boot_id'].item()) != self.boot:
                    raise ValueError('ACP protocol or host boot mismatch')
                frame = str(data['frame_id'].item())
                if frame != self.frame_id:
                    raise ValueError('ACP localization frame mismatch')
                sequence_raw = data['sequence'].item()
                if isinstance(sequence_raw, bool) or not isinstance(sequence_raw, (int, np.integer)) or sequence_raw < 0:
                    raise ValueError('Invalid ACP state sequence')
                sequence = int(sequence_raw)
                stamp = float(data['captured_at'].item())
                pose_stamp = float(data['pose_captured_at'].item()) if 'pose_captured_at' in data else stamp
                terrain_stamp = float(data['terrain_captured_at'].item()) if 'terrain_captured_at' in data else stamp
                stamps = np.array([stamp, pose_stamp, terrain_stamp])
                if (not np.isfinite(stamps).all() or np.any(time.monotonic()-stamps < 0)
                        or np.any(time.monotonic()-stamps > self.max_age)
                        or np.ptp(stamps) > self.max_skew):
                    raise ValueError('ACP state stale, future, or unsynchronized')
                if self.previous is not None and (sequence <= self.previous[0] or stamp <= self.previous[1]):
                    raise ValueError('Repeated, reordered, or re-stamped ACP state packet')
                position = np.array(data['pelvis_pos_w'], dtype=np.float32, copy=True)
                quat = np.array(data['pelvis_quat_w'], dtype=np.float32, copy=True)
                if (position.shape != (3,) or quat.shape != (4,)
                        or not np.isfinite(position).all() or not np.isfinite(quat).all()
                        or abs(np.linalg.norm(quat)-1) > .01):
                    raise ValueError('Invalid ACP pelvis localization')
                mode = str(data['terrain_mode'].item()) if 'terrain_mode' in data else 'grid'
                if mode == 'plane':
                    ground = float(data['ground_height'].item())
                    if not np.isfinite(ground):
                        raise ValueError('Invalid measured plane height')
                    terrain = flat_grid(position[2], ground)
                elif mode == 'grid':
                    terrain = np.array(data['terrain'], dtype=np.float32, copy=True)
                else:
                    raise ValueError('Unknown ACP terrain mode')
            if (terrain.shape != (187, 3) or not np.isfinite(terrain).all()
                    or not np.allclose(terrain[:, :2], grid_xy(), atol=1e-5, rtol=0)):
                raise ValueError('Invalid ACP terrain grid or grid order')
            for array in (position, quat, terrain):
                array.setflags(write=False)
            self.previous = (sequence, stamp)
            sample = StateSample(position, quat, terrain, stamp, pose_stamp, terrain_stamp, sequence, frame, '')
        except Exception as error:
            sample = StateSample(error='ACP state IPC: ' + str(error))
        with self.lock:
            self.sample = sample
        return sample

    def latest(self):
        with self.lock:
            sample = self.sample
        if not sample.error:
            ages = time.monotonic()-np.array([sample.captured_at, sample.pose_captured_at, sample.terrain_captured_at])
            if np.any(ages < 0) or np.any(ages > self.max_age):
                return StateSample(error='ACP state stream stalled/stale')
        return sample

    def _run(self):
        while not self.stop.is_set():
            self.poll()
            self.stop.wait(.005)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('ACP state reader already started')
        self.thread = threading.Thread(target=self._run, name='acp-state-reader', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        with self.lock:
            self.sample = StateSample(error='ACP state reader stopped')
