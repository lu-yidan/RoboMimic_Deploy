"""Calibrated RealSense depth -> training camera -> PHP. No DDS or motor I/O.

SDK import and USB access occur only in the explicitly started capture thread.
The policy consumes an immutable latest-frame snapshot without waiting for USB.
"""
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from policy.php_parkour.depth_processing import preprocess_depth

TRAIN_POSITION = np.array([.01, .01, .44])
TRAIN_RPY = np.array([1., 27., 1.])
# RealSense optical (right, down, forward) -> body (forward, left, up).
OPTICAL_TO_BODY = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def load_camera_config(path):
    with Path(path).expanduser().open() as stream:
        cfg = yaml.safe_load(stream)
    for name in ('position_m', 'rpy_deg'):
        value = np.asarray(cfg['mount'][name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid camera mount ' + name)
    if not isinstance(cfg.get('serial'), str):
        raise ValueError('Camera serial must be a string')
    for key in ('width', 'height', 'fps'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError('Camera '+key+' must be a positive integer')
    if not 0 < float(cfg['max_age_s']) <= .15:
        raise ValueError('max_age_s must be in (0, 0.15]')
    if not .5 <= float(cfg['min_valid_fraction']) <= 1:
        raise ValueError('min_valid_fraction must be in [0.5, 1]')
    limits = np.asarray(cfg['effort_limits'], dtype=float)
    if limits.shape != (29,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Expected 29 positive finite effort limits')
    # Do not silently relax the validated simulation effort envelope.
    maximum = np.array([88,139,88,139,50,50]*2+[88,50,50]+[25,25,25,25,25,5,5]*2)
    if np.any(limits > maximum):
        raise ValueError('PHP real effort limits cannot exceed simulation limits')
    return cfg


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class DepthSample:
    image: object = None
    captured_at: float = 0.
    sequence: int = -1
    valid_fraction: float = 0.
    error: str = 'no depth frame'


class DepthProjector:
    """Z-buffer rectified optical-Z depth into the FIXED training camera.

    Configuring a different mount changes this transform, not the model's target
    extrinsics. Unobserved pixels remain invalid and count against coverage.
    """
    def __init__(self, config):
        self.config = config
        actual = Rotation.from_euler('xyz', config['mount']['rpy_deg'], degrees=True).as_matrix() @ OPTICAL_TO_BODY
        target = Rotation.from_euler('xyz', TRAIN_RPY, degrees=True).as_matrix() @ OPTICAL_TO_BODY
        self.rotation = (target.T @ actual).astype(np.float32)
        self.translation = (target.T @ (np.asarray(config['mount']['position_m']) - TRAIN_POSITION)).astype(np.float32)
        self.focal = 53. / np.tan(np.deg2rad(89.5/2))
        self._key = None
        self._rays = None

    def project(self, metric_depth, intrinsics):
        depth = np.asarray(metric_depth, dtype=np.float32)
        k = intrinsics
        if depth.shape != (k.height, k.width) or min(k.width, k.height) < 1:
            raise ValueError('Depth resolution does not match stream intrinsics')
        if not np.isfinite([k.fx,k.fy,k.cx,k.cy]).all() or min(k.fx,k.fy) <= 0:
            raise ValueError('Invalid stream intrinsics')
        if self._key != k:
            v,u = np.indices(depth.shape, dtype=np.float32)
            self._rays = np.stack(((u-k.cx)/k.fx, (v-k.cy)/k.fy, np.ones_like(u)), -1).reshape(-1,3)
            self._key = k
        flat = depth.ravel()
        good = np.isfinite(flat) & (flat > 0)
        points = np.einsum('ij,kj->ki', self.rotation, self._rays[good]*flat[good,None], optimize=False) + self.translation
        points = points[np.isfinite(points).all(axis=1) & (points[:,2] > 0)]
        # Training pinhole uses width/2, height/2 and pixel-centre coordinates.
        uv = points[:,:2]/points[:,2,None]*self.focal + [53.,30.]
        inside = (uv[:,0]>=-.5)&(uv[:,0]<105.5)&(uv[:,1]>=-.5)&(uv[:,1]<59.5)
        uv = np.floor(uv[inside]+.5).astype(int)
        projected = np.full(60*106, np.inf, dtype=np.float32)
        np.minimum.at(projected, uv[:,1]*106+uv[:,0], points[inside,2])
        projected = projected.reshape(60,106)
        coverage = float(np.isfinite(projected[2:,4:-4]).mean())
        if coverage < self.config['min_valid_fraction']:
            raise ValueError('Depth coverage {:.1%} below {:.1%}'.format(coverage,self.config['min_valid_fraction']))
        # Training far-range encoding for the small admitted missing region.
        projected[~np.isfinite(projected)] = 3.
        return preprocess_depth(projected), coverage


class DepthBuffer:
    def __init__(self, config):
        self.config = config
        self.projector = DepthProjector(config)
        self.lock = threading.Lock()
        self.sample = DepthSample()
        self._last_sequence = -1
        self._last_timestamp = 0.
        self._recording = None

    def invalidate(self, error):
        with self.lock:
            self.sample = DepthSample(error=str(error))
            self._recording = None

    def submit(self, metric_depth, intrinsics, captured_at, sequence):
        """Single producer; captured_at is capture time in host MONOTONIC seconds."""
        try:
            now = time.monotonic()
            if not np.isfinite(captured_at) or not 0 <= now-captured_at <= self.config['max_age_s']:
                raise ValueError('Depth timestamp stale, future or invalid')
            if sequence <= self._last_sequence or captured_at <= self._last_timestamp:
                raise ValueError('Repeated/out-of-order depth frame')
            self._last_sequence, self._last_timestamp = sequence, captured_at
            image, coverage = self.projector.project(metric_depth, intrinsics)
            if time.monotonic()-captured_at > self.config['max_age_s']:
                raise ValueError('Depth processing exceeded frame age limit')
            image.setflags(write=False)
            raw = np.array(metric_depth,copy=True)
            with self.lock:
                self.sample = DepthSample(image,captured_at,sequence,coverage,'')
                self._recording = (raw,intrinsics,captured_at)
        except Exception as error:
            self.invalidate(error)

    def latest(self):
        with self.lock:
            sample = self.sample
        if sample.image is not None and not 0 <= time.monotonic()-sample.captured_at <= self.config['max_age_s']:
            return DepthSample(error='Depth stream stalled/stale')
        return sample

    def recording(self):
        """Copy a coherent raw frame for offline replay; never used by control."""
        with self.lock:
            if self._recording is None:
                raise ValueError('No accepted raw frame available')
            depth,k,stamp = self._recording
            return depth.copy(),k,stamp

    def start(self):
        """Replay buffers have no transport; RealSense overrides this."""

    def close(self):
        pass


class RealSenseDepth(DepthBuffer):
    def __init__(self, config):
        super().__init__(config)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Camera already started')
        self._thread = threading.Thread(target=self._capture, name='php-depth', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)
        self.invalidate('Camera stopped')

    def _capture(self):
        pipeline = None
        try:
            import pyrealsense2 as rs
            context = rs.context()
            devices = list(context.query_devices())
            serial = self.config['serial']
            if not serial:
                if len(devices) != 1:
                    raise RuntimeError('Set camera serial explicitly unless exactly one device is connected')
                serial = devices[0].get_info(rs.camera_info.serial_number)
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, self.config['width'],self.config['height'],rs.format.z16,self.config['fps'])
            pipeline = rs.pipeline(context)
            profile = pipeline.start(cfg)
            sensor = profile.get_device().first_depth_sensor()
            if not sensor.supports(rs.option.global_time_enabled):
                raise RuntimeError('Camera requires host/global timestamp support')
            sensor.set_option(rs.option.global_time_enabled, 1.)
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1.)
            scale = float(sensor.get_depth_scale())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid camera depth scale')
            while not self._stop.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=100)
                    frame = frames.get_depth_frame()
                    if not frame:
                        raise ValueError('No depth frame')
                    received_wall, received_mono = time.time(), time.monotonic()
                    if frame.get_frame_timestamp_domain() not in (rs.timestamp_domain.global_time,rs.timestamp_domain.system_time):
                        raise ValueError('Waiting for host/global timestamp synchronization')
                    captured = received_mono + frame.get_timestamp()/1000. - received_wall
                    k = frame.profile.as_video_stream_profile().get_intrinsics()
                    if k.model != rs.distortion.none and not np.allclose(k.coeffs,0,atol=1e-12,rtol=0):
                        raise ValueError('Requires rectified depth with distortion=none; do not use color-aligned depth')
                    intrinsics = Intrinsics(k.width,k.height,k.fx,k.fy,k.ppx,k.ppy)
                    depth = np.asanyarray(frame.get_data()).astype(np.float32)*scale
                    self.submit(depth,intrinsics,captured,int(frame.get_frame_number()))
                except Exception as error:
                    self.invalidate(error)
        except Exception as error:
            self.invalidate(error)
            print('[PHP camera] {}'.format(error),flush=True)
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
