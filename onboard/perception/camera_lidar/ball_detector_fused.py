"""ball_detector_fused.py — Chest camera (D435 + YOLO) + LiDAR (MID360) fused ball detection.

Source priority:
  1. Chest camera (YOLO hit + D435 depth)  →  camera depth measurement
  2. Camera miss / quiet                   →  LiDAR centroid (Livox MID360)
  A shared 3-D constant-velocity Kalman filter smooths the fused output and
  bridges missing frames from either source.

Head joint is FIXED at 2.3° downward for the LiDAR → pelvis transform.
  (Change HEAD_JOINT_ANGLE if your robot's head pitch differs.)

Subscribes:
  /livox/lidar   (livox_ros_driver2 CustomMsg, ~10 Hz)
  /lowstate      (unitree_hg LowState,          500 Hz)

Publishes via DDS:
  rt/ball_state  (BallState — ball position in pelvis body frame, ~50 Hz)

Usage:
    python onboard/perception/ball_detector_fused.py
    python onboard/perception/ball_detector_fused.py --show
    python onboard/perception/ball_detector_fused.py --chest-serial 123456789
    python onboard/perception/ball_detector_fused.py --model models/yolo11m.engine
"""

import sys
import numpy as _np_compat
# TensorRT 8.5 Python binding was built against numpy <1.24 and uses removed aliases.
if not hasattr(_np_compat, 'bool'):   _np_compat.bool   = bool
if not hasattr(_np_compat, 'int'):    _np_compat.int    = int
if not hasattr(_np_compat, 'float'):  _np_compat.float  = float
if not hasattr(_np_compat, 'object'): _np_compat.object = object
del _np_compat

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import argparse
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowState
from livox_ros_driver2.msg import CustomMsg

from onboard.perception.camera.camera_to_base import (
    transform_point_chest_camera_to_base,
    optical_to_body,
)
from onboard.perception.lidar.mid360_to_base import transform_point_mid360_to_base
from common.ball_state_dds import BallStatePublisher, SOURCE_CAM, SOURCE_LIDAR, SOURCE_NONE


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Head joint angle for LiDAR → pelvis transform.
# Previous value was 0.593412 rad ≈ 34°. Now the head is at 2.3° downward.
HEAD_JOINT_ANGLE  = np.radians(2.3)
HEAD_MID360_JOINT = 0.593412           # MID360 rotating joint (fixed at 0)

# ── Camera detection ──────────────────────────────────────────────────────────
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD       = 0.30
DEPTH_SAMPLE_RADIUS  = 5
DEPTH_MIN            = 0.10    # m — discard depth readings below this
DEPTH_MAX            = 10.0   # m — discard depth readings above this
BALL_RADIUS          = 0.115  # m — physical radius; depth sensor sees front surface
EMA_ALPHA            = 0.6    # exponential moving-average weight for camera
EMA_GATE             = 0.6    # m — jump larger than this resets EMA
COAST_FRAMES         = 10     # frames to hold last bbox after YOLO miss

# ── LiDAR detection ──────────────────────────────────────────────────────────
LIDAR_REFLECT_THR   = 130    # minimum reflectivity to be considered ball candidate
LIDAR_MIN_POINTS    = 4      # minimum candidate points to attempt center fit
LIDAR_MAX_RANGE     = 1.8    # m (MID360 frame)
LIDAR_MIN_RANGE     = 0.20   # m
LIDAR_Z_LOW         = -1.5   # m
LIDAR_Z_HIGH        =  1.5   # m
LIDAR_X_LOW         =  0.0   # m
LIDAR_X_HIGH        =  5.0   # m
LIDAR_CENTER_OFFSET =  0.085 # m — offset from centroid toward sensor origin

# ── Fusion ────────────────────────────────────────────────────────────────────
CAM_STALE_SEC   = 0.40  # camera slot expires after this long without a write
LIDAR_STALE_SEC = 0.50  # lidar slot expires after this long without a write
FUSION_HZ       = 50    # fusion loop rate [Hz]

# Kalman filter measurement noise covariance (R) per source.
# Camera is more precise laterally; LiDAR noisier but works at longer range.
_R_CAM   = np.diag([0.010, 0.010, 0.010]).astype(np.float64)
_R_LIDAR = np.diag([0.040, 0.040, 0.040]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Kalman filter
# ══════════════════════════════════════════════════════════════════════════════

class _FusionKF:
    """3-D constant-velocity Kalman filter with per-update measurement noise R.

    State: [x, y, z, vx, vy, vz]  (pelvis body frame)
    Measurement: [x, y, z]

    Call predict(dt) every fusion tick, then update(z, R) whenever a fresh
    measurement is available (camera or lidar).
    """
    # Process noise Q (scaled by dt at prediction time)
    _Q0 = np.diag([0.02, 0.02, 0.02, 0.50, 0.50, 0.50]).astype(np.float64)
    # Observation matrix H: z = H @ x
    _H  = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)])

    MAX_JUMP = 1.2   # gating: jump larger than this resets the filter [m]

    def __init__(self):
        self.x           = np.zeros((6, 1), dtype=np.float64)
        self.P           = np.eye(6,        dtype=np.float64)
        self.initialized = False

    def predict(self, dt: float):
        dt = max(dt, 1e-3)
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q0 * dt

    def update(self, z: np.ndarray, R: np.ndarray):
        """Update with measurement z (3,) and noise covariance R (3×3)."""
        z3 = np.asarray(z, dtype=np.float64).flatten()[:3]

        if not self.initialized:
            self.x[:3, 0] = z3
            self.P        = np.eye(6, dtype=np.float64) * 0.5
            self.initialized = True
            return

        # Gating: large jump → reset to new measurement
        if np.linalg.norm(z3 - self.x[:3, 0]) > self.MAX_JUMP:
            self.x[:3, 0] = z3
            self.x[3:, 0] = 0.0
            self.P        = np.eye(6, dtype=np.float64) * 0.5
            print("\n[KF] large jump → filter reset", flush=True)
            return

        zc = z3.reshape(3, 1)
        y  = zc - self._H @ self.x
        S  = self._H @ self.P @ self._H.T + R
        K  = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ self._H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3, 0].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Thread-safe measurement slot
# ══════════════════════════════════════════════════════════════════════════════

class _MeasSlot:
    """Holds the latest measurement from one sensor, thread-safely."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._pos   = np.zeros(3, dtype=np.float32)
        self._valid = False
        self._t     = 0.0  # time.monotonic() of last write

    def put(self, pos: np.ndarray):
        with self._lock:
            self._pos[:] = np.asarray(pos, dtype=np.float32)
            self._valid  = True
            self._t      = time.monotonic()

    def get(self):
        """Returns (pos_copy (3,), age_sec).  age=inf when never written."""
        with self._lock:
            age = (time.monotonic() - self._t) if self._valid else float('inf')
            return self._pos.copy(), age

    def invalidate(self):
        with self._lock:
            self._valid = False


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════

class _FPS:
    def __init__(self, window: int = 30):
        self._t: list = []
        self._w = window

    def tick(self):
        self._t.append(time.perf_counter())
        if len(self._t) > self._w:
            self._t.pop(0)

    @property
    def fps(self) -> float:
        return 0.0 if len(self._t) < 2 else (len(self._t) - 1) / (self._t[-1] - self._t[0])


class _CamState:
    def __init__(self):
        self.center_ema: np.ndarray | None = None
        self.last_bbox:  tuple | None      = None
        self.miss_count: int               = 0
        self.fps = _FPS()


class _CamBuf:
    def __init__(self):
        self.lock    = threading.Lock()
        self.frames  = None
        self.updated = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# Camera depth helper  (identical to ball_detector_dual.py)
# ══════════════════════════════════════════════════════════════════════════════

def _sample_depth(depth_arr, cx, cy,
                  color_intrin, depth_intrin, c2d_extr, depth_scale) -> float:
    """Return depth [m] at colour pixel (cx,cy) with parallax correction."""
    dh, dw  = depth_arr.shape
    ndcx = (cx - color_intrin.ppx) / color_intrin.fx
    ndcy = (cy - color_intrin.ppy) / color_intrin.fy

    # Coarse depth (intrinsics-only mapping, no parallax)
    dx0 = int(np.clip(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5, 0, dw - 1))
    dy0 = int(np.clip(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5, 0, dh - 1))
    raw0         = depth_arr[dy0, dx0]
    depth_coarse = float(raw0 * depth_scale) if raw0 > 0 else 1.0

    # Parallax correction
    tx = c2d_extr.translation[0]
    ty = c2d_extr.translation[1]
    dx = int(np.clip(
        ndcx * depth_intrin.fx + depth_intrin.ppx + tx / depth_coarse * depth_intrin.fx + 0.5,
        0, dw - 1))
    dy = int(np.clip(
        ndcy * depth_intrin.fy + depth_intrin.ppy + ty / depth_coarse * depth_intrin.fy + 0.5,
        0, dh - 1))

    R     = DEPTH_SAMPLE_RADIUS
    patch = (depth_arr[max(0, dy-R):min(dh, dy+R+1),
                       max(0, dx-R):min(dw, dx+R+1)]
             .astype(np.float32) * depth_scale)
    valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR ball center estimator  (from lidar/ball_detector.py)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_lidar_center(points: np.ndarray,
                           offset: float = LIDAR_CENTER_OFFSET) -> np.ndarray:
    """Centroid of high-reflectivity cluster + small forward offset."""
    center = points.mean(axis=0).astype(np.float64)
    mask   = np.linalg.norm(points - center, axis=1) < 0.20
    c      = points[mask].mean(axis=0).astype(np.float64) if mask.any() else center
    n      = np.linalg.norm(c)
    return (c + c / n * offset) if n > 1e-6 else c


# ══════════════════════════════════════════════════════════════════════════════
# ROS2 node — handles /lowstate and /livox/lidar
# ══════════════════════════════════════════════════════════════════════════════

class _DetectorNode(Node):
    """ROS2 node that subscribes to joint states and the LiDAR.

    LiDAR processing runs in a background worker thread so that the
    500 Hz /lowstate callback never starves the lidar pipeline.
    Detected ball pelvis-frame positions are written to `lidar_slot`.
    """

    def __init__(self, lidar_slot: _MeasSlot):
        super().__init__("ball_detector_fused")
        self._slot = lidar_slot

        # Joint angles — updated at 500 Hz from /lowstate
        self.q_wy = 0.0
        self.q_wr = 0.0
        self.q_wp = 0.0

        # Lidar message buffer (shared between ROS callback and worker)
        self._buf_lock   = threading.Lock()
        self._buf_msg    = None
        self._buf_event  = threading.Event()
        self._stop_flag  = threading.Event()

        self.create_subscription(
            LowState, "/lowstate", self._cb_low, qos_profile_sensor_data
        )
        self.create_subscription(
            CustomMsg, "/livox/lidar", self._cb_lidar, 5
        )

        self._worker = threading.Thread(target=self._lidar_worker, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"FusedDetectorNode ready. HEAD_JOINT_ANGLE={np.degrees(HEAD_JOINT_ANGLE):.1f}°"
        )

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _cb_low(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]

    def _cb_lidar(self, msg: CustomMsg):
        with self._buf_lock:
            self._buf_msg = msg
        self._buf_event.set()

    # ── LiDAR worker ──────────────────────────────────────────────────────────

    def _lidar_worker(self):
        frame_n = 0
        while not self._stop_flag.is_set():
            if not self._buf_event.wait(timeout=1.0):
                self._slot.invalidate()
                continue
            self._buf_event.clear()

            with self._buf_lock:
                msg = self._buf_msg
            if msg is None:
                continue

            pts = msg.points
            n   = len(pts)
            if n == 0:
                self._slot.invalidate()
                continue

            # Pass 1 — reflectivity (cheap: 1 attribute per point)
            refl    = np.array([p.reflectivity for p in pts], dtype=np.uint8)
            hi_idx  = np.where(refl >= LIDAR_REFLECT_THR)[0].tolist()

            # Pass 2 — XYZ only for high-reflectivity candidates
            if hi_idx:
                cand_raw = np.array(
                    [(pts[i].x, pts[i].y, pts[i].z) for i in hi_idx],
                    dtype=np.float32,
                )
                d   = np.linalg.norm(cand_raw, axis=1)
                roi = (
                    (d >= LIDAR_MIN_RANGE) & (d <= LIDAR_MAX_RANGE) &
                    (cand_raw[:, 2] >= LIDAR_Z_LOW)  & (cand_raw[:, 2] <= LIDAR_Z_HIGH) &
                    (cand_raw[:, 0] >= LIDAR_X_LOW)  & (cand_raw[:, 0] <= LIDAR_X_HIGH)
                )
                cand = cand_raw[roi]
            else:
                cand = np.zeros((0, 3), dtype=np.float32)

            if cand.shape[0] < LIDAR_MIN_POINTS:
                self._slot.invalidate()
                if frame_n % 30 == 0:
                    print(f"\r[lidar] no ball  hi={len(hi_idx)} cand={cand.shape[0]}  n={n}",
                          end="", flush=True)
                frame_n += 1
                continue

            center_lidar = _estimate_lidar_center(cand)
            center_base  = transform_point_mid360_to_base(
                center_lidar,
                self.q_wy, self.q_wr, self.q_wp,
                HEAD_JOINT_ANGLE, HEAD_MID360_JOINT,
            )
            self._slot.put(center_base)

            if frame_n % 5 == 0:
                x, y, z = float(center_base[0]), float(center_base[1]), float(center_base[2])
                print(f"\r[lidar] pelvis=({x:+.3f},{y:+.3f},{z:+.3f})  cand={cand.shape[0]}",
                      end="", flush=True)
            frame_n += 1

    def destroy_node(self):
        self._stop_flag.set()
        self._buf_event.set()   # unblock worker if it is waiting
        self._worker.join(timeout=2)
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# Fusion loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_fusion(cam_slot: _MeasSlot, lidar_slot: _MeasSlot,
                dds: BallStatePublisher, stop_flag: threading.Event):
    """50 Hz loop: pick best measurement → KF predict+update → DDS publish."""
    kf     = _FusionKF()
    prev_t = time.perf_counter()
    source = "none"

    while not stop_flag.is_set():
        time.sleep(1.0 / FUSION_HZ)
        now = time.perf_counter()
        dt  = now - prev_t
        prev_t = now

        cam_pos,   cam_age   = cam_slot.get()
        lidar_pos, lidar_age = lidar_slot.get()

        # Predict
        kf.predict(dt)

        # Update — camera takes priority over LiDAR
        if cam_age < CAM_STALE_SEC:
            kf.update(cam_pos, _R_CAM)
            source = "cam  "
            source_id = SOURCE_CAM
            valid  = True
        elif lidar_age < LIDAR_STALE_SEC:
            kf.update(lidar_pos, _R_LIDAR)
            source = "lidar"
            source_id = SOURCE_LIDAR
            valid  = True
        else:
            source = "none "
            source_id = SOURCE_NONE
            valid  = False

        pos = kf.position if kf.initialized else np.zeros(3, dtype=np.float32)
        dds.publish(float(pos[0]), float(pos[1]), float(pos[2]), valid=valid, source=source_id)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fused ball detector: chest D435 + MID360 LiDAR → rt/ball_state"
    )
    parser.add_argument("--model",        default="onboard/perception/camera/models/yolo11m.pt",
                        help="YOLO model path (.pt or .engine)")
    parser.add_argument("--imgsz",        type=int, default=320)
    parser.add_argument("--width",        type=int, default=640)
    parser.add_argument("--height",       type=int, default=480)
    parser.add_argument("--chest-serial", default=None,
                        help="Serial number of the CHEST D435 camera")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List connected RealSense cameras and exit")
    parser.add_argument("--show",         action="store_true",
                        help="Stream annotated chest camera feed via HTTP on port 8080")
    args = parser.parse_args()

    # ── Enumerate RealSense cameras ───────────────────────────────────────
    ctx      = rs.context()
    devs     = ctx.query_devices()
    all_sns  = [d.get_info(rs.camera_info.serial_number) for d in devs]
    all_names= [d.get_info(rs.camera_info.name)          for d in devs]

    print("[INFO] Connected RealSense devices:")
    for i, (sn, nm) in enumerate(zip(all_sns, all_names)):
        print(f"  [{i}] serial={sn}  {nm}")

    if args.list_cameras:
        return

    if len(all_sns) == 0:
        raise RuntimeError("No RealSense device found. Check USB connection.")

    serial_chest = args.chest_serial or all_sns[0]
    print(f"[INFO] CHEST camera → serial {serial_chest}")

    # ── ROS2 ─────────────────────────────────────────────────────────────
    rclpy.init()
    lidar_slot = _MeasSlot()
    cam_slot   = _MeasSlot()
    node       = _DetectorNode(lidar_slot)

    # spin_once at ~500 Hz so /lowstate and /livox/lidar are processed promptly
    def _ros_spin():
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.002)
    threading.Thread(target=_ros_spin, daemon=True).start()
    print("[INFO] ROS2 spin thread started (/lowstate + /livox/lidar)")

    # ── DDS publisher ─────────────────────────────────────────────────────
    dds = BallStatePublisher(domain_id=0)
    print("[INFO] DDS publisher ready on 'rt/ball_state'")

    # ── YOLO ──────────────────────────────────────────────────────────────
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_path = args.model
    if model_path.endswith('.pt'):
        engine_path = model_path.replace('.pt', '.engine')
        if __import__('os').path.exists(engine_path):
            model_path = engine_path
            print(f"[INFO] TensorRT engine found: {engine_path}")
        else:
            print(f"[INFO] No .engine at {engine_path}, using .pt")

    print(f"[INFO] Loading YOLO: {model_path}  (device={device})")
    model  = YOLO(model_path)
    is_trt = model_path.endswith('.engine')
    infer_kw = {} if is_trt else dict(device=device, half=True)

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] Warming up YOLO (~5 s)...")
    for i in range(5):
        t0 = time.perf_counter()
        model(dummy, verbose=False, **infer_kw)
        print(f"[INFO]   warmup[{i}]: {(time.perf_counter()-t0)*1000:.1f} ms")

    t0 = time.perf_counter()
    for _ in range(10):
        model(dummy, verbose=False, **infer_kw)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"[INFO] YOLO: {ms:.1f} ms/frame  ≈ {1000/ms:.0f} FPS")

    # ── RealSense pipeline ────────────────────────────────────────────────
    def _start_pipeline(serial, name):
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 60)
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,  90)
        for attempt in range(2):
            print(f"[INFO] Starting [{name}] pipeline (attempt {attempt + 1})...")
            profile = pipe.start(cfg)
            try:
                pipe.wait_for_frames(timeout_ms=5000)
                print(f"[INFO] [{name}] pipeline OK")
                return pipe, profile
            except RuntimeError:
                print(f"[WARN] [{name}] frame timeout — hardware reset...")
                pipe.stop()
                for dev in rs.context().query_devices():
                    if dev.get_info(rs.camera_info.serial_number) == serial:
                        dev.hardware_reset()
                        break
                time.sleep(3)
        raise RuntimeError(f"[{name}] pipeline failed to start.")

    pipe_chest, prof_chest = _start_pipeline(serial_chest, "chest")

    # Camera intrinsics / extrinsics
    def _get_cam_params(profile):
        cp  = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dp  = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        return cp.get_intrinsics(), dp.get_intrinsics(), cp.get_extrinsics_to(dp), \
               profile.get_device().first_depth_sensor().get_depth_scale()

    ci_c, di_c, c2d_c, ds_c = _get_cam_params(prof_chest)
    print(f"[INFO] Chest — color fx={ci_c.fx:.1f}  depth_scale={ds_c:.4f}")

    # ── MJPEG server (--show) ─────────────────────────────────────────────
    _mjpeg_frame = [None]
    _mjpeg_lock  = threading.Lock()

    if args.show:
        import http.server, socketserver

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path not in ('/', '/stream'):
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                last_sent = None
                try:
                    while True:
                        with _mjpeg_lock:
                            jpg = _mjpeg_frame[0]
                        if jpg is None or jpg is last_sent:
                            time.sleep(0.02); continue
                        last_sent = jpg
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                        )
                except Exception:
                    pass

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        _httpd = socketserver.ThreadingTCPServer(('0.0.0.0', 8080), _Handler)
        _httpd.daemon_threads = True
        threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        print("[INFO] MJPEG stream → http://192.168.123.164:8080/stream")

    # ── Camera capture thread ─────────────────────────────────────────────
    stop_flag = threading.Event()
    buf_chest = _CamBuf()

    def _capture_loop(pipe, buf, name):
        while not stop_flag.is_set():
            try:
                frames = pipe.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print(f"\n[WARN] [{name}] frame timeout, retrying...", flush=True)
                continue
            with buf.lock:
                buf.frames = frames
            buf.updated.set()

    threading.Thread(
        target=_capture_loop, args=(pipe_chest, buf_chest, "chest"), daemon=True
    ).start()

    # ── YOLO / depth worker for chest camera ──────────────────────────────
    def _chest_worker():
        state = _CamState()
        x = y = z = 0.0
        depth_m = 0.0

        while not stop_flag.is_set():
            if not buf_chest.updated.wait(timeout=1.0):
                continue
            buf_chest.updated.clear()

            with buf_chest.lock:
                frames = buf_chest.frames
            if frames is None:
                continue

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color     = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()

            orig_h, orig_w = color.shape[:2]
            color_small    = cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            results = model.predict(
                color_small, conf=CONF_THRESHOLD, verbose=False, **infer_kw
            )

            best_box, best_conf = None, 0.0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            if best_box is not None:
                state.miss_count = 0
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                state.last_bbox = (
                    int(x1s * sx), int(y1s * sy),
                    int(x2s * sx), int(y2s * sy),
                )
            else:
                state.miss_count += 1

            published = False
            depth_m   = 0.0
            if state.last_bbox is not None and state.miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = state.last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                depth_surface = _sample_depth(
                    depth_arr, cx, cy, ci_c, di_c, c2d_c, ds_c
                )
                depth_m = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0

                if depth_m > 0:
                    p_opt = rs.rs2_deproject_pixel_to_point(ci_c, [cx, cy], depth_m)
                    p_cam = optical_to_body(p_opt)

                    # EMA smoothing in camera body frame before KF hand-off
                    if state.center_ema is None:
                        gate_dist = 0.0
                        state.center_ema = p_cam.copy()
                    else:
                        gate_dist = np.linalg.norm(p_cam - state.center_ema)
                        if gate_dist < EMA_GATE:
                            state.center_ema = (EMA_ALPHA * p_cam
                                                + (1 - EMA_ALPHA) * state.center_ema)
                        else:
                            print(f"\n[WARN] [chest] EMA gate {gate_dist:.2f}m → reset",
                                  flush=True)
                            state.center_ema = p_cam.copy()

                    # Transform to pelvis frame and write to shared slot
                    p_base = transform_point_chest_camera_to_base(
                        state.center_ema,
                        node.q_wy, node.q_wr, node.q_wp,
                    )
                    x, y, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
                    cam_slot.put(p_base)
                    published = True

                    status = "BALL " if best_box is not None else "COAST"
                    print(
                        f"\r[chest/{status}] pelvis=({x:+.3f},{y:+.3f},{z:+.3f})  "
                        f"d={depth_m:.2f}m  conf={best_conf:.2f}  fps={state.fps.fps:.1f}",
                        end="", flush=True,
                    )

            if not published:
                # Don't call cam_slot.invalidate() — let age-based expiry handle it.
                if state.fps.fps > 0:
                    print(f"\r[chest/     ] no ball  fps={state.fps.fps:.1f}" + " " * 30,
                          end="", flush=True)

            # Annotate and push to MJPEG if --show
            if args.show:
                vis = color.copy()
                if state.last_bbox is not None and state.miss_count <= COAST_FRAMES:
                    bx1, by1, bx2, by2 = state.last_bbox
                    col = (0, 255, 0) if best_box is not None else (0, 165, 255)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, 2)
                    cv2.circle(vis, ((bx1+bx2)//2, (by1+by2)//2), 4, col, -1)
                    lbl = (f"ball {best_conf:.2f}" if best_box is not None
                           else f"coast {state.miss_count}/{COAST_FRAMES}")
                    cv2.putText(vis, lbl, (bx1, by1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    if published:
                        cv2.putText(vis,
                                    f"pelvis ({x:+.2f},{y:+.2f},{z:+.2f})m  d={depth_m:.2f}m",
                                    (10, vis.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(vis, f"chest {state.fps.fps:.1f}fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                _, jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
                with _mjpeg_lock:
                    _mjpeg_frame[0] = jpg.tobytes()

            state.fps.tick()

    threading.Thread(target=_chest_worker, daemon=True).start()

    # ── Fusion + publish thread ───────────────────────────────────────────
    threading.Thread(
        target=_run_fusion,
        args=(cam_slot, lidar_slot, dds, stop_flag),
        daemon=True,
    ).start()
    print("[INFO] Fusion thread started (chest camera > LiDAR, KF @ 50 Hz)")
    print("[INFO] Running. Press Ctrl-C to stop.")

    # ── Main thread: wait for Ctrl-C ─────────────────────────────────────
    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        pipe_chest.stop()
        if args.show:
            _httpd.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ball_detector_fused.py  --  Head D435 (YOLO) + Head MID360 (LiDAR) fused ball detection.

Priority:
  1. Camera (YOLO hit + D435 depth)  -->  camera depth measurement
  2. Camera miss / quiet             -->  LiDAR centroid (MID360)

Threading (7 threads + main):
  ros_spin        -- ROS2 spin_once at ~500 Hz (/lowstate + /livox/lidar)
  lidar_worker    -- reflectivity filter -> ROI -> center -> pelvis transform
  rs_capture      -- wait_for_frames -> copy color + depth
  yolo_infer      -- resize -> model.predict -> best bbox
  cam_geo         -- depth sample -> EMA -> pelvis transform
  fusion          -- 50 Hz: cam_slot vs lidar_slot -> DDS publish
  mjpeg_enc       -- (optional, --show) imencode -> HTTP stream

Queues between camera stages are maxsize=1 with latest-only semantics so
YOLO never accumulates a backlog of stale frames.

Subscribes:
  /livox/lidar   (livox_ros_driver2 CustomMsg, ~10 Hz)
  /lowstate      (unitree_hg LowState, 500 Hz)

Publishes via DDS:
  rt/ball_state  (BallState -- ball position in pelvis body frame, ~50 Hz)

Usage:
    bash onboard/perception/camera_lidar/run_fused.sh
    bash onboard/perception/camera_lidar/run_fused.sh --show
    bash onboard/perception/camera_lidar/run_fused.sh --head-serial 334622071404
"""

import sys
import warnings
import numpy as _np

# TRT 8.5 on JetPack calls np.bool / np.int / np.float inside tensorrt.__init__.
# numpy >=1.24 removed these aliases. hasattr() doesn't catch it because the
# attribute exists but raises AttributeError on access.  Force-assign unconditionally.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    _np.bool = bool
    _np.int = int
    _np.float = float
    _np.object = object
del _np

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowState
from livox_ros_driver2.msg import CustomMsg

from onboard.perception.camera.camera_to_base import (
    transform_point_camera_to_base,
    optical_to_body,
)
from onboard.perception.lidar.mid360_to_base import transform_point_mid360_to_base
from common.ball_state_dds import BallStatePublisher, SOURCE_CAM, SOURCE_LIDAR, SOURCE_NONE
from onboard.perception.yolo_util import resolve_model_path

# ==============================================================================
# Constants
# ==============================================================================

# Head pitch for both LiDAR and head-camera -> pelvis transforms [rad].
HEAD_JOINT_ANGLE_DEG = 34.0
HEAD_JOINT_ANGLE = np.radians(HEAD_JOINT_ANGLE_DEG)
HEAD_MID360_JOINT = 0.593412

# -- Camera detection ----------------------------------------------------------
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD = 0.30
DEPTH_SAMPLE_RADIUS = 5
DEPTH_MIN = 0.10
DEPTH_MAX = 10.0
BALL_RADIUS = 0.115
EMA_ALPHA = 0.6
EMA_GATE = 0.6
COAST_FRAMES = 10

# -- LiDAR detection -----------------------------------------------------------
LIDAR_REFLECT_THR = 150
LIDAR_MIN_POINTS = 4
LIDAR_MAX_RANGE = 4.0
LIDAR_MIN_RANGE = 0.20
LIDAR_Z_LOW = -1.5
LIDAR_Z_HIGH = 1.5
LIDAR_X_LOW = 0.0
LIDAR_X_HIGH = 5.0
LIDAR_Y_LOW = -1.2
LIDAR_Y_HIGH = 1.2
LIDAR_CENTER_OFFSET = 0.085

# -- Fusion --------------------------------------------------------------------
CAM_STALE_SEC = 0.30
LIDAR_STALE_SEC = 0.50
FUSION_HZ = 50


# ==============================================================================
# Thread-safe measurement slot
# ==============================================================================


class _MeasSlot:
    """Latest 3-D position from one sensor, with timestamp for staleness check."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pos = np.zeros(3, dtype=np.float32)
        self._valid = False
        self._t = 0.0

    def put(self, pos: np.ndarray):
        with self._lock:
            self._pos[:] = np.asarray(pos, dtype=np.float32)
            self._valid = True
            self._t = time.monotonic()

    def get(self):
        with self._lock:
            age = (time.monotonic() - self._t) if self._valid else float("inf")
            return self._pos.copy(), age

    def invalidate(self):
        with self._lock:
            self._valid = False


# ==============================================================================
# Helpers
# ==============================================================================


class _FPS:
    def __init__(self, window: int = 30):
        self._t: list = []
        self._w = window

    def tick(self):
        self._t.append(time.perf_counter())
        if len(self._t) > self._w:
            self._t.pop(0)

    @property
    def fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


class _CamState:
    def __init__(self):
        self.center_ema = None
        self.last_bbox = None
        self.miss_count = 0
        self.fps = _FPS()


def _queue_put_latest(q, item, lock: threading.Lock):
    """Replace anything pending so the consumer always sees the freshest item."""
    with lock:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


# ==============================================================================
# Camera depth helper
# ==============================================================================


def _sample_depth(depth_arr, cx, cy, color_intrin, depth_intrin, c2d_extr, depth_scale):
    dh, dw = depth_arr.shape
    ndcx = (cx - color_intrin.ppx) / color_intrin.fx
    ndcy = (cy - color_intrin.ppy) / color_intrin.fy

    dx0 = int(np.clip(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5, 0, dw - 1))
    dy0 = int(np.clip(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5, 0, dh - 1))
    raw0 = depth_arr[dy0, dx0]
    depth_coarse = float(raw0 * depth_scale) if raw0 > 0 else 1.0

    tx = c2d_extr.translation[0]
    ty = c2d_extr.translation[1]
    dx = int(
        np.clip(
            ndcx * depth_intrin.fx
            + depth_intrin.ppx
            + tx / depth_coarse * depth_intrin.fx
            + 0.5,
            0,
            dw - 1,
        )
    )
    dy = int(
        np.clip(
            ndcy * depth_intrin.fy
            + depth_intrin.ppy
            + ty / depth_coarse * depth_intrin.fy
            + 0.5,
            0,
            dh - 1,
        )
    )

    R = DEPTH_SAMPLE_RADIUS
    patch = (
        depth_arr[max(0, dy - R) : min(dh, dy + R + 1), max(0, dx - R) : min(dw, dx + R + 1)]
        .astype(np.float32)
        * depth_scale
    )
    valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ==============================================================================
# LiDAR ball center estimator
# ==============================================================================


def _estimate_lidar_center(points: np.ndarray, offset: float = LIDAR_CENTER_OFFSET):
    center = points.mean(axis=0).astype(np.float64)
    mask = np.linalg.norm(points - center, axis=1) < 0.20
    c = points[mask].mean(axis=0).astype(np.float64) if mask.any() else center
    n = np.linalg.norm(c)
    return (c + c / n * offset) if n > 1e-6 else c


# ==============================================================================
# ROS2 node -- /lowstate + /livox/lidar  (LiDAR worker writes lidar_slot)
# ==============================================================================


class _DetectorNode(Node):
    def __init__(self, lidar_slot: _MeasSlot):
        super().__init__("camera_lidar_fused_detector")
        self._slot = lidar_slot

        self.q_wy = 0.0
        self.q_wr = 0.0
        self.q_wp = 0.0

        self._buf_lock = threading.Lock()
        self._buf_msg = None
        self._buf_event = threading.Event()
        self._stop_flag = threading.Event()

        self.create_subscription(LowState, "/lowstate", self._cb_low, qos_profile_sensor_data)
        self.create_subscription(CustomMsg, "/livox/lidar", self._cb_lidar, 5)

        self._worker = threading.Thread(target=self._lidar_worker, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"_DetectorNode ready  HEAD_JOINT_ANGLE={HEAD_JOINT_ANGLE_DEG:.1f} deg"
        )

    def _cb_low(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]

    def _cb_lidar(self, msg: CustomMsg):
        with self._buf_lock:
            self._buf_msg = msg
        self._buf_event.set()

    def _lidar_worker(self):
        frame_n = 0
        while not self._stop_flag.is_set():
            if not self._buf_event.wait(timeout=1.0):
                self._slot.invalidate()
                continue
            self._buf_event.clear()

            with self._buf_lock:
                msg = self._buf_msg
            if msg is None:
                continue

            pts = msg.points
            n = len(pts)
            if n == 0:
                self._slot.invalidate()
                continue

            # Single-pass filter: reflectivity + ROI in one loop.
            # Process in chunks and yield the GIL between them so YOLO
            # pre/post processing can run.  Each chunk is pure-Python
            # attribute access on ROS msg objects (~20k pts total).
            CHUNK = 2000
            thr = LIDAR_REFLECT_THR
            cand_list = []
            for start in range(0, n, CHUNK):
                for p in pts[start : start + CHUNK]:
                    if p.reflectivity >= thr:
                        x_, y_, z_ = p.x, p.y, p.z
                        d2 = x_ * x_ + y_ * y_ + z_ * z_
                        if (
                            LIDAR_MIN_RANGE * LIDAR_MIN_RANGE <= d2 <= LIDAR_MAX_RANGE * LIDAR_MAX_RANGE
                            and LIDAR_Z_LOW <= z_ <= LIDAR_Z_HIGH
                            and LIDAR_X_LOW <= x_ <= LIDAR_X_HIGH
                            and LIDAR_Y_LOW <= y_ <= LIDAR_Y_HIGH
                        ):
                            cand_list.append((x_, y_, z_))
                time.sleep(0)  # yield GIL between chunks

            if cand_list:
                cand = np.array(cand_list, dtype=np.float32)
            else:
                cand = np.zeros((0, 3), dtype=np.float32)

            if cand.shape[0] < LIDAR_MIN_POINTS:
                self._slot.invalidate()
                if frame_n % 30 == 0:
                    print(
                        f"\r[lidar] no ball  hi={len(hi_idx)} cand={cand.shape[0]}  n={n}",
                        end="",
                        flush=True,
                    )
                frame_n += 1
                continue

            center_lidar = _estimate_lidar_center(cand)
            center_base = transform_point_mid360_to_base(
                center_lidar,
                self.q_wy,
                self.q_wr,
                self.q_wp,
                HEAD_JOINT_ANGLE,
                HEAD_MID360_JOINT,
            )
            self._slot.put(center_base)

            if frame_n % 5 == 0:
                x, y, z = float(center_base[0]), float(center_base[1]), float(center_base[2])
                print(
                    f"\r[lidar] pelvis=({x:+.3f},{y:+.3f},{z:+.3f})  cand={cand.shape[0]}",
                    end="",
                    flush=True,
                )
            frame_n += 1

    def destroy_node(self):
        self._stop_flag.set()
        self._buf_event.set()
        self._worker.join(timeout=2)
        super().destroy_node()


# ==============================================================================
# Fusion loop
# ==============================================================================


def _run_fusion(cam_slot, lidar_slot, dds, stop_flag):
    """Camera-priority fusion at FUSION_HZ -> DDS publish."""
    prev_t = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fusion_slot") as pool:
        while not stop_flag.is_set():
            time.sleep(1.0 / FUSION_HZ)
            now = time.perf_counter()
            prev_t = now

            fut_c = pool.submit(cam_slot.get)
            fut_l = pool.submit(lidar_slot.get)
            cam_pos, cam_age = fut_c.result()
            lidar_pos, lidar_age = fut_l.result()

            if cam_age < CAM_STALE_SEC:
                pos, valid, source_id = cam_pos, True, SOURCE_CAM
            elif lidar_age < LIDAR_STALE_SEC:
                pos, valid, source_id = lidar_pos, True, SOURCE_LIDAR
            else:
                pos, valid, source_id = np.zeros(3, dtype=np.float32), False, SOURCE_NONE

            dds.publish(float(pos[0]), float(pos[1]), float(pos[2]), valid=valid, source=source_id)


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fused ball detector: head D435 (YOLO) + MID360 LiDAR -> rt/ball_state"
    )
    parser.add_argument(
        "--model",
        default="onboard/perception/camera/models/yolo11m.pt",
        help="YOLO model path (.pt or .engine)",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--head-serial", default=None, help="D435 serial number")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Stream annotated video via HTTP on port 8080",
    )
    args = parser.parse_args()

    # -- Enumerate RealSense cameras -------------------------------------------
    ctx = rs.context()
    devs = ctx.query_devices()
    all_sns = [d.get_info(rs.camera_info.serial_number) for d in devs]
    all_names = [d.get_info(rs.camera_info.name) for d in devs]
    print("[INFO] Connected RealSense devices:")
    for i, (sn, nm) in enumerate(zip(all_sns, all_names)):
        print(f"  [{i}] serial={sn}  {nm}")
    if args.list_cameras:
        return
    if not all_sns:
        raise RuntimeError("No RealSense device found.")
    serial_head = args.head_serial or all_sns[0]
    print(f"[INFO] HEAD camera -> serial {serial_head}")

    # -- ROS2 ------------------------------------------------------------------
    rclpy.init()
    lidar_slot = _MeasSlot()
    cam_slot = _MeasSlot()
    node = _DetectorNode(lidar_slot)

    def _ros_spin():
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.002)

    threading.Thread(target=_ros_spin, name="ros_spin", daemon=True).start()
    print("[INFO] ROS2 spin thread started (/lowstate + /livox/lidar)")

    # -- DDS publisher ---------------------------------------------------------
    dds = BallStatePublisher(domain_id=0)
    print("[INFO] DDS publisher ready on 'rt/ball_state'")

    # -- YOLO ------------------------------------------------------------------
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path(args.model)
    print(f"[INFO] Loading YOLO: {model_path}  (device={device})")
    model = YOLO(model_path)
    is_trt = model_path.endswith(".engine")
    infer_kw = {} if is_trt else dict(device=device, half=True)

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] Warming up YOLO (~5 s)...")
    for i in range(5):
        t0 = time.perf_counter()
        model(dummy, verbose=False, **infer_kw)
        print(f"[INFO]   warmup[{i}]: {(time.perf_counter() - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    for _ in range(10):
        model(dummy, verbose=False, **infer_kw)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"[INFO] YOLO: {ms:.1f} ms/frame  ({1000 / ms:.0f} FPS)")

    # -- RealSense pipeline ----------------------------------------------------
    _FPS_TRIES = [(60, 60), (30, 30), (15, 15)]

    def _start_pipeline(serial, name):
        pipe = rs.pipeline()
        last_err = None
        for c_fps, d_fps in _FPS_TRIES:
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps)
            cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, d_fps)
            for attempt in range(2):
                print(
                    f"[INFO] Starting [{name}] (color {c_fps} Hz, depth {d_fps} Hz, "
                    f"attempt {attempt + 1})..."
                )
                try:
                    profile = pipe.start(cfg)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "resolve" in msg or "couldn't" in msg:
                        last_err = e
                        print(f"[WARN] [{name}] profile not supported: {e}")
                        break
                    raise
                try:
                    pipe.wait_for_frames(timeout_ms=5000)
                    print(
                        f"[INFO] [{name}] pipeline OK "
                        f"(color {c_fps} Hz, depth {d_fps} Hz)"
                    )
                    return pipe, profile
                except RuntimeError:
                    print(f"[WARN] [{name}] frame timeout -- hardware reset...")
                    pipe.stop()
                    for dev in rs.context().query_devices():
                        if dev.get_info(rs.camera_info.serial_number) == serial:
                            dev.hardware_reset()
                            break
                    time.sleep(3)
        err_msg = f"[{name}] pipeline failed. Tried Hz pairs {_FPS_TRIES}."
        if last_err is not None:
            err_msg += f" Last: {last_err!r}"
        raise RuntimeError(err_msg)

    pipe_head, prof_head = _start_pipeline(serial_head, "head")

    def _get_cam_params(profile):
        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        return (
            cp.get_intrinsics(),
            dp.get_intrinsics(),
            cp.get_extrinsics_to(dp),
            profile.get_device().first_depth_sensor().get_depth_scale(),
        )

    ci_h, di_h, c2d_h, ds_h = _get_cam_params(prof_head)
    print(f"[INFO] Head D435 -- color fx={ci_h.fx:.1f}  depth_scale={ds_h:.4f}")

    # -- MJPEG server (--show) -------------------------------------------------
    _mjpeg_frame = [None]
    _mjpeg_lock = threading.Lock()

    if args.show:
        import http.server
        import socketserver

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path not in ("/", "/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_sent = None
                try:
                    while True:
                        with _mjpeg_lock:
                            jpg = _mjpeg_frame[0]
                        if jpg is None or jpg is last_sent:
                            time.sleep(0.02)
                            continue
                        last_sent = jpg
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                        )
                except Exception:
                    pass

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        _httpd = socketserver.ThreadingTCPServer(("0.0.0.0", 8080), _Handler)
        _httpd.daemon_threads = True
        threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        print("[INFO] MJPEG stream -> http://192.168.123.164:8080/stream")

    # -- Camera pipeline -------------------------------------------------------
    stop_flag = threading.Event()
    frame_q = queue.Queue(maxsize=1)
    det_q = queue.Queue(maxsize=1)
    frame_q_lock = threading.Lock()
    det_q_lock = threading.Lock()
    vis_q = queue.Queue(maxsize=1) if args.show else None
    vis_q_lock = threading.Lock() if args.show else None

    # Stage 1: RealSense capture
    def _capture_loop():
        while not stop_flag.is_set():
            try:
                frames = pipe_head.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print("\n[WARN] [head] frame timeout, retrying...", flush=True)
                continue
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()
            _queue_put_latest(frame_q, (color, depth_arr), frame_q_lock)

    # Stage 2: YOLO inference (GPU)
    def _infer_loop():
        while not stop_flag.is_set():
            try:
                color, depth_arr = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            orig_h, orig_w = color.shape[:2]
            color_small = cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz
            results = model.predict(color_small, conf=CONF_THRESHOLD, verbose=False, **infer_kw)

            best_box, best_conf = None, 0.0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            xyxy_small = None
            if best_box is not None:
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                xyxy_small = (float(x1s), float(y1s), float(x2s), float(y2s))

            bundle = {
                "color": color,
                "depth": depth_arr,
                "sx": sx,
                "sy": sy,
                "yolo_hit": best_box is not None,
                "xyxy_small": xyxy_small,
                "best_conf": best_conf,
            }
            _queue_put_latest(det_q, bundle, det_q_lock)

    # Stage 3: geometry (depth + EMA + transform)
    def _geo_loop():
        state = _CamState()
        x = y = z = 0.0

        while not stop_flag.is_set():
            try:
                b = det_q.get(timeout=0.5)
            except queue.Empty:
                continue

            color = b["color"]
            depth_arr = b["depth"]
            sx, sy = b["sx"], b["sy"]
            yolo_hit = b["yolo_hit"]
            best_conf = b["best_conf"]

            if yolo_hit and b["xyxy_small"] is not None:
                state.miss_count = 0
                x1s, y1s, x2s, y2s = b["xyxy_small"]
                state.last_bbox = (
                    int(x1s * sx),
                    int(y1s * sy),
                    int(x2s * sx),
                    int(y2s * sy),
                )
            else:
                state.miss_count += 1

            published = False
            depth_m = 0.0
            best_box_this = yolo_hit

            if state.last_bbox is not None and state.miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = state.last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                depth_surface = _sample_depth(depth_arr, cx, cy, ci_h, di_h, c2d_h, ds_h)
                depth_m = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0

                if depth_m > 0:
                    p_opt = rs.rs2_deproject_pixel_to_point(ci_h, [cx, cy], depth_m)
                    p_cam = optical_to_body(p_opt)

                    if state.center_ema is None:
                        state.center_ema = p_cam.copy()
                    else:
                        gate_dist = np.linalg.norm(p_cam - state.center_ema)
                        if gate_dist < EMA_GATE:
                            state.center_ema = (
                                EMA_ALPHA * p_cam + (1 - EMA_ALPHA) * state.center_ema
                            )
                        else:
                            print(
                                f"\n[WARN] [head] EMA gate {gate_dist:.2f}m -> reset",
                                flush=True,
                            )
                            state.center_ema = p_cam.copy()

                    p_base = transform_point_camera_to_base(
                        state.center_ema,
                        node.q_wy,
                        node.q_wr,
                        node.q_wp,
                        HEAD_JOINT_ANGLE,
                    )
                    x, y, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
                    cam_slot.put(p_base)
                    published = True

                    status = "BALL " if best_box_this else "COAST"
                    print(
                        f"\r[head/{status}] pelvis=({x:+.3f},{y:+.3f},{z:+.3f})  "
                        f"d={depth_m:.2f}m  conf={best_conf:.2f}  fps={state.fps.fps:.1f}",
                        end="",
                        flush=True,
                    )

            if not published and state.fps.fps > 0:
                print(
                    f"\r[head/     ] no ball  fps={state.fps.fps:.1f}" + " " * 30,
                    end="",
                    flush=True,
                )

            if args.show and vis_q is not None:
                vis = color.copy()
                if state.last_bbox is not None and state.miss_count <= COAST_FRAMES:
                    bx1, by1, bx2, by2 = state.last_bbox
                    col = (0, 255, 0) if best_box_this else (0, 165, 255)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, 2)
                    cv2.circle(vis, ((bx1 + bx2) // 2, (by1 + by2) // 2), 4, col, -1)
                    lbl = (
                        f"ball {best_conf:.2f}"
                        if best_box_this
                        else f"coast {state.miss_count}/{COAST_FRAMES}"
                    )
                    cv2.putText(
                        vis, lbl, (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2
                    )
                    if published:
                        cv2.putText(
                            vis,
                            f"pelvis ({x:+.2f},{y:+.2f},{z:+.2f})m  d={depth_m:.2f}m",
                            (10, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2,
                        )
                cv2.putText(
                    vis,
                    f"head {state.fps.fps:.1f}fps",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                _queue_put_latest(vis_q, vis, vis_q_lock)

            state.fps.tick()

    # MJPEG encode (offloaded from geometry thread)
    def _mjpeg_encode_loop():
        while not stop_flag.is_set():
            try:
                vis = vis_q.get(timeout=0.15)
            except queue.Empty:
                continue
            _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
            with _mjpeg_lock:
                _mjpeg_frame[0] = jpg.tobytes()

    # -- Start all threads -----------------------------------------------------
    threading.Thread(target=_capture_loop, name="rs_capture", daemon=True).start()
    threading.Thread(target=_infer_loop, name="yolo_infer", daemon=True).start()
    threading.Thread(target=_geo_loop, name="cam_geo", daemon=True).start()
    if args.show:
        threading.Thread(target=_mjpeg_encode_loop, name="mjpeg_enc", daemon=True).start()

    threading.Thread(
        target=_run_fusion,
        args=(cam_slot, lidar_slot, dds, stop_flag),
        name="fusion",
        daemon=True,
    ).start()

    thread_names = "rs_capture, yolo_infer, cam_geo, lidar_worker, ros_spin, fusion"
    if args.show:
        thread_names += ", mjpeg_enc"
    print(f"[INFO] Threads: {thread_names}")
    print("[INFO] Running. Press Ctrl-C to stop.")

    # -- Main thread: wait for Ctrl-C -----------------------------------------
    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        pipe_head.stop()
        if args.show:
            _httpd.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
