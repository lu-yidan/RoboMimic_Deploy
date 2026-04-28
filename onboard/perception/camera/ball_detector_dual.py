"""Dual RealSense D435 ball detector — runs on G1 onboard computer.

Two D435 cameras (head + chest).  Any camera that detects the ball immediately
publishes rt/ball_state.  Both workers share ONE YOLO model and ONE GPU lock so
GPU throughput is barely slower than single-camera mode (~24 fps/camera vs ~30).

Subscribes to:
  /lowstate  (Unitree G1 joint states)

Publishes via DDS:
  "rt/ball_state"  (BallState — ball position in pelvis frame)

Usage:
    bash onboard/perception/camera/run_dual.sh
    bash onboard/perception/camera/run_dual.sh --show          # open browser on port 8080
    bash onboard/perception/camera/run_dual.sh --list-cameras  # print serials and exit
    bash onboard/perception/camera/run_dual.sh --head-serial 12345 --chest-serial 67890

Camera assignment:
  By default the first enumerated device = head, second = chest.
  Override with --head-serial / --chest-serial if ordering is wrong.

Chest camera extrinsics:
  Update _CHEST_XYZ and _CHEST_RPY in camera_to_base.py once measured.

--show mode:
  Opens an HTTP server on port 8080.
  Visit http://<robot-ip>:8080 to see both cameras side-by-side with detections.
  Individual streams: http://<robot-ip>:8080/stream/head
                      http://<robot-ip>:8080/stream/chest
"""

import sys
import warnings
import numpy as _np_compat
# TensorRT Python binding may need legacy numpy aliases
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(_np_compat, 'bool'):   _np_compat.bool   = bool
    if not hasattr(_np_compat, 'int'):    _np_compat.int    = int
    if not hasattr(_np_compat, 'float'):  _np_compat.float  = float
    if not hasattr(_np_compat, 'object'): _np_compat.object = object
del _np_compat

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

import argparse
import threading
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowState

from onboard.perception.camera.camera_to_base import (
    transform_point_camera_to_base,
    transform_point_chest_camera_to_base,
    optical_to_body,
)
from common.ball_state_dds import BallStatePublisher, SOURCE_CAM, SOURCE_NONE


# ── Detection parameters ──────────────────────────────────────────────────────
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD       = 0.3
DEPTH_SAMPLE_RADIUS  = 5
DEPTH_MIN            = 0.1    # m
DEPTH_MAX            = 10.0   # m
BALL_RADIUS          = 0.115  # m  (depth sensor sees front surface → add radius)
EMA_ALPHA            = 0.6
EMA_GATE             = 0.6    # m  (jump larger than this resets EMA)
COAST_FRAMES         = 15     # frames to hold last bbox after YOLO miss
VALID_HOLD_SEC       = 0.5    # publish valid=False only after both cameras are
                               # quiet for this long


# ── FPS counter ───────────────────────────────────────────────────────────────
class _FPS:
    def __init__(self, window=30):
        self._t, self._w = [], window

    def tick(self):
        now = time.perf_counter()
        self._t.append(now)
        if len(self._t) > self._w:
            self._t.pop(0)

    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


# ── ROS2 joint listener ───────────────────────────────────────────────────────
class _JointListener(Node):
    def __init__(self):
        super().__init__("dual_camera_ball_detector_joints")
        self.q_wy   = 0.0
        self.q_wr   = 0.0
        self.q_wp   = 0.0
        self.q_head = 0.593412   # URDF default; head is not commanded at runtime
        self.create_subscription(
            LowState, "/lowstate", self._cb, qos_profile_sensor_data
        )

    def _cb(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]


# ── Per-camera tracking state ─────────────────────────────────────────────────
class _CamState:
    def __init__(self):
        self.center_ema = None
        self.last_bbox  = None
        self.miss_count = 0
        self.fps        = _FPS()


# ── Camera frame buffer ───────────────────────────────────────────────────────
class _CamBuf:
    def __init__(self):
        self.lock    = threading.Lock()
        self.frames  = None
        self.updated = threading.Event()


# ── Depth sampling (color pixel → corrected depth value in metres) ────────────
def _sample_depth(depth_arr, cx, cy,
                  color_intrin, depth_intrin, c2d_extr, depth_scale):
    """Return depth (metres) at the colour pixel (cx,cy), with parallax correction."""
    dh, dw = depth_arr.shape
    ndcx = (cx - color_intrin.ppx) / color_intrin.fx
    ndcy = (cy - color_intrin.ppy) / color_intrin.fy

    # Coarse depth (no parallax)
    dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
    dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
    dx0 = max(0, min(dw - 1, dx0))
    dy0 = max(0, min(dh - 1, dy0))
    raw0 = depth_arr[dy0, dx0]
    depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0

    # Parallax correction
    tx = c2d_extr.translation[0]
    ty = c2d_extr.translation[1]
    dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx
             + tx / depth_coarse * depth_intrin.fx + 0.5)
    dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy
             + ty / depth_coarse * depth_intrin.fy + 0.5)
    dx = max(0, min(dw - 1, dx))
    dy = max(0, min(dh - 1, dy))

    R = DEPTH_SAMPLE_RADIUS
    patch = (depth_arr[max(0, dy-R):min(dh, dy+R+1),
                       max(0, dx-R):min(dw, dx+R+1)]
             .astype(np.float32) * depth_scale)
    valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Dual D435 + YOLO ball detector → rt/ball_state"
    )
    parser.add_argument("--model", default="onboard/perception/camera/models/yolo11m.pt",
                        help="YOLO model (.pt or .engine)")
    parser.add_argument("--imgsz",        type=int, default=320)
    parser.add_argument("--width",        type=int, default=640)
    parser.add_argument("--height",       type=int, default=480)
    parser.add_argument("--list-cameras", action="store_true",
                        help="Print connected RealSense serials and exit")
    parser.add_argument("--head-serial",  default=None,
                        help="Serial number of the HEAD camera (default: first enumerated)")
    parser.add_argument("--chest-serial", default=None,
                        help="Serial number of the CHEST camera (default: second enumerated)")
    parser.add_argument("--show", action="store_true",
                        help="Stream annotated video from both cameras via HTTP (port 8080)")
    args = parser.parse_args()

    # ── Enumerate cameras ─────────────────────────────────────────────────
    ctx      = rs.context()
    devs     = ctx.query_devices()
    all_sns  = [d.get_info(rs.camera_info.serial_number) for d in devs]
    all_names = [d.get_info(rs.camera_info.name) for d in devs]

    print("[INFO] Connected RealSense devices:")
    for i, (sn, nm) in enumerate(zip(all_sns, all_names)):
        print(f"  [{i}] serial={sn}  {nm}")

    if args.list_cameras:
        return

    if len(all_sns) < 2:
        raise RuntimeError(
            f"Need at least 2 RealSense cameras, found {len(all_sns)}. "
            "Check USB connections."
        )

    serial_head  = args.head_serial  or all_sns[0]
    serial_chest = args.chest_serial or all_sns[1]
    print(f"[INFO] HEAD  camera → serial {serial_head}")
    print(f"[INFO] CHEST camera → serial {serial_chest}")

    if serial_head == serial_chest:
        raise RuntimeError("Head and chest serial numbers are the same. "
                           "Use --head-serial / --chest-serial to assign correctly.")

    # ── ROS2 joint listener ───────────────────────────────────────────────
    rclpy.init()
    joint = _JointListener()

    def _spin_loop():
        while True:
            rclpy.spin_once(joint, timeout_sec=0.0)
            time.sleep(0.02)   # 50 Hz — adequate for joint angle updates
    threading.Thread(target=_spin_loop, daemon=True).start()
    print("[INFO] ROS2 /lowstate listener started")

    # ── DDS publisher (shared, protected by dds_lock) ─────────────────────
    dds      = BallStatePublisher(domain_id=0)
    dds_lock = threading.Lock()
    print("[INFO] DDS publisher ready on 'rt/ball_state'")

    # Shared "last valid detection" timestamp — used to decide when to send
    # valid=False (only after BOTH cameras have been quiet long enough).
    last_valid_t    = [0.0]
    last_valid_lock = threading.Lock()

    # ── YOLO model (shared across both camera workers) ────────────────────
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
    model    = YOLO(model_path)
    is_trt   = str(model_path).endswith('.engine')
    infer_kw = {} if is_trt else dict(device=device, half=True)

    # GPU lock: only one YOLO inference at a time across both workers.
    # On Jetson the GPU is shared; serialising here avoids contention and
    # keeps the two workers from corrupting each other's model state.
    gpu_lock = threading.Lock()

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] Warming up YOLO (~5 s)...")
    for i in range(5):
        t_w = time.perf_counter()
        model(dummy, verbose=False, **infer_kw)
        print(f"[INFO]   warmup[{i}]: {(time.perf_counter()-t_w)*1000:.1f} ms")

    t0 = time.perf_counter()
    for _ in range(10):
        model(dummy, verbose=False, **infer_kw)
    ms_per = (time.perf_counter() - t0) / 10 * 1000
    print(f"[INFO] YOLO single-inference: {ms_per:.1f} ms  "
          f"(≈ {1000/ms_per:.0f} FPS; each camera gets ~{500/ms_per:.0f} FPS with dual-worker)")

    # ── MJPEG display server (--show) ─────────────────────────────────────
    # Shared JPEG buffers — each YOLO worker writes its latest annotated frame.
    # One HTTP server serves both streams plus a composite HTML page.
    import io, http.server, socketserver as _ss
    import cv2 as _cv2

    _mjpeg = {"head": None, "chest": None}
    _mjpeg_lock = threading.Lock()

    _HTML_PAGE = b"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dual D435 Ball Detector</title>
  <style>
    body  { background:#111; color:#eee; font-family:monospace;
            margin:0; padding:12px; }
    h2    { margin:0 0 8px; font-size:14px; color:#aaa; }
    .wrap { display:flex; gap:10px; align-items:flex-start; }
    .cam  { flex:1; }
    img   { width:100%; border:1px solid #444; display:block; }
    .lbl  { text-align:center; font-size:12px; color:#888; margin-top:4px; }
  </style>
</head>
<body>
  <h2>Dual D435 &mdash; Ball Detector</h2>
  <div class="wrap">
    <div class="cam">
      <img src="/stream/head" />
      <div class="lbl">HEAD camera</div>
    </div>
    <div class="cam">
      <img src="/stream/chest" />
      <div class="lbl">CHEST camera</div>
    </div>
  </div>
</body>
</html>
"""

    class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # silence per-request access log

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(_HTML_PAGE)))
                self.end_headers()
                self.wfile.write(_HTML_PAGE)
                return

            cam = None
            if self.path == '/stream/head':
                cam = 'head'
            elif self.path == '/stream/chest':
                cam = 'chest'
            if cam is None:
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last_sent = None
            try:
                while True:
                    with _mjpeg_lock:
                        jpg = _mjpeg[cam]
                    if jpg is None or jpg is last_sent:
                        time.sleep(0.02)
                        continue
                    last_sent = jpg
                    self.wfile.write(
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                    )
            except Exception:
                pass

    if args.show:
        _ss.ThreadingTCPServer.allow_reuse_address = True
        _httpd = _ss.ThreadingTCPServer(('0.0.0.0', 8080), _MJPEGHandler)
        _httpd.daemon_threads = True
        threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        print("[INFO] MJPEG server → http://192.168.123.164:8080")
        print("[INFO]   /stream/head   — head camera feed")
        print("[INFO]   /stream/chest  — chest camera feed")

    # ── RealSense pipeline factory ────────────────────────────────────────
    stop_flag = threading.Event()

    # Same as single-camera: color@60 + depth@90 often fails ("Couldn't resolve").
    _FPS_TRIES = [(60, 60), (30, 30), (15, 15)]

    def _start_pipeline(serial, name):
        pipe = rs.pipeline()
        last_err = None
        for c_fps, d_fps in _FPS_TRIES:
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps,
            )
            cfg.enable_stream(
                rs.stream.depth, args.width, args.height, rs.format.z16, d_fps,
            )
            for attempt in range(2):
                try:
                    print(
                        f"[INFO] Starting [{name}] (color {c_fps} Hz, depth {d_fps} Hz, "
                        f"attempt {attempt + 1})...",
                    )
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
                        f"[INFO] [{name}] pipeline OK (serial={serial}, "
                        f"color {c_fps} Hz, depth {d_fps} Hz)",
                    )
                    return pipe, profile
                except RuntimeError:
                    print(f"[WARN] [{name}] frame timeout — hardware reset...")
                    pipe.stop()
                    for dev in rs.context().query_devices():
                        if dev.get_info(rs.camera_info.serial_number) == serial:
                            dev.hardware_reset()
                            break
                    time.sleep(3)
        msg = f"[{name}] pipeline failed. Tried Hz pairs {_FPS_TRIES}."
        if last_err is not None:
            msg += f" Last resolve error: {last_err!r}"
        raise RuntimeError(msg)

    pipe_head,  prof_head  = _start_pipeline(serial_head,  "head")
    pipe_chest, prof_chest = _start_pipeline(serial_chest, "chest")

    # ── Per-camera intrinsics / extrinsics ────────────────────────────────
    def _get_cam_params(profile):
        cp  = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dp  = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci  = cp.get_intrinsics()
        di  = dp.get_intrinsics()
        c2d = cp.get_extrinsics_to(dp)
        ds  = profile.get_device().first_depth_sensor().get_depth_scale()
        return ci, di, c2d, ds

    ci_h, di_h, c2d_h, ds_h = _get_cam_params(prof_head)
    ci_c, di_c, c2d_c, ds_c = _get_cam_params(prof_chest)
    print(f"[INFO] Head  — color fx={ci_h.fx:.1f}  depth_scale={ds_h:.4f}")
    print(f"[INFO] Chest — color fx={ci_c.fx:.1f}  depth_scale={ds_c:.4f}")

    # ── Camera capture threads ────────────────────────────────────────────
    buf_head  = _CamBuf()
    buf_chest = _CamBuf()

    def _capture_loop(pipe, buf, name):
        while not stop_flag.is_set():
            try:
                frames = pipe.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print(f"\n[WARN] [{name}] frame timeout, retrying...", flush=True)
                continue
            with buf.lock:
                buf.frames = frames   # swap reference — O(1), non-blocking
            buf.updated.set()

    # ── Generic YOLO worker (one instance per camera) ─────────────────────
    def _yolo_worker(name, buf,
                     color_intrin, depth_intrin, c2d_extr, depth_scale,
                     transform_fn):
        """
        Waits for a new frame, runs YOLO (with gpu_lock), samples depth,
        and publishes ball position if detected.

        transform_fn(p_cam_body) → p_pelvis   (captures joint via closure)
        """
        state = _CamState()
        x = y = z = 0.0   # last published pelvis coords (for OSD)

        while not stop_flag.is_set():
            if not buf.updated.wait(timeout=1.0):
                continue
            buf.updated.clear()

            with buf.lock:
                frames = buf.frames
            if frames is None:
                continue

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color     = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()

            orig_h, orig_w = color.shape[:2]
            color_small    = _cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            # GPU-serialised YOLO inference.
            # Using predict() (not track()) because two workers sharing one
            # model instance would corrupt the persistent tracker's state.
            with gpu_lock:
                results = model.predict(
                    color_small, conf=CONF_THRESHOLD,
                    verbose=False, **infer_kw,
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

            published_valid = False
            depth_surface   = 0.0
            depth_m         = 0.0

            if state.last_bbox is not None and state.miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = state.last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                depth_surface = _sample_depth(
                    depth_arr, cx, cy,
                    color_intrin, depth_intrin, c2d_extr, depth_scale,
                )
                depth_m = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0

                if depth_m > 0:
                    p_opt = rs.rs2_deproject_pixel_to_point(
                        color_intrin, [cx, cy], depth_m
                    )
                    p_cam = optical_to_body(p_opt)

                    if state.center_ema is None:
                        gate_dist = 0.0
                        state.center_ema = p_cam.copy()
                    else:
                        gate_dist = np.linalg.norm(p_cam - state.center_ema)
                        if gate_dist < EMA_GATE:
                            state.center_ema = (EMA_ALPHA * p_cam
                                                + (1 - EMA_ALPHA) * state.center_ema)
                        else:
                            print(f"\n[WARN] [{name}] EMA gate {gate_dist:.2f}m → reset",
                                  flush=True)
                            state.center_ema = p_cam.copy()

                    p_base = transform_fn(state.center_ema)
                    x, y, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
                    detected = best_box is not None

                    with dds_lock:
                        dds.publish(x, y, z, valid=detected, source=SOURCE_CAM)
                    with last_valid_lock:
                        last_valid_t[0] = time.time()
                    published_valid = True

                    status = "BALL " if detected else "COAST"
                    print(
                        f"\r[{name}/{status}] pelvis=({x:+.3f},{y:+.3f},{z:+.3f})  "
                        f"surf={depth_surface:.2f}m  ctr={depth_m:.2f}m  "
                        f"conf={best_conf:.2f}  fps={state.fps.fps:.1f}",
                        end="", flush=True,
                    )

            if not published_valid:
                # Only publish valid=False if BOTH cameras have been quiet.
                # This prevents one camera's miss from overwriting the other's
                # valid detection.
                with last_valid_lock:
                    quiet_sec = time.time() - last_valid_t[0]
                if quiet_sec > VALID_HOLD_SEC:
                    with dds_lock:
                        dds.publish(0.0, 0.0, 0.0, valid=False, source=SOURCE_NONE)
                    if state.fps.fps > 0:
                        print(
                            f"\r[{name}/     ] no ball  fps={state.fps.fps:.1f}" + " " * 20,
                            end="", flush=True,
                        )

            # ── Annotate frame and push to MJPEG server (--show) ─────────
            if args.show:
                vis = color.copy()
                if state.last_bbox is not None and state.miss_count <= COAST_FRAMES:
                    bx1, by1, bx2, by2 = state.last_bbox
                    bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    col_box  = (0, 255, 0) if best_box is not None else (0, 165, 255)
                    _cv2.rectangle(vis, (bx1, by1), (bx2, by2), col_box, 2)
                    _cv2.circle(vis, (bcx, bcy), 4, col_box, -1)
                    lbl = (f"ball {best_conf:.2f}" if best_box is not None
                           else f"coast {state.miss_count}/{COAST_FRAMES}")
                    _cv2.putText(vis, lbl, (bx1, by1 - 8),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_box, 2)
                    if published_valid:
                        info = f"pelvis ({x:+.2f}, {y:+.2f}, {z:+.2f})m  d={depth_m:.2f}m"
                        _cv2.putText(vis, info, (10, vis.shape[0] - 10),
                                     _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                fps_txt = f"{name}  {state.fps.fps:.1f} fps"
                _cv2.putText(vis, fps_txt, (10, 24),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                _, jpg = _cv2.imencode('.jpg', vis, [_cv2.IMWRITE_JPEG_QUALITY, 60])
                with _mjpeg_lock:
                    _mjpeg[name] = jpg.tobytes()

            state.fps.tick()

    # ── Transform closures (capture joint reference) ──────────────────────
    def tf_head(p_cam):
        return transform_point_camera_to_base(
            p_cam, joint.q_wy, joint.q_wr, joint.q_wp, joint.q_head
        )

    def tf_chest(p_cam):
        return transform_point_chest_camera_to_base(
            p_cam, joint.q_wy, joint.q_wr, joint.q_wp
        )

    # ── Launch all threads ────────────────────────────────────────────────
    threads = [
        threading.Thread(
            target=_capture_loop, args=(pipe_head,  buf_head,  "head"),  daemon=True
        ),
        threading.Thread(
            target=_capture_loop, args=(pipe_chest, buf_chest, "chest"), daemon=True
        ),
        threading.Thread(
            target=_yolo_worker,
            args=("head",  buf_head,  ci_h, di_h, c2d_h, ds_h, tf_head),
            daemon=True,
        ),
        threading.Thread(
            target=_yolo_worker,
            args=("chest", buf_chest, ci_c, di_c, c2d_c, ds_c, tf_chest),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    print("[INFO] Dual camera running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        pipe_head.stop()
        pipe_chest.stop()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
