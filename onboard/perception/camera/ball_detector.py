"""Ball detector service — RealSense D435 + YOLOv8, runs on G1 onboard computer.

Subscribes to:
  /lowstate   (Unitree G1 joint states via ROS2, for waist/head angles)

Publishes via DDS:
  "rt/ball_state"  (ball position in pelvis body frame, ~30 Hz)

Usage (on G1 onboard):
    cd RoboMimicDeploy_G1
    source /opt/ros/foxy/setup.bash
    source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh
    python onboard/perception/camera/ball_detector.py
    python onboard/perception/camera/ball_detector.py --model yolov8s.pt --imgsz 320
"""

import sys
import warnings
import numpy as _np_compat
# TensorRT 8.5 Python binding was built against numpy <1.24 and uses removed
# aliases (np.bool, np.int, np.float).  Patch them back before any TRT import.
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
    optical_to_body,
)
from common.ball_state_dds import BallStatePublisher


# ── Detection params ─────────────────────────────────────────────────────────
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD       = 0.3
DEPTH_SAMPLE_RADIUS  = 5
DEPTH_MIN            = 0.1   # m
DEPTH_MAX            = 10.0  # m
BALL_RADIUS          = 0.115 # m — physical radius; depth sensor sees the front
                              #     surface, so we add this to get the ball center
EMA_ALPHA            = 0.6
EMA_GATE             = 0.6   # m — jump larger than this skips EMA update
COAST_FRAMES         = 10    # frames to hold last position after YOLO misses


# ── FPS counter ──────────────────────────────────────────────────────────────

class _FPS:
    def __init__(self, window=30):
        self._t = []
        self._w = window

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
    """Background ROS2 node — reads waist joint angles from /lowstate."""

    def __init__(self):
        super().__init__("camera_ball_detector_joint_listener")
        self.q_wy   = 0.0
        self.q_wr   = 0.0
        self.q_wp   = 0.0
        # head_joint is not commanded at runtime; use URDF default.
        self.q_head = 0.593412
        self.create_subscription(
            LowState, "/lowstate", self._cb, qos_profile_sensor_data
        )
        self.get_logger().info("camera_ball_detector: subscribed to /lowstate")

    def _cb(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RealSense D435 + YOLO ball detector → rt/ball_state"
    )
    parser.add_argument("--model", default="onboard/perception/camera/models/yolo11m.pt",
                        help="YOLO model path (default: models/yolo11m.pt; auto-uses .engine if found)")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="YOLO input size in pixels (smaller = faster, default 320)")
    parser.add_argument("--width",  type=int, default=640,
                        help="camera capture width  (default 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="camera capture height (default 480)")
    parser.add_argument("--show", action="store_true",
                        help="stream annotated video via MJPEG HTTP (open browser on port 8080)")
    args = parser.parse_args()

    # ── ROS2 joint listener ───────────────────────────────────────────────
    # spin_once at 50 Hz instead of rclpy.spin() to avoid saturating the GIL
    # with 500 Hz /lowstate deserialization (which crushes CUDA throughput).
    rclpy.init()
    joint = _JointListener()
    def _spin_loop():
        while True:
            rclpy.spin_once(joint, timeout_sec=0.0)
            time.sleep(0.02)   # 50 Hz — enough for joint angle updates
    threading.Thread(target=_spin_loop, daemon=True).start()
    print("[INFO] ROS2 joint listener started (/lowstate)")

    # ── DDS publisher ─────────────────────────────────────────────────────
    dds = BallStatePublisher(domain_id=0)
    print("[INFO] DDS publisher ready on 'rt/ball_state'")

    # ── YOLO ─────────────────────────────────────────────────────────────
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Auto-detect TensorRT engine: if a .engine file exists alongside the
    # requested .pt model, prefer it (3-4× faster on Jetson).
    model_path = args.model
    if model_path.endswith('.pt'):
        engine_path = model_path.replace('.pt', '.engine')
        if __import__('os').path.exists(engine_path):
            model_path = engine_path
            print(f"[INFO] TensorRT engine found, using: {engine_path}")
        else:
            print(f"[INFO] No .engine found at {engine_path}, using .pt")
    print(f"[INFO] Loading YOLO model: {model_path}  (device={device})")
    model = YOLO(model_path)

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    is_trt = str(model_path).endswith('.engine')
    warmup_kw = {} if is_trt else dict(device=device, half=True)
    print("[INFO] YOLO warming up (first run triggers CUDA JIT / TRT init, takes ~5s)...")
    for i in range(5):
        t_w = time.perf_counter()
        model(dummy, verbose=False, **warmup_kw)
        print(f"[INFO]   warmup[{i}]: {(time.perf_counter()-t_w)*1000:.1f}ms")
    t0 = time.perf_counter()
    for _ in range(10):
        model(dummy, verbose=False, **warmup_kw)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"[INFO] YOLO inference: {ms:.1f} ms/frame  (≈ {1000/ms:.0f} FPS upper bound)")

    # ── RealSense ─────────────────────────────────────────────────────────
    # We do NOT use rs.align() — it costs ~100 ms on ARM CPU and holds the GIL.
    # Instead we read color and depth frames separately and map the ball-center
    # pixel from color space to depth space via rs2_project_color_pixel_to_depth_pixel,
    # which is a O(1) operation (<0.1 ms).
    pipeline = rs.pipeline()
    # D435/D435I: requesting color@60 Hz + depth@90 Hz together often fails with
    # RuntimeError: Couldn't resolve requests — the firmware cannot satisfy
    # mismatched rates.  Use equal FPS (60/60 preferred), then fall back.
    _FPS_TRIES = [(60, 60), (30, 30), (15, 15)]

    def _start_pipeline():
        last_err = None
        for c_fps, d_fps in _FPS_TRIES:
            rs_cfg = rs.config()
            rs_cfg.enable_stream(
                rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps,
            )
            rs_cfg.enable_stream(
                rs.stream.depth, args.width, args.height, rs.format.z16, d_fps,
            )
            for attempt in range(2):
                try:
                    print(
                        f"[INFO] Starting RealSense (color {c_fps} Hz, depth {d_fps} Hz, "
                        f"attempt {attempt + 1})...",
                    )
                    profile = pipeline.start(rs_cfg)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "resolve" in msg or "couldn't" in msg:
                        last_err = e
                        print(f"[WARN] Stream profile not supported: {e}")
                        break  # try next fps pair
                    raise
                try:
                    pipeline.wait_for_frames(timeout_ms=5000)
                    print(
                        f"[INFO] RealSense pipeline OK (color {c_fps} Hz, "
                        f"depth {d_fps} Hz)",
                    )
                    return profile
                except RuntimeError:
                    print("[WARN] Frame timeout — performing hardware reset...")
                    pipeline.stop()
                    ctx  = rs.context()
                    devs = ctx.query_devices()
                    if len(devs) == 0:
                        raise RuntimeError("No RealSense device found.")
                    devs[0].hardware_reset()
                    time.sleep(3)
        msg = (
            "RealSense failed to start. Tried color/depth Hz pairs: "
            f"{_FPS_TRIES}."
        )
        if last_err is not None:
            msg += f" Last resolve error: {last_err!r}"
        raise RuntimeError(msg)

    profile = _start_pipeline()

    # ── Intrinsics and extrinsics ─────────────────────────────────────────
    # The D435 has SEPARATE color and depth sensors (measured at 640×480):
    #   Color camera:  FOV ~55.6°H,  fx ≈ 607  ppx ≈ 317
    #   Depth camera:  FOV ~79.3°H,  fx ≈ 386  ppx ≈ 320  ← wider FOV!
    #   Color→Depth baseline: tx ≈ -14.5 mm horizontal
    #
    # Naively using color pixel (cx,cy) directly in depth_arr[cy,cx] is WRONG:
    #   - FOV scale error: at cx=500 (183px from center), depth px offset=436 vs naive 500 → 64px error
    #   - Baseline parallax at 1m: 14.5mm/1000mm * 386 ≈ 6px additional error
    #   Total error at image edge: ~70 pixels = ~18cm lateral error at 1m!
    #
    # Correct approach:
    #   1. Map color pixel → depth pixel via intrinsics + extrinsics
    #   2. Sample depth_arr at the mapped depth pixel
    #   3. Deproject using COLOR intrinsics + color pixel (gives correct 3-D ray)
    color_profile  = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile  = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin   = color_profile.get_intrinsics()
    depth_intrin   = depth_profile.get_intrinsics()
    # Extrinsics: rotation + translation that maps a point from color frame → depth frame
    color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
    intrinsics     = color_intrin   # alias used by rs2_deproject_pixel_to_point below

    # Depth scale converts raw uint16 → metres (typically 0.001 for D435)
    depth_sensor   = profile.get_device().first_depth_sensor()
    depth_scale    = depth_sensor.get_depth_scale()
    print(f"[INFO] Color intrinsics — fx={color_intrin.fx:.1f}  fy={color_intrin.fy:.1f}  "
          f"ppx={color_intrin.ppx:.1f}  ppy={color_intrin.ppy:.1f}")
    print(f"[INFO] Depth intrinsics — fx={depth_intrin.fx:.1f}  fy={depth_intrin.fy:.1f}  "
          f"ppx={depth_intrin.ppx:.1f}  ppy={depth_intrin.ppy:.1f}  "
          f"depth_scale={depth_scale:.4f}")
    t_cd = color_to_depth_extr.translation   # [tx, ty, tz] in metres
    print(f"[INFO] Color→Depth translation — tx={t_cd[0]*1000:.1f}mm  "
          f"ty={t_cd[1]*1000:.1f}mm  tz={t_cd[2]*1000:.1f}mm")

    # ── Shared state (camera thread → YOLO thread) ────────────────────────
    # We only share the raw frameset; alignment is done in the YOLO thread
    # to avoid holding the GIL during align.process() on the camera thread.
    buf_lock    = threading.Lock()
    buf_frames  = None   # raw rs2.composite_frame
    buf_updated = threading.Event()
    stop_flag   = threading.Event()

    # ── Display queue (YOLO thread → MJPEG server thread) ────────────────
    import queue as _queue
    import cv2
    disp_queue  = _queue.Queue(maxsize=1) if args.show else None

    if args.show:
        import http.server, socketserver, io

        _mjpeg_frame = [None]
        _mjpeg_lock  = threading.Lock()

        class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass  # silence access logs

            def do_GET(self):
                if self.path not in ('/', '/stream'):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    last_sent = None
                    while True:
                        with _mjpeg_lock:
                            jpg = _mjpeg_frame[0]
                        if jpg is None or jpg is last_sent:
                            time.sleep(0.02)   # 50 Hz poll, avoid busy-loop
                            continue
                        last_sent = jpg
                        self.wfile.write(
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                        )
                except Exception:
                    pass

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        _httpd = socketserver.ThreadingTCPServer(('0.0.0.0', 8080), _MJPEGHandler)
        _httpd.daemon_threads = True
        threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        print("[INFO] MJPEG stream started → open http://192.168.123.164:8080 in your browser")

    # ── YOLO thread ───────────────────────────────────────────────────────
    def yolo_worker():
        import os
        # Raise thread priority if permitted.
        try:
            param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
            print("[INFO] YOLO thread: SCHED_FIFO priority set.")
        except (PermissionError, OSError):
            try:
                os.nice(-10)
            except PermissionError:
                pass

        center_ema = None
        last_bbox  = None
        miss_count = 0
        yolo_fps   = _FPS()
        import cv2 as _cv2

        _frame_n = 0
        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                frames = buf_frames   # grab frameset reference

            # Get color and depth frames directly — NO align.process() needed.
            # align.process() remaps the full depth image (~100ms on ARM, holds GIL).
            # We only need depth at one small patch (ball center), sampled later.
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data()).copy()
            # Copy the full depth array once from DMA into Python heap.
            # A single numpy copy of the full 640×480 uint16 frame (0.6 MB) takes
            # ~2ms but is much faster than 121 individual get_distance() calls
            # (each of which incurs Python call overhead).
            depth_raw = np.asanyarray(df.get_data())   # DMA view
            depth_arr = depth_raw.copy()               # move to cache-friendly heap

            orig_h, orig_w = color.shape[:2]
            color_small = _cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            # YOLO detection — TRT engine ignores device/half (fixed at export)
            track_kw = {} if is_trt else dict(device=device, half=True)
            results = model.track(color_small, conf=CONF_THRESHOLD,
                                  persist=True, verbose=False,
                                  **track_kw)
            best_box  = None
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            if best_box is not None:
                miss_count = 0
                # 将检测框坐标从缩放图映射回原图
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                last_bbox = (int(x1s * sx), int(y1s * sy),
                             int(x2s * sx), int(y2s * sy))
            else:
                miss_count += 1

            # Depth + coordinate transform
            published_valid = False
            if last_bbox is not None and miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ── Color pixel → Depth pixel mapping ──────────────────────
                # The D435 color and depth sensors have different FOV and a
                # ~55mm physical baseline.  We must map (cx,cy) in color space
                # to (dx,dy) in depth space before reading depth_arr.
                #
                # Step 1 — first-pass depth estimate (use current depth_arr at
                #   the rough position for a coarse depth needed to correct parallax).
                dh, dw = depth_arr.shape
                # Intrinsics-only mapping (no parallax yet):
                ndcx = (cx - color_intrin.ppx) / color_intrin.fx
                ndcy = (cy - color_intrin.ppy) / color_intrin.fy
                dx0  = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
                dy0  = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
                dx0  = max(0, min(dw - 1, dx0))
                dy0  = max(0, min(dh - 1, dy0))
                raw0 = depth_arr[dy0, dx0]
                depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0  # fallback 1m

                # Step 2 — parallax correction.
                # The extrinsics give translation from color frame to depth frame.
                # At depth Z, a colour-ray direction (ndcx, ndcy) originates from
                # the colour optical centre.  The depth sensor's optical centre is
                # offset by (tx, ty, tz) from the colour sensor (from extrinsics).
                # The additional pixel shift in the depth image is:
                #   Δx = tx / Z * fx_depth   (and similarly for y, tz negligible)
                tx, ty = color_to_depth_extr.translation[0], color_to_depth_extr.translation[1]
                dx_parallax = tx / depth_coarse * depth_intrin.fx
                dy_parallax = ty / depth_coarse * depth_intrin.fy

                dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx + dx_parallax + 0.5)
                dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy + dy_parallax + 0.5)
                dx = max(0, min(dw - 1, dx))
                dy = max(0, min(dh - 1, dy))

                # Step 3 — sample depth patch at the corrected depth pixel.
                x0d = max(0, dx - DEPTH_SAMPLE_RADIUS)
                x1d = min(dw, dx + DEPTH_SAMPLE_RADIUS + 1)
                y0d = max(0, dy - DEPTH_SAMPLE_RADIUS)
                y1d = min(dh, dy + DEPTH_SAMPLE_RADIUS + 1)
                patch   = depth_arr[y0d:y1d, x0d:x1d].astype(np.float32) * depth_scale
                valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
                depth_surface = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

                # The depth reading is to the front surface of the ball.
                # Shift by BALL_RADIUS along the optical axis to reach the ball center.
                depth_m = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0

                if depth_m > 0:
                    p_opt = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [cx, cy], depth_m)
                    p_cam_arr = optical_to_body(p_opt)

                    if center_ema is None:
                        gate_dist = 0.0
                        center_ema = p_cam_arr.copy()
                    else:
                        gate_dist = np.linalg.norm(p_cam_arr - center_ema)
                        if gate_dist < EMA_GATE:
                            center_ema = (EMA_ALPHA * p_cam_arr
                                          + (1 - EMA_ALPHA) * center_ema)
                        else:
                            # Jump larger than gate — reset EMA to avoid permanent freeze
                            print(f"\n[WARN] EMA gate {gate_dist:.2f}m > {EMA_GATE}m, resetting",
                                  flush=True)
                            center_ema = p_cam_arr.copy()

                    p_base = transform_point_camera_to_base(
                        center_ema,
                        joint.q_wy, joint.q_wr, joint.q_wp, joint.q_head,
                    )
                    x, y, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
                    detected = best_box is not None
                    dds.publish(x, y, z, valid=detected)
                    published_valid = True

                    status = "BALL " if detected else "COAST"
                    print(f"\r[{status}] pelvis=({x:+.3f}, {y:+.3f}, {z:+.3f})  "
                          f"surf={depth_surface:.2f}m  ctr={depth_m:.2f}m  "
                          f"gate={gate_dist:.2f}m  YOLO={yolo_fps.fps:4.1f}fps",
                          end="", flush=True)

            if not published_valid:
                dds.publish(0.0, 0.0, 0.0, valid=False)
                print(f"\r[     ] no ball  YOLO={yolo_fps.fps:4.1f}fps" + " " * 30,
                      end="", flush=True)

            # ── Annotate frame and push to MJPEG server ───────────────────
            if disp_queue is not None:
                vis = color.copy()
                if last_bbox is not None and miss_count <= COAST_FRAMES:
                    x1, y1, x2, y2 = last_bbox
                    cx2, cy2 = (x1 + x2) // 2, (y1 + y2) // 2
                    color_box = (0, 255, 0) if best_box is not None else (0, 165, 255)
                    _cv2.rectangle(vis, (x1, y1), (x2, y2), color_box, 2)
                    _cv2.circle(vis, (cx2, cy2), 4, color_box, -1)
                    label = (f"ball {best_conf:.2f}" if best_box is not None
                             else f"coast {miss_count}/{COAST_FRAMES}")
                    _cv2.putText(vis, label, (x1, y1 - 8),
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                    if published_valid:
                        info = f"pelvis ({x:+.2f}, {y:+.2f}, {z:+.2f})m"
                        _cv2.putText(vis, info, (10, vis.shape[0] - 10),
                                    _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                fps_txt = f"YOLO {yolo_fps.fps:.1f} fps"
                _cv2.putText(vis, fps_txt, (10, 24),
                            _cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                _, jpg_buf = _cv2.imencode('.jpg', vis, [_cv2.IMWRITE_JPEG_QUALITY, 60])
                with _mjpeg_lock:
                    _mjpeg_frame[0] = jpg_buf.tobytes()
            yolo_fps.tick()
            _frame_n += 1

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()

    # ── Main thread: camera capture ───────────────────────────────────────
    print("[INFO] Camera running. Press Ctrl+C to stop.")
    try:
        while True:
            # pipeline.wait_for_frames() is a C extension that releases the
            # GIL while blocking — camera capture does NOT starve YOLO.
            frames = pipeline.wait_for_frames()
            with buf_lock:
                buf_frames = frames   # just swap reference, no copies
            buf_updated.set()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        yolo_thread.join(timeout=2)
        pipeline.stop()
        if args.show:
            _httpd.shutdown()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
