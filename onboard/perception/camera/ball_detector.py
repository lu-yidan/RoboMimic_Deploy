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
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="YOLO input size in pixels (smaller = faster)")
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
    print(f"[INFO] Loading YOLO model: {args.model}  (device={device})")
    model = YOLO(args.model)

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] YOLO warming up (first run triggers CUDA JIT, takes ~10s)...")
    for i in range(5):
        t_w = time.perf_counter()
        model(dummy, verbose=False, device=device, half=True)
        print(f"[INFO]   warmup[{i}]: {(time.perf_counter()-t_w)*1000:.1f}ms")
    t0 = time.perf_counter()
    for _ in range(10):
        model(dummy, verbose=False, device=device, half=True)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"[INFO] YOLO inference: {ms:.1f} ms/frame  (≈ {1000/ms:.0f} FPS upper bound)")

    # ── RealSense ─────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
    rs_cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  60)

    def _start_pipeline():
        for attempt in range(2):
            print(f"[INFO] Starting RealSense pipeline (attempt {attempt + 1})...")
            profile = pipeline.start(rs_cfg)
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
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
        raise RuntimeError("RealSense failed to start after hardware reset.")

    profile    = _start_pipeline()
    align      = rs.align(rs.stream.color)
    intrinsics = (
        profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics()
    )
    print(f"[INFO] Intrinsics — fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"ppx={intrinsics.ppx:.1f}  ppy={intrinsics.ppy:.1f}")

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
            # align.process() is CPU-intensive (~100ms) and holds the GIL.
            # Do it here in the YOLO thread so it does NOT compete with the
            # camera main thread; the main thread only calls
            # pipeline.wait_for_frames() (GIL-free) + a quick lock swap.
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data()).copy()
            depth = np.asanyarray(df.get_data()).copy()

            orig_h, orig_w = color.shape[:2]
            color_small = _cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            # YOLO detection
            results = model.track(color_small, conf=CONF_THRESHOLD,
                                  persist=True, verbose=False,
                                  device=device, half=True)
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
                h, w   = depth.shape

                x0d = max(0, cx - DEPTH_SAMPLE_RADIUS)
                x1d = min(w - 1, cx + DEPTH_SAMPLE_RADIUS)
                y0d = max(0, cy - DEPTH_SAMPLE_RADIUS)
                y1d = min(h - 1, cy + DEPTH_SAMPLE_RADIUS)
                patch   = depth[y0d:y1d+1, x0d:x1d+1].astype(np.float32) * 0.001
                valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
                depth_m = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

                if depth_m > 0:
                    p_opt     = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [cx, cy], depth_m)
                    p_cam_arr = optical_to_body(p_opt)

                    if center_ema is None:
                        center_ema = p_cam_arr.copy()
                    elif np.linalg.norm(p_cam_arr - center_ema) < EMA_GATE:
                        center_ema = (EMA_ALPHA * p_cam_arr
                                      + (1 - EMA_ALPHA) * center_ema)

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
                          f"d={depth_m:.2f}m  YOLO={yolo_fps.fps:4.1f}fps",
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
