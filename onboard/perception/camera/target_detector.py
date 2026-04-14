"""Chest-camera target detector service for G1 onboard computer.

Subscribes to:
  /lowstate  (Unitree G1 joint states via ROS2, for waist angles)

Publishes via DDS:
  "rt/target_state"  (target position in pelvis body frame)

The detector is intentionally separate from ball_detector.py so camera-target
logic can evolve independently of the ball pipeline.
"""

import sys
import warnings
import numpy as _np_compat

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(_np_compat, "bool"):
        _np_compat.bool = bool
    if not hasattr(_np_compat, "int"):
        _np_compat.int = int
    if not hasattr(_np_compat, "float"):
        _np_compat.float = float
    if not hasattr(_np_compat, "object"):
        _np_compat.object = object
del _np_compat

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

import argparse
import queue
import socket
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

from onboard.perception.camera.camera_to_base import (
    get_default_chest_extrinsics,
    optical_to_body,
    transform_point_chest_camera_to_base_with_extrinsics,
)
from common.target_state_dds import (
    INVALID_CLASS_ID,
    SOURCE_CHEST_CAMERA,
    TargetStatePublisher,
)


DEFAULT_TARGET_CLASSES = ("bottle",)
DEFAULT_CONF_THRESHOLD = 0.25
DEPTH_SAMPLE_RADIUS = 5
DEPTH_MIN = 0.1
DEPTH_MAX = 10.0
EMA_ALPHA = 0.5
EMA_GATE = 0.5
COAST_FRAMES = 8
DEPTH_BIAS_Y = 0.5


class _FPS:
    def __init__(self, window=30):
        self._timestamps = []
        self._window = window

    def tick(self):
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    @property
    def fps(self):
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / (self._timestamps[-1] - self._timestamps[0])


class _JointListener(Node):
    """Background ROS2 node that reads waist joint angles from /lowstate."""

    def __init__(self):
        super().__init__("camera_target_detector_joint_listener")
        self.q_wy = 0.0
        self.q_wr = 0.0
        self.q_wp = 0.0
        self.create_subscription(LowState, "/lowstate", self._cb, qos_profile_sensor_data)
        self.get_logger().info("camera_target_detector: subscribed to /lowstate")

    def _cb(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]


def _parse_target_classes(raw_values):
    if not raw_values:
        return list(DEFAULT_TARGET_CLASSES)
    parsed = []
    for raw in raw_values:
        for item in raw.split(","):
            name = item.strip().lower()
            if name:
                parsed.append(name)
    return parsed or list(DEFAULT_TARGET_CLASSES)


def _resolve_target_ids(model_names, target_names):
    if isinstance(model_names, dict):
        pairs = model_names.items()
    else:
        pairs = enumerate(model_names)
    by_name = {str(name).lower(): int(class_id) for class_id, name in pairs}
    ids = {}
    missing = []
    for name in target_names:
        if name in by_name:
            ids[name] = by_name[name]
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Target classes not found in model labels: {missing}. "
            f"Available labels include: {sorted(list(by_name.keys()))[:20]}"
        )
    return ids


def _sample_depth(depth_arr, cx, cy, color_intrin, depth_intrin, c2d_extr, depth_scale):
    """Return depth in metres at a color pixel with intrinsics + parallax correction."""
    dh, dw = depth_arr.shape
    ndcx = (cx - color_intrin.ppx) / color_intrin.fx
    ndcy = (cy - color_intrin.ppy) / color_intrin.fy

    dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
    dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
    dx0 = max(0, min(dw - 1, dx0))
    dy0 = max(0, min(dh - 1, dy0))
    raw0 = depth_arr[dy0, dx0]
    depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0

    tx = c2d_extr.translation[0]
    ty = c2d_extr.translation[1]
    dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx + tx / depth_coarse * depth_intrin.fx + 0.5)
    dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy + ty / depth_coarse * depth_intrin.fy + 0.5)
    dx = max(0, min(dw - 1, dx))
    dy = max(0, min(dh - 1, dy))

    radius = DEPTH_SAMPLE_RADIUS
    patch = (
        depth_arr[max(0, dy - radius):min(dh, dy + radius + 1),
                  max(0, dx - radius):min(dw, dx + radius + 1)].astype(np.float32)
        * depth_scale
    )
    valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def _start_camera_pipeline(args):
    ctx = rs.context()
    devs = ctx.query_devices()
    all_sns = [d.get_info(rs.camera_info.serial_number) for d in devs]
    all_names = [d.get_info(rs.camera_info.name) for d in devs]

    print("[INFO] Connected RealSense devices:")
    for idx, (sn, nm) in enumerate(zip(all_sns, all_names)):
        print(f"  [{idx}] serial={sn}  {nm}")

    if args.list_cameras:
        return None, None
    if len(all_sns) == 0:
        raise RuntimeError("No RealSense device found.")

    serial = args.camera_serial or all_sns[0]
    print(f"[INFO] CHEST target camera → serial {serial}")

    pipeline = rs.pipeline()
    fps_tries = [(60, 60), (30, 30), (15, 15)]
    last_err = None
    for color_fps, depth_fps in fps_tries:
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, color_fps)
        rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, depth_fps)
        for attempt in range(2):
            try:
                print(
                    f"[INFO] Starting RealSense serial={serial} "
                    f"(color {color_fps} Hz, depth {depth_fps} Hz, attempt {attempt + 1})..."
                )
                profile = pipeline.start(rs_cfg)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "resolve" in msg or "couldn't" in msg:
                    last_err = exc
                    print(f"[WARN] Stream profile not supported: {exc}")
                    break
                raise
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                print(
                    f"[INFO] RealSense pipeline OK for serial={serial} "
                    f"(color {color_fps} Hz, depth {depth_fps} Hz)"
                )
                return pipeline, profile
            except RuntimeError:
                print("[WARN] Frame timeout. Performing hardware reset...")
                pipeline.stop()
                reset_ctx = rs.context()
                matched = [d for d in reset_ctx.query_devices()
                           if d.get_info(rs.camera_info.serial_number) == serial]
                if not matched:
                    raise RuntimeError(f"RealSense serial {serial} disappeared during reset.")
                matched[0].hardware_reset()
                time.sleep(3)

    msg = f"RealSense failed to start for serial={serial}. Tried FPS pairs: {fps_tries}."
    if last_err is not None:
        msg += f" Last resolve error: {last_err!r}"
    raise RuntimeError(msg)


def _start_mjpeg_server():
    import http.server
    import socketserver

    mjpeg_frame = [None]
    mjpeg_lock = threading.Lock()

    class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/stream")
                self.end_headers()
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last_sent = None
                while True:
                    with mjpeg_lock:
                        jpg = mjpeg_frame[0]
                    if jpg is None or jpg is last_sent:
                        time.sleep(0.02)
                        continue
                    last_sent = jpg
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
            except Exception:
                pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", 8080), _MJPEGHandler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, mjpeg_frame, mjpeg_lock


def _get_stream_url(port=8080, path="/stream"):
    host = "127.0.0.1"
    sock = None
    try:
        # Ask the OS which outbound interface would be used for a LAN address.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("192.168.123.1", 1))
        host = sock.getsockname()[0]
    except OSError:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError:
            pass
    finally:
        if sock is not None:
            sock.close()
    return f"http://{host}:{port}{path}"


def main():
    parser = argparse.ArgumentParser(
        description="Chest D435 + YOLO target detector -> rt/target_state"
    )
    parser.add_argument("--model", default="onboard/perception/camera/models/yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-serial", default=None,
                        help="RealSense serial for the chest camera.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Print connected RealSense serials and exit.")
    parser.add_argument("--show", action="store_true",
                        help="Stream annotated video via MJPEG on port 8080.")
    parser.add_argument("--dds-topic", default="rt/target_state",
                        help="DDS topic name to publish to.")
    parser.add_argument("--target-class", action="append", dest="target_classes",
                        help="Target class label(s). Can be repeated or comma-separated.")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--depth-bias-y", type=float, default=DEPTH_BIAS_Y,
                        help="Sample depth at bbox y = y1 + bias * (y2 - y1).")
    parser.add_argument("--coast-frames", type=int, default=COAST_FRAMES)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--ema-gate", type=float, default=EMA_GATE)
    parser.add_argument("--chest-xyz", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Override chest camera translation in waist_pitch frame (metres).")
    parser.add_argument("--chest-rpy", type=float, nargs=3, default=None,
                        metavar=("ROLL", "PITCH", "YAW"),
                        help="Override chest camera rotation in radians.")
    args = parser.parse_args()

    target_names = _parse_target_classes(args.target_classes)
    print(f"[INFO] Target classes requested: {target_names}")

    pipeline, profile = _start_camera_pipeline(args)
    if args.list_cameras:
        return

    rclpy.init()
    joint = _JointListener()

    def _spin_loop():
        while True:
            rclpy.spin_once(joint, timeout_sec=0.0)
            time.sleep(0.02)

    threading.Thread(target=_spin_loop, daemon=True).start()
    print("[INFO] ROS2 joint listener started (/lowstate)")

    dds = TargetStatePublisher(domain_id=0, topic_name=args.dds_topic)
    print(f"[INFO] DDS publisher ready on '{args.dds_topic}'")

    default_xyz, default_rpy = get_default_chest_extrinsics()
    chest_xyz = tuple(args.chest_xyz) if args.chest_xyz is not None else default_xyz
    chest_rpy = tuple(args.chest_rpy) if args.chest_rpy is not None else default_rpy
    print(f"[INFO] Chest extrinsics xyz={tuple(round(v, 5) for v in chest_xyz)}")
    print(f"[INFO] Chest extrinsics rpy={tuple(round(v, 5) for v in chest_rpy)}")

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = args.model
    if model_path.endswith(".pt"):
        engine_path = model_path.replace(".pt", ".engine")
        if __import__("os").path.exists(engine_path):
            model_path = engine_path
            print(f"[INFO] TensorRT engine found, using: {engine_path}")
        else:
            print(f"[INFO] No .engine found at {engine_path}, using .pt")

    print(f"[INFO] Loading YOLO model: {model_path} (device={device})")
    model = YOLO(model_path)
    is_trt = str(model_path).endswith(".engine")
    infer_kw = {} if is_trt else dict(device=device, half=True)

    target_ids_by_name = _resolve_target_ids(model.names, target_names)
    target_ids = set(target_ids_by_name.values())
    id_to_name = {class_id: name for name, class_id in target_ids_by_name.items()}
    print(f"[INFO] Resolved target classes: {target_ids_by_name}")

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] YOLO warming up...")
    for idx in range(3):
        t0 = time.perf_counter()
        model(dummy, verbose=False, **infer_kw)
        print(f"[INFO]   warmup[{idx}]: {(time.perf_counter() - t0) * 1000:.1f}ms")

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    depth_intrin = depth_profile.get_intrinsics()
    color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
    intrinsics = color_intrin
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Color intrinsics fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f}")
    print(f"[INFO] Depth intrinsics fx={depth_intrin.fx:.1f} fy={depth_intrin.fy:.1f} scale={depth_scale:.4f}")

    buf_lock = threading.Lock()
    buf_frames = None
    buf_updated = threading.Event()
    stop_flag = threading.Event()
    disp_queue = queue.Queue(maxsize=1) if args.show else None

    if args.show:
        httpd, mjpeg_frame, mjpeg_lock = _start_mjpeg_server()
        print(f"[INFO] MJPEG stream started -> open {_get_stream_url()}")
    else:
        httpd = None
        mjpeg_frame = None
        mjpeg_lock = None

    def yolo_worker():
        center_ema = None
        last_bbox = None
        last_cls_id = INVALID_CLASS_ID
        last_conf = 0.0
        miss_count = 0
        yolo_fps = _FPS()

        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                frames = buf_frames
            if frames is None:
                continue

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()
            orig_h, orig_w = color.shape[:2]
            color_small = cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            results = model(color_small, conf=args.conf_threshold, verbose=False, **infer_kw)
            best_box = None
            best_conf = 0.0
            best_cls_id = INVALID_CLASS_ID

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in target_ids:
                        continue
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_box = box
                        best_conf = conf
                        best_cls_id = cls_id

            if best_box is not None:
                miss_count = 0
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                last_bbox = (
                    int(x1s * sx), int(y1s * sy),
                    int(x2s * sx), int(y2s * sy),
                )
                last_cls_id = best_cls_id
                last_conf = best_conf
            else:
                miss_count += 1

            published = False
            published_valid = False
            p_cam_arr = None
            depth_m = 0.0
            depth_surface = 0.0
            target_name = id_to_name.get(last_cls_id, "target")
            pelvis_xyz = (0.0, 0.0, 0.0)

            if last_bbox is not None and miss_count <= args.coast_frames:
                x1, y1, x2, y2 = last_bbox
                cx = (x1 + x2) // 2
                cy = int(y1 + args.depth_bias_y * (y2 - y1))
                cy = max(0, min(orig_h - 1, cy))

                depth_surface = _sample_depth(
                    depth_arr,
                    cx,
                    cy,
                    color_intrin,
                    depth_intrin,
                    color_to_depth_extr,
                    depth_scale,
                )
                depth_m = depth_surface

                if depth_m > 0:
                    p_opt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
                    p_cam_arr = optical_to_body(p_opt)

                    if center_ema is None:
                        gate_dist = 0.0
                        center_ema = p_cam_arr.copy()
                    else:
                        gate_dist = np.linalg.norm(p_cam_arr - center_ema)
                        if gate_dist < args.ema_gate:
                            center_ema = args.ema_alpha * p_cam_arr + (1.0 - args.ema_alpha) * center_ema
                        else:
                            center_ema = p_cam_arr.copy()

                    p_base = transform_point_chest_camera_to_base_with_extrinsics(
                        center_ema,
                        joint.q_wy,
                        joint.q_wr,
                        joint.q_wp,
                        chest_xyz=chest_xyz,
                        chest_rpy=chest_rpy,
                    )
                    pelvis_xyz = (float(p_base[0]), float(p_base[1]), float(p_base[2]))
                    published_valid = best_box is not None
                    dds.publish(
                        *pelvis_xyz,
                        valid=published_valid,
                        class_id=last_cls_id,
                        confidence=last_conf,
                        source=SOURCE_CHEST_CAMERA,
                    )
                    published = True
                    status = target_name.upper() if published_valid else "COAST"
                    print(
                        f"\r[{status}] pelvis=({pelvis_xyz[0]:+.3f}, {pelvis_xyz[1]:+.3f}, {pelvis_xyz[2]:+.3f}) "
                        f"depth={depth_surface:.2f}m conf={last_conf:.2f} "
                        f"cls={target_name} yolo={yolo_fps.fps:4.1f}fps",
                        end="",
                        flush=True,
                    )

            if not published:
                dds.publish(
                    0.0,
                    0.0,
                    0.0,
                    valid=False,
                    class_id=INVALID_CLASS_ID,
                    confidence=0.0,
                    source=SOURCE_CHEST_CAMERA,
                )
                print(
                    f"\r[     ] no target ({'/'.join(target_names)}) yolo={yolo_fps.fps:4.1f}fps" + " " * 24,
                    end="",
                    flush=True,
                )

            if disp_queue is not None:
                vis = color.copy()
                if last_bbox is not None and miss_count <= args.coast_frames:
                    x1, y1, x2, y2 = last_bbox
                    color_box = (0, 255, 0) if best_box is not None else (0, 165, 255)
                    cx = (x1 + x2) // 2
                    cy = int(y1 + args.depth_bias_y * (y2 - y1))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color_box, 2)
                    cv2.circle(vis, (cx, cy), 4, color_box, -1)
                    label = (
                        f"{target_name} {last_conf:.2f}" if best_box is not None
                        else f"coast {miss_count}/{args.coast_frames}"
                    )
                    cv2.putText(vis, label, (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                    if published:
                        info = (
                            f"pelvis ({pelvis_xyz[0]:+.2f}, {pelvis_xyz[1]:+.2f}, {pelvis_xyz[2]:+.2f})m "
                            f"depth {depth_surface:.2f}m"
                        )
                        cv2.putText(vis, info, (10, vis.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                fps_txt = f"YOLO {yolo_fps.fps:.1f} fps"
                cls_txt = f"targets: {', '.join(target_names)}"
                extr_txt = f"xyz={tuple(round(v, 3) for v in chest_xyz)}"
                cv2.putText(vis, fps_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(vis, cls_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(vis, extr_txt, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                _, jpg_buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
                with mjpeg_lock:
                    mjpeg_frame[0] = jpg_buf.tobytes()

            yolo_fps.tick()

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()

    print("[INFO] Camera running. Press Ctrl+C to stop.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            with buf_lock:
                nonlocal_buf = frames
                buf_frames = nonlocal_buf
            buf_updated.set()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        yolo_thread.join(timeout=2)
        pipeline.stop()
        if httpd is not None:
            httpd.shutdown()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
