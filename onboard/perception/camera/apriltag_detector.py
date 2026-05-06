"""Chest-camera AprilTag target detector service for G1 onboard computer.

Subscribes to:
  /lowstate  (Unitree G1 joint states via ROS2, for waist angles)

Publishes via DDS:
  "rt/target_state"  (target position in pelvis body frame)

Usage:
    bash onboard/perception/camera/run_apriltag_target.sh
    bash onboard/perception/camera/run_apriltag_target.sh --show
    bash onboard/perception/camera/run_apriltag_target.sh --tag-id 0 --tag-size 0.12
    bash onboard/perception/camera/run_apriltag_target.sh --tag-size 0.12 --show
    bash onboard/perception/camera/run_apriltag_target.sh \
        --tag-id 5 --tag-id 8 --tag-size 0.10 \
        --tag-offset 5 0.20 0.00 0.00 \
        --tag-offset 8 -0.20 0.00 0.00
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowState

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from common.target_state_dds import (  # noqa: E402
    INVALID_CLASS_ID,
    SOURCE_CHEST_CAMERA,
    TargetStatePublisher,
)
from common.ball_state_dds import BallStatePublisher, SOURCE_CAM, SOURCE_NONE  # noqa: E402
from onboard.perception.camera.camera_to_base import (  # noqa: E402
    get_default_chest_extrinsics,
    optical_to_body,
    transform_point_chest_camera_to_base_with_extrinsics,
)


COAST_FRAMES = 8
EMA_ALPHA = 0.5
EMA_GATE = 0.4

# ── Ball detection constants (used when --ball is active) ─────────────────
_BALL_DEPTH_SAMPLE_R = 5
_BALL_DEPTH_MIN      = 0.1    # m — minimum valid depth
_BALL_DEPTH_MAX      = 10.0   # m — maximum valid depth
_BALL_RADIUS         = 0.115  # m — physical ball radius; depth sensor sees front surface
_BALL_CONF_THRESH    = 0.25   # YOLO confidence threshold
_BALL_SPORTS_ID      = 32     # COCO class index for sports ball
_BALL_COAST          = 10     # frames to hold last position after YOLO miss
_BALL_EMA_ALPHA      = 0.5
_BALL_EMA_GATE       = 0.6    # m — EMA reset threshold

# HSV ball detection constants (used when --ball-hsv is active)
# Tuned for a soccer ball with blue/purple hexagonal patches on white background.
# Adjust via --ball-hsv-h-low / --ball-hsv-h-high / --ball-hsv-s-min / --ball-hsv-v-min.
_HSV_H_LOW_DEFAULT  = 90
_HSV_H_HIGH_DEFAULT = 150
_HSV_S_MIN_DEFAULT  = 40
_HSV_V_MIN_DEFAULT  = 50
_HSV_DILATION       = 17    # px — merges scattered color patches into one blob; half=8
_HSV_MIN_R          = 4     # px — smallest allowed ball radius (~8m max range)
_HSV_MAX_R          = 280   # px — largest allowed ball radius
_HSV_FILL_MIN       = 0.05  # fraction of enclosing circle covered by original mask pixels


def _detect_ball_hsv(
    color_bgr: np.ndarray,
    hsv_low: np.ndarray,
    hsv_high: np.ndarray,
    min_r: int = _HSV_MIN_R,
    max_r: int = _HSV_MAX_R,
    fill_min: float = _HSV_FILL_MIN,
) -> tuple:
    """Detect soccer ball by HSV color patch matching.

    Works for balls with scattered color patches (e.g. blue hexagons on white):
    dilates the mask heavily to merge patches into one blob, finds the largest
    circular cluster, then corrects the radius for the dilation offset.

    Returns (cx_px, cy_px, r_est_px) or (None, None, 0.0).
    r_est_px is the corrected ball radius in the color image (≈ actual ball edge).
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # Merge separated color patches with a large dilation.
    k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_HSV_DILATION, _HSV_DILATION))
    merged = cv2.dilate(mask, k_merge)
    # Remove stray noise that survived dilation.
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k_clean)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    dilation_half = _HSV_DILATION // 2
    h, w = mask.shape
    best = None

    for cnt in contours:
        (blob_cx, blob_cy), blob_r = cv2.minEnclosingCircle(cnt)
        blob_cx, blob_cy, blob_r = int(blob_cx), int(blob_cy), float(blob_r)

        # Estimated true ball radius (subtract the dilation that inflated the blob).
        r_est = blob_r - dilation_half
        if not (min_r <= r_est <= max_r):
            continue

        # Validate with original (undilated) mask: check what fraction of the
        # enclosing circle contains actual color pixels.
        circle_area = np.pi * blob_r * blob_r
        y0 = max(0, blob_cy - int(blob_r))
        y1 = min(h, blob_cy + int(blob_r) + 1)
        x0 = max(0, blob_cx - int(blob_r))
        x1 = min(w, blob_cx + int(blob_r) + 1)
        roi = mask[y0:y1, x0:x1]
        orig_pixels = int(np.count_nonzero(roi))
        fill = orig_pixels / circle_area
        if fill < fill_min:
            continue

        score = orig_pixels * fill
        if best is None or score > best[0]:
            best = (score, blob_cx, blob_cy, r_est)

    if best is None:
        return None, None, 0.0
    _, cx, cy, r_est = best
    return cx, cy, r_est


def _sample_ball_depth(
    cx: int, cy: int,
    depth_arr: np.ndarray,
    depth_scale: float,
    color_intrin,
    depth_intrin,
    color_to_depth_extr,
    sample_r: int = _BALL_DEPTH_SAMPLE_R,
) -> float:
    """3-step color→depth pixel mapping + patch median → ball-center depth in metres.

    Returns 0.0 if depth is unavailable or out of range.
    """
    dh, dw = depth_arr.shape
    ndcx = (cx - color_intrin.ppx) / color_intrin.fx
    ndcy = (cy - color_intrin.ppy) / color_intrin.fy

    # Step 1: FOV-only mapping (no parallax correction yet).
    dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
    dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
    dx0 = max(0, min(dw - 1, dx0))
    dy0 = max(0, min(dh - 1, dy0))
    raw0 = depth_arr[dy0, dx0]
    depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0

    # Step 2: Parallax correction using color→depth baseline.
    t_cx = color_to_depth_extr.translation[0]
    t_cy = color_to_depth_extr.translation[1]
    dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx
             + t_cx / depth_coarse * depth_intrin.fx + 0.5)
    dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy
             + t_cy / depth_coarse * depth_intrin.fy + 0.5)
    dx = max(0, min(dw - 1, dx))
    dy = max(0, min(dh - 1, dy))

    # Step 3: Median of a small depth patch.
    patch = (depth_arr[max(0, dy - sample_r):dy + sample_r + 1,
                       max(0, dx - sample_r):dx + sample_r + 1]
             .astype(np.float32) * depth_scale)
    valid_d = patch[(patch > _BALL_DEPTH_MIN) & (patch < _BALL_DEPTH_MAX)]
    if len(valid_d) == 0:
        return 0.0
    # depth to ball center = surface depth + ball radius
    return float(np.median(valid_d)) + _BALL_RADIUS


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
        super().__init__("camera_apriltag_detector_joint_listener")
        self.q_wy = 0.0
        self.q_wr = 0.0
        self.q_wp = 0.0
        self.create_subscription(LowState, "/lowstate", self._cb, qos_profile_sensor_data)
        self.get_logger().info("camera_apriltag_detector: subscribed to /lowstate")

    def _cb(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]


def _parse_tag_ids(raw_values):
    if not raw_values:
        return [0]
    parsed = []
    for raw in raw_values:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    if not parsed:
        raise ValueError("At least one valid --tag-id is required.")
    return parsed


def _parse_tag_offsets(raw_values):
    offsets = {}
    if not raw_values:
        return offsets
    for raw in raw_values:
        if len(raw) != 4:
            raise ValueError("--tag-offset expects: TAG_ID DX DY DZ")
        tag_id = int(raw[0])
        if tag_id in offsets:
            raise ValueError(f"Duplicate --tag-offset for tag id {tag_id}.")
        offsets[tag_id] = np.array(
            [float(raw[1]), float(raw[2]), float(raw[3])],
            dtype=np.float32,
        )
    return offsets


def _get_apriltag_dictionary(tag_family):
    family_key = str(tag_family).strip().lower()
    families = {
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    if family_key not in families:
        raise ValueError(
            f"Unsupported AprilTag family '{tag_family}'. "
            f"Choose one of: {sorted(families.keys())}"
        )
    return cv2.aruco.getPredefinedDictionary(families[family_key]), family_key


def _start_camera_pipeline(args, with_depth: bool = False):
    ctx = rs.context()
    devs = ctx.query_devices()
    all_sns = [d.get_info(rs.camera_info.serial_number) for d in devs]
    all_names = [d.get_info(rs.camera_info.name) for d in devs]

    print("[INFO] Connected RealSense devices:")
    for idx, (d, sn, nm) in enumerate(zip(devs, all_sns, all_names)):
        try:
            usb = d.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb = "?"
        usb_warn = "  ⚠ USB2 – frame drops likely!" if usb.startswith("2") else ""
        print(f"  [{idx}] serial={sn}  {nm}  USB {usb}{usb_warn}")

    if args.list_cameras:
        return None, None
    if not all_sns:
        raise RuntimeError("No RealSense device found.")

    serial = args.camera_serial or all_sns[0]
    # Print USB speed for the selected device so frame-drop issues are easy to spot.
    sel_dev = next((d for d, s in zip(devs, all_sns) if s == serial), None)
    if sel_dev is not None:
        try:
            usb = sel_dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb = "?"
        usb_ok = usb.startswith("3")
        tag = "OK" if usb_ok else "WARN – frame drops likely if USB2!"
        print(f"[INFO] CHEST AprilTag camera -> serial {serial}  USB {usb}  [{tag}]")

    pipeline = rs.pipeline()
    fps_tries = [30, 15, 10, 5]
    last_err = None
    for color_fps in fps_tries:
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, color_fps)
        if with_depth:
            # Depth at 848×480 — D455 supports 1280×720 color + 848×480 depth @ 30fps.
            # 424×240 is NOT a valid paired mode at 30fps and causes fallback to 15fps.
            rs_cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, color_fps)
        for attempt in range(2):
            try:
                print(
                    f"[INFO] Starting RealSense serial={serial} "
                    f"(color {color_fps} Hz, attempt {attempt + 1})..."
                )
                profile = pipeline.start(rs_cfg)
            except RuntimeError as exc:
                last_err = exc
                print(f"[WARN] Stream profile not supported: {exc}")
                break
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                print(
                    f"[INFO] RealSense pipeline OK for serial={serial} "
                    f"(color {color_fps} Hz)"
                )
                return pipeline, profile
            except RuntimeError:
                print("[WARN] Frame timeout. Performing hardware reset...")
                pipeline.stop()
                reset_ctx = rs.context()
                matched = [
                    d
                    for d in reset_ctx.query_devices()
                    if d.get_info(rs.camera_info.serial_number) == serial
                ]
                if not matched:
                    raise RuntimeError(f"RealSense serial {serial} disappeared during reset.")
                matched[0].hardware_reset()
                time.sleep(3)

    msg = f"RealSense failed to start for serial={serial}. Tried FPS: {fps_tries}."
    if last_err is not None:
        msg += f" Last error: {last_err!r}"
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


def _build_camera_matrix(intrinsics):
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _build_dist_coeffs(intrinsics):
    coeffs = list(intrinsics.coeffs)
    if not coeffs:
        coeffs = [0.0] * 5
    if len(coeffs) < 5:
        coeffs += [0.0] * (5 - len(coeffs))
    return np.array(coeffs[:5], dtype=np.float32).reshape(-1, 1)


def _estimate_tag_pose(corners, camera_matrix, dist_coeffs, tag_size_m):
    # Define the printed tag plane in its own local frame. The tag center is the
    # published target point in this first version of the pipeline.
    half = 0.5 * float(tag_size_m)
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    solvepnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=solvepnp_flag,
    )
    if not ok:
        return None, None
    # solvePnP returns the pose in the camera optical frame.
    return rvec, tvec.reshape(3)


def _select_detection(detections, target_tag_ids):
    if not detections:
        return None
    priority = {tag_id: idx for idx, tag_id in enumerate(target_tag_ids)}
    ranked = sorted(
        detections,
        key=lambda det: (
            priority.get(det["tag_id"], len(priority)),
            -det["area"],
            det["distance_m"],
        ),
    )
    return ranked[0]


def _detection_confidence(area_px, image_shape):
    img_h, img_w = image_shape
    area_norm = float(area_px) / max(1.0, float(img_h * img_w))
    return max(0.05, min(1.0, 5.0 * math.sqrt(max(0.0, area_norm))))


def _resolve_target_point_optical(detection, tag_offsets):
    if detection["tvec"] is None:
        return None, False

    target_optical = np.asarray(detection["tvec"], dtype=np.float32).reshape(3)
    offset_tag = tag_offsets.get(detection["tag_id"])
    if offset_tag is None:
        return target_optical, False

    rot_optical_from_tag, _ = cv2.Rodrigues(
        np.asarray(detection["rvec"], dtype=np.float32).reshape(3, 1)
    )
    return target_optical + rot_optical_from_tag @ offset_tag, True


def _fuse_target_points_optical(detections):
    if not detections:
        return None, 0

    weights = []
    points = []
    for det in detections:
        # Larger tags are usually more reliable, so use image area as the
        # dominant fusion weight while keeping a small floor for single-tag use.
        weights.append(max(1.0, float(det["area"])))
        points.append(np.asarray(det["target_tvec"], dtype=np.float32).reshape(3))

    weights_arr = np.asarray(weights, dtype=np.float32)
    points_arr = np.stack(points, axis=0)
    fused = (weights_arr[:, None] * points_arr).sum(axis=0) / weights_arr.sum()
    return fused, len(detections)


def _make_yolo_ball_thread(
    model_path, topic, imgsz, joint,
    color_intrin, depth_intrin, color_to_depth_extr, depth_scale,
    chest_xyz, chest_rpy,
    buf_frames, buf_lock, buf_event,
    overlay_state=None,   # optional dict shared with MJPEG renderer
):
    """Start a background YOLO ball-detection thread.

    The main loop pre-copies (color_small, depth_arr) into buf_frames[0] before
    signalling buf_event, so this thread never touches librealsense frame objects.
    That releases the librealsense frame buffer immediately, letting
    pipeline.wait_for_frames() return at full camera fps instead of waiting for
    this thread to finish inference.

    GIL notes:
    - buf_event.wait() releases the GIL while idle — no contention with AprilTag.
    - YOLO CUDA inference (~25ms) releases the GIL — AprilTag can run freely.
    - Only Python overhead (~2ms) and depth sampling (~1ms) hold the GIL per frame.
    """
    def _run():
        # TensorRT / ultralytics need np.bool / np.int aliases removed in Py ≥ 3.9.
        import numpy as _np_compat
        for _attr in ("bool", "int", "float", "complex", "object", "str"):
            if not hasattr(_np_compat, _attr):
                setattr(_np_compat, _attr, getattr(__builtins__, _attr, None))

        from ultralytics import YOLO

        # Prefer .engine (TRT) over .pt if available alongside the requested path.
        _mp = model_path
        if _mp.endswith(".pt"):
            _eng = _mp.replace(".pt", ".engine")
            if __import__("os").path.exists(_eng):
                _mp = _eng
        _is_trt = _mp.endswith(".engine")

        print(f"[BALL] Loading YOLO model: {_mp}")
        _model = YOLO(_mp)
        _dummy = __import__("numpy").zeros((imgsz, imgsz, 3), dtype=__import__("numpy").uint8)
        _kw = {} if _is_trt else dict(device="cuda:0", half=True)
        print("[BALL] YOLO warming up...")
        for _ in range(3):
            _model.track(_dummy, persist=True, verbose=False, **_kw)
        print("[BALL] YOLO ready.")

        ball_dds = BallStatePublisher(domain_id=0, topic_name=topic)
        print(f"[BALL] DDS publisher ready on '{topic}'")

        # State
        center_ema = None
        last_bbox  = None
        miss_count = 0
        ball_fps   = _FPS()

        t_cx = color_to_depth_extr.translation[0]
        t_cy = color_to_depth_extr.translation[1]
        # Scale factors are fixed for the lifetime of the thread.
        sx = color_intrin.width  / imgsz
        sy = color_intrin.height / imgsz

        while True:
            if not buf_event.wait(timeout=1.0):
                ball_dds.publish(0.0, 0.0, 0.0, valid=False, source=SOURCE_NONE)
                continue
            buf_event.clear()

            with buf_lock:
                buf_item = buf_frames[0]
            if buf_item is None:
                continue

            # Main loop pre-copies (color_small, depth_arr) so we never hold
            # a librealsense frame object — that would block pipeline.wait_for_frames().
            color_small, depth_arr = buf_item
            if depth_arr is None:
                continue

            results = _model.track(color_small, conf=_BALL_CONF_THRESH,
                                   persist=True, verbose=False, **_kw)
            best_box  = None
            best_conf = 0.0
            for _r in results:
                for _b in _r.boxes:
                    if int(_b.cls[0]) == _BALL_SPORTS_ID:
                        _c = float(_b.conf[0])
                        if _c > best_conf:
                            best_conf, best_box = _c, _b

            import numpy as np
            if best_box is not None:
                miss_count = 0
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                last_bbox = (int(x1s * sx), int(y1s * sy),
                             int(x2s * sx), int(y2s * sy))
            else:
                miss_count += 1

            published_valid = False
            if last_bbox is not None and miss_count <= _BALL_COAST:
                x1, y1, x2, y2 = last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Color pixel → depth pixel (3-step, see TROUBLESHOOTING.md §9)
                dh, dw = depth_arr.shape
                ndcx = (cx - color_intrin.ppx) / color_intrin.fx
                ndcy = (cy - color_intrin.ppy) / color_intrin.fy
                dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
                dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
                dx0 = max(0, min(dw - 1, dx0))
                dy0 = max(0, min(dh - 1, dy0))
                raw0 = depth_arr[dy0, dx0]
                depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0

                dx_par = t_cx / depth_coarse * depth_intrin.fx
                dy_par = t_cy / depth_coarse * depth_intrin.fy
                dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx + dx_par + 0.5)
                dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy + dy_par + 0.5)
                dx = max(0, min(dw - 1, dx))
                dy = max(0, min(dh - 1, dy))

                r = _BALL_DEPTH_SAMPLE_R
                patch = depth_arr[max(0, dy - r):dy + r + 1,
                                  max(0, dx - r):dx + r + 1].astype(np.float32) * depth_scale
                valid_d = patch[(patch > _BALL_DEPTH_MIN) & (patch < _BALL_DEPTH_MAX)]
                depth_surface = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0
                depth_m = depth_surface + _BALL_RADIUS if depth_surface > 0 else 0.0

                if depth_m > 0:
                    p_opt = rs.rs2_deproject_pixel_to_point(color_intrin, [cx, cy], depth_m)
                    p_cam = optical_to_body(p_opt)

                    if center_ema is None:
                        center_ema = p_cam.copy()
                    else:
                        gate = float(np.linalg.norm(p_cam - center_ema))
                        if gate < _BALL_EMA_GATE:
                            center_ema = (_BALL_EMA_ALPHA * p_cam
                                          + (1 - _BALL_EMA_ALPHA) * center_ema)
                        else:
                            center_ema = p_cam.copy()

                    p_base = transform_point_chest_camera_to_base_with_extrinsics(
                        center_ema,
                        joint.q_wy, joint.q_wr, joint.q_wp,
                        chest_xyz=chest_xyz, chest_rpy=chest_rpy,
                    )
                    bx, by, bz = float(p_base[0]), float(p_base[1]), float(p_base[2])
                    is_det = best_box is not None
                    ball_dds.publish(bx, by, bz, valid=is_det, source=SOURCE_CAM)
                    published_valid = True

                    if overlay_state is not None:
                        overlay_state["bbox"]   = last_bbox
                        overlay_state["pelvis"] = (bx, by, bz)
                        overlay_state["depth"]  = depth_m
                        overlay_state["valid"]  = is_det
                        overlay_state["miss"]   = miss_count

                    tag = "BALL " if is_det else "COAST"
                    print(
                        f"\r[{tag}] cam_ball pelvis=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
                        f"dist={depth_m:.2f}m ball={ball_fps.fps:4.1f}fps" + " " * 5,
                        end="", flush=True,
                    )

            if not published_valid:
                center_ema = None
                if overlay_state is not None:
                    overlay_state["bbox"] = None
                ball_dds.publish(0.0, 0.0, 0.0, valid=False, source=SOURCE_NONE)
                print(f"\r[     ] cam_ball: no ball  fps={ball_fps.fps:4.1f}" + " " * 20,
                      end="", flush=True)

            ball_fps.tick()

    t = threading.Thread(target=_run, daemon=True, name="yolo-ball")
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser(
        description="Chest D455 + AprilTag target detector -> rt/target_state"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--detect-scale", type=float, default=1.0,
        help="Scale factor applied to gray frame before ArUco detection "
             "(default 1.0 = full resolution). Set <1.0 if bad_alloc occurs "
             "on low-memory systems; corners are scaled back automatically.",
    )
    parser.add_argument(
        "--camera-serial",
        default=None,
        help="RealSense serial for the chest camera.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print connected RealSense serials and exit.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Stream annotated video via MJPEG on port 8080.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable video recording (recording is on by default).",
    )
    parser.add_argument(
        "--record-dir",
        default="recordings",
        help="Directory for recorded MP4 files, relative to repo root (default: recordings/).",
    )
    parser.add_argument(
        "--dds-topic",
        default="rt/target_state",
        help="DDS topic name to publish to.",
    )
    parser.add_argument(
        "--tag-id",
        action="append",
        dest="tag_ids",
        help="AprilTag id(s) to track. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--tag-family",
        default="tag36h11",
        help="AprilTag family, e.g. tag36h11.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.15,
        help="Printed tag edge length in metres, excluding any white paper margin.",
    )
    parser.add_argument(
        "--tag-offset",
        action="append",
        nargs=4,
        default=None,
        metavar=("TAG_ID", "DX", "DY", "DZ"),
        help=(
            "Optional shared target offset in the tag frame (metres). Repeat per tag as: "
            "--tag-offset TAG_ID DX DY DZ"
        ),
    )
    parser.add_argument("--coast-frames", type=int, default=COAST_FRAMES)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--ema-gate", type=float, default=EMA_GATE)

    # ── Ball detection (optional, --ball enables YOLO thread) ─────────────
    parser.add_argument(
        "--ball", action="store_true",
        help="Also run YOLO ball detection on the same chest camera. "
             "Publishes to --ball-topic (default rt/cam_ball_state).",
    )
    parser.add_argument(
        "--ball-model",
        default="onboard/perception/camera/models/yolo11m.pt",
        help="YOLO model path for ball detection (default: yolo11m.pt; .engine preferred).",
    )
    parser.add_argument(
        "--ball-topic", default="rt/cam_ball_state",
        help="DDS topic for camera ball detection output (default: rt/cam_ball_state).",
    )
    parser.add_argument(
        "--ball-imgsz", type=int, default=320,
        help="YOLO input resolution for ball detection (default 320).",
    )

    # ── HSV ball detection (alternative to YOLO, runs in main loop, no thread) ─
    parser.add_argument(
        "--ball-hsv", action="store_true",
        help="Enable HSV color-matching ball detection instead of YOLO. "
             "Runs synchronously in the main loop — no fps penalty. "
             "Publishes to --ball-hsv-topic (default rt/cam_ball_state).",
    )
    parser.add_argument("--ball-hsv-topic", default="rt/cam_ball_state")
    parser.add_argument("--ball-hsv-h-low",  type=int, default=_HSV_H_LOW_DEFAULT,
                        help=f"HSV hue lower bound (0-180, default {_HSV_H_LOW_DEFAULT})")
    parser.add_argument("--ball-hsv-h-high", type=int, default=_HSV_H_HIGH_DEFAULT,
                        help=f"HSV hue upper bound (0-180, default {_HSV_H_HIGH_DEFAULT})")
    parser.add_argument("--ball-hsv-s-min",  type=int, default=_HSV_S_MIN_DEFAULT,
                        help=f"HSV saturation minimum (default {_HSV_S_MIN_DEFAULT})")
    parser.add_argument("--ball-hsv-v-min",  type=int, default=_HSV_V_MIN_DEFAULT,
                        help=f"HSV value minimum (default {_HSV_V_MIN_DEFAULT})")
    parser.add_argument("--ball-hsv-show-mask", action="store_true",
                        help="Draw HSV mask outline on the MJPEG stream for tuning.")

    parser.add_argument(
        "--chest-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override chest camera translation in waist_pitch frame (metres).",
    )
    parser.add_argument(
        "--chest-rpy",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Override chest camera rotation in radians.",
    )
    args = parser.parse_args()

    target_tag_ids = _parse_tag_ids(args.tag_ids)
    tag_offsets = _parse_tag_offsets(args.tag_offset)
    print(f"[INFO] Target AprilTag ids requested: {target_tag_ids}")
    if tag_offsets:
        print(
            "[INFO] Tag-frame target offsets: "
            + ", ".join(
                f"id={tag_id}->{tuple(round(float(v), 4) for v in offset)}"
                for tag_id, offset in sorted(tag_offsets.items())
            )
        )

    april_dict, family_name = _get_apriltag_dictionary(args.tag_family)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(april_dict, detector_params)
    print(f"[INFO] AprilTag family: {family_name}")
    print(f"[INFO] Tag size: {args.tag_size:.4f} m")

    # Init ROS2 + DDS before the camera pipeline — mirroring ball_detector.py.
    # Starting the camera first lets the frame queue fill during DDS setup
    # (several seconds), which overflows librealsense's frame pool → bad_alloc.
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

    # Depth stream only needed for YOLO mode; HSV mode uses visual depth (apparent ball size).
    pipeline, profile = _start_camera_pipeline(args, with_depth=args.ball)
    if args.list_cameras:
        return

    default_xyz, default_rpy = get_default_chest_extrinsics()
    chest_xyz = tuple(args.chest_xyz) if args.chest_xyz is not None else default_xyz
    chest_rpy = tuple(args.chest_rpy) if args.chest_rpy is not None else default_rpy
    print(f"[INFO] Chest extrinsics xyz={tuple(round(v, 5) for v in chest_xyz)}")
    print(f"[INFO] Chest extrinsics rpy={tuple(round(v, 5) for v in chest_rpy)}")

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    camera_matrix = _build_camera_matrix(color_intrin)
    dist_coeffs = _build_dist_coeffs(color_intrin)
    print(f"[INFO] Color intrinsics fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f}")

    # ── Ball detection setup ───────────────────────────────────────────────
    _yolo_buf_frames = [None]
    _yolo_buf_lock   = threading.Lock()
    _yolo_buf_event  = threading.Event()
    # Shared dict written by YOLO/HSV path, read by MJPEG renderer (dict-key writes
    # are atomic in CPython; stale-by-one-frame reads are acceptable).
    _ball_overlay = {"bbox": None, "pelvis": None, "depth": 0.0, "valid": False, "miss": 0}

    # Depth stream resources (YOLO mode only; HSV mode uses visual depth).
    depth_intrin        = None
    color_to_depth_extr = None
    depth_scale         = 1.0
    if args.ball:
        depth_profile       = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrin        = depth_profile.get_intrinsics()
        color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
        depth_scale         = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"[INFO] Depth intrinsics fx={depth_intrin.fx:.1f} fy={depth_intrin.fy:.1f}  "
              f"depth_scale={depth_scale:.4f}")

    if args.ball:
        _make_yolo_ball_thread(
            model_path=args.ball_model,
            topic=args.ball_topic,
            imgsz=args.ball_imgsz,
            joint=joint,
            color_intrin=color_intrin,
            depth_intrin=depth_intrin,
            color_to_depth_extr=color_to_depth_extr,
            depth_scale=depth_scale,
            chest_xyz=chest_xyz,
            chest_rpy=chest_rpy,
            buf_frames=_yolo_buf_frames,
            buf_lock=_yolo_buf_lock,
            buf_event=_yolo_buf_event,
            overlay_state=_ball_overlay,
        )
        print(f"[INFO] YOLO ball detection thread started -> topic '{args.ball_topic}'")

    # HSV ball detection state (main-loop, no thread).
    _hsv_ball_dds   = None
    _hsv_low        = None
    _hsv_high       = None
    _hsv_center_ema = None
    _hsv_miss_count = 0
    if args.ball_hsv:
        _hsv_low  = np.array([args.ball_hsv_h_low,  args.ball_hsv_s_min, args.ball_hsv_v_min],
                             dtype=np.uint8)
        _hsv_high = np.array([args.ball_hsv_h_high, 255, 255], dtype=np.uint8)
        _hsv_ball_dds = BallStatePublisher(domain_id=0, topic_name=args.ball_hsv_topic)
        print(f"[INFO] HSV ball detection active  H=[{args.ball_hsv_h_low},{args.ball_hsv_h_high}] "
              f"S≥{args.ball_hsv_s_min} V≥{args.ball_hsv_v_min} -> topic '{args.ball_hsv_topic}'")

    if args.show:
        httpd, mjpeg_frame, mjpeg_lock = _start_mjpeg_server()
        print(f"[INFO] MJPEG stream started -> open {_get_stream_url()}")
    else:
        httpd = None
        mjpeg_frame = None
        mjpeg_lock = None

    video_writer = None
    if not args.no_record:
        rec_fps = color_profile.fps()
        rec_dir = Path(args.record_dir)
        rec_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rec_path = rec_dir / f"apriltag_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(rec_path), fourcc, rec_fps, (args.width, args.height)
        )
        print(f"[INFO] Recording -> {rec_path}  ({args.width}×{args.height} @ {rec_fps:.0f}fps)")

    center_ema = None
    last_detection = None
    last_pelvis_xyz = None
    miss_count = 0
    fps = _FPS()

    print("[INFO] Camera running. Press Ctrl+C to stop.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data()).copy()

            # YOLO thread: pre-copy depth + downscaled color; release frame buffer ASAP.
            if args.ball:
                _df = frames.get_depth_frame()
                _ball_depth_np = np.asanyarray(_df.get_data()).copy() if _df else None
                _color_small = cv2.resize(color, (args.ball_imgsz, args.ball_imgsz))
                with _yolo_buf_lock:
                    _yolo_buf_frames[0] = (_color_small, _ball_depth_np)
                _yolo_buf_event.set()

            # HSV ball detection runs synchronously here — pure OpenCV, no thread needed.
            if args.ball_hsv:
                _hsv_cx, _hsv_cy, _hsv_r = _detect_ball_hsv(
                    color, _hsv_low, _hsv_high)

                if _hsv_cx is not None:
                    _hsv_miss_count = 0

                    # Visual depth from apparent ball radius (no depth stream needed).
                    # depth = fx * R_physical / r_px  (accurate to ±10% for r_px > 10)
                    _hsv_depth = (color_intrin.fx * _BALL_RADIUS / _hsv_r
                                  if _hsv_r > 0 else 0.0)

                    if _hsv_depth > 0:
                        _p_opt = rs.rs2_deproject_pixel_to_point(
                            color_intrin, [_hsv_cx, _hsv_cy], _hsv_depth)
                        _p_cam = optical_to_body(_p_opt)
                        _p_arr = np.array(_p_cam, dtype=np.float32)

                        if _hsv_center_ema is None:
                            _hsv_center_ema = _p_arr.copy()
                        elif np.linalg.norm(_p_arr - _hsv_center_ema) < _BALL_EMA_GATE:
                            _hsv_center_ema = (_BALL_EMA_ALPHA * _p_arr
                                               + (1 - _BALL_EMA_ALPHA) * _hsv_center_ema)
                        else:
                            _hsv_center_ema = _p_arr.copy()

                        _p_base = transform_point_chest_camera_to_base_with_extrinsics(
                            _hsv_center_ema,
                            joint.q_wy, joint.q_wr, joint.q_wp,
                            chest_xyz=chest_xyz, chest_rpy=chest_rpy,
                        )
                        _hbx, _hby, _hbz = (float(_p_base[0]),
                                             float(_p_base[1]),
                                             float(_p_base[2]))
                        _hsv_ball_dds.publish(_hbx, _hby, _hbz,
                                              valid=True, source=SOURCE_CAM)

                        _r_int = int(_hsv_r)
                        _ball_overlay["bbox"]   = (_hsv_cx - _r_int, _hsv_cy - _r_int,
                                                   _hsv_cx + _r_int, _hsv_cy + _r_int)
                        _ball_overlay["pelvis"] = (_hbx, _hby, _hbz)
                        _ball_overlay["depth"]  = _hsv_depth
                        _ball_overlay["valid"]  = True
                        _ball_overlay["miss"]   = 0

                        if args.ball_hsv_show_mask:
                            cv2.circle(color, (_hsv_cx, _hsv_cy), _r_int,
                                       (0, 255, 180), 2)

                        print(f"\r[HSV ] cam_ball "
                              f"pelvis=({_hbx:+.3f},{_hby:+.3f},{_hbz:+.3f}) "
                              f"dist={_hsv_depth:.2f}m",
                              end="", flush=True)
                else:
                    _hsv_miss_count += 1
                    if _hsv_miss_count <= _BALL_COAST and _hsv_center_ema is not None:
                        _p_base = transform_point_chest_camera_to_base_with_extrinsics(
                            _hsv_center_ema,
                            joint.q_wy, joint.q_wr, joint.q_wp,
                            chest_xyz=chest_xyz, chest_rpy=chest_rpy,
                        )
                        _hbx, _hby, _hbz = (float(_p_base[0]),
                                             float(_p_base[1]),
                                             float(_p_base[2]))
                        _hsv_ball_dds.publish(_hbx, _hby, _hbz,
                                              valid=False, source=SOURCE_CAM)
                        _ball_overlay["miss"]  = _hsv_miss_count
                        _ball_overlay["valid"] = False
                    else:
                        if _hsv_miss_count > _BALL_COAST:
                            _hsv_center_ema = None
                            _ball_overlay["bbox"] = None
                        _hsv_ball_dds.publish(0.0, 0.0, 0.0,
                                              valid=False, source=SOURCE_NONE)

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            # Downscale before ArUco to avoid bad_alloc in OpenCV C++ on Jetson
            # at high resolutions/fps. Corners are scaled back so pose estimation
            # uses the original camera_matrix unchanged.
            if args.detect_scale != 1.0:
                h, w = gray.shape
                gray_det = cv2.resize(
                    gray, (int(w * args.detect_scale), int(h * args.detect_scale))
                )
            else:
                gray_det = gray
            corners_list, ids, _ = detector.detectMarkers(gray_det)
            if args.detect_scale != 1.0 and corners_list:
                corners_list = tuple(c / args.detect_scale for c in corners_list)
            ids_flat = ids.reshape(-1).tolist() if ids is not None else []
            all_detections = []
            target_detections = []

            for corners, tag_id in zip(corners_list, ids_flat):
                corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
                area = abs(cv2.contourArea(corners))
                center_xy = corners.mean(axis=0)
                detection = {
                    "tag_id": int(tag_id),
                    "corners": corners,
                    "center_xy": center_xy,
                    "area": float(area),
                    "distance_m": 0.0,
                    "rvec": None,
                    "tvec": None,
                    "confidence": _detection_confidence(area, color.shape[:2]),
                }
                rvec, tvec = _estimate_tag_pose(corners, camera_matrix, dist_coeffs, args.tag_size)
                if tvec is not None:
                    detection["rvec"] = rvec
                    detection["tvec"] = tvec
                    detection["distance_m"] = float(np.linalg.norm(tvec))
                    target_tvec, uses_offset = _resolve_target_point_optical(
                        detection,
                        tag_offsets,
                    )
                    detection["target_tvec"] = target_tvec
                    detection["target_distance_m"] = float(np.linalg.norm(target_tvec))
                    detection["uses_offset"] = uses_offset
                else:
                    detection["target_tvec"] = None
                    detection["target_distance_m"] = 0.0
                    detection["uses_offset"] = False
                all_detections.append(detection)
                if (
                    detection["tag_id"] in target_tag_ids
                    and detection["target_tvec"] is not None
                ):
                    target_detections.append(detection)

            representative = _select_detection(target_detections, target_tag_ids)
            fused_target_tvec, fused_count = _fuse_target_points_optical(target_detections)
            published = False
            pelvis_xyz = last_pelvis_xyz if last_pelvis_xyz is not None else (0.0, 0.0, 0.0)
            tag_conf = 0.0
            tag_distance = 0.0
            fused_tag_ids = []
            overlay_mode_txt = "mode: waiting"

            if representative is not None and fused_target_tvec is not None:
                miss_count = 0
                last_detection = representative
                fused_tag_ids = [det["tag_id"] for det in target_detections]
                # Each visible tag proposes the same shared target point in the
                # optical frame. Fuse all visible proposals, then convert once.
                p_cam_arr = optical_to_body(fused_target_tvec)

                if center_ema is None:
                    center_ema = p_cam_arr.copy()
                else:
                    gate_dist = float(np.linalg.norm(p_cam_arr - center_ema))
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
                last_pelvis_xyz = pelvis_xyz
                tag_conf = float(max(det["confidence"] for det in target_detections))
                tag_distance = float(np.linalg.norm(fused_target_tvec))
                published = True
                dds.publish(
                    *pelvis_xyz,
                    valid=True,
                    class_id=representative["tag_id"],
                    confidence=tag_conf,
                    source=SOURCE_CHEST_CAMERA,
                )
                target_mode = "fused" if fused_count > 1 else ("offset" if representative["uses_offset"] else "center")
                contributors = ",".join(str(tag_id) for tag_id in fused_tag_ids)
                if fused_count > 1:
                    overlay_mode_txt = f"mode: fused tags {contributors}"
                else:
                    overlay_mode_txt = f"mode: single tag {representative['tag_id']} ({target_mode})"
                print(
                    f"\r[TAG {representative['tag_id']}] pelvis=({pelvis_xyz[0]:+.3f}, {pelvis_xyz[1]:+.3f}, {pelvis_xyz[2]:+.3f}) "
                    f"dist={tag_distance:.2f}m mode={target_mode} tags={contributors} conf={tag_conf:.2f} apriltag={fps.fps:4.1f}fps",
                    end="",
                    flush=True,
                )
            else:
                miss_count += 1
                if last_detection is not None and last_pelvis_xyz is not None and miss_count <= args.coast_frames:
                    published = True
                    tag_conf = float(last_detection["confidence"])
                    tag_distance = float(last_detection["target_distance_m"])
                    overlay_mode_txt = f"mode: coast from tag {last_detection['tag_id']}"
                    dds.publish(
                        *last_pelvis_xyz,
                        valid=False,
                        class_id=last_detection["tag_id"],
                        confidence=tag_conf,
                        source=SOURCE_CHEST_CAMERA,
                    )
                    print(
                        f"\r[COAST {last_detection['tag_id']}] pelvis=({last_pelvis_xyz[0]:+.3f}, {last_pelvis_xyz[1]:+.3f}, {last_pelvis_xyz[2]:+.3f}) "
                        f"dist={tag_distance:.2f}m miss={miss_count}/{args.coast_frames} apriltag={fps.fps:4.1f}fps",
                        end="",
                        flush=True,
                    )
                else:
                    center_ema = None
                    last_detection = None
                    last_pelvis_xyz = None
                    overlay_mode_txt = "mode: no visible target tags"
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
                        f"\r[     ] no tag ({','.join(str(tag_id) for tag_id in target_tag_ids)}) apriltag={fps.fps:4.1f}fps" + " " * 20,
                        end="",
                        flush=True,
                    )

            if args.show or video_writer is not None:
                vis = color.copy()
                for det in all_detections:
                    corners = det["corners"].astype(np.int32)
                    is_target = det["tag_id"] in target_tag_ids
                    is_selected = representative is not None and det["tag_id"] == representative["tag_id"] and np.allclose(
                        det["center_xy"], representative["center_xy"]
                    )
                    color_box = (0, 255, 0) if is_selected else ((255, 160, 0) if is_target else (160, 160, 160))
                    cv2.polylines(vis, [corners], True, color_box, 2)
                    cx, cy = det["center_xy"].astype(int)
                    cv2.circle(vis, (cx, cy), 4, color_box, -1)
                    label = f"id={det['tag_id']} d={det['distance_m']:.2f}m"
                    cv2.putText(
                        vis,
                        label,
                        (int(corners[0][0]), max(24, int(corners[0][1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color_box,
                        2,
                    )

                if representative is None and last_detection is not None and miss_count <= args.coast_frames:
                    coast_corners = last_detection["corners"].astype(np.int32)
                    cv2.polylines(vis, [coast_corners], True, (0, 165, 255), 2)
                    cv2.putText(
                        vis,
                        f"coast {last_detection['tag_id']} {miss_count}/{args.coast_frames}",
                        (int(coast_corners[0][0]), max(24, int(coast_corners[0][1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )

                fps_txt = f"AprilTag {fps.fps:.1f} fps"
                ids_txt = f"target ids: {', '.join(str(tag_id) for tag_id in target_tag_ids)}"
                fam_txt = f"family: {family_name} size={args.tag_size:.3f}m"
                extr_txt = f"xyz={tuple(round(v, 3) for v in chest_xyz)}"
                cv2.putText(
                    vis, fps_txt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                )
                cv2.putText(
                    vis, ids_txt, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )
                cv2.putText(
                    vis, fam_txt, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                cv2.putText(
                    vis, extr_txt, (10, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                if tag_offsets:
                    cv2.putText(
                        vis, "target mode: per-tag offset", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )
                cv2.putText(
                    vis,
                    overlay_mode_txt,
                    (10, 142),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                if published and last_pelvis_xyz is not None:
                    info = (
                        f"pelvis ({pelvis_xyz[0]:+.2f}, {pelvis_xyz[1]:+.2f}, {pelvis_xyz[2]:+.2f})m "
                        f"dist {tag_distance:.2f}m"
                    )
                    cv2.putText(vis, info, (10, vis.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                # Draw ball detection overlay (from YOLO thread, if --ball active)
                if args.ball:
                    bo = _ball_overlay
                    if bo["bbox"] is not None:
                        bx1, by1, bx2, by2 = bo["bbox"]
                        is_ball_valid = bo["valid"]
                        box_color = (0, 255, 128) if is_ball_valid else (0, 165, 255)
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_color, 2)
                        bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                        cv2.circle(vis, (bcx, bcy), 5, box_color, -1)
                        miss = bo["miss"]
                        label = (f"ball d={bo['depth']:.2f}m" if is_ball_valid
                                 else f"ball coast {miss}")
                        cv2.putText(vis, label, (bx1, max(16, by1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                        if bo["pelvis"] is not None:
                            px, py, pz = bo["pelvis"]
                            cv2.putText(
                                vis,
                                f"ball pelvis ({px:+.2f},{py:+.2f},{pz:+.2f})m",
                                (10, vis.shape[0] - 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2,
                            )

                if video_writer is not None:
                    video_writer.write(vis)

                if args.show:
                    ok, jpg_buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ok:
                        with mjpeg_lock:
                            mjpeg_frame[0] = jpg_buf.tobytes()

            fps.tick()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Recording saved.")
        if httpd is not None:
            httpd.shutdown()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
