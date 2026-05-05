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
from onboard.perception.camera.camera_to_base import (  # noqa: E402
    get_default_chest_extrinsics,
    optical_to_body,
    transform_point_chest_camera_to_base_with_extrinsics,
)


COAST_FRAMES = 8
EMA_ALPHA = 0.5
EMA_GATE = 0.4


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
    if not all_sns:
        raise RuntimeError("No RealSense device found.")

    serial = args.camera_serial or all_sns[0]
    print(f"[INFO] CHEST AprilTag camera -> serial {serial}")

    pipeline = rs.pipeline()
    fps_tries = [30, 15, 10, 5]
    last_err = None
    for color_fps in fps_tries:
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, color_fps)
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

    pipeline, profile = _start_camera_pipeline(args)
    if args.list_cameras:
        return

    # Drain incoming frames while ROS2/DDS initializes — the realsense pipeline
    # streams at 30fps from this point and its internal queue will overflow (bad_alloc)
    # if frames are not consumed during the several-second DDS setup.
    _draining = True
    def _drain_frames():
        while _draining:
            try:
                pipeline.wait_for_frames(timeout_ms=50)
            except Exception:
                pass
    threading.Thread(target=_drain_frames, daemon=True).start()

    rclpy.init()
    joint = _JointListener()
    _draining = False

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

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    camera_matrix = _build_camera_matrix(color_intrin)
    dist_coeffs = _build_dist_coeffs(color_intrin)
    print(f"[INFO] Color intrinsics fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f}")

    if args.show:
        httpd, mjpeg_frame, mjpeg_lock = _start_mjpeg_server()
        print(f"[INFO] MJPEG stream started -> open {_get_stream_url()}")
    else:
        httpd = None
        mjpeg_frame = None
        mjpeg_lock = None

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

            if args.show:
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
                ok, jpg_buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    with mjpeg_lock:
                        mjpeg_frame[0] = jpg_buf.tobytes()

            fps.tick()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()
        if httpd is not None:
            httpd.shutdown()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
