#!/usr/bin/env python3
"""Measure ball-to-target distance from one RGB image or video frame.

The script estimates:
1. The shared AprilTag target point from known tag size + board offsets.
2. The ball centre from a detected image circle + known physical ball radius.
3. The Euclidean distance between the two 3-D points.

Both points are reported in the camera optical frame and camera body frame.
No depth image is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from onboard.perception.camera.camera_to_base import (  # noqa: E402
    optical_to_body,
)


APRILTAG_FAMILIES = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the 3-D distance between a ball centre and an "
            "AprilTag-defined target point from one RGB frame."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to an RGB image.")
    source.add_argument("--video", type=Path, help="Path to a video file.")
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index for --video input (default: 0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "frame_ball_target_distance.example.yaml"
        ),
        help="YAML config path.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Optional path for annotated output image.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for structured JSON results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the annotated frame in an OpenCV window.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return cfg


def load_image_or_frame(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return image, args.image.name

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame {args.frame_idx} from video {args.video}."
        )
    label = f"{args.video.name}[frame={args.frame_idx}]"
    return frame, label


def build_camera_matrix(intr_cfg: dict[str, Any]) -> np.ndarray:
    fx = float(intr_cfg["fx"])
    fy = float(intr_cfg["fy"])
    cx = float(intr_cfg["cx"])
    cy = float(intr_cfg["cy"])
    if min(fx, fy) <= 0.0:
        raise ValueError("Camera intrinsics fx/fy must be positive.")
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_dist_coeffs(values: Any) -> np.ndarray:
    coeffs = [float(v) for v in (values or [0.0] * 5)]
    if len(coeffs) < 5:
        coeffs += [0.0] * (5 - len(coeffs))
    return np.asarray(coeffs[:5], dtype=np.float32).reshape(-1, 1)


def resolve_camera(
    cfg: dict[str, Any],
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    camera_cfg = cfg.get("camera", {})
    intr_cfg = camera_cfg.get("intrinsics", {})
    camera_matrix = build_camera_matrix(intr_cfg)
    dist_coeffs = build_dist_coeffs(camera_cfg.get("dist_coeffs"))

    cfg_w = int(camera_cfg.get("width", image_shape[1]))
    cfg_h = int(camera_cfg.get("height", image_shape[0]))
    img_h, img_w = image_shape[:2]
    if (cfg_w, cfg_h) != (img_w, img_h):
        print(
            "[WARN] Config resolution "
            f"{cfg_w}x{cfg_h} differs from input frame {img_w}x{img_h}. "
            "Metric accuracy may be poor unless the intrinsics match "
            "the frame.",
            flush=True,
        )
    return camera_matrix, dist_coeffs


def get_apriltag_dictionary(family_name: str) -> cv2.aruco.Dictionary:
    key = str(family_name).strip().lower()
    if key not in APRILTAG_FAMILIES:
        raise ValueError(
            f"Unsupported AprilTag family '{family_name}'. "
            f"Choose one of: {sorted(APRILTAG_FAMILIES)}"
        )
    return cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[key])


def create_aruco_detector(
    dictionary: cv2.aruco.Dictionary,
) -> tuple[Any, Any]:
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector, None
    return None, params


def estimate_tag_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    solvepnp_flag = getattr(
        cv2,
        "SOLVEPNP_IPPE_SQUARE",
        cv2.SOLVEPNP_ITERATIVE,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=solvepnp_flag,
    )
    if not ok:
        return None, None
    return rvec.reshape(3), tvec.reshape(3)


def parse_tag_offsets(raw_offsets: dict[str, Any]) -> dict[int, np.ndarray]:
    offsets: dict[int, np.ndarray] = {}
    for key, value in raw_offsets.items():
        tag_id = int(key)
        if len(value) != 3:
            raise ValueError(f"Tag offset for id {tag_id} must have 3 values.")
        offsets[tag_id] = np.asarray(value, dtype=np.float32).reshape(3)
    return offsets


def project_camera_point(
    point_xyz: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    object_points = np.asarray(point_xyz, dtype=np.float32).reshape(1, 1, 3)
    image_points, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        camera_matrix,
        dist_coeffs,
    )
    return image_points.reshape(2)


def detect_apriltag_target(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    tag_cfg = cfg.get("apriltag", {})
    family = tag_cfg.get("family", "tag36h11")
    tag_size_m = float(tag_cfg["tag_size_m"])
    track_ids = [int(tag_id) for tag_id in tag_cfg.get("track_ids", [])]
    tag_offsets = parse_tag_offsets(tag_cfg.get("offsets_m", {}))
    if not track_ids:
        track_ids = sorted(tag_offsets.keys())
    if not track_ids:
        raise ValueError(
            "apriltag.track_ids or apriltag.offsets_m is required."
        )

    dictionary = get_apriltag_dictionary(family)
    detector, legacy_params = create_aruco_detector(dictionary)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        corners_list, ids, _ = detector.detectMarkers(gray)
    else:
        corners_list, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=legacy_params,
        )
    if ids is None:
        raise RuntimeError(
            f"No AprilTag detected for family {family} in the input frame."
        )

    detections: list[dict[str, Any]] = []
    for corners, tag_id_raw in zip(corners_list, ids.flatten()):
        tag_id = int(tag_id_raw)
        if tag_id not in track_ids:
            continue
        corners_arr = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        area = float(abs(cv2.contourArea(corners_arr)))
        center_xy = corners_arr.mean(axis=0)
        rvec, tvec = estimate_tag_pose(
            corners_arr,
            camera_matrix,
            dist_coeffs,
            tag_size_m,
        )
        if tvec is None:
            continue

        target_tvec = np.asarray(tvec, dtype=np.float32)
        uses_offset = False
        offset_tag = tag_offsets.get(tag_id)
        if offset_tag is not None:
            rot_optical_from_tag, _ = cv2.Rodrigues(
                np.asarray(rvec, dtype=np.float32).reshape(3, 1)
            )
            target_tvec = target_tvec + rot_optical_from_tag @ offset_tag
            uses_offset = True

        target_xy = project_camera_point(
            target_tvec,
            camera_matrix,
            dist_coeffs,
        )
        detections.append(
            {
                "tag_id": tag_id,
                "corners": corners_arr,
                "center_xy": center_xy,
                "area": area,
                "rvec": rvec,
                "tvec": tvec,
                "target_tvec": target_tvec,
                "target_xy": target_xy,
                "uses_offset": uses_offset,
            }
        )

    if not detections:
        raise RuntimeError(
            "AprilTags were detected, but none matched apriltag.track_ids."
        )

    weights = np.asarray(
        [max(1.0, det["area"]) for det in detections],
        dtype=np.float32,
    )
    points = np.stack([det["target_tvec"] for det in detections], axis=0)
    fused_target = (weights[:, None] * points).sum(axis=0) / weights.sum()
    fused_pixel = project_camera_point(
        fused_target,
        camera_matrix,
        dist_coeffs,
    )

    return {
        "family": family,
        "tag_size_m": tag_size_m,
        "detections": detections,
        "fused_target_optical": fused_target,
        "fused_target_pixel": fused_pixel,
        "tag_ids_used": [det["tag_id"] for det in detections],
    }


def clip_roi(
    roi_cfg: list[int] | None,
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    if not roi_cfg or len(roi_cfg) != 4:
        return 0, 0, img_w, img_h

    x0, y0, x1, y1 = [int(v) for v in roi_cfg]
    if x1 <= 0:
        x1 = img_w
    if y1 <= 0:
        y1 = img_h
    x0 = max(0, min(img_w, x0))
    y0 = max(0, min(img_h, y0))
    x1 = max(x0 + 1, min(img_w, x1))
    y1 = max(y0 + 1, min(img_h, y1))
    return x0, y0, x1, y1


def score_hough_circle(
    gray_roi: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    expect_dark: bool,
) -> float:
    x0 = max(0, int(cx - 1.5 * radius))
    y0 = max(0, int(cy - 1.5 * radius))
    x1 = min(gray_roi.shape[1], int(cx + 1.5 * radius) + 1)
    y1 = min(gray_roi.shape[0], int(cy + 1.5 * radius) + 1)
    patch = gray_roi[y0:y1, x0:x1]
    if patch.size == 0:
        return -1e9

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = patch[dist <= 0.75 * radius]
    ring = patch[(dist >= 1.05 * radius) & (dist <= 1.35 * radius)]
    if inner.size == 0 or ring.size == 0:
        return -1e9

    contrast = float(ring.mean() - inner.mean())
    if not expect_dark:
        contrast = abs(contrast)
    return contrast * float(radius)


def detect_ball_hough(
    image_bgr: np.ndarray,
    ball_cfg: dict[str, Any],
) -> dict[str, Any]:
    det_cfg = ball_cfg.get("detection", {})
    hough_cfg = det_cfg.get("hough", {})
    x0, y0, x1, y1 = clip_roi(det_cfg.get("roi"), image_bgr.shape)
    roi = image_bgr[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_ksize = int(hough_cfg.get("blur_ksize", 9))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_sigma = float(hough_cfg.get("blur_sigma", 2.0))
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

    max_working_size_px = int(hough_cfg.get("max_working_size_px", 1600))
    scale = 1.0
    gray_h, gray_w = gray.shape[:2]
    max_side = max(gray_h, gray_w)
    if max_working_size_px > 0 and max_side > max_working_size_px:
        scale = float(max_working_size_px) / float(max_side)
        new_w = max(1, int(round(gray_w * scale)))
        new_h = max(1, int(round(gray_h * scale)))
        gray_work = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_work = gray

    circles = cv2.HoughCircles(
        gray_work,
        cv2.HOUGH_GRADIENT,
        dp=float(hough_cfg.get("dp", 1.2)),
        minDist=max(1.0, float(hough_cfg.get("min_dist_px", 200)) * scale),
        param1=float(hough_cfg.get("param1", 100)),
        param2=float(hough_cfg.get("param2", 20)),
        minRadius=max(1, int(round(hough_cfg.get("min_radius_px", 20) * scale))),
        maxRadius=max(1, int(round(hough_cfg.get("max_radius_px", 500) * scale))),
    )
    if circles is None:
        raise RuntimeError("Hough circle detection did not find a ball.")

    expect_dark = bool(hough_cfg.get("expect_dark", True))
    best = None
    for circle in circles[0]:
        cx_work, cy_work, radius_work = [float(v) for v in circle]
        cx_roi = cx_work / scale
        cy_roi = cy_work / scale
        radius = radius_work / scale
        score = score_hough_circle(gray, cx_roi, cy_roi, radius, expect_dark)
        if best is None or score > best["score"]:
            best = {
                "cx": cx_roi + x0,
                "cy": cy_roi + y0,
                "radius_px": radius,
                "score": score,
                "method": "hough",
                "roi": [x0, y0, x1, y1],
            }
    if best is None:
        raise RuntimeError(
            "Hough circle detection did not yield a usable ball."
        )
    return best


def detect_ball_hsv(
    image_bgr: np.ndarray,
    ball_cfg: dict[str, Any],
) -> dict[str, Any]:
    det_cfg = ball_cfg.get("detection", {})
    hsv_cfg = det_cfg.get("hsv", {})
    x0, y0, x1, y1 = clip_roi(det_cfg.get("roi"), image_bgr.shape)
    roi = image_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.asarray(hsv_cfg.get("lower", [0, 0, 0]), dtype=np.uint8)
    upper = np.asarray(hsv_cfg.get("upper", [180, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_size = int(hsv_cfg.get("kernel_size", 5))
    if kernel_size < 1:
        kernel_size = 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    open_iterations = int(hsv_cfg.get("open_iterations", 1))
    close_iterations = int(hsv_cfg.get("close_iterations", 2))
    if close_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )
    if open_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations,
        )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best = None
    min_circularity = float(hsv_cfg.get("min_circularity", 0.6))
    min_radius_px = float(hsv_cfg.get("min_radius_px", 10))
    max_radius_px = float(hsv_cfg.get("max_radius_px", 500))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity < min_circularity:
            continue
        (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(contour)
        if not (min_radius_px <= radius <= max_radius_px):
            continue

        score = area * circularity
        if best is None or score > best["score"]:
            best = {
                "cx": float(cx_roi + x0),
                "cy": float(cy_roi + y0),
                "radius_px": float(radius),
                "score": float(score),
                "method": "hsv",
                "roi": [x0, y0, x1, y1],
            }
    if best is None:
        raise RuntimeError("HSV ball detection did not find a usable circle.")
    return best


def detect_ball_manual(ball_cfg: dict[str, Any]) -> dict[str, Any]:
    det_cfg = ball_cfg.get("detection", {})
    manual_cfg = det_cfg.get("manual", {})
    circle = manual_cfg.get("circle_px")
    if not circle or len(circle) != 3:
        raise RuntimeError(
            "ball.detection.manual.circle_px must contain [cx, cy, radius]."
        )
    return {
        "cx": float(circle[0]),
        "cy": float(circle[1]),
        "radius_px": float(circle[2]),
        "score": 0.0,
        "method": "manual",
        "roi": None,
    }


def detect_ball(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ball_cfg = cfg.get("ball", {})
    det_cfg = ball_cfg.get("detection", {})
    method = str(det_cfg.get("method", "hough")).strip().lower()
    if method == "hough":
        return detect_ball_hough(image_bgr, ball_cfg)
    if method == "hsv":
        return detect_ball_hsv(image_bgr, ball_cfg)
    if method == "manual":
        return detect_ball_manual(ball_cfg)
    raise ValueError(
        f"Unsupported ball detection method '{method}'. "
        "Choose from: hough, hsv, manual."
    )


def estimate_ball_depth_z(
    radius_px: float,
    camera_matrix: np.ndarray,
    ball_radius_m: float,
    formula: str,
) -> float:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    f_mean = 0.5 * (fx + fy)
    radius_px = max(float(radius_px), 1e-6)

    if formula == "small_angle":
        return f_mean * float(ball_radius_m) / radius_px
    if formula == "exact_sphere":
        ratio = f_mean / radius_px
        return float(ball_radius_m) * math.sqrt(1.0 + ratio * ratio)
    raise ValueError(
        "ball.depth_formula must be 'exact_sphere' or 'small_angle'."
    )


def deproject_pixel(
    pixel_xy: tuple[float, float],
    depth_z: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(
        pixel,
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    ).reshape(2)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (float(undistorted[0]) - cx) / fx * depth_z
    y = (float(undistorted[1]) - cy) / fy * depth_z
    return np.asarray([x, y, depth_z], dtype=np.float32)


def draw_cross(
    image: np.ndarray,
    center_xy: np.ndarray | tuple[float, float],
    color: tuple[int, int, int],
    size: int = 18,
    thickness: int = 2,
) -> None:
    cx = int(round(float(center_xy[0])))
    cy = int(round(float(center_xy[1])))
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)


def annotate_image(
    image_bgr: np.ndarray,
    tag_result: dict[str, Any],
    ball_result: dict[str, Any],
    distance_m: float,
) -> np.ndarray:
    vis = image_bgr.copy()

    for det in tag_result["detections"]:
        corners = det["corners"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
        center_xy = det["center_xy"]
        cv2.circle(
            vis,
            tuple(np.round(center_xy).astype(int)),
            4,
            (0, 255, 0),
            -1,
        )
        label = f"id={det['tag_id']}"
        if det["uses_offset"]:
            label += " -> target"
        text_xy = tuple(
            np.round(det["center_xy"]).astype(int) + np.array([8, -8])
        )
        cv2.putText(
            vis,
            label,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    draw_cross(vis, tag_result["fused_target_pixel"], (255, 0, 255), size=24)
    cv2.putText(
        vis,
        "target",
        tuple(
            np.round(tag_result["fused_target_pixel"]).astype(int)
            + np.array([8, 26])
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    ball_center = (
        int(round(ball_result["cx"])),
        int(round(ball_result["cy"])),
    )
    ball_radius = int(round(ball_result["radius_px"]))
    cv2.circle(vis, ball_center, ball_radius, (0, 165, 255), 3)
    cv2.circle(vis, ball_center, 4, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"ball ({ball_result['method']})",
        (
            ball_center[0] - ball_radius,
            max(20, ball_center[1] - ball_radius - 12),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
    )

    target_center = tuple(
        np.round(tag_result["fused_target_pixel"]).astype(int).tolist()
    )
    cv2.line(vis, ball_center, target_center, (255, 255, 0), 2)
    cv2.putText(
        vis,
        f"ball-target = {distance_m:.3f} m",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2,
    )
    return vis


def to_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values).reshape(-1).tolist()]


def build_results(
    source_label: str,
    ball_result: dict[str, Any],
    ball_optical: np.ndarray,
    tag_result: dict[str, Any],
    distance_m: float,
) -> dict[str, Any]:
    ball_body = optical_to_body(ball_optical)
    target_optical = tag_result["fused_target_optical"]
    target_body = optical_to_body(target_optical)

    return {
        "source": source_label,
        "ball": {
            "method": ball_result["method"],
            "center_px": [float(ball_result["cx"]), float(ball_result["cy"])],
            "radius_px": float(ball_result["radius_px"]),
            "optical_xyz_m": to_float_list(ball_optical),
            "body_xyz_m": to_float_list(ball_body),
        },
        "target": {
            "tag_ids_used": [int(v) for v in tag_result["tag_ids_used"]],
            "center_px": to_float_list(tag_result["fused_target_pixel"]),
            "optical_xyz_m": to_float_list(target_optical),
            "body_xyz_m": to_float_list(target_body),
        },
        "distance_m": float(distance_m),
    }


def default_output_path(
    args: argparse.Namespace,
    stem_suffix: str,
    extension: str,
) -> Path | None:
    if args.image is not None:
        base = args.image
        return base.with_name(base.stem + stem_suffix + extension)
    if args.video is not None:
        base = args.video
        name = f"{base.stem}_frame{args.frame_idx:06d}{stem_suffix}{extension}"
        return base.with_name(name)
    return None


def save_outputs(
    args: argparse.Namespace,
    annotated: np.ndarray,
    results: dict[str, Any],
) -> None:
    output_image = args.output_image or default_output_path(
        args,
        "_annotated",
        ".jpg",
    )
    output_json = args.output_json or default_output_path(
        args,
        "_distance",
        ".json",
    )

    if output_image is not None:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image), annotated)
        print(f"[INFO] Saved annotated image to {output_image}")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"[INFO] Saved JSON results to {output_json}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    image_bgr, source_label = load_image_or_frame(args)
    camera_matrix, dist_coeffs = resolve_camera(cfg, image_bgr.shape)

    tag_result = detect_apriltag_target(
        image_bgr,
        camera_matrix,
        dist_coeffs,
        cfg,
    )
    ball_result = detect_ball(image_bgr, cfg)

    ball_cfg = cfg.get("ball", {})
    ball_radius_m = float(ball_cfg["radius_m"])
    depth_formula = str(ball_cfg.get("depth_formula", "exact_sphere"))
    ball_depth_z = estimate_ball_depth_z(
        ball_result["radius_px"],
        camera_matrix,
        ball_radius_m,
        depth_formula,
    )
    ball_optical = deproject_pixel(
        (ball_result["cx"], ball_result["cy"]),
        ball_depth_z,
        camera_matrix,
        dist_coeffs,
    )

    target_optical = np.asarray(
        tag_result["fused_target_optical"],
        dtype=np.float32,
    )
    distance_m = float(np.linalg.norm(ball_optical - target_optical))
    results = build_results(
        source_label,
        ball_result,
        ball_optical,
        tag_result,
        distance_m,
    )

    print(f"[INFO] Source: {source_label}")
    print(
        "[INFO] Ball centre (optical, m): "
        f"{np.round(ball_optical, 4).tolist()}  "
        f"radius_px={ball_result['radius_px']:.2f}"
    )
    print(
        "[INFO] Target centre (optical, m): "
        f"{np.round(target_optical, 4).tolist()}  "
        f"tags={tag_result['tag_ids_used']}"
    )
    print(f"[INFO] Ball-target distance: {distance_m:.4f} m")

    annotated = annotate_image(image_bgr, tag_result, ball_result, distance_m)
    save_outputs(args, annotated, results)

    if args.show:
        cv2.imshow("frame_ball_target_distance", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
