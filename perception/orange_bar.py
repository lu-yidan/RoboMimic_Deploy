"""Controlled orange-bar RGB-D baseline; no MuJoCo/model access in estimator.

Inputs are image samples, calibrated pinhole intrinsics, camera pose, and a
known 15 mm radius. Scores below are quality heuristics, not probabilities.
The endpoints describe the visible capsule span; occluded ends are uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float):
        focal = .5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
        return cls(width, height, float(focal), float(focal),
                   (width - 1) / 2, (height - 1) / 2)


def backproject(depth, calibration: CameraCalibration, camera_position,
                camera_rotation, mask=None):
    """MuJoCo optical-axis depth -> world, OpenGL x right/y up/z backwards."""
    depth = np.asarray(depth, dtype=float)
    if depth.shape != (calibration.height, calibration.width):
        raise ValueError("Depth shape does not match calibration")
    valid = np.isfinite(depth) & (depth > .05) & (depth < 5.)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    xyz = np.column_stack(((u - calibration.cx) * z / calibration.fx,
                           -(v - calibration.cy) * z / calibration.fy, -z))
    world = xyz @ np.asarray(camera_rotation).reshape(3, 3).T
    world += np.asarray(camera_position)
    return world, np.column_stack((u, v))


def project(points, calibration: CameraCalibration, camera_position,
            camera_rotation):
    local = (np.asarray(points) - np.asarray(camera_position)) @ np.asarray(camera_rotation).reshape(3, 3)
    z = -local[:, 2]
    return np.column_stack((calibration.fx * local[:, 0] / z + calibration.cx,
                            -calibration.fy * local[:, 1] / z + calibration.cy)), z


def orange_mask(rgb):
    """Fixed controlled-color threshold; not arbitrary/unpainted bar recognition."""
    rgb = np.asarray(rgb).astype(float)
    r, g, b = np.moveaxis(rgb, -1, 0)
    return (r > 55) & (r > 1.35 * g) & (g > 1.25 * b) & (r - b > 35)


def _axis_fit(points, radius_m, camera_position):
    mean = points.mean(axis=0)
    covariance = np.cov(points - mean, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    span = (points - mean) @ axis
    # Remove capsule endcaps from the known-radius cylindrical surface fit.
    lo, hi = np.quantile(span, [.08, .92])
    inner = points[(span >= lo) & (span <= hi)]
    away = mean - np.asarray(camera_position)
    away -= axis * np.dot(away, axis)
    away /= max(np.linalg.norm(away), 1e-9)
    e1 = away
    e2 = np.cross(axis, e1)
    uv = np.column_stack(((inner - mean) @ e1, (inner - mean) @ e2))
    fit = least_squares(lambda c: (np.linalg.norm(uv - c, axis=1) - radius_m),
                        [radius_m * .75, 0.], loss="soft_l1", f_scale=.001,
                        bounds=([-3 * radius_m, -3 * radius_m],
                                [3 * radius_m, 3 * radius_m]))
    transverse_center = mean + fit.x[0] * e1 + fit.x[1] * e2
    all_t = (points - transverse_center) @ axis
    # Robust visible limits. Centerline endpoints subtract the assumed radius.
    t0, t1 = np.quantile(all_t, [.001, .999])
    visible_endpoints = np.stack((transverse_center + t0 * axis,
                                  transverse_center + t1 * axis))
    if t1 - t0 > 2 * radius_m:
        endpoints = np.stack((transverse_center + (t0 + radius_m) * axis,
                              transverse_center + (t1 - radius_m) * axis))
    else:
        endpoints = visible_endpoints.copy()
    radial_residual = np.linalg.norm(uv - fit.x, axis=1) - radius_m
    return {"center_world_m": endpoints.mean(axis=0).tolist(),
            "axis_world": axis.tolist(), "endpoints_world_m": endpoints.tolist(),
            "visible_surface_endpoints_world_m": visible_endpoints.tolist(),
            "visible_span_m": float(t1 - t0),
            "radial_rmse_m": float(np.sqrt(np.mean(radial_residual ** 2))),
            "elongation": float(values[-1] / max(values[-2], 1e-12))}


def estimate_bars(rgb, depth, calibration: CameraCalibration, camera_position,
                  camera_rotation, radius_m=.015):
    """Return finite supported visible candidates. No ground-truth arguments.

    camera_position/rotation must come from calibration/state estimation;
    ideal simulation pose is supplied by the benchmark and reported as such.
    """
    if np.asarray(rgb).shape != (calibration.height, calibration.width, 3):
        raise ValueError("RGB shape does not match calibration")
    color_mask = orange_mask(rgb)
    valid_depth = np.isfinite(depth) & (np.asarray(depth) > .05) & (np.asarray(depth) < 5.)
    # Hole closing only aids image grouping; only original measured pixels fit.
    grouping = ndimage.binary_closing(color_mask, iterations=2)
    labels, count = ndimage.label(grouping)
    candidates = []
    fragments = []
    for label in range(1, count + 1):
        component = (labels == label) & color_mask & valid_depth
        if np.count_nonzero(component) < 60:
            continue
        points, pixels = backproject(depth, calibration, camera_position,
                                     camera_rotation, component)
        median = np.median(points, axis=0)
        # Reject color/noise outliers outside a plausible local object region.
        points = points[np.linalg.norm(points - median, axis=1) < .5]
        if len(points) < 60:
            continue
        fragments.append({"points": points, "pixels": pixels,
                          "count": int(np.count_nonzero(component))})
    # A hook can split one orange rod into disconnected regions. Merge only
    # collinear nearby observed regions; never extrapolate across a large gap.
    fragments.sort(key=lambda f: len(f["points"]), reverse=True)
    while fragments:
        fragment = fragments.pop(0)
        points, pixels = fragment["points"], fragment["pixels"]
        count = fragment["count"]
        initial = _axis_fit(points, radius_m, camera_position)
        if initial["elongation"] >= 8:
            axis = np.asarray(initial["axis_world"])
            center = np.asarray(initial["center_world_m"])
            keep = []
            for other in fragments:
                delta = other["points"] - center
                t_other = delta @ axis
                t_main = (points - center) @ axis
                gap = max(float(t_other.min() - t_main.max()),
                          float(t_main.min() - t_other.max()), 0.)
                residual = np.linalg.norm(delta - t_other[:, None] * axis, axis=1)
                if np.median(residual) < radius_m * 1.35 and gap < .095:
                    points = np.vstack((points, other["points"]))
                    pixels = np.vstack((pixels, other["pixels"]))
                    count += other["count"]
                else:
                    keep.append(other)
            fragments = keep
        if len(points) > 3000:
            points = points[np.linspace(0, len(points) - 1, 3000).astype(int)]
        result = _axis_fit(points, radius_m, camera_position)
        boundary = bool(np.any((pixels[:, 0] <= 1) | (pixels[:, 0] >= calibration.width - 2)
                               | (pixels[:, 1] <= 1) | (pixels[:, 1] >= calibration.height - 2)))
        result["pixel_count"] = count
        result["image_clipped"] = boundary
        result["valid"] = bool(result["visible_span_m"] >= .06 and
                                result["elongation"] >= 8 and
                                result["radial_rmse_m"] <= .003 and not boundary)
        result["quality_score"] = float(np.clip((1 - result["radial_rmse_m"] / .003)
                                                * min(1, result["visible_span_m"] / .12), 0, 1))
        result["endpoint_status"] = "visible_span_only_not_occlusion_certified"
        candidates.append(result)
    return {"candidates": candidates,
            "valid_candidate_count": sum(c["valid"] for c in candidates),
            "color_pixel_count": int(color_mask.sum()),
            "measured_color_pixel_count": int((color_mask & valid_depth).sum()),
            "assumptions": ["controlled orange color", "known capsule radius",
                            "calibrated pinhole", "supplied camera pose"],
            "calibration": asdict(calibration)}
