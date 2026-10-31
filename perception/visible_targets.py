"""Conservative visible-segment targets and plane/cylinder model challenge.

The frozen v1 estimator remains unchanged. This layer never infers hidden ends,
never assigns grasp/load permission, and receives no MuJoCo objects or labels.
"""
from __future__ import annotations
import numpy as np
from .orange_bar import (CameraCalibration, backproject, estimate_bars,
                               orange_mask)


def select_visible_grasp_targets(rgb, depth, calibration: CameraCalibration,
                                camera_position, camera_rotation,
                                radius_m=.015, end_margin_m=.025,
                                min_visible_length_m=.06, bin_width_m=.002):
    """Return only points inside continuously measured local colored rod spans.

    The default 25 mm end margin is a development choice, not a validated hook
    footprint/safety clearance. A future controller must set it from hardware.
    Visibility is assessed from ≥3 depth samples per 2 mm axis bin; no empty-bin
    closing. Depth holes/occlusion split intervals. Cylinder must explain the
    sampled surface better than a best-fit plane by at least 0.25 mm RMS, a
    frozen developer criterion that is not a calibrated identity probability.
    """
    v1 = estimate_bars(rgb, depth, calibration, camera_position, camera_rotation,
                       radius_m=radius_m)
    points, _ = backproject(depth, calibration, camera_position, camera_rotation,
                           orange_mask(rgb))
    diagnostics, targets = [], []
    for candidate_index, candidate in enumerate(v1["candidates"]):
        center = np.asarray(candidate["center_world_m"])
        axis = np.asarray(candidate["axis_world"])
        surface_ends = np.asarray(candidate["visible_surface_endpoints_world_m"])
        extent = (surface_ends - center) @ axis
        delta = points - center
        t = delta @ axis
        radial = np.linalg.norm(delta - t[:, None] * axis, axis=1)
        sample_mask = (t >= extent.min() - .002) & (t <= extent.max() + .002)
        sample_mask &= np.abs(radial - radius_m) <= .006
        observed = points[sample_mask]
        observed_t = t[sample_mask]
        diagnostic = {"candidate_index": candidate_index,
                      "v1_local_geometry_valid": candidate["valid"],
                      "measured_samples": int(len(observed)),
                      "target_identity_confirmed": False,
                      "load_permission": False, "visible_intervals": []}
        if len(observed) < 60:
            diagnostic["rejection"] = "insufficient measured samples"
            diagnostics.append(diagnostic)
            continue
        # Compare on the same central axial surface samples, excluding caps.
        q0, q1 = np.quantile(observed_t, [.08, .92])
        inner = observed[(observed_t >= q0) & (observed_t <= q1)]
        cloud_mean = inner.mean(axis=0)
        cov = np.cov(inner - cloud_mean, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        plane_rmse = float(np.sqrt(max(eigenvalues[0], 0)))
        inner_delta = inner - center
        inner_t = inner_delta @ axis
        cylinder_residual = np.linalg.norm(inner_delta - inner_t[:, None] * axis, axis=1) - radius_m
        cylinder_rmse = float(np.sqrt(np.mean(cylinder_residual ** 2)))
        model_margin = plane_rmse - cylinder_rmse
        cylinder_preferred = bool(model_margin >= .00025)
        diagnostic.update(plane_rmse_m=plane_rmse,
                          cylinder_rmse_m=cylinder_rmse,
                          cylinder_improvement_m=model_margin,
                          cylinder_model_preferred=cylinder_preferred)
        if not candidate["valid"] or not cylinder_preferred:
            diagnostic["rejection"] = ("v1 local geometry rejected" if not candidate["valid"]
                                       else "plane or ambiguous surface model")
            diagnostics.append(diagnostic)
            continue
        # Continuous measured axial support. No gap interpolation or hidden
        # endpoint recovery. All intervals are tagged as visible limits only.
        start = np.floor(observed_t.min() / bin_width_m) * bin_width_m
        bins = np.floor((observed_t - start) / bin_width_m).astype(int)
        counts = np.bincount(bins)
        occupied = counts >= 3
        padded = np.pad(occupied.astype(int), (1, 1))
        begins = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        for begin, end in zip(begins, ends):
            # Conservative pixel supported inner limits rather than edge bins.
            segment_t0 = start + (begin + 1) * bin_width_m
            segment_t1 = start + (end - 1) * bin_width_m
            length = segment_t1 - segment_t0
            if length < max(min_visible_length_m, 2 * end_margin_m):
                continue
            endpoints = np.stack((center + segment_t0 * axis,
                                  center + segment_t1 * axis))
            target = endpoints.mean(axis=0)
            interval = {"visible_endpoints_world_m": endpoints.tolist(),
                        "visible_length_m": float(length),
                        "endpoint_status": "measured_visible_limits_not_physical_rod_ends"}
            diagnostic["visible_intervals"].append(interval)
            targets.append({"candidate_index": candidate_index,
                            "target_world_m": target.tolist(),
                            "axis_world": axis.tolist(), **interval,
                            "end_margin_m": end_margin_m,
                            "local_visible_segment_supported": True,
                            "target_identity_confirmed": False,
                            "load_permission": False})
        if not diagnostic["visible_intervals"]:
            diagnostic["rejection"] = "no sufficiently long continuous visible span"
        diagnostics.append(diagnostic)
    return {"targets": targets, "candidate_diagnostics": diagnostics,
            "v1_prediction": v1,
            "load_permission": False,
            "scope": "local visible geometric proposals, identity and load require independent validation",
            "parameters": {"radius_m": radius_m, "end_margin_m": end_margin_m,
                           "min_visible_length_m": min_visible_length_m,
                           "bin_width_m": bin_width_m,
                           "min_samples_per_bin": 3,
                           "cylinder_rms_improvement_threshold_m": .00025}}
