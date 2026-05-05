#!/usr/bin/env python3
"""Ball-state sensor fusion: lidar + chest camera → single rt/ball_state.

Subscribes to:
  rt/lidar_ball_state  — MID360 lidar ball detector (0.3–2 m, w/ Kalman)
  rt/cam_ball_state    — chest D455 YOLO ball detector (1–5 m+)

Publishes:
  rt/ball_state        — fused authoritative ball position for deploy_policy.py

Fusion strategy (range-based priority):
  dist < 0.3 m   : lidar blind zone — use cam if available, else Kalman prediction
  0.3 – 2.0 m    : prefer lidar real detection; fall back to cam if lidar invalid
  2.0 – 5.0 m+   : camera only (lidar out of range)
  no valid source : publish valid=False

Lidar publishes valid=False + source=SOURCE_LIDAR when it is in Kalman-prediction
mode (ball occluded near feet).  The fuser uses that prediction in the close range
where the camera is also unreliable, but overrides it with camera if the camera has
a real detection and the ball is further away.

Usage (from repo root):
    bash onboard/perception/run_ball_fuser.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from common.ball_state_dds import (
    BallState,
    BallStatePublisher,
    BallStateSubscriber,
    SOURCE_CAM,
    SOURCE_LIDAR,
    SOURCE_NONE,
)

# ── Thresholds ──────────────────────────────────────────────────────────────
LIDAR_MAX_RANGE  = 2.0   # m — beyond this, lidar is unreliable / out of range
CAM_MIN_RANGE    = 0.8   # m — closer than this, camera depth is noisy
FUSE_HZ          = 50    # Hz — fuser publish rate
STALE_MS         = 300   # ms — treat sensor data as stale beyond this

# SOURCE_FUSED is not a new constant in the IDL — we reuse SOURCE_CAM / SOURCE_LIDAR
# to indicate which sensor was authoritative, so deploy_policy.py can distinguish.


def _dist(s: BallState) -> float:
    return (s.x ** 2 + s.y ** 2 + s.z ** 2) ** 0.5


def fuse(lidar: BallState, cam: BallState) -> BallState:
    """Return the best BallState given the two sensor readings."""
    lidar_real   = lidar.valid == 1 and lidar.source == SOURCE_LIDAR
    lidar_kalman = lidar.valid == 0 and lidar.source == SOURCE_LIDAR
    cam_valid    = cam.valid == 1 and cam.source == SOURCE_CAM

    d_lidar = _dist(lidar) if (lidar_real or lidar_kalman) else 0.0
    d_cam   = _dist(cam)   if cam_valid else 0.0

    # ── Case 1: lidar real detection ──────────────────────────────────────
    if lidar_real:
        if d_lidar <= LIDAR_MAX_RANGE:
            # Lidar in range — authoritative.
            return BallState(
                timestamp_us=lidar.timestamp_us,
                x=lidar.x, y=lidar.y, z=lidar.z,
                valid=1, source=SOURCE_LIDAR,
            )
        # Lidar real but surprisingly far (shouldn't happen) — trust cam if available
        if cam_valid:
            return BallState(
                timestamp_us=cam.timestamp_us,
                x=cam.x, y=cam.y, z=cam.z,
                valid=1, source=SOURCE_CAM,
            )

    # ── Case 2: lidar in Kalman-prediction mode ───────────────────────────
    if lidar_kalman:
        if d_lidar < CAM_MIN_RANGE:
            # Ball very close — camera unreliable, use Kalman prediction.
            return BallState(
                timestamp_us=lidar.timestamp_us,
                x=lidar.x, y=lidar.y, z=lidar.z,
                valid=0, source=SOURCE_LIDAR,   # valid=0 signals "predicted, not observed"
            )
        # Ball in overlap zone (CAM_MIN_RANGE – LIDAR_MAX_RANGE) and lidar is predicting.
        if cam_valid:
            return BallState(
                timestamp_us=cam.timestamp_us,
                x=cam.x, y=cam.y, z=cam.z,
                valid=1, source=SOURCE_CAM,
            )
        # No camera — propagate Kalman prediction anyway.
        return BallState(
            timestamp_us=lidar.timestamp_us,
            x=lidar.x, y=lidar.y, z=lidar.z,
            valid=0, source=SOURCE_LIDAR,
        )

    # ── Case 3: lidar invalid (SOURCE_NONE or stale) ─────────────────────
    if cam_valid:
        return BallState(
            timestamp_us=cam.timestamp_us,
            x=cam.x, y=cam.y, z=cam.z,
            valid=1, source=SOURCE_CAM,
        )

    # ── Case 4: nothing valid ─────────────────────────────────────────────
    return BallState(valid=0, source=SOURCE_NONE)


def main():
    parser = argparse.ArgumentParser(
        description="Fuse lidar + camera ball estimates → rt/ball_state"
    )
    parser.add_argument("--lidar-topic",  default="rt/lidar_ball_state")
    parser.add_argument("--cam-topic",    default="rt/cam_ball_state")
    parser.add_argument("--output-topic", default="rt/ball_state")
    parser.add_argument("--hz", type=float, default=FUSE_HZ)
    parser.add_argument("--lidar-max-range", type=float, default=LIDAR_MAX_RANGE)
    parser.add_argument("--cam-min-range",   type=float, default=CAM_MIN_RANGE)
    args = parser.parse_args()

    global LIDAR_MAX_RANGE, CAM_MIN_RANGE
    LIDAR_MAX_RANGE = args.lidar_max_range
    CAM_MIN_RANGE   = args.cam_min_range

    lidar_sub = BallStateSubscriber(topic_name=args.lidar_topic)
    cam_sub   = BallStateSubscriber(topic_name=args.cam_topic)
    out_pub   = BallStatePublisher(topic_name=args.output_topic)

    lidar_sub.start()
    cam_sub.start()

    print(f"[fuser] Subscribing: {args.lidar_topic}  +  {args.cam_topic}")
    print(f"[fuser] Publishing:  {args.output_topic}  @ {args.hz:.0f} Hz")
    print(f"[fuser] Ranges: lidar ≤ {LIDAR_MAX_RANGE}m | cam ≥ {CAM_MIN_RANGE}m")

    dt = 1.0 / args.hz
    try:
        while True:
            t0 = time.monotonic()

            lidar = lidar_sub.latest()
            cam   = cam_sub.latest()
            result = fuse(lidar, cam)

            out_pub.publish(
                result.x, result.y, result.z,
                valid=result.valid == 1,
                source=result.source,
            )

            # Brief status line
            src_str = {SOURCE_LIDAR: "lidar", SOURCE_CAM: "cam", SOURCE_NONE: "none"}.get(
                result.source, "?"
            )
            v_str = "valid" if result.valid else "pred " if result.source == SOURCE_LIDAR else "inval"
            print(
                f"\r[fuser] src={src_str:<5} {v_str}  "
                f"xyz=({result.x:+.2f},{result.y:+.2f},{result.z:+.2f})"
                f"  lidar={'R' if lidar.valid and lidar.source==SOURCE_LIDAR else 'K' if lidar.source==SOURCE_LIDAR else '-'}"
                f"  cam={'V' if cam.valid and cam.source==SOURCE_CAM else '-'}",
                end="", flush=True,
            )

            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[fuser] Stopped.")


if __name__ == "__main__":
    main()
