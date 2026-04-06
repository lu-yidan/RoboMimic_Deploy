#!/usr/bin/env python3
"""fusion_node.py — Lightweight camera-priority fusion of two DDS ball_state topics.

Subscribes to:
  rt/cam_ball_state    (from camera/ball_detector.py, ~20-30 Hz)
  rt/lidar_ball_state  (from lidar/ball_detector.py,  ~10 Hz)

Publishes:
  rt/ball_state        (fused output, ~50 Hz, with source field)

This process has NO YOLO, NO RealSense, NO ROS — just DDS reads + writes.
CPU usage is negligible (~0.1%).
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from common.ball_state_dds import (
    BallState,
    BallStatePublisher,
    BallStateSubscriber,
    SOURCE_NONE,
    SOURCE_CAM,
    SOURCE_LIDAR,
)

CAM_STALE_MS = 200
LIDAR_STALE_MS = 400
FUSION_HZ = 50

_SOURCE_LABEL = {SOURCE_NONE: "none ", SOURCE_CAM: "cam  ", SOURCE_LIDAR: "lidar"}


def main():
    parser = argparse.ArgumentParser(description="Camera-priority fusion node")
    parser.add_argument("--cam-topic", default="rt/cam_ball_state")
    parser.add_argument("--lidar-topic", default="rt/lidar_ball_state")
    parser.add_argument("--out-topic", default="rt/ball_state")
    parser.add_argument("--fusion-hz", type=int, default=FUSION_HZ)
    parser.add_argument("--cam-stale-ms", type=int, default=CAM_STALE_MS)
    parser.add_argument("--lidar-stale-ms", type=int, default=LIDAR_STALE_MS)
    args = parser.parse_args()

    cam_sub = BallStateSubscriber(domain_id=0, topic_name=args.cam_topic)
    lidar_sub = BallStateSubscriber(domain_id=0, topic_name=args.lidar_topic)
    out_pub = BallStatePublisher(domain_id=0, topic_name=args.out_topic)

    cam_sub.start()
    lidar_sub.start()

    print(f"[fusion] cam={args.cam_topic}  lidar={args.lidar_topic}  out={args.out_topic}")
    print(f"[fusion] {args.fusion_hz} Hz | cam<{args.cam_stale_ms}ms  lidar<{args.lidar_stale_ms}ms")
    print("[fusion] Running. Ctrl-C to stop.")

    dt = 1.0 / args.fusion_hz
    frame_n = 0

    try:
        while True:
            time.sleep(dt)

            cam = cam_sub.latest()
            lidar = lidar_sub.latest()

            now_us = int(time.time() * 1e6)
            cam_age_ms = (now_us - cam.timestamp_us) / 1000 if cam.valid else float("inf")
            lidar_age_ms = (now_us - lidar.timestamp_us) / 1000 if lidar.valid else float("inf")

            if cam.valid and cam_age_ms < args.cam_stale_ms:
                x, y, z = cam.x, cam.y, cam.z
                src = SOURCE_CAM
                valid = True
            elif lidar.valid and lidar_age_ms < args.lidar_stale_ms:
                x, y, z = lidar.x, lidar.y, lidar.z
                src = SOURCE_LIDAR
                valid = True
            else:
                x = y = z = 0.0
                src = SOURCE_NONE
                valid = False

            out_pub.publish(x, y, z, valid=valid, source=src)

            if frame_n % 10 == 0:
                lbl = _SOURCE_LABEL[src]
                if valid:
                    print(
                        f"\r[fusion/{lbl}] pelvis=({x:+.3f},{y:+.3f},{z:+.3f})  "
                        f"cam_age={cam_age_ms:5.0f}ms  lidar_age={lidar_age_ms:5.0f}ms   ",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r[fusion/{lbl}] no ball  "
                        f"cam_age={cam_age_ms:5.0f}ms  lidar_age={lidar_age_ms:5.0f}ms   ",
                        end="",
                        flush=True,
                    )
            frame_n += 1

    except KeyboardInterrupt:
        print("\n[fusion] Stopped.")
    finally:
        cam_sub.stop()
        lidar_sub.stop()


if __name__ == "__main__":
    main()
