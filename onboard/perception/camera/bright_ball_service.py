#!/usr/bin/env python3
"""Bright grayscale camera ball service entrypoint."""

from __future__ import annotations

import sys

from onboard.perception.camera import apriltag_detector
from onboard.perception.camera.config import BRIGHT_BALL_ARGS, GRAY_V4L2_ARGS


def main() -> None:
    # Reuse the mature camera loop and bright-ball detector while disabling
    # AprilTag target publication by tracking an impossible tag id.
    sys.argv = [
        sys.argv[0],
        *GRAY_V4L2_ARGS,
        *BRIGHT_BALL_ARGS,
        "--tag-id", "-1",
        "--coast-frames", "0",
        *sys.argv[1:],
    ]
    apriltag_detector.main()


if __name__ == "__main__":
    main()
