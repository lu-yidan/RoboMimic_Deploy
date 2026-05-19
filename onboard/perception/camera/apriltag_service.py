#!/usr/bin/env python3
"""AprilTag target service entrypoint.

This keeps the existing detector implementation as the compatibility backend,
but gives launch scripts a clear AprilTag-only service name.
"""

from __future__ import annotations

import sys

from onboard.perception.camera import apriltag_detector


def main() -> None:
    sys.argv = [sys.argv[0], *sys.argv[1:]]
    apriltag_detector.main()


if __name__ == "__main__":
    main()
