#!/usr/bin/env bash
# run_fused.sh — Launch the fused chest-camera + LiDAR ball detector.
#
# Usage:
#   bash onboard/perception/run_fused.sh               # default settings
#   bash onboard/perception/run_fused.sh --show        # open MJPEG stream on port 8080
#   bash onboard/perception/run_fused.sh --list-cameras
#   bash onboard/perception/run_fused.sh --chest-serial 123456789
#   bash onboard/perception/run_fused.sh --model onboard/perception/camera/models/yolo11m.engine
#
# Environment:
#   Requires ROS2 (foxy/humble) + ws_livox overlay to be sourced.
#   All additional args are forwarded to ball_detector_fused.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── ROS2 environment ─────────────────────────────────────────────────────────
ROS_DISTRO="${ROS_DISTRO:-foxy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
LIVOX_SETUP="${HOME}/yixuan/yichao-deploy/ws_livox/install/setup.sh"

if [[ -f "$ROS_SETUP" ]]; then
    # shellcheck source=/dev/null
    source "$ROS_SETUP"
    echo "[run_fused] sourced $ROS_SETUP"
else
    echo "[run_fused] WARNING: $ROS_SETUP not found — ROS2 may not be available"
fi

if [[ -f "$LIVOX_SETUP" ]]; then
    # shellcheck source=/dev/null
    source "$LIVOX_SETUP"
    echo "[run_fused] sourced $LIVOX_SETUP"
else
    echo "[run_fused] WARNING: $LIVOX_SETUP not found — /livox/lidar topic may not be available"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
echo "[run_fused] Starting ball_detector_fused.py  args: $*"
echo "[run_fused] HEAD_JOINT_ANGLE fixed at 2.3° (edit ball_detector_fused.py to change)"
echo "──────────────────────────────────────────────────────────────────────────"

exec python onboard/perception/ball_detector_fused.py "$@"
