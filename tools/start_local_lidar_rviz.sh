#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
RVIZ_CONFIG="${RVIZ_CONFIG:-${REPO_DIR}/onboard/perception/lidar/ball_detector.rviz}"

detect_robot_iface() {
  ip -o -4 addr show scope global 2>/dev/null \
    | awk '/192\.168\.123\./ {print $2; exit}'
}

CYCLONEDDS_IFACE="${CYCLONEDDS_IFACE:-${ROBOT_IFACE:-$(detect_robot_iface)}}"
CYCLONEDDS_IFACE="${CYCLONEDDS_IFACE:-enp6s0}"

if ! ip link show "$CYCLONEDDS_IFACE" >/dev/null 2>&1; then
  echo "[rviz] ERROR: interface '$CYCLONEDDS_IFACE' does not exist on this machine." >&2
  echo "[rviz] Available interfaces:" >&2
  ip -o link show | awk -F': ' '{print "  " $2}' >&2
  exit 1
fi

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
  echo "[rviz] ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found." >&2
  exit 1
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${CYCLONEDDS_IFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"

echo "[rviz] CycloneDDS interface: ${CYCLONEDDS_IFACE}"
echo "[rviz] ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
echo "[rviz] Checking /ball_detector topics..."
if ! timeout 8 ros2 topic list | grep -q '^/ball_detector/cloud_all$'; then
  echo "[rviz] ERROR: cannot see /ball_detector/cloud_all." >&2
  echo "[rviz] Check that lidar detector is running with --show and both machines are on 192.168.123.x." >&2
  exit 1
fi

exec rviz2 -d "$RVIZ_CONFIG"
