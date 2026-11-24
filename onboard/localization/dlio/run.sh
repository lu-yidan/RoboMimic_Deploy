#!/usr/bin/env bash
# Sensor/estimator processes only. User starts this onsite; never starts a policy.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${CONDA_PREFIX:-}" ]] || { echo 'Use a clean system/ROS shell outside conda' >&2; exit 1; }
export ACP_DLIO_SITE="${ACP_DLIO_SITE:-$HOME/.config/robomimic/acp-dlio/site.json}"
WS="${ACP_DLIO_WS:-$HOME/acp_dlio_ws}"
[[ -f "$WS/install/setup.bash" ]] || { echo 'Run build.sh first' >&2; exit 1; }
# Never kill an existing Livox owner: its SDK uses the same physical UDP ports.
if pgrep -f '[/]livox_ros_driver2_node|[m]sg_MID360_launch' >/dev/null; then
  echo 'A Livox driver is already running. Stop its own launch before using ACP DLIO.' >&2
  exit 1
fi
set +u
source /opt/ros/foxy/setup.bash
source "$WS/install/setup.bash"
set -u
source "$HERE/transport_env.sh"
export OMP_NUM_THREADS="${ACP_OMP_THREADS:-2}"
exec ros2 launch "$HERE/acp.launch.py"
