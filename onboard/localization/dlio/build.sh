#!/usr/bin/env bash
# Build software only; no devices, nodes, sudo, or existing workspaces touched.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${ACP_DLIO_WS:-$HOME/acp_dlio_ws}"
[[ "${ROS_DISTRO:-foxy}" == foxy ]] || { echo 'Use a clean ROS2 Foxy shell' >&2; exit 1; }
[[ -z "${CONDA_PREFIX:-}" ]] || { echo 'Run outside conda (system Python / ROS2)' >&2; exit 1; }
set +u  # ROS Foxy setup files reference optional, unset shell variables.
source /opt/ros/foxy/setup.bash
set -u
export CMAKE_BUILD_PARALLEL_LEVEL="${ACP_BUILD_JOBS:-2}"
export MAKEFLAGS="-j${ACP_BUILD_JOBS:-2}"
mkdir -p "$WS/src"
/usr/bin/python3 "$HERE/prepare_sources.py" "$WS"
cd "$WS"
colcon build --executor sequential --packages-select livox_ros_driver2 direct_lidar_inertial_odometry \
  --cmake-args -DROS_EDITION=ROS2 -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_CXX_FLAGS="-O2" --parallel-workers 1
echo "Software built at $WS/install; no sensor or control processes started."
