#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/foxy/setup.bash
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ACP_ROS_DOMAIN_ID:-37}"
export ROS_LOCALHOST_ONLY=1
exec rviz2 -d "$HERE/config/acp.rviz"
