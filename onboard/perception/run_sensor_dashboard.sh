#!/usr/bin/env bash
# ============================================================
# 启动传感器全览仪表盘（target + ball 所有传感器 -> browser）
#
# 订阅：
#   rt/target_state      (apriltag 目标)
#   rt/cam_ball_state    (相机球，当前未启用)
#   rt/ball_state        (lidar 直接输出，同时作为 lidar + fused 两栏)
#
# 注：lidar 直接发布到 rt/ball_state（无 fuser 中间层），
#     故 --lidar-topic 指向 rt/ball_state。
#     相机球未启用，cam 栏显示 wait 属正常。
#
# 用法：
#   bash onboard/perception/run_sensor_dashboard.sh
#   bash onboard/perception/run_sensor_dashboard.sh --port 8091 --range-m 5
#
# 默认端口：8091
# ============================================================
set -e
cd "$(dirname "$0")/../.."

source onboard/perception/setup_runtime_env.sh

ROS_DISTRO="${ROS_DISTRO:-humble}"
if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
    source /opt/ros/foxy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
fi

LIVOX_WS="${LIVOX_WS:-$HOME/ws_livox}"
source "$LIVOX_WS/install/setup.sh" 2>/dev/null || true
source "$LIVOX_WS/install/setup.bash" 2>/dev/null || true
source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_IFACE="${CYCLONEDDS_IFACE:-$(ip -o -4 addr show scope global 2>/dev/null | awk '/192\.168\.123\./ {print $2; exit}')}"
CYCLONEDDS_IFACE="${CYCLONEDDS_IFACE:-enP8p1s0}"
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$CYCLONEDDS_IFACE\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/debug/sensor_dashboard.py \
        --lidar-topic rt/ball_state \
        "$@"
