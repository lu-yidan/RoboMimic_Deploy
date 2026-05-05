#!/usr/bin/env bash
# ============================================================
# 启动传感器全览仪表盘（target + ball 所有传感器 -> browser）
#
# 订阅：
#   rt/target_state      (apriltag 目标)
#   rt/cam_ball_state    (相机球)
#   rt/lidar_ball_state  (雷达球)
#   rt/ball_state        (融合球)
#
# 用法：
#   bash onboard/perception/run_sensor_dashboard.sh
#   bash onboard/perception/run_sensor_dashboard.sh --port 8091 --range-m 5
#
# 默认端口：8091（ball_web_viewer 使用 8090，互不冲突）
# ============================================================
set -e
cd "$(dirname "$0")/../.."

source /opt/ros/foxy/setup.bash 2>/dev/null || true
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true
source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/debug/sensor_dashboard.py "$@"
