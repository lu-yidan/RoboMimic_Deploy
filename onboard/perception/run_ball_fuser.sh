#!/usr/bin/env bash
# ============================================================
# 启动 ball_fuser — 融合雷达 + 胸部相机球位置估计
#
# 订阅：
#   rt/lidar_ball_state  (lidar ball_detector 输出)
#   rt/cam_ball_state    (apriltag_detector --ball 输出)
#
# 发布：
#   rt/ball_state        (deploy_policy.py 读取)
#
# 用法：
#   bash onboard/perception/run_ball_fuser.sh
#   bash onboard/perception/run_ball_fuser.sh --lidar-max-range 1.5
# ============================================================
set -e
cd "$(dirname "$0")/../.."

source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/ball_fuser.py "$@"
