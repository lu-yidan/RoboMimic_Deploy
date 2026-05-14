#!/usr/bin/env bash
# ============================================================
# 启动 YOLO 全类别网页调试器（RealSense + MJPEG）
#
# 用法：
#   bash onboard/perception/camera/run_yolo_web.sh
#   bash onboard/perception/camera/run_yolo_web.sh --list-cameras
#   bash onboard/perception/camera/run_yolo_web.sh --camera-serial 244622070281
#   bash onboard/perception/camera/run_yolo_web.sh --conf-threshold 0.15
#   bash onboard/perception/camera/run_yolo_web.sh --class-filter bottle,cup,vase
#
# 默认网页端口：8081
# 启动后访问终端打印出的 http://<robot-ip>:8081/stream
# ============================================================
set -e
cd "$(dirname "$0")/../../.."

echo "[run_yolo_web.sh] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0 2>/dev/null || true
echo "123" | sudo -S jetson_clocks 2>/dev/null || true

source onboard/perception/setup_runtime_env.sh

source /opt/ros/foxy/setup.bash 2>/dev/null || true
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true
source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/debug/yolo_web_viewer.py "$@"
