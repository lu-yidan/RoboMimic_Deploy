#!/usr/bin/env bash
# run.sh — 启动 LiDAR 球检测（自动启动 Livox MID360 驱动）
#
# 用法：
#   bash onboard/perception/lidar/run.sh
#   bash onboard/perception/lidar/run.sh --dds-topic rt/lidar_ball_state
#
# Ctrl-C 退出时自动关闭 Livox 驱动和检测器。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── ROS2 + Livox 环境 ────────────────────────────────────────────────────────
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

cd "$ROOT_DIR"

# ── 清理 ─────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[run_lidar] Stopping..."
    pkill -f "msg_MID360_launch" 2>/dev/null || true
    pkill -f "livox_ros_driver2_node" 2>/dev/null || true
    sleep 1
    pkill -9 -f "livox_ros_driver2_node" 2>/dev/null || true
    echo "[run_lidar] Done."
}
trap cleanup EXIT INT TERM

# ── 启动 Livox MID360 驱动 ───────────────────────────────────────────────────
echo "[run_lidar] Starting Livox MID360 driver..."
ros2 launch livox_ros_driver2 msg_MID360_launch.py > /tmp/livox_driver.log 2>&1 &
sleep 3
echo "[run_lidar] Livox driver started"

# ── 启动 LiDAR 检测器（前台） ────────────────────────────────────────────────
echo "[run_lidar] Starting lidar ball detector..."
echo "──────────────────────────────────────────────────────────────────────────"

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/lidar/ball_detector.py "$@"
