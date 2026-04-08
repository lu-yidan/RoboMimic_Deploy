#!/usr/bin/env bash
# run_fused.sh — 双进程融合：camera + lidar 独立进程，fusion 节点合并输出
#
# 架构：
#   1. Livox MID360 驱动       → /livox/lidar topic
#   2. camera/ball_detector.py → DDS rt/cam_ball_state   (~20-30 Hz)
#   3. lidar/ball_detector.py  → DDS rt/lidar_ball_state (~10 Hz)
#   4. fusion_node.py          → DDS rt/ball_state       (~50 Hz, 相机优先)
#
# 用法：
#   bash onboard/perception/camera_lidar/run_fused.sh
#   bash onboard/perception/camera_lidar/run_fused.sh --show

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAM_LOG=/tmp/cam_detector.log
LIDAR_LOG=/tmp/lidar_detector.log
LIVOX_LOG=/tmp/livox_driver.log

# ── 1. 解锁 GPU / CPU 频率 ──────────────────────────────────────────────────
echo "[run_fused] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0  2>/dev/null || true
echo "123" | sudo -S jetson_clocks  2>/dev/null || true

# ── 2. TensorRT ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH:-}

# ── 3. ROS2 + Livox 环境 ────────────────────────────────────────────────────
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

cd "$ROOT_DIR"

# ── 4. 清理函数：按名称杀掉所有相关进程 ──────────────────────────────────────
cleanup() {
    echo ""
    echo "[run_fused] Stopping all processes..."
    # Kill by specific command patterns (catches conda wrappers + python children)
    pkill -f "ball_detector.py.*--dds-topic rt/cam_ball_state" 2>/dev/null || true
    pkill -f "ball_detector.py.*--dds-topic rt/lidar_ball_state" 2>/dev/null || true
    pkill -f "fusion_node.py" 2>/dev/null || true
    pkill -f "msg_MID360_launch" 2>/dev/null || true
    pkill -f "livox_ros_driver2_node" 2>/dev/null || true
    sleep 1
    # Force kill any survivors
    pkill -9 -f "ball_detector.py.*--dds-topic rt/cam_ball_state" 2>/dev/null || true
    pkill -9 -f "ball_detector.py.*--dds-topic rt/lidar_ball_state" 2>/dev/null || true
    pkill -9 -f "fusion_node.py" 2>/dev/null || true
    pkill -9 -f "msg_MID360_launch" 2>/dev/null || true
    pkill -9 -f "livox_ros_driver2_node" 2>/dev/null || true
    echo "[run_fused] Logs: $CAM_LOG  $LIDAR_LOG  $LIVOX_LOG"
    echo "[run_fused] Done."
}
trap cleanup EXIT INT TERM

# ── 5. Livox MID360 驱动（PointCloud2 模式，Python lidar 订阅更快） ─────────
echo "[run_fused] Starting Livox MID360 driver..."
ros2 run livox_ros_driver2 livox_ros_driver2_node \
    --ros-args \
    -p xfer_format:=0 \
    -p multi_topic:=0 \
    -p data_src:=0 \
    -p publish_freq:=10.0 \
    -p output_data_type:=0 \
    -p frame_id:=livox_frame \
    -p user_config_path:=/home/unitree/yixuan/yichao-deploy/ws_livox/src/livox_ros_driver2/config/MID360_config.json \
    -p cmdline_input_bd_code:=livox0000000001 > "$LIVOX_LOG" 2>&1 &
sleep 3
echo "[run_fused] Livox driver started"

# ── 6. LiDAR 检测进程 → rt/lidar_ball_state ──────────────────────────────────
echo "[run_fused] Starting lidar detector  (log: $LIDAR_LOG)"
conda run -n robomimic --no-capture-output \
    python -u onboard/perception/lidar/ball_detector.py \
        --msg-type pc2 \
        --dds-topic rt/lidar_ball_state > "$LIDAR_LOG" 2>&1 &
sleep 1

# ── 7. Camera 检测进程 → rt/cam_ball_state ───────────────────────────────────
echo "[run_fused] Starting camera detector  (log: $CAM_LOG)"
conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/ball_detector.py \
        --dds-topic rt/cam_ball_state "$@" > "$CAM_LOG" 2>&1 &
sleep 2

# ── 8. Fusion 节点（前台） ───────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "[run_fused] Camera  → rt/cam_ball_state    (log: $CAM_LOG)"
echo "[run_fused] LiDAR   → rt/lidar_ball_state  (log: $LIDAR_LOG)"
echo "[run_fused] Fusion  → rt/ball_state  (output below)"
echo "[run_fused] Ctrl-C to stop all."
echo "══════════════════════════════════════════════════════════════════════════"

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera_lidar/fusion_node.py
