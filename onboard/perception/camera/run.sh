#!/usr/bin/env bash
# ============================================================
# 启动 camera ball_detector（含 TensorRT + GPU 解锁）
# 默认模型：models/yolo11m（自动检测 .engine，优先使用 TRT）
#
# 首次使用请先运行：
#   bash onboard/perception/camera/models/download_and_export.sh
#
# 用法：bash onboard/perception/camera/run.sh [额外参数]
#   bash onboard/perception/camera/run.sh --show
#   bash onboard/perception/camera/run.sh --model models/yolov8n.pt  # 最快
#   bash onboard/perception/camera/run.sh --model models/yolo11n.pt  # 均衡
#   bash onboard/perception/camera/run.sh --imgsz 224 --width 424 --height 240
# ============================================================
set -e
cd "$(dirname "$0")/../../.."   # → RoboMimic_Deploy 根目录

# ── 1. 解锁 GPU / CPU 频率 ────────────────────────────────
echo "[run.sh] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0  2>/dev/null || true
echo "123" | sudo -S jetson_clocks  2>/dev/null || true

# ── 2. 加载 TensorRT 所需的 libnvcudla ──────────────────
#    Also sets JetPack 6 TensorRT Python path and CycloneDDS 0.10.5.
source onboard/perception/setup_runtime_env.sh

# ── 4. ROS2 + livox 环境 ─────────────────────────────────
source /opt/ros/foxy/setup.bash 2>/dev/null || true
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'

# ── 5. 启动 ───────────────────────────────────────────────
conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/ball_detector.py "$@"
