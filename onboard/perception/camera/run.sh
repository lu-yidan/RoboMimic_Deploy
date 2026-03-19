#!/usr/bin/env bash
# ============================================================
# 启动 camera ball_detector（含 TensorRT + GPU 解锁）
# 默认模型：yolo11m（自动检测 .engine，优先使用 TRT）
#
# 用法：bash onboard/perception/camera/run.sh [额外参数]
#   bash onboard/perception/camera/run.sh --show
#   bash onboard/perception/camera/run.sh --model yolo11n.pt   # 更快
#   bash onboard/perception/camera/run.sh --model yolov8n.pt   # 回退
#   bash onboard/perception/camera/run.sh --imgsz 224 --width 424 --height 240
# ============================================================
set -e
cd "$(dirname "$0")/../../.."   # → RoboMimic_Deploy 根目录

# ── 1. 解锁 GPU / CPU 频率 ────────────────────────────────
echo "[run.sh] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0  2>/dev/null || true
echo "123" | sudo -S jetson_clocks  2>/dev/null || true

# ── 2. 加载 TensorRT 所需的 libnvcudla ──────────────────
#    TRT 8.5.2 on JetPack 5.1.2 needs CUDA 12.1 compat stub
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:${LD_LIBRARY_PATH:-}

# ── 3. 让 conda Python 能 import tensorrt ───────────────
export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH:-}

# ── 4. ROS2 + livox 环境 ─────────────────────────────────
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

# ── 5. 启动 ───────────────────────────────────────────────
conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/ball_detector.py "$@"
