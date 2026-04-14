#!/usr/bin/env bash
# ============================================================
# 启动 chest camera target_detector（含 TensorRT + GPU 解锁）
# 默认目标类别：bottle
#
# 用法：
#   bash onboard/perception/camera/run_target.sh
#   bash onboard/perception/camera/run_target.sh --show
#   bash onboard/perception/camera/run_target.sh --target-class bottle
#   bash onboard/perception/camera/run_target.sh --camera-serial 123456789
#   bash onboard/perception/camera/run_target.sh --chest-xyz 0.13 0.0 0.06
# ============================================================
set -e
cd "$(dirname "$0")/../../.."

echo "[run_target.sh] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0 2>/dev/null || true
echo "123" | sudo -S jetson_clocks 2>/dev/null || true

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH:-}

source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/target_detector.py "$@"
