#!/usr/bin/env bash
# ============================================================
# 启动 chest camera apriltag_detector（含 GPU/Jetson 解锁）
# 默认目标 tag id：0
#
# 用法：
#   bash onboard/perception/camera/run_apriltag_target.sh
#   bash onboard/perception/camera/run_apriltag_target.sh --show
#   bash onboard/perception/camera/run_apriltag_target.sh --tag-id 5 --tag-size 0.08
#   bash onboard/perception/camera/run_apriltag_target.sh --tag-id 5 --tag-id 8
#   bash onboard/perception/camera/run_apriltag_target.sh \
#       --tag-id 5 --tag-id 8 --tag-size 0.10 \
#       --tag-offset 5 0.20 0.00 0.00 \
#       --tag-offset 8 -0.20 0.00 0.00
#
#   其中 --tag-offset TAG_ID DX DY DZ 表示：
#     从该 tag 中心出发，在 tag 自身坐标系里走 (DX, DY, DZ) 后到达公共目标点。
#     方向定义相对 tag 本身，不是相机系/机器人系：
#       +X = tag 右边
#       +Y = tag 上边
#       +Z = 垂直纸面向外
#
#   上面的例子里：
#     tag 5 在公共目标点左侧 20 cm，所以目标点在 tag 5 的右边 -> DX = +0.20
#     tag 8 在公共目标点右侧 20 cm，所以目标点在 tag 8 的左边 -> DX = -0.20
# ============================================================
set -e
cd "$(dirname "$0")/../../.."

echo "[run_apriltag_target.sh] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0 2>/dev/null || true
echo "123" | sudo -S jetson_clocks 2>/dev/null || true

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH:-}

source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh 2>/dev/null || true

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/apriltag_detector.py "$@"
