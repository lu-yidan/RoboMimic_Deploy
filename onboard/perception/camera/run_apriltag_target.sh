#!/usr/bin/env bash
# ============================================================
# 启动 chest camera apriltag_detector（含 GPU/Jetson 解锁）
# 默认使用 4-tag 板：
#   tag 0 -> target offset = (+14cm, +10cm, 0)
#   tag 1 -> target offset = (+14cm, -10cm, 0)
#   tag 2 -> target offset = (-14cm, -10cm, 0)
#   tag 3 -> target offset = (-14cm, +10cm, 0)
#
# 用法：
#   只检测 tag 0：
#   bash onboard/perception/camera/run_apriltag_target.sh --tag-id 0 --tag-size 0.12 --show
#
#   使用默认 4-tag 板（0/1/2/3）并融合估计主目标点：
#   bash onboard/perception/camera/run_apriltag_target.sh --show
#   bash onboard/perception/camera/run_apriltag_target.sh --tag-size 0.12 --show
#   bash onboard/perception/camera/run_apriltag_target.sh --tag-size 0.10
#
#   如需覆盖默认 4-tag 板配置，可显式传入你自己的 --tag-id / --tag-offset：
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

use_default_board_layout=1
for arg in "$@"; do
    case "$arg" in
        --tag-id|--tag-id=*|--tag-offset|--tag-offset=*)
            use_default_board_layout=0
            break
            ;;
    esac
done

default_board_args=(
    --tag-id 0
    --tag-id 1
    --tag-id 2
    --tag-id 3
    --tag-offset 0 0.14 0.10 0.00
    --tag-offset 1 0.14 -0.10 0.00
    --tag-offset 2 -0.14 -0.10 0.00
    --tag-offset 3 -0.14 0.10 0.00
)

cmd=(
    conda run -n robomimic --no-capture-output
    python -u onboard/perception/camera/apriltag_detector.py
)

if [[ "$use_default_board_layout" -eq 1 ]]; then
    echo "[run_apriltag_target.sh] Using default 4-tag board layout (ids 0/1/2/3, offsets in metres)."
    cmd+=("${default_board_args[@]}")
fi

cmd+=("$@")
"${cmd[@]}"
