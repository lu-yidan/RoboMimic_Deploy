#!/usr/bin/env bash
# ============================================================
# 启动 ZED X ball_detector（含 TensorRT + GPU 解锁）
#
# 用法（从 RoboMimic_Deploy 根目录运行）：
#   bash onboard/perception/zed-x/run.sh --show
#   bash onboard/perception/zed-x/run.sh --ball tennis --show
#   bash onboard/perception/zed-x/run.sh --ball soccer --show
#   bash onboard/perception/zed-x/run.sh --ball soccer-m --show   ← 默认，最准
#
# 浏览器打开 http://<robot-ip>:8080/stream 查看实时流
# ============================================================
set -e
cd "$(dirname "$0")/../../.."   # → RoboMimic_Deploy 根目录

# ── 1. 解锁 GPU / CPU 频率 ────────────────────────────────────────────────────
echo "[run.sh] 解锁 Jetson 性能模式（需要 sudo）..."
if sudo nvpmodel -m 0 2>/dev/null; then
    echo "  nvpmodel: MAXN 模式已启用"
else
    echo "  nvpmodel: 跳过（无权限或已设置）"
fi
if sudo jetson_clocks 2>/dev/null; then
    echo "  jetson_clocks: CPU/GPU/EMC 已解锁"
else
    echo "  jetson_clocks: 跳过"
fi

GPU_FREQ=$(cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>/dev/null | head -1)
CPU_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)
[ -n "$GPU_FREQ" ] && echo "[run.sh] GPU 当前频率: $((GPU_FREQ / 1000000)) MHz"
[ -n "$CPU_FREQ" ] && echo "[run.sh] CPU 当前频率: $((CPU_FREQ / 1000)) MHz"

# ── 2. 启动 ──────────────────────────────────────────────────────────────────
echo "[run.sh] 启动 ZED X ball_detector ..."
python3 -u onboard/perception/zed-x/ball_detector.py "$@"
