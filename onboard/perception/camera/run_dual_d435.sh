#!/usr/bin/env bash
# ============================================================
# 启动双 D435 ball_detector_dual（含 TensorRT + GPU 解锁）
#
# 用法：
#   bash onboard/perception/camera/run_dual.sh
#   bash onboard/perception/camera/run_dual.sh --show
#       → 浏览器打开 http://192.168.123.164:8080 查看双路画面
#   bash onboard/perception/camera/run_dual.sh --list-cameras
#       → 打印两台 D435 的序列号后退出
#   bash onboard/perception/camera/run_dual.sh --head-serial 12345 --chest-serial 67890
#       → 手动指定哪台是 head / chest（推荐，避免 USB 枚举顺序变化）
#
# 外参更新：编辑 onboard/perception/camera/camera_to_base.py
#   → 修改 _CHEST_XYZ 和 _CHEST_RPY
# ============================================================
set -e
cd "$(dirname "$0")/../../.."   # → RoboMimic_Deploy 根目录

# ── 1. 解锁 GPU / CPU 频率 ────────────────────────────────
echo "[run_dual.sh] Unlocking Jetson clocks..."
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

# ── 4b. conda init for non-interactive shells (e.g. SSH) ──
source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

# ── 4c. Use CycloneDDS — Unitree's intended RMW ───────────
# FastDDS (the Foxy default) pre-allocates ~14 GB of shared memory on a 16 GB
# system, immediately triggering the OOM killer before any Python code runs.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'

# ── 5. 启动双相机检测器 ───────────────────────────────────
conda run -n robomimic --no-capture-output \
    python -u onboard/perception/camera/ball_detector_dual.py "$@"
