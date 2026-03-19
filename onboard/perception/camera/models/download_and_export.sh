#!/usr/bin/env bash
# =============================================================================
# download_and_export.sh — 一键下载 YOLO 模型并导出为 TensorRT engine
#
# 用法（在 RoboMimic_Deploy 根目录执行）：
#   bash onboard/perception/camera/models/download_and_export.sh
#   bash onboard/perception/camera/models/download_and_export.sh --model yolov8n
#
# 默认导出：yolo11m（精度最高，TRT 约 8.6ms）
# 可选模型：yolov8n（最快，TRT 约 4.9ms）| yolo11n | yolo11s | yolo11m
#
# 依赖：
#   - conda 环境 robomimic（含 ultralytics, torch）
#   - TensorRT 8.5.2（JetPack 5.1.2 自带）
#   - CUDA 12.1 compat stubs（/usr/local/cuda-12.1/compat）
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MODEL_DIR="$SCRIPT_DIR"
MODEL_NAME="${1:-yolo11m}"   # 默认 yolo11m；可传 yolov8n / yolo11n / yolo11s

# 解析 --model 参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift 2 ;;
        *)       shift ;;
    esac
done

MODEL_PT="$MODEL_DIR/${MODEL_NAME}.pt"
MODEL_ENGINE="$MODEL_DIR/${MODEL_NAME}.engine"
IMGSZ=320

echo "================================================================"
echo "[download_and_export] 目标模型: $MODEL_NAME"
echo "[download_and_export] 模型目录: $MODEL_DIR"
echo "================================================================"

# ── 1. 环境变量（TRT 需要 libnvcudla + tensorrt Python 绑定）─────────
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH:-}

# ── 2. 解锁 GPU 最大频率 ──────────────────────────────────────────────
echo "[download_and_export] Unlocking Jetson clocks..."
echo "123" | sudo -S nvpmodel -m 0 2>/dev/null || echo "  (nvpmodel skip)"
echo "123" | sudo -S jetson_clocks  2>/dev/null || echo "  (jetson_clocks skip)"

# ── 3. 下载 .pt（ultralytics 自动从 GitHub releases 下载）────────────
if [[ ! -f "$MODEL_PT" ]]; then
    echo "[download_and_export] 下载 ${MODEL_NAME}.pt ..."
    conda run -n robomimic --no-capture-output python -c "
from ultralytics import YOLO
import shutil, os
m = YOLO('${MODEL_NAME}.pt')   # 下载到当前目录
src = os.path.abspath('${MODEL_NAME}.pt')
dst = '${MODEL_PT}'
if src != dst:
    shutil.move(src, dst)
print('Downloaded:', dst)
"
else
    echo "[download_and_export] 已存在: $MODEL_PT，跳过下载"
fi

# ── 4. 导出 TRT engine ────────────────────────────────────────────────
if [[ -f "$MODEL_ENGINE" ]]; then
    echo "[download_and_export] 已存在: $MODEL_ENGINE"
    read -t 10 -p "  是否重新导出？[y/N] " ans || ans="N"
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "跳过导出。"; exit 0; }
fi

echo "[download_and_export] 导出 TRT engine（imgsz=$IMGSZ, FP16）..."
echo "  预计耗时：10-15 分钟（TensorRT kernel profiling）"

conda run -n robomimic --no-capture-output python -u -c "
import sys, os, numpy as np, shutil, time

# numpy 1.24 compatibility patch for TRT 8.5
if not hasattr(np, 'bool'):   np.bool   = bool
if not hasattr(np, 'int'):    np.int    = int
if not hasattr(np, 'float'):  np.float  = float
if not hasattr(np, 'object'): np.object = object

sys.path.insert(0, '/usr/lib/python3.8/dist-packages')
from ultralytics import YOLO

model_pt     = '${MODEL_PT}'
model_dir    = '${MODEL_DIR}'
imgsz        = ${IMGSZ}

print(f'Loading {model_pt} ...')
m = YOLO(model_pt)

print(f'Exporting to TRT FP16, imgsz={imgsz} ...')
t0   = time.time()
path = m.export(format='engine', device=0, half=True, imgsz=imgsz, workspace=4, verbose=False)
elapsed = time.time() - t0
print(f'Export done in {elapsed:.0f}s')

# ultralytics saves .engine next to .pt; move to model_dir if needed
src = os.path.abspath(path)
dst = os.path.join(model_dir, os.path.basename(src))
if src != dst:
    shutil.move(src, dst)
    print(f'Moved: {src} -> {dst}')
print(f'Engine: {dst}')
"

echo ""
echo "================================================================"
echo "[download_and_export] 完成！"
echo "  启动 ball_detector："
echo "    bash onboard/perception/camera/run.sh"
echo "================================================================"
