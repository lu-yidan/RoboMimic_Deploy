# YOLO 模型目录

该目录存放 ball_detector 使用的 YOLO 模型文件。

## 文件说明

| 文件 | 大小 | 是否在 git | 说明 |
|------|------|-----------|------|
| `*.pt` | 6–39 MB | ❌ | PyTorch 权重，ultralytics 会自动下载 |
| `*.onnx` | 12–80 MB | ❌ | ONNX 中间格式，export 时自动生成 |
| `*.engine` | 8–42 MB | ❌ | **TRT 二进制，硬件绑定，不可跨机器** |
| `download_and_export.sh` | 2 KB | ✅ | 一键复现脚本 |

## 首次使用

```bash
# 在 RoboMimic_Deploy 根目录执行
bash onboard/perception/camera/models/download_and_export.sh
```

约需 **15 分钟**（下载 + TRT kernel profiling）。

## 模型精度与速度对比（Jetson Orin NX，imgsz=320，FP16）

| 后端 | 推理时间 | 整体 FPS | COCO mAP50-95 |
|------|---------|---------|---------------|
| yolov8n.pt | 17 ms | ~25 FPS | 37.3 |
| yolov8n.engine | 4.9 ms | **~40 FPS** | 37.3 |
| yolo11m.pt | 28 ms | ~18 FPS | 51.5 |
| **yolo11m.engine** | **8.6 ms** | **~35 FPS** | **51.5 (+38%)** |

## 为什么 .engine 不能直接复用？

TRT Engine 在编译时被"烧录"进当前机器的：
- **GPU 架构**（Jetson Orin NX = Ampere，Jetson Nano = Maxwell，不互通）
- **CUDA 版本**（JetPack 5.1.2 = CUDA 11.4）
- **TRT 版本**（8.5.2）

换机器、升级 JetPack 都需要重新执行 `download_and_export.sh`。
