# ZED X 球检测模块

> 硬件：**Unitree G1**，ZED X GMSL 相机安装于**胸部**，NVIDIA Jetson AGX Orin  
> 模型：**YOLO11m → TensorRT FP16 engine**（~66 FPS @ imgsz=320）  
> 当前阶段：MJPEG 验证（body 系坐标）；DDS 发布为 P1

---

## 文件结构

```
onboard/perception/zed-x/
├── ball_detector.py   ← 主程序：ZED X + YOLO TRT，三线程，MJPEG 输出
├── zed_to_base.py     ← 坐标变换：ZED X body 系 → G1 pelvis 系（胸部外参待标定）
├── run.sh             ← 启动脚本（GPU 解锁 + python3）
├── README.md          ← 本文档
└── weights/           ← 符号链接 → detect_tennis_zed/weights/
    ├── yolo11n.engine   （COCO，快速验证）
    ├── yolo11m.engine   ← 默认使用
    └── yolov8_best.engine （自训网球，class 0）
```

---

## 快速启动

```bash
cd ~/workspace/RoboMimic_Deploy

# 足球检测（默认，最准）
bash onboard/perception/zed-x/run.sh --ball soccer-m --show

# 足球检测（更快，精度稍低）
bash onboard/perception/zed-x/run.sh --ball soccer --show

# 网球检测
bash onboard/perception/zed-x/run.sh --ball tennis --show
```

浏览器打开 `http://<robot-ip>:8080/stream` 查看实时流。

---

## 球类配置

| `--ball`   | 模型            | class | 置信阈值 | 推理速度  |
|------------|-----------------|-------|----------|-----------|
| `tennis`   | yolov8_best     | 0     | 0.30     | ~90 fps   |
| `soccer`   | yolo11n         | 32    | 0.20     | ~100 fps  |
| `soccer-m` | yolo11m（默认） | 32    | 0.15     | ~66 fps   |

---

## 与 D435 版本（camera/）的主要差异

| 项目 | D435（camera/） | ZED X（zed-x/） |
|------|----------------|-----------------|
| 深度对齐 | 需要 color→depth 像素映射 | 深度已对齐左目，直接采样 |
| 深度格式 | `uint16 × depth_scale` | `float32` 直接是米，NaN=无效 |
| 深度模式 | 硬件固定 | `PERFORMANCE`（SDK 5.x 中 QUALITY/NEURAL 占 GPU）|
| 相机初始化 | `rs.pipeline()` | `sl.Camera().open()`，自动降级 SVGA→HD1200→HD1080 |
| 坐标变换 | `camera_to_base.py`（头部，已标定） | `zed_to_base.py`（胸部，**外参待标定**）|

---

## 坐标输出

当前输出为 **ZED X body 系**（X-forward, Y-left, Z-up，相对相机光心）：

```
[BALL ] body=(+1.234,-0.012,+0.045)m  conf=0.72  YOLO=66fps
```

**P1 完成后**将输出 pelvis 系坐标（需标定 `zed_to_base._ZED_XYZ/_ZED_RPY`）。

---

## P1 待完成

- [ ] 标定胸部安装外参（`zed_to_base._ZED_XYZ` / `_ZED_RPY`）
- [ ] 加 ROS2 `_JointListener` 订阅 `/lowstate`（参考 `camera/ball_detector.py`）
- [ ] 加 `BallStatePublisher` DDS 发布 `rt/ball_state`
- [ ] 在 `yolo_worker` 中调用 `zed_to_base.transform_point_zed_to_base()`

---

## 检测精度说明

`yolo11m` 使用 COCO 预训练权重，class 32（sports ball）在真实场景下置信度偏低。  
如需更高检测率，建议在足球专用数据集上微调（Roboflow 有公开足球数据集）。

---

## 常见问题

**ZED 无法打开 / CAMERA STREAM FAILED**  
重新插拔 GMSL 线缆后重启，或执行：
```bash
sudo systemctl restart zed_x_daemon.service
```

**PERFORMANCE is deprecated 警告**  
ZED SDK 5.x 废弃了 PERFORMANCE/QUALITY，但实测 PERFORMANCE 仍走传统立体匹配，不占 GPU。  
NEURAL 模式会与 YOLO TRT 争 GPU，导致帧率从 ~66fps 降至 ~21fps。

**YOLO 推理慢（首帧 ~3秒）**  
正常，TRT engine 首次反序列化需要初始化 cuDNN。后续帧 ~6ms。
