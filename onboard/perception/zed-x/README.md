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
如需更高检测率，建议使用以下数据集微调（按推荐优先级排序）：

### 推荐训练数据集

#### ⭐ 最推荐：RoboCup 机器人视角数据集

**1. TORSO-21**（最适合 ZED X 胸部安装场景）
- 地址：https://github.com/bit-bots/TORSO_21_dataset
- 下载：https://data.bit-bots.de/TORSO-21/
- 内容：**10,464 张真实图 + 24,000 张仿真图**，6,081 个球标注
- 视角：人形机器人胸部/头部摄像头，地面近距离侧视 ← **和 ZED X 胸部完全一致**
- 格式：YAML，自带 YOLO 转换脚本
- 训练：`cd weights && python3 prepare_torso21.py`（全自动，约 1 小时）

**2. Hamburg Bit-Bots Ball Dataset 2018**（数量最多）
- 地址：https://robocup.informatik.uni-hamburg.de/en/bit-bots-ball-dataset-2018/
- 内容：**35,327 张训练图 + 14,886 张负样本**
- 视角：机器人地面近距离，4 种球类型，各角度

#### Roboflow Universe（YOLO 格式，直接可用）

**3. Soccer Ball Dataset**（4,377 张）
- 地址：https://universe.roboflow.com/queendev9516-gmail-com/soccer-ball-oa830
- 下载：`roboflow download queendev9516-gmail-com/soccer-ball-oa830/1 -f yolov8`

**4. yolo_soccer_ball_tracker**（250 张，含预训练模型）
- 地址：https://universe.roboflow.com/ball-tracker/yolo_soccer_ball_tracker
- 最小，快速验证用

> **注意**：`football-ball-detection-rejhg`（Roboflow，1237 张）为广播俯视视角，  
> 与胸部相机侧视场景不匹配，**不推荐**用于本项目。

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
