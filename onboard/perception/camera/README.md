# Camera 感知模块

> 硬件：**Unitree G1**，机载电脑 NVIDIA Jetson Orin NX 16 GB，JetPack 5.1.2  
> 相机：**Intel RealSense D435I**（USB 3.0）× 1（单相机）或 × 2（双相机）  
> 模型：**YOLO11m → TensorRT FP16 engine**（单相机 ~35 FPS；双相机每路 ~24 FPS）  
> DDS Topic：`rt/ball_state`（与 Lidar 方案相同，可互换）

---

## 目录

1. [文件结构](#一文件结构)
2. [快速启动](#二快速启动)
   - [单相机（默认）](#21-单相机默认)
   - [双相机](#22-双相机)
3. [浏览器预览 --show](#三浏览器预览---show)
4. [检测流程](#四检测流程)
   - [单相机流程](#41-单相机流程)
   - [双相机流程](#42-双相机流程)
5. [坐标变换与外参](#五坐标变换与外参)
   - [头部相机（已标定）](#51-头部相机已标定)
   - [胸部相机（待标定）](#52-胸部相机待标定)
6. [参数速查](#六参数速查)
7. [模型选择](#七模型选择)
8. [深度精度说明](#八深度精度说明)
9. [常见问题](#九常见问题)

---

## 一、文件结构

```
onboard/perception/camera/
├── apriltag_detector.py    ← 胸前相机主检测器：AprilTag + 可选球检测 → DDS
│                              --ball       YOLO 球检测（后台线程，需 depth 流）
│                              --ball-hsv   HSV 色块球检测（主循环，仅需彩色流）
├── ball_detector.py        ← 单相机球检测（头部/外置 D435 + YOLO → rt/ball_state）
├── ball_detector_dual.py   ← 双相机球检测：2×D435 + 共享 YOLO → rt/ball_state
├── target_detector.py      ← 胸前相机 YOLO target 检测 → rt/target_state
├── camera_to_base.py       ← 坐标变换：相机系 → pelvis 系（含胸部外参）
├── run.sh                  ← 单相机球检测启动脚本
├── run_dual.sh             ← 双相机球检测启动脚本
├── run_target.sh           ← 胸前相机 YOLO target 检测启动脚本
├── run_apriltag_target.sh  ← 胸前相机 AprilTag 检测启动脚本（支持 --ball/--ball-hsv）
├── debug/
│   ├── hsv_tuner.py              ← HSV 参数交互调试工具（滑条 + 4 格预览）
│   ├── generate_apriltag_template.py ← 生成可打印 A4 AprilTag 模板
│   ├── target_state_echo.py      ← 打印 rt/target_state 最新值
│   └── target_extrinsics_eval.py ← 统计 target 位姿均值/方差，辅助外参标定
├── README.md               ← 本文档
├── TROUBLESHOOTING.md      ← 性能优化全记录（GIL/DMA/TRT/Color-Depth 映射等）
└── models/
    ├── download_and_export.sh  ← 一键下载 .pt + 导出 TRT engine
    ├── README.md               ← 模型精度/速度对比
    └── .gitignore              ← 排除 *.pt / *.onnx / *.engine
```

相关感知模块（`onboard/perception/` 层）：

| 文件 | 说明 |
|------|------|
| `ball_fuser.py` | 融合 `rt/lidar_ball_state` + `rt/cam_ball_state` → `rt/ball_state` |
| `run_ball_fuser.sh` | 启动融合器 |
| `run_sensor_dashboard.sh` | 启动传感器全览仪表盘（port 8091） |
| `run_ball_web_viewer.sh` | 启动单 topic 球位置可视化（port 8090） |
| `debug/sensor_dashboard.py` | 订阅 4 路 DDS topic，浏览器显示 target + 各 sensor 球位置 |
| `debug/ball_web_viewer.py` | 订阅单 topic 球位置 |

---

## 二、快速启动

### 首次使用：下载模型并导出 TRT engine（约 15 分钟，只需一次）

```bash
bash onboard/perception/camera/models/download_and_export.sh
```

---

### 2.1 单相机（默认）

将 D435 插入 USB 3.0 接口，在项目根目录执行：

```bash
bash onboard/perception/camera/run.sh
```

可选参数：

```bash
bash onboard/perception/camera/run.sh --show                        # 开启浏览器预览 (port 8080)
bash onboard/perception/camera/run.sh --model models/yolov8n.pt     # 切换轻量模型（更快）
bash onboard/perception/camera/run.sh --imgsz 224 --width 424 --height 240  # 低分辨率模式
```

正常运行输出：

```
[INFO] TensorRT engine found: models/yolo11m.engine
[INFO] YOLO single-inference: 8.6 ms  (≈ 116 FPS upper bound)
[BALL ] pelvis=(+0.823, -0.012, -0.673)  surf=0.71m  ctr=0.82m  gate=0.03m  YOLO=35.2fps
[COAST] pelvis=(+0.821, -0.011, -0.672)  surf=0.71m  ctr=0.82m  gate=0.03m  YOLO=35.2fps
[     ] no ball  YOLO=35.2fps
```

| 状态标志 | 含义 |
|----------|------|
| `BALL`   | 本帧 YOLO 检测到球，发布 valid=1 |
| `COAST`  | 本帧漏检，沿用上帧位置，最多保持 10 帧，发布 valid=1 |
| `(空)`   | 超过 10 帧未检测到，发布 valid=0 |

---

### 2.1a 实时球位置网页可视化

如果只想看球相对于机器人的位置，不需要打开 RViz。先启动任意会发布 `rt/ball_state` 的检测节点，再另开一个终端：

```bash
bash onboard/perception/run_ball_web_viewer.sh
```

默认端口是 **8090**，启动后访问终端打印出的 `http://<robot-ip>:8090/`。页面显示：
- 俯视图：`+X` 为机器人前方，`+Y` 为机器人左侧，机器人固定在原点。
- 侧视图：`+X` 为机器人前方，`+Z` 为上方。
- 数值面板：`x/y/z`、平面距离、3D 距离、DDS sample age、valid/stale 状态和 source。
- 最近轨迹：保留最近几秒的有效球位置，方便观察抖动和漂移。

常用参数：

```bash
bash onboard/perception/run_ball_web_viewer.sh --port 8090
bash onboard/perception/run_ball_web_viewer.sh --topic rt/ball_state
bash onboard/perception/run_ball_web_viewer.sh --range-m 4.0 --history-sec 5.0
```

注意：该页面读取的是已经变换到 pelvis/base body frame 的 `rt/ball_state`，所以第一版只画机器人基座和坐标轴，不需要完整 MuJoCo ghost 姿态。

---

### 2.1b 胸前相机 target 检测

在项目根目录执行：

```bash
bash onboard/perception/camera/run_target.sh --target-class bottle
```

常用调试命令：

```bash
bash onboard/perception/camera/run_target.sh --show
python onboard/perception/camera/debug/target_state_echo.py
python onboard/perception/camera/debug/target_extrinsics_eval.py
```

说明：
- `target_detector.py` 与球检测逻辑分离，方便后续更换胸前相机角度、位置或目标类别。
- `debug/` 下脚本仅用于联调、观察和外参校准，不参与正式感知链路。

---

### 2.1c 胸前相机 AprilTag 检测

在项目根目录执行：

```bash
bash onboard/perception/camera/run_apriltag_target.sh --tag-id 0 --tag-size 0.12
```

带网页预览：

```bash
bash onboard/perception/camera/run_apriltag_target.sh --tag-id 0 --tag-size 0.12 --show
```

多标签候选：

```bash
bash onboard/perception/camera/run_apriltag_target.sh --tag-id 5 --tag-id 8 --tag-size 0.10
```

多标签共享同一个目标点：

```bash
# 例子：tag 5 在公共目标点左侧 20 cm，tag 8 在右侧 20 cm
bash onboard/perception/camera/run_apriltag_target.sh \
    --tag-id 5 --tag-id 8 --tag-size 0.10 \
    --tag-offset 5 0.20 0.00 0.00 \
    --tag-offset 8 -0.20 0.00 0.00
```

说明：
- `--tag-size` 单位是米，填写的是黑色方形 tag 本体边长，不是整张 A4 纸大小。
- 默认情况下将 `tag 中心` 作为目标点发布到 `rt/target_state`。
- 若提供 `--tag-offset TAG_ID DX DY DZ`，则改为发布该 tag 坐标系下的偏移目标点，适合把多个 tag 映射到同一个公共踢球目标。
- `--tag-offset` 单位是米，原点在 tag 中心；`+X` 指向 tag 右边，`+Y` 指向 tag 上边，`+Z` 为 tag 平面法向。`+Z` 不等价于“离机器人更远”，它会随 tag 朝向一起旋转；对同一平面上的公共目标点，通常填 `DZ=0`。
- 检测成功时，`class_id` 字段复用为 `tag_id`，便于下游和调试脚本继续沿用现有接口。
- AprilTag 方案更适合贴了已知 tag 的受控目标；如果需要识别任意 `bottle`、`suitcase`，仍建议使用 YOLO 方案。

输出语义：
- 发布到 `rt/target_state` 的坐标已经完成 `RealSense optical -> chest body -> pelvis` 的变换；其中胸前外参默认使用 `camera_to_base.py` 里的 `_CHEST_XYZ / _CHEST_RPY`，也可以通过 `--chest-xyz / --chest-rpy` 覆盖。
- 检测成功时发布 `pelvis` 系目标点，`valid=1`，`class_id=tag_id`，`confidence` 为当前 tag 置信度，`source=SOURCE_CHEST_CAMERA`。
- 短时丢失目标但仍在 `--coast-frames` 范围内时，继续发布上一次 `pelvis` 坐标，但 `valid=0`，便于下游区分“沿用旧值”和“当前帧重新检测到”。
- 超过 `--coast-frames` 后仍未检测到 tag，则发布 `(0, 0, 0)`、`valid=0`、`class_id=-1`、`confidence=0`。
- `camera_to_base.py` 里的胸前外参当前仍是代码内置值，若后续要直接用于控制，建议先完成实测或标定。

生成打印模板：

```bash
# 默认生成 PDF，适合直接打印
conda run -n robomimic --no-capture-output \
    python onboard/perception/camera/debug/generate_apriltag_template.py \
    --family tag36h11 --tag-id 0 --tag-size-mm 120

# 如需同时保留一份 PNG 预览图，额外加 --also-png
conda run -n robomimic --no-capture-output \
    python onboard/perception/camera/debug/generate_apriltag_template.py \
    --family tag36h11 --tag-id 0 --tag-size-mm 80 --also-png
```

打印建议：
- 首选 `tag36h11`
- 起步尺寸建议 `120 mm`
- 脚本现在默认输出 `PDF`；如需留一份预览图，可额外加 `--also-png`
- 若显式输出 `PNG`，脚本也会写入 `DPI` 元数据；但最终打印仍优先直接使用生成的 `PDF`
- 打印时选择 `100%` / `actual size`，禁止 `fit to page` / `shrink to printable area`
- 用尺子实测黑色外框边长；检测代码里的 `--tag-size` 应填写这个黑框边长
- 打印后最好贴到硬纸板上，避免纸张弯曲带来的姿态抖动

---

### 2.1d 胸前相机同时检测 AprilTag + 球（推荐方式）

`apriltag_detector.py` 支持在同一进程、同一彩色流上**同时**做 AprilTag 目标检测和球检测，无需额外进程或第二台相机。

#### 方案 A：HSV 色块匹配（推荐，无 YOLO 依赖，30fps）

适合：球有明显区别于背景的颜色（如蓝/紫色花纹），不需要 GPU 推理。

```bash
# 基本启动
bash onboard/perception/camera/run_apriltag_target.sh --ball-hsv --show

# 调试时加 --ball-hsv-show-mask，在 MJPEG 画面上画出检测到的圆
bash onboard/perception/camera/run_apriltag_target.sh --ball-hsv --show --ball-hsv-show-mask

# 指定 HSV 范围（用 hsv_tuner.py 离线调好后粘贴）
bash onboard/perception/camera/run_apriltag_target.sh --ball-hsv \
  --ball-hsv-h-low 95 --ball-hsv-h-high 135 \
  --ball-hsv-s-min 30 --ball-hsv-v-min 130
```

**HSV 参数调试**（不需要机器人在线）：

```bash
# 对图片目录调参（n/p 切换图片，s 保存参数）
conda run -n robomimic python onboard/perception/camera/debug/hsv_tuner.py \
    onboard/perception/camera/debug/test/

# 或连接 D455 实时调参
conda run -n robomimic python onboard/perception/camera/debug/hsv_tuner.py --camera
```

界面显示 4 格面板：原图 / HSV mask / 合并后 / 检测结果，7 个滑条实时调整。按 `s` 将参数保存为 `hsv_params.txt`，直接粘贴到启动命令中。

**深度说明**：HSV 模式**不开 depth 流**，球的深度用视觉测距（`depth = fx × R_ball / r_px`），精度约 ±10%，1–5m 范围内满足控制需求。这也是为什么 HSV 模式能稳定保持 30fps——不受 USB 带宽限制。

**默认 HSV 范围**（蓝/紫色足球花纹，H=90–150, S≥30, V≥130）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--ball-hsv-h-low` | 90 | HSV Hue 下界（OpenCV 0-180） |
| `--ball-hsv-h-high` | 150 | HSV Hue 上界 |
| `--ball-hsv-s-min` | 40 | 饱和度下界（越高越严格） |
| `--ball-hsv-v-min` | 50 | 亮度下界（提高可减少暗背景误检） |
| `--ball-hsv-topic` | `rt/cam_ball_state` | 发布 topic |

**调参建议**：
- 先用 `hsv_tuner.py` 在机器人运行环境的照片上确认 HSV 范围
- 误检多：提高 `--ball-hsv-v-min`（如 130）或 `--ball-hsv-s-min`（如 50）
- 检测不到：降低 `--ball-hsv-s-min`（如 25）或扩大 Hue 范围

#### 方案 B：YOLO（适合多类别目标，需 GPU）

```bash
bash onboard/perception/camera/run_apriltag_target.sh --ball --show
```

YOLO 在后台线程运行，需要 depth 流，会额外占用 GPU 和 ~10ms GIL/帧。TRT engine 需提前导出（见 `models/download_and_export.sh`）。

| 对比项 | `--ball-hsv` | `--ball` (YOLO) |
|--------|-------------|-----------------|
| 算法 | HSV mask + 圆形验证 | YOLO11m TRT |
| 是否需要 depth 流 | 否 | 是 |
| 对 FPS 影响 | 几乎无（~2ms 主循环） | 轻微（~5ms GIL） |
| 对球颜色的依赖 | 强（需调参） | 弱（COCO 泛化） |
| 常见球的适配 | 蓝紫花纹足球效果好 | 标准黑白足球效果好 |

#### 方案 A+B 同时运行

```bash
bash onboard/perception/camera/run_apriltag_target.sh --ball --ball-hsv --show
```

两种检测共用一次 depth 拷贝，发布到同一个 `rt/cam_ball_state` topic（互相覆盖，不需要 fuser）。一般不建议同时启用，除非用于对比调试。

---

### 2.1e 传感器全览仪表盘（sensor dashboard）

相比 MJPEG 视频叠加，仪表盘是**纯数据** web UI，不闪烁，可在电脑浏览器上看机器人上的实时状态。

```bash
# 启动仪表盘（通常已在 tmux 左下角自动启动）
bash onboard/perception/run_sensor_dashboard.sh

# 然后在电脑浏览器访问
http://<robot-ip>:8091/
```

订阅 4 个 DDS topic，显示在同一俯视图画布上：

| 符号 | 颜色 | 含义 |
|------|------|------|
| 金色菱形 | `--target` | AprilTag 目标（rt/target_state） |
| 蓝色圆 | `cam` | 相机球估计（rt/cam_ball_state） |
| 橙色圆 | `lidar` | 雷达球估计（rt/lidar_ball_state） |
| 绿色圆 | `fused` | 融合球（rt/ball_state） |

右侧面板显示每个传感器的 x/y/z、距离、消息频率（Hz）、数据时效。

---

### 2.2 双相机

将两台 D435 分别插入 USB 3.0 接口（建议用 USB Hub 或主板上的两个独立 USB 控制器），在项目根目录执行：

```bash
# 第一步：查看两台相机的序列号
bash onboard/perception/camera/run_dual.sh --list-cameras

# 第二步：启动（默认第一台=head，第二台=chest）
bash onboard/perception/camera/run_dual.sh

# 推荐：明确指定序列号，防止 USB 枚举顺序变化
bash onboard/perception/camera/run_dual.sh \
    --head-serial  334622071404 \
    --chest-serial 244622070281 \
    --show

# 带预览
bash onboard/perception/camera/run_dual.sh --show
```

正常运行输出：

```
[INFO] Found 2 RealSense devices:
  [0] serial=117322071089  Intel RealSense D435I
  [1] serial=334622071404  Intel RealSense D435I
[INFO] HEAD  camera → serial 117322071089
[INFO] CHEST camera → serial 334622071404
[INFO] YOLO single-inference: 8.6 ms  (each camera ~24 FPS with dual-worker)
[head/BALL ] pelvis=(+0.823, -0.012, -0.673)  surf=0.71m  ctr=0.82m  conf=0.91  fps=24.1
[chest/     ] no ball  fps=23.8
```

任意一台相机检测到球即立即发布到 `rt/ball_state`，两台同时检测到时**最后写入的结果优先**（±1ms 内发生的竞争在机器人控制周期内无实际影响）。

---

## 三、浏览器预览 `--show`

启动时加 `--show` 参数后，程序在 **port 8080** 开启 HTTP 服务。

这里的 `--show` 是相机画面 MJPEG 预览；如果要看球在机器人坐标系中的实时位置，请使用上面的 `run_ball_web_viewer.sh`（默认 port 8090）。

| URL | 内容 |
|-----|------|
| `http://192.168.123.164:8080` | 主页面：两路画面左右并排显示 |
| `http://192.168.123.164:8080/stream/head` | head 相机 MJPEG 单流 |
| `http://192.168.123.164:8080/stream/chest` | chest 相机 MJPEG 单流 |

单相机 `--show` 时直接访问 `http://192.168.123.164:8080` 即可。

**画面说明：**
- 绿色框：本帧 YOLO 检测到球，框内标注置信度
- 橙色框：YOLO 漏检（COAST），框内标注剩余 coast 帧数
- 左下角黄字：球心在 pelvis 系坐标 + 深度距离
- 左上角白字：相机名称 + 实时 FPS

> **注意**：`--show` 开启后每帧额外进行一次 JPEG 编码（~2ms），对检测 FPS 影响极小。

---

## 四、检测流程

### 4.1 单相机流程

```
D435（color 60Hz + depth 90Hz，USB 3.0）
         │
         ▼ [主线程] pipeline.wait_for_frames()  ← 释放 GIL，~16ms 阻塞
raw frameset
         │
         ▼ [YOLO 线程] get_color/depth_frame + .copy()
color(640×480×3 BGR)   depth_arr(640×480 uint16)
         │
         ▼ cv2.resize → (320×320)，model.predict()  ← GPU TRT ~8.6ms
BBox 中心 (cx, cy)（640×480 坐标系）
         │
         ▼ Color→Depth 像素映射（修正 FOV 差异 + 基线视差，见第八章）
depth 采样 → 中位数 → + BALL_RADIUS(0.115m) = depth_m
         │
         ▼ rs2_deproject_pixel_to_point(color_intrin, [cx,cy], depth_m)
p_optical（光学系：Z前，X右，Y下）
         │
         ▼ optical_to_body()
p_cam（body系：X前，Y左，Z上）
         │
         ▼ EMA 平滑（α=0.6，跳变门限 0.6m）
p_cam_smooth
         │
         ▼ transform_point_camera_to_base(q_wy, q_wr, q_wp, q_head)
球心（pelvis body 系，单位 m）
         │
         ▼ DDS publish "rt/ball_state"  valid=1
```

---

### 4.2 双相机流程

```
D435-head（USB 3.0）          D435-chest（USB 3.0）
      │                               │
      ▼ [capture_loop-head]           ▼ [capture_loop-chest]
  buf_head（最新帧）              buf_chest（最新帧）
      │                               │
      ▼ [yolo_worker-head]            ▼ [yolo_worker-chest]
      │  ← ─ ─ ─ gpu_lock ─ ─ ─ → │   ← GPU 推理串行，各自独立处理
      │                               │
  depth → p_cam → EMA            depth → p_cam → EMA
      │                               │
  transform_head_to_base()       transform_chest_to_base()
      │                               │
      └───────────┬───────────────────┘
                  │  任意一路检测到即发布
                  ▼ dds_lock → DDS publish "rt/ball_state"
```

**双相机速度影响**：

| 配置 | GPU 推理耗时 | 每路有效 FPS |
|------|-------------|-------------|
| 单相机 | 8.6 ms | ~35 FPS |
| 双相机（共享 GPU 锁） | 8.6 ms × 2 轮流 | ~24 FPS/路 |

两路 worker 互相填补对方的非 GPU 时间（深度采样 ~2ms、EMA ~0.1ms、DDS 发布 ~0.5ms），实际吞吐约为 50 次推理/秒，每路约 24 FPS。

---

## 五、坐标变换与外参

所有变换实现在 `camera_to_base.py`。

### 5.1 头部相机（已标定）

运动学链（来自 URDF `g1_sysid_23dof.urdf`）：

```
pelvis
  └─ waist_yaw_joint    (Rz, q_wy)  t=[0, 0, 0]
      └─ waist_roll_joint  (Rx, q_wr)  t=[-0.0039635, 0, 0.044]
          └─ waist_pitch_joint (Ry, q_wp)  t=[0, 0, 0]
              └─ head_joint   (Ry, q_head=0.593412)  t=[0.0039635, 0, 0.3159]
                  └─ head_camera_joint (fixed)
                         xyz=[0.0448353662, 0.01, 0.1219029938]
                         rpy=[0.0119142, 0.8377475, 0.0053045]
```

运行时使用实时关节角 `q_wy / q_wr / q_wp`（从 `/lowstate` 读取），`q_head` 固定为 URDF 默认值 `0.593412 rad`。

### 5.2 胸部相机（待标定）

运动学链（截止到 `waist_pitch_joint`，之后为待测固定关节）：

```
pelvis
  └─ waist_yaw_joint    (Rz, q_wy)  t=[0, 0, 0]
      └─ waist_roll_joint  (Rx, q_wr)  t=[-0.0039635, 0, 0.044]
          └─ waist_pitch_joint (Ry, q_wp)  t=[0, 0, 0]
              └─ chest_camera_joint (fixed)  ← ⚠️ 占位值，需要测量/标定
                     xyz=[0.10, 0.00, 0.12]  (X前 / Y左 / Z上，单位 m)
                     rpy=[0.00, 0.30, 0.00]  (roll / pitch / yaw，单位 rad)
```

**如何更新外参**：打开 `camera_to_base.py`，修改文件顶部的两个变量：

```python
_CHEST_XYZ = [0.10, 0.00, 0.12]   # TODO: 实测后替换（单位 m，waist_pitch_link 系）
_CHEST_RPY = (0.00, 0.30, 0.00)   # TODO: 标定后替换（roll, pitch, yaw，单位 rad）
```

**测量方法（手工）**：
1. 将机器人静止站立（全关节 q=0），在 waist_pitch_link 坐标原点（约腰部中心）打标记
2. 用尺子量取相机光学中心相对该点的 x（向前）/ y（向左）/ z（向上）偏移，填入 `_CHEST_XYZ`
3. 用量角器或 CAD 模型量取安装俯仰角，填入 `_CHEST_RPY` 的 pitch 分量

**标定方法（棋盘格，精度更高）**：
1. 在机器人全关节 q=0 状态下，用棋盘格做手眼标定
2. 将标定结果转换为相对 waist_pitch_link 的 SE(3) 变换，替换上述两个变量

---

## 六、参数速查

### 代码常量（修改源文件后重启生效）

| 常量 | 位置 | 默认值 | 含义 |
|------|------|--------|------|
| `CONF_THRESHOLD` | 两个检测器 | 0.3 | YOLO 置信度阈值 |
| `DEPTH_SAMPLE_RADIUS` | 两个检测器 | 5 px | 深度采样半径，采样区 (2R+1)² |
| `DEPTH_MIN / MAX` | 两个检测器 | 0.1 / 10.0 m | 有效深度范围 |
| `BALL_RADIUS` | 两个检测器 | 0.115 m | 球半径（深度传感器测前表面，需加此值到球心） |
| `EMA_ALPHA` | 两个检测器 | 0.6 | EMA 平滑系数（越大响应越快，越小越平滑） |
| `EMA_GATE` | 两个检测器 | 0.6 m | 跳变重置门限 |
| `COAST_FRAMES` | 两个检测器 | 10 帧 | YOLO 漏检时保持位置的最大帧数 |
| `VALID_HOLD_SEC` | `ball_detector_dual.py` | 0.5 s | 双相机模式：两路都静默超过此时长才发布 valid=0 |
| `SPORTS_BALL_CLASS_ID` | 两个检测器 | 32 | COCO 类别 ID，不可修改 |

### 命令行参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--model` | `models/yolo11m.pt` | 模型路径（自动检测同名 `.engine` 优先） |
| `--imgsz` | 320 | YOLO 输入边长（像素），越小越快越粗 |
| `--width / --height` | 640 / 480 | 相机采集分辨率 |
| `--show` | 关闭 | 开启 HTTP MJPEG 预览（port 8080） |
| `--head-serial` | 第一台枚举 | 双相机：指定 head 相机序列号 |
| `--chest-serial` | 第二台枚举 | 双相机：指定 chest 相机序列号 |
| `--list-cameras` | — | 双相机：打印所有 D435 序列号后退出 |

---

## 七、模型选择

| 后端 | 推理时间 | 整体 FPS（单相机） | COCO mAP50-95 | 推荐场景 |
|------|---------|---------|---------------|---------|
| `yolov8n.engine` | 4.9 ms | ~40 FPS | 37.3 | 最高速，远距离精度略低 |
| `yolov8n.pt` | 17 ms | ~25 FPS | 37.3 | 无 GPU 解锁时备用 |
| **`yolo11m.engine`（默认）** | **8.6 ms** | **~35 FPS** | **51.5** | **精度/速度均衡，推荐** |
| `yolo11m.pt` | 28 ms | ~18 FPS | 51.5 | TRT 导出失败时备用 |

切换模型：

```bash
bash onboard/perception/camera/run.sh --model models/yolov8n.pt
```

> TRT engine 与硬件绑定，换机器或升级 JetPack 后需重新执行 `download_and_export.sh`。

---

## 八、深度精度说明

D435 的 Color 和 Depth 传感器**不共光心**（基线约 -14.5 mm，FOV 也不同），直接将 Color 像素坐标用于查找 `depth_arr` 会引入最大约 70 px / 18 cm 的横向误差。

代码中采用两步视差修正：

```
Step 1 — 内参映射（去 Color FOV，加 Depth FOV）
  ndcx = (cx - ppx_c) / fx_c
  dx0  = ndcx × fx_d + ppx_d              ← 不含视差

Step 2 — 基线视差修正
  dx   = ndcx × fx_d + ppx_d + tx/Z × fx_d  ← tx = -14.5mm，Z = Step1 粗深度
```

详细推导见 `TROUBLESHOOTING.md` 第九章。

---

## 九、常见问题

### Q1：`RuntimeError: Couldn't resolve requests`（RealSense pipeline.start 失败）

常见原因：同时请求 **color@60 Hz + depth@90 Hz** 等**帧率不匹配**的组合，多数 D435/D435I 固件无法同时满足。

当前代码已改为优先 **60/60 Hz**，失败则依次尝试 **30/30**、**15/15 Hz**。若仍失败，检查 USB 3.0、线缆、`rs-enumerate-devices`。

---

### Q2：`ModuleNotFoundError: No module named 'tensorrt'`

必须通过 `run.sh` / `run_dual.sh` 启动，**不能直接 `python ball_detector.py`**。

`run.sh` 设置了：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH
```

这两个路径让 conda Python 能找到系统级 TensorRT 绑定（`libnvinfer`）和 `import tensorrt`。

---

### Q3：两台相机开机后枚举顺序颠倒

USB 枚举顺序不稳定，建议**固定序列号**：

```bash
# 先查序列号
bash onboard/perception/camera/run_dual.sh --list-cameras

# 启动时明确指定
bash onboard/perception/camera/run_dual.sh \
    --head-serial 117322071089 \
    --chest-serial 334622071404
```

也可以将序列号固定写入启动脚本，避免每次输入。

---

### Q4：双相机模式 chest 坐标偏差很大

胸部相机外参当前为**占位值**，仅用于调试流程，实测前输出的 pelvis 坐标**不可用于控制**。请先完成第五章 5.2 节的外参标定。

---

### Q5：YOLO 无法检测到球（all `no ball`）

1. `--show` 查看画面，确认球在视野内
2. 降低 `CONF_THRESHOLD`（如 `0.15`）临时测试
3. 确认模型包含 COCO class 32（sports ball）—— YOLO 模型需在含 COCO 数据集上训练

---

### Q6：深度值为 0 或球心 z 坐标异常

1. 检查深度流是否正常：`--show` 下观察球周围是否有深度孔洞
2. 调整 `DEPTH_SAMPLE_RADIUS`（适当增大）
3. 检查 USB 是否接 3.0 口（2.0 口带宽不足，Depth 流可能掉帧）

---

### Q7：性能优化参考

详见 `TROUBLESHOOTING.md`：
- 第三章：每步耗时分析
- 第五章：从 1 FPS 到 35 FPS 的完整优化历程
- 第八章：进一步优化方向（多进程/DLA）
