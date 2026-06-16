# Camera Ball Detector — 排障与性能调优全记录

> 硬件：**Unitree G1**，机载电脑 NVIDIA Jetson Orin NX 16 GB，JetPack 5.1.2  
> 相机：**Intel RealSense D435**（USB 3.0）  
> 模型：**YOLO11m → TensorRT FP16 engine**（ultralytics；yolov8n 也已保留作备用）  
> 最终帧率：**~35 FPS**（YOLO11m TRT 8.6ms；精度 mAP 51.5，从最初 1 FPS 提升 35 倍）

---

## 0. 现场速查手册（AprilTag + IR 白球）

这一节记录 2026-05-14 在 G1 机载电脑上调 chest camera 的实际处理流程。现场先看这里，后面章节是历史性能调优细节。

### 0.1 当前推荐启动命令

```bash
pkill -f "onboard/perception/camera/apriltag_detector.py" 2>/dev/null || true
./onboard/perception/camera/run_apriltag_gray_ball.sh --show
```

默认脚本会做这些事：

- 使用 V4L2 读取 RealSense 的 IR/灰度 UVC 流：`/dev/video3` + `GREY`
- AprilTag 发布到 `rt/target_state`
- 白球 bright detector 发布到 `rt/cam_ball_state`
- 浏览器预览端口：`8080`
- 默认四 tag 板：id `0/1/2/3`

如果需要彩色画面：

```bash
./onboard/perception/camera/run_apriltag_color_ball.sh --show
```

两个接口不要混用：

- 灰度/IR：`run_apriltag_gray_ball.sh`，默认 `/dev/video3` + `GREY`
- 彩色：`run_apriltag_color_ball.sh`，默认 `/dev/video4` + `YUYV`

打开预览：

```text
http://192.168.123.164:8080/stream
```

如果端口被占用：

```bash
./onboard/perception/camera/run_apriltag_target.sh --show --show-port 8081
```

### 0.2 现在这路相机到底是什么模式？

灰度入口看到的是 RealSense 的 **IR/灰度 UVC 流**；彩色入口看到的是普通 V4L2 彩色流。

表现：

- 画面是灰度/黑白
- AprilTag 对比度很好，识别稳定
- 白色足球、反光点阵球会很亮
- HSV 颜色检测没有意义，因为没有真实颜色

两条入口当前都使用 `--ball-bright`，不是 YOLO，也不是 HSV。灰度入口通常对白球/反光点阵球更敏感；彩色入口方便人工观察。

### 0.3 黑屏但程序不崩

现象：

```text
RealSense pipeline OK
MJPEG 正常打开
画面全黑
```

原因：`librealsense` 的 RGB color stream 在这台 Jetson/G1 上可能返回黑帧。

解决：改用 V4L2 UVC 节点。

检查节点：

```bash
v4l2-ctl --list-devices
for d in /dev/video*; do
  echo "--- $d"
  v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | sed -n '1,60p'
done
```

常见可用结果：

- `/dev/video0`：Depth，格式 `Z16`
- `/dev/video3`：IR/灰度 UVC，格式 `GREY/UYVY`
- `/dev/video4`：彩色 UVC，格式 `YUYV`

脚本默认：

```bash
--color-backend v4l2 --v4l2-device /dev/video3 --v4l2-fourcc GREY --v4l2-fps 30
```

OpenCV 在 Jetson 上用字符串 `/dev/video2` 可能打不开，所以代码内部会把它转成数字 index `2`。

### 0.4 `failed to set power state`

常见栈：

```text
RuntimeError: failed to set power state
```

原因通常是：

- 同一个 RealSense 被旧进程占用
- 或者重复访问 RealSense device handle 时 librealsense/USB 状态不稳定

先清旧进程：

```bash
pkill -f "apriltag_detector.py" 2>/dev/null || true
pkill -f "ball_detector.py" 2>/dev/null || true
```

确认相机还在：

```bash
lsusb | grep -i RealSense
```

代码中已经避免在选择 serial 后重复 `get_info()`，减少这个问题。

### 0.5 `unitree_hg ROS msg not found`

现象：

```text
unitree_hg ROS msg not found; using default waist angles
```

原因：机器上只有 Livox workspace，没有 Unitree G1/H1 的 ROS2 message workspace。

已采用修复方式：

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros2.git ~/unitree_ros2
echo "123" | sudo -S apt install -y ros-humble-rosidl-generator-dds-idl libyaml-cpp-dev
cd ~/unitree_ros2/cyclonedds_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select unitree_hg \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
```

验证：

```bash
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
/home/unitree/miniconda3/envs/robomimic/bin/python -c \
  "from unitree_hg.msg import LowState; print('unitree_hg OK')"
```

`run_apriltag_target.sh` 已默认 source：

```bash
~/unitree_ros2/cyclonedds_ws/install/setup.bash
```

### 0.6 DDS / ROS2 网络注意事项

脚本默认使用 CycloneDDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

接口自动优先选 `192.168.123.*`：

```bash
CYCLONEDDS_IFACE="$(ip -o -4 addr show scope global | awk '/192\.168\.123\./ {print $2; exit}')"
```

如果日志出现：

```text
selected interface "lo" is not multicast-capable
```

说明当前没找到机器人网段接口。检查：

```bash
ip -o -4 addr show scope global
```

必要时手动指定：

```bash
CYCLONEDDS_IFACE=enP8p1s0 ./onboard/perception/camera/run_apriltag_target.sh --show
```

### 0.7 8080 端口占用

现象：

```text
OSError: [Errno 98] Address already in use
```

查占用：

```bash
ss -ltnp 'sport = :8080'
pgrep -af "apriltag_detector.py"
```

解决：

```bash
pkill -f "onboard/perception/camera/apriltag_detector.py"
```

或者换端口：

```bash
./onboard/perception/camera/run_apriltag_target.sh --show --show-port 8081
```

### 0.8 白球 bright detector

当前有两个稳定入口：

```bash
# 灰度 / IR，适合 AprilTag + 白球/反光点阵球
./onboard/perception/camera/run_apriltag_gray_ball.sh --show

# 彩色，适合人工观察正常 RGB-like 画面
./onboard/perception/camera/run_apriltag_color_ball.sh --show
```

截图参考：

- 灰度 / IR 画面
- 彩色 V4L2 画面

灰度入口默认：

```bash
--camera-profile gray-ir
--color-backend v4l2
--v4l2-device /dev/video3
--v4l2-fourcc GREY
--ball-bright
```

彩色入口默认：

```bash
--camera-profile color-v4l2
--color-backend v4l2
--v4l2-device /dev/video4
--v4l2-fourcc YUYV
--ball-bright
```

代码里 V4L2 fallback 会按当前 `--v4l2-fourcc` 过滤节点：

- 灰度入口只找支持 `GREY` 的节点，不会跳到彩色节点
- 彩色入口只找支持 `YUYV` 的节点，不会跳到 depth/灰度节点 Modification on

两条入口现在也会分开使用 `--camera-profile`：

- `gray-ir`：IR/灰度流使用更宽 FOV 的近似内参，并给 chest 外参加 `+0.020m` Y 偏置。这个 profile 更适合 AprilTag 位姿和白球单目深度估计。
- `color-v4l2`：彩色 YUYV 流保留原来的彩色近似内参和默认 chest 外参，并使用更严格的 bright-ball 形状过滤，减少白墙/鞋子误检。

如果现场标定出更准的值，仍可直接覆盖：

```bash
./onboard/perception/camera/run_apriltag_gray_ball.sh --show \
  --fx 675 --fy 650 --cx 640 --cy 360 \
  --chest-xyz 0.13444 0.020 0.06228 \
  --chest-rpy 0.0 0.2902482546 0.0
```

原理：

1. 在 IR/灰度图下半部分找亮斑
2. 对亮点做形态学合并，适配反光点阵球
3. 检查候选是否像球：
  - 外接圆半径范围
  - contour circularity
  - minAreaRect 宽高比
  - 轮廓质心是否接近圆心
4. 使用已知球半径 `0.115 m` 做单目距离估计和候选几何过滤：

```text
depth = fx * 0.115 / r_px
```

1. 转到 pelvis 坐标后继续过滤：
  - 深度范围
  - 左右范围
  - 高度范围

默认发布：

```text
rt/cam_ball_state
```

### 0.9 关于 depth 验证的结论

我们尝试过在 V4L2 灰度画面旁边再开一个 RealSense depth-only pipeline，用真实 depth 验证：

```text
R_measured = r_px * depth_surface / fx
```

理想逻辑是：

- bright detector 只给候选圆
- depth patch 给候选的真实表面距离
- 如果 `R_measured` 接近 `0.115m`，才认为是球
- 发布球心：`depth_center = depth_surface + 0.115`

但实测这台 G1 + D435I 上 **V4L2 UVC 和 librealsense depth-only 不能稳定同时打开同一台相机**：

```text
Starting RealSense depth-only ...
Frame didn't arrive within 5000
Failed to resolve the request ...
VIDEOIO(V4L2): failed VIDIOC_REQBUFS: errno=19 (No such device)
```

后果：

- depth-only 启动失败
- `/dev/video*` 会重新枚举
- 正在工作的灰度 V4L2 节点可能消失
- 预览从灰度变黑或跳到彩色节点

最终决策：

- 默认 **不启用** `--ball-bright-use-depth`
- 保留该参数作为实验接口
- 现场稳定方案使用灰度/彩色 V4L2 + 单目球半径估计

如果以后接入第二个相机，或确认 depth/color 可同步，再考虑重新启用 depth 验证。

### 0.10 白球识别率/误检调参

如果经常丢球，降低阈值：

```bash
./onboard/perception/camera/run_apriltag_gray_ball.sh --show --ball-bright-threshold 160
```

如果误检白鞋、墙面、反光物，提高阈值：

```bash
./onboard/perception/camera/run_apriltag_gray_ball.sh --show --ball-bright-threshold 190
```

如果白鞋仍被识别成球，优先调这些参数：

```bash
--ball-bright-z-min -1.2
--ball-bright-z-max 0.1
--ball-bright-max-abs-y 2.0
```

判断逻辑：

- 鞋通常是长条/不对称亮斑，会被宽高比和质心偏移过滤
- 球应该接近圆形，`aspect≈1`，`center_offset≈0`
- 如果鞋尖刚好很圆，只能依赖位置、高度、时序稳定性进一步过滤

半径估计注意：

为了合并反光点阵球，算法会对亮点做 dilation，外接圆半径可能偏大，导致单目深度偏近。代码提供：

```bash
--ball-bright-radius-correction 4
```

含义：深度估计时使用 `r_depth = r_px - correction`。如果球的 x 明显比 AprilTag 近，增大 correction；如果球比实际偏远，减小 correction。

### 0.11 `rs2_deproject_pixel_to_point` 类型错误

现象：

```text
TypeError: rs2_deproject_pixel_to_point(): incompatible function arguments
Invoked with: <__main__._ApproxIntrinsics object ...>
```

原因：V4L2 模式使用 `_ApproxIntrinsics`，不是 RealSense 原生 `rs.intrinsics`。

修复：代码里已改成纯 Python pinhole 反投影：

```text
x = (u - ppx) / fx * depth
y = (v - ppy) / fy * depth
z = depth
```

### 0.12 RealSense USB buffer

如果遇到相机 `bad_alloc`、帧启动失败、流不稳定，检查：

```bash
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

推荐：

```bash
echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

不要盲目设成 1000MB，Jetson 16GB 上可能和 DDS 共享内存一起造成系统不稳定。

### 0.13 V4L2 节点和格式

常见枚举：

```bash
v4l2-ctl --list-devices
for d in /dev/video*; do
  echo "--- $d"
  v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | sed -n '1,80p'
done
```

当前观察到的节点：

- `/dev/video0`：depth，`Z16`
- `/dev/video3`：IR/灰度，`GREY/UYVY`
- `/dev/video4`：彩色，`YUYV`

`/dev/video3 + GREY` 第一帧可能是全黑，代码里已做 warmup，最多丢弃 10 帧，直到拿到非空帧。

如果 V4L2 运行中出现：

```text
select() timeout
```

代码会连续失败 5 次后释放当前 cap，重新扫描同类 FOURCC 节点并继续。

### 0.14 常用确认命令

```bash
# 相机进程
pgrep -af "apriltag_detector.py|ball_detector.py|sensor_dashboard.py"

# 相机 USB
lsusb | grep -i RealSense

# V4L2 节点
v4l2-ctl --list-devices

# 8080 预览端口
ss -ltnp 'sport = :8080'

# Unitree ROS2 msg
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
/home/unitree/miniconda3/envs/robomimic/bin/python -c \
  "from unitree_hg.msg import LowState; print('OK')"

# Python 语法检查
python -m py_compile onboard/perception/camera/apriltag_detector.py
bash -n onboard/perception/camera/run_apriltag_target.sh
bash -n onboard/perception/camera/run_apriltag_gray_ball.sh
bash -n onboard/perception/camera/run_apriltag_color_ball.sh
```

---

## 目录

1. [硬件与环境配置](#一硬件与环境配置)
2. [灰白足球颜色/形态检测实验总结](#14-灰白足球颜色形态检测实验总结结论当前不采用)
3. [完整 Pipeline 流程图](#二完整-pipeline-流程图)
4. [每步耗时与阻塞分析](#三每步耗时与阻塞分析)
5. [核心概念：GIL / DMA / 多线程](#四核心概念gil--dma--多线程)
6. [优化历程（按问题顺序）](#五优化历程按问题顺序)
7. [性能演变总览](#六性能演变总览)
8. [当前架构说明](#七当前架构说明)
9. [进一步优化方向](#八进一步优化方向)
10. [Color→Depth 像素映射详解](#九colordepth-像素映射详解)
11. [诊断命令速查](#十诊断命令速查)

---

## 一、硬件与环境配置

### 1.1 硬件规格


| 项目      | 规格                                      |
| ------- | --------------------------------------- |
| SoC     | NVIDIA Jetson Orin NX 16 GB             |
| CPU     | 8× ARM Cortex-A78AE @ 1.5 GHz           |
| GPU     | 1024-core Ampere（最高 918 MHz）            |
| 内存      | 16 GB LPDDR5，CPU/GPU **统一内存**（无独立显存）    |
| 存储      | eMMC + microSD                          |
| JetPack | 5.1.2（CUDA 11.4，cuDNN 8.6）              |
| Python  | 3.8（conda 环境 `robomimic`）               |
| ROS2    | Foxy + CycloneDDS（`rmw_cyclonedds_cpp`） |


> **统一内存**意味着 CPU 和 GPU 共享同一块物理内存，PyTorch 无需 CPU→GPU 拷贝，
> 但也意味着 GPU 显存和系统 RAM 互相竞争同一带宽。

### 1.2 依赖安装

```bash
conda activate robomimic

# RealSense（必须 conda-forge，pip 版无 ARM .so）
conda install -c conda-forge pyrealsense2 -y

# YOLOv8
pip install ultralytics

# PyTorch（见 1.3）
# torchvision（见 1.4，必须源码编译）
```

### 1.3 安装 Jetson 专属 PyTorch（必须，否则无 CUDA）

PyPI 上的 `pip install torch` 只提供 x86_64 版，没有 aarch64+CUDA 支持。
必须从 NVIDIA 开发者网站下载 Jetson 专用 wheel：

```bash
# JetPack 5.1.2 → PyTorch 2.1.0
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/\
torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

python -c "import torch; print(torch.cuda.is_available())"
# True ← 必须为 True，否则 YOLO 只能 CPU
```

参考下载页：[https://developer.nvidia.com/embedded/downloads#?search=pytorch](https://developer.nvidia.com/embedded/downloads#?search=pytorch)

### 1.4 编译 torchvision v0.16.0（必须与 torch 2.1.0 对齐）

PyPI 的 torchvision 0.19.x 与 torch 2.1.0 不兼容，会报
`RuntimeError: Couldn't load custom C++ ops`，必须源码编译：

```bash
export CUDA_HOME=/usr/local/cuda-11.4   # 与 JetPack 5.1.2 一致
export PATH=$CUDA_HOME/bin:$PATH
pip uninstall torchvision -y

git clone --branch v0.16.0 --depth 1 https://github.com/pytorch/vision
cd vision && python setup.py install && cd .. && rm -rf vision

python -c "import torchvision; print(torchvision.__version__)"  # 0.16.0
```

---

## 二、完整 Pipeline 流程图

### 2.1 当前架构（最终优化版）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 进程：ball_detector.py（3 个常驻线程）                                    │
│                                                                         │
│  ┌──────────────────┐   ┌──────────────────────────────────────────┐   │
│  │  主线程（相机捕获）│   │  YOLO 线程（推理 + 发布）                 │   │
│  │                  │   │                                          │   │
│  │  D435 USB 3.0    │   │  ① buf_updated.wait()    ← 等新帧 ~16ms  │   │
│  │        ↓         │   │         ↓                                │   │
│  │  wait_for_frames │   │  ② get_color/depth_frame  ~0.1ms        │   │
│  │  【释放 GIL】     │──▶│         ↓                                │   │
│  │  ~16ms（60 FPS） │   │  ③ color.copy()（DMA→堆）  ~2ms          │   │
│  │        ↓         │   │         ↓                                │   │
│  │  buf_lock swap   │   │  ④ cv2.resize → imgsz×imgsz  ~1ms       │   │
│  │  ~0.1ms          │   │         ↓                                │   │
│  │  buf_updated.set │   │  ⑤ model.track()（GPU）  ~25ms 【主耗时】 │   │
│  └──────────────────┘   │         ↓                                │   │
│                          │  ⑥ get_distance() 11×11 patch  ~0.5ms  │   │
│  ┌──────────────────┐   │         ↓                                │   │
│  │  DDS lowstate     │   │  ⑦ rs2_deproject + EMA + FK  ~0.5ms    │   │
│  │                  │   │         ↓                                │   │
│  │  rt/lowstate     │   │  ⑧ dds.publish()         ~0.5ms        │   │
│  │  读关节角 q_wy 等 │   │         ↓                                │   │
│  │  避免 ROS msg 依赖 │   │  ⑨ MJPEG 编码（--show）  ~5ms（可选）    │   │
│  └──────────────────┘   └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流（像素坐标 → pelvis body 坐标）

```
RealSense D435（60 Hz）
    │  color: 640×480 BGR   depth: 640×480 Z16
    │
    ▼ [① wait_for_frames, 释放 GIL]
raw frameset（内存在 USB DMA 缓冲区）
    │
    ▼ [② get_color_frame() + color.copy()]
color: numpy uint8 (640×480×3)   depth: rs2_depth_frame 对象（保持 DMA 引用）
    │
    ▼ [④ cv2.resize]
color_small: numpy uint8 (imgsz×imgsz×3)   sx/sy: 缩放比例
    │
    ▼ [⑤ model.track()，CUDA GPU]
results → best_box: xyxy in (imgsz×imgsz)   → last_bbox: xyxy in (640×480)
    │
    ▼ [⑥ color→depth 像素映射 + patch 采样]
    │   a) 归一化: ndcx=(cx-ppx_c)/fx_c, ndcy=(cy-ppy_c)/fy_c
    │   b) 映射到 depth 图: dx=ndcx*fx_d+ppx_d + tx/Z*fx_d
    │   c) depth_arr[dy-5:dy+6, dx-5:dx+6] median
depth_m: float（球心深度，单位 m）
    │
    ▼ [⑦a rs2_deproject_pixel_to_point]
p_optical: [X,Y,Z]（光学坐标系：Z前，X右，Y下）
    │
    ▼ [⑦b optical_to_body()]
p_cam: [X,Y,Z]（相机 body 系：X前，Y左，Z上）
    │
    ▼ [⑦c EMA 滤波（α=0.6，跳变门限 0.6m）]
center_ema: 平滑后的相机 body 系坐标
    │
    ▼ [⑦d transform_point_camera_to_base() 正运动学链]
    │   pelvis → waist_yaw(q_wy) → waist_roll(q_wr) → waist_pitch(q_wp)
    │                                                → head(q_head) → camera
p_base: [x,y,z]（pelvis body 系）
    │
    ▼ [⑧ dds.publish()]
"rt/ball_state"（DDS BestEffort KeepLast(1)）
    │
    ▼
deploy_real.py / bridge policy runtime → FreeKick._build_obs() → state_cmd.ball_pos_b
```

---

## 三、每步耗时与阻塞分析

### 3.1 当前稳态耗时（640×480，imgsz=320）


| 步骤                       | 耗时        | 占比                 | 说明                             |
| ------------------------ | --------- | ------------------ | ------------------------------ |
| ① buf_updated.wait       | ~16ms     | 等相机帧（60FPS=16ms/帧） | 主要等待时间，不是"慢"                   |
| ② get_color/depth_frame  | <0.1ms    | —                  | SDK 返回引用，无拷贝                   |
| ③ color.copy()（DMA→堆）    | ~2ms      | —                  | 640×480×3=0.9MB，从 USB DMA 缓冲区拷 |
| ④ cv2.resize             | ~1ms      | —                  | 640×480 → 320×320              |
| **⑤ model.track()（GPU）** | **~25ms** | **~55%**           | **当前主瓶颈**                      |
| ⑥ get_distance() patch   | <0.5ms    | —                  | 11×11=121 个点，O(1) SDK 调用       |
| ⑦ deproject+EMA+FK       | <0.5ms    | —                  | 纯 numpy 数学                     |
| ⑧ dds.publish            | <0.5ms    | —                  | CycloneDDS UDP                 |
| ⑨ MJPEG 编码（--show）       | ~5ms      | —                  | imencode JPEG Q=60             |
| **单帧合计**                 | **~45ms** | —                  | **≈ 22 FPS**                   |


> **为什么实测 25 FPS 而不是 1000/45≈22 FPS？**  
> 因为 YOLO 处理帧时，主线程已经在采集下一帧，两者并行运行，
> 实际吞吐量由 max(GPU时间, 相机帧周期) 决定。

### 3.2 历史各版本耗时对比

```
版本                  copy    resize   track    total    FPS
─────────────────────────────────────────────────────────────
v1 无CUDA             —       —        670ms    670ms    1
v2 有CUDA但GIL满      —       —        260ms    260ms    4
v3 align在YOLO线程    102ms   0.1ms    34000ms  ∞        0.1  ← 最差
v4 align在YOLO线程    97ms    97ms     44ms     148ms    7    ← align是瓶颈
v5 去掉align          2ms     1ms      25ms     45ms     25   ← 当前
```

> v3 的 `track=34000ms` 是 GIL 争抢导致的，见第五节。  
> v4 的 `copy=97ms, resize=97ms` 是 DMA 内存 + buf_lock 竞争导致的，见第五节。

---

## 四、核心概念：GIL / DMA / 多线程

### 4.1 Python GIL（全局解释器锁）

Python 进程内部有一把"令牌"——GIL（Global Interpreter Lock）。规则：

```
同一时刻，只有持有令牌的线程能执行 Python 字节码。
```

C 扩展可以选择**主动交出令牌**让其他线程并行运行：

```
函数                          GIL 行为           对我们的影响
─────────────────────────────────────────────────────────────────
pipeline.wait_for_frames()    释放 GIL（等硬件）  ✅ YOLO 可同时运行
torch CUDA 核心计算            释放 GIL（等GPU）   ✅ 相机可同时采集
align.process()               持有 GIL（CPU运算） ❌ 阻塞所有其他线程
rclpy.spin()（原来）           持有 GIL（高频）    ❌ 500Hz 反序列化占满
np.asanyarray().copy()         持有 GIL（CPU运算） ❌ 大 DMA 拷贝时阻塞
```

**GIL 导致的典型问题**：

```
时间轴 →（修复前，align 在主线程）

主线程:  [wait 16ms]──[align.process() 持GIL 126ms]──[wait 16ms]──[align 126ms]──...
YOLO线程:[Python 5ms]──[等GIL 126ms]──[CUDA 20ms]──[Python 5ms]──[等GIL 126ms]──...
                        ↑ 每次Python代码都要等 126ms
```

model.track() 内部有数十次 Python/CUDA 切换，每次等 126ms → 单帧 34 秒。

### 4.2 DMA 内存与 ARM 缓存

相机帧的传输路径：

```
RealSense D435
    │ USB 3.0（5 Gbps）
    ▼
USB 控制器（硬件）
    │ DMA（Direct Memory Access）—— 硬件直接写内存，不经 CPU
    ▼
DMA 缓冲区（uncacheable 内存）
    │ 属性：不走 CPU L1/L2/L3 缓存
    │       每次访问都要经过内存总线
    ▼
np.asanyarray(frame.get_data())  ← 这是 DMA 区的"视图"，不是拷贝！
```

**uncacheable 内存 vs 普通内存的读速差异（ARM Cortex-A78）**：

```
普通堆内存（有 cache）：~20 GB/s
DMA uncacheable 内存：  ~8 MB/s（慢 2500 倍）
```

所以 `np.asanyarray(df.get_data()).copy()` 拷贝 0.8MB 深度图：

```
有 cache：0.8MB ÷ 20GB/s = 0.04ms   ✅
DMA 区：  0.8MB ÷ 8MB/s  = 100ms    ❌（之前 v4 的问题）
```

**当前的处理方式**：  
深度数据完全**不拷贝到 Python 堆**，直接调用
`depth_frame.get_distance(px, py)`（SDK 内部读一个 uint16，
只有 2 字节 = 可忽略），从而绕过了大块 DMA 读取的问题。

彩色帧仍然需要拷贝（YOLO 推理需要 numpy 数组），但 640×480×3 = 0.9MB，
耗时约 2ms（在允许范围内）。

### 4.3 多线程在这里的作用

当前 3 个线程：

```
线程              工作内容                              GIL 占用
─────────────────────────────────────────────────────────────────
主线程（相机）     pipeline.wait_for_frames()           绝大多数时间释放 GIL
YOLO 线程         align→resize→track→deproject→publish  顺序执行，是 GIL 持有者
spin_loop 线程    rclpy.spin_once()，50 Hz             每 20ms 短暂持有 GIL
```

**多线程的收益**：主线程在等下一帧（释放 GIL 16ms）时，YOLO 线程可以同时
做推理，形成真正的流水线。

**多线程的开销**：线程切换 <5µs，buf_lock 竞争 <0.1µs，相比 45ms 帧时间可忽略。

**多线程≠多核 Python 并行**：由于 GIL，两个线程不能同时执行 Python 字节码。
但当一个线程在执行 C 扩展（且 C 扩展释放了 GIL）时，另一个线程可以运行 Python。
这就是为什么 `pipeline.wait_for_frames()`（释放 GIL）和 `model.track()`（CUDA
期间释放 GIL）能真正并行的原因。

---

## 五、优化历程（按问题顺序）

### 问题 1：EMA 公式写反，lidar 球心跳变

**现象**：`check_ball_state.py` 观察到坐标在大范围内随机跳变。

**根因**：`alpha=0.9` 给新测量值 90% 权重，完全没有平滑效果。

```python
# ❌ 错误：alpha 越大跳变越厉害
self.center_ema = alpha * center_new + (1-alpha) * self.center_ema

# ✅ 修复：alpha 对应"历史权重"，越大越平滑
self.center_ema = alpha * self.center_ema + (1-alpha) * center_new
```

同时 alpha 从 0.9 → 0.5（响应更快）。

---

### 问题 2：本地 check_ball_state 显示全零（时钟不同步）

**根因**：`BallStateSubscriber.latest()` 用机器人时钟戳与本地时间做差判断新鲜度，
两台机器时钟差异导致所有消息被判为"过期"。

**修复**（`common/ball_state_dds.py`）：收到消息时记录**本地接收时刻**，用本地时间判断：

```python
self._received_at = int(time.time() * 1e6)   # 记录本地接收时间
# latest() 中：
age_us = now_us - self._received_at           # 而不是 now_us - s.timestamp_us
```

---

### 问题 3：YOLO 在 CPU 跑，670 ms/帧

**根因**：PyPI 的 `torch` 是 x86，`torch.cuda.is_available()` = False。

**修复**：安装 NVIDIA Jetson 专属 wheel（见 §1.3）。修复后：

```
warmup[0]: 3845ms  ← 首次 CUDA JIT 编译
warmup[1]:   27ms  ← 正常
YOLO inference: 23 ms/frame ✅
```

---

### 问题 4：GPU 被限速（115 MHz），YOLO 260 ms/帧

**诊断**：

```bash
cat /sys/class/devfreq/*/cur_freq  # 输出 115200000（115 MHz）
```

**修复**：

```bash
echo "123" | sudo -S nvpmodel -m 0    # 最大性能模式
echo "123" | sudo -S jetson_clocks    # 解锁所有时钟
# GPU 恢复 918 MHz，YOLO 回到 23 ms
```

---

### 问题 5：/lowstate 500 Hz 饱和 GIL，YOLO 降速 10×

**根因**：`rclpy.spin(joint)` 持续反序列化 500 Hz 的 LowState 消息，
几乎全时占用 GIL。YOLO 的 Python/CUDA 切换每次都需要等待。

```
YOLO 线程：[等GIL 2ms][CUDA 20ms][等GIL 2ms][CUDA 20ms]...
spin线程：  [GIL 2ms][GIL 2ms][GIL 2ms]...（500 Hz 连续反序列化）
```

**修复**：改为 50 Hz 轮询，每次 spin 后主动 sleep 让出 GIL：

```python
def _spin_loop():
    while True:
        rclpy.spin_once(joint, timeout_sec=0.0)
        time.sleep(0.02)   # 50 Hz：关节角更新足够，不需要 500 Hz
```

---

### 问题 6：align.process() 在主线程，YOLO track 需 34 秒/帧（最严重）

这是整个调试过程中最隐蔽、耗时最长的问题。

#### 6a. 诊断过程

加入逐步 TIMING 打印，追踪到 `model.track()` 耗时 34 秒：

```
[TIMING] copy=97ms  resize=97ms  track=33979ms  total=34414ms  fps=0.1
```

独立测试 `model.track()` 在后台线程中只需 30ms，说明问题来自**并发竞争**。

#### 6b. 根因：align.process() 持有 GIL 126ms，阻断 YOLO

```python
# 测量 align.process() 耗时：
t0 = time.perf_counter()
aligned = align.process(frames)
print(f"{(time.perf_counter()-t0)*1e3:.1f}ms")
# 输出：126.1ms
```

`align.process()` 是 Intel RealSense 的 C 扩展，在 ARM 上对
640×480 深度图做逐像素几何重投影，**不释放 GIL**。

原来的架构（错误的）：

```
主线程:  wait(16ms)──align.process(持GIL 126ms)──wait(16ms)──align(126ms)──...
YOLO线程: 每次 Python/CUDA 切换都要等 126ms
         model.track() 内部约 270 次切换 × 126ms ≈ 34 秒
```

#### 6c. 中间修复：align 移到 YOLO 线程（7 FPS）

把 align 移到 YOLO 线程顺序执行，消除 GIL 争抢。
但 align 本身 100ms 仍是瓶颈：

```
YOLO 线程：align(100ms) → resize(1ms) → track(44ms) → ... = 148ms = 7 FPS
```

#### 6d. 最终修复：完全去掉 align（25 FPS）

**核心洞察**：我们根本不需要对整张深度图做对齐。
`align.process()` 的目的是让每个彩色像素都能查到对应的深度值。
但我们只有一个需要深度的像素：球心 `(cx, cy)`。

对于单点查深度，D435 的 SDK 提供了 `depth_frame.get_distance(x, y)`，
O(1)，<0.1ms。D435 的彩色和深度传感器几何位置非常接近（对准误差 <5px 在 1m 处），
对球心定位（cm 级精度）完全可以接受。

```python
# ❌ 旧方案：全图对齐，100ms
aligned = align.process(frames)
depth_patch = aligned_depth[y0:y1, x0:x1] * 0.001

# ✅ 新方案：单点查深度，<0.1ms
samples = [depth_frame.get_distance(px, py)
           for py in range(y0d, y1d+1)
           for px in range(x0d, x1d+1)]  # 11×11 = 121 个点
depth_m = np.median([v for v in samples if DEPTH_MIN < v < DEPTH_MAX])
```

同时：

- 相机分辨率从 848×480 降为 640×480（更小，更快）
- 主线程只保留 `pipeline.wait_for_frames()`（释放 GIL）+ 引用交换

**优化后各步骤耗时**：

```
copy=2ms  resize=1ms  track=25ms  depth=0.5ms  post=0.5ms  total=≈30ms  → 25 FPS ✅
```

---

### 问题 7：DMA 缓冲区读取慢（v4 中 copy=97ms）

在 v4（align 移到 YOLO 线程但未彻底去掉的中间版本）中：

```python
# 相机线程（v4 错误版本）：
buf_depth = np.asanyarray(df.get_data())   # ← 无 .copy()，是 DMA 区视图！

# YOLO 线程：
depth = buf_depth.copy()   # 从 DMA uncacheable 内存读 0.8MB → 100ms
```

**修复**（当时）：在相机线程立即 copy 到 Python 堆：

```python
buf_depth = np.asanyarray(df.get_data()).copy()   # 相机线程内完成 DMA→堆 copy
```

**最终修复**：去掉 align 后，彻底不再需要拷贝完整深度图，
只用 `get_distance(x,y)` 读取 2 字节（可忽略），这个问题自然消失。

---

### 问题 8：MJPEG 端口冲突 / busy-loop

**现象**：`OSError: [Errno 98] Address already in use`；浏览器刷新慢。

**修复**：

```python
# 端口快速复用
socketserver.ThreadingTCPServer.allow_reuse_address = True

# 避免 busy-loop，只推新帧
last_sent = None
while True:
    with _mjpeg_lock:
        jpg = _mjpeg_frame[0]
    if jpg is None or jpg is last_sent:
        time.sleep(0.02)   # 50 Hz 轮询
        continue
    last_sent = jpg
    self.wfile.write(...)
```

---

## 六、性能演变总览

```
                         copy    resize   track      FPS   主要问题
──────────────────────────────────────────────────────────────────────
v1  无CUDA PyTorch         —       —       670ms      1    PyPI torch，无GPU
v2  有CUDA，spin全速        —       —       260ms      4    /lowstate GIL 占满
v3  spin限速，align在主线程  —       —      34000ms    0.1  align持GIL 126ms
v4  align移到YOLO线程      102ms   97ms     44ms       7    DMA读+align=100ms
v5  去掉align（当前）        2ms    1ms      25ms      25   GPU推理，可接受
──────────────────────────────────────────────────────────────────────
提升倍数：25× 相比初始版本
```

---

## 七、当前架构说明

### 7.1 线程职责分工

```
线程              任务                        GIL 特性
─────────────────────────────────────────────────────
主线程            wait_for_frames() + 引用交换  绝大多数时间释放 GIL
YOLO 线程         帧处理→推理→发布（顺序执行）   持有 GIL，是计算主体
spin_loop 线程    关节角订阅（50 Hz）            每 20ms 短暂占用 ~0.1ms
MJPEG 线程（可选） HTTP 推流                    等待 socket write
```

### 7.2 关键设计原则

1. **主线程只做 `wait_for_frames()`**：这个 C 扩展会释放 GIL，让 YOLO
  线程可以真正并行运行。主线程不做任何耗时计算。
2. **不对齐全图**：depth 只在球心处查询一个 patch（O(1)），
  不调用 `align.process()`（O(像素数)）。
3. **深度帧不拷贝**：depth 保持 rs2 对象，通过 `get_distance()` 按需读取，
  完全避免 DMA 大块读取问题。
4. **彩色帧必须拷贝**：YOLO 需要 numpy 数组，且 DMA 视图在下一帧到来后可能
  失效，所以必须立即 `.copy()` 到 Python 堆（2ms）。
5. **spin_once + sleep(0.02)**：每 20ms 处理一次 ROS2 消息，50 Hz 足够
  读取关节角，且 GIL 持有时间极短。

### 7.3 命令行参数

```bash
python onboard/perception/camera/ball_detector.py \
    --model yolov8n.pt  \  # YOLO 模型（n=最快，s/m=更精确）
    --imgsz 320         \  # YOLO 输入尺寸（更小=更快，默认320）
    --width  640        \  # 相机宽度（默认640）
    --height 480        \  # 相机高度（默认480）
    --show                 # 开启 MJPEG 流（浏览器查看）
```

---

## 八、进一步优化方向

### 8.1 分辨率对速度的影响


| 相机分辨率   | YOLO imgsz | TRT FPS（估） | PT FPS（估） | 说明           |
| ------- | ---------- | ---------- | --------- | ------------ |
| 848×480 | 480        | ~30        | ~12       | YOLO 输入大，精度高 |
| 640×480 | 320        | **~40**    | ~25       | **当前默认**     |
| 424×240 | 224        | ~55        | ~35       | 近距离够用        |
| 424×240 | 160        | ~65        | ~45       | 球较小时可能漏检     |


调整命令：

```bash
python onboard/perception/camera/ball_detector.py --width 424 --height 240 --imgsz 224
```

### 8.2 其他可尝试的优化


| 方案                                           | 预期效果                      | 复杂度 |
| -------------------------------------------- | ------------------------- | --- |
| 跳帧（每 N 帧跑一次 YOLO）                            | FPS 不变，GPU 负载↓            | 低   |
| TensorRT 导出（`model.export(format="engine")`） | YOLO 加速 2-4×，约 10ms       | 中   |
| YOLOv8n-pose 换成更小模型                          | 视模型而定                     | 低   |
| 减少 `DEPTH_SAMPLE_RADIUS`（5→2）                | patch 从 121→25 点，节约 0.3ms | 低   |
| 相机帧率降到 30 Hz                                 | 省 USB 带宽，不影响 FPS          | 低   |


### 8.3 模型选择：精度 vs 速度（实测，Jetson Orin NX，imgsz=320，FP16）✅

> 模型文件统一存放于 `onboard/perception/camera/models/`


| 后端                 | 推理时间       | 整体估算 FPS    | COCO mAP50-95   | 备注                   |
| ------------------ | ---------- | ----------- | --------------- | -------------------- |
| yolov8n.pt         | 17 ms      | ~25 FPS     | 37.3            | 无需环境配置，随时可用          |
| yolov8n.engine     | 4.9 ms     | **~40 FPS** | 37.3            | TRT 加速 3.5×，仅限本机     |
| yolo11n.pt         | 24 ms      | ~20 FPS     | 39.5            | 比 v8n 略准             |
| yolo11s.pt         | 25 ms      | ~20 FPS     | 47.0            | 同速但精度高               |
| yolo11m.pt         | 28 ms      | ~18 FPS     | 51.5            | 高精度                  |
| **yolo11m.engine** | **8.6 ms** | **~35 FPS** | **51.5 (+38%)** | **当前默认，TRT 加速 3.3×** |


> **如何选择：**  
>
> - 精度优先 → **yolo11m.engine**（默认）  
> - 速度优先 → **yolov8n.engine**（`--model models/yolov8n.pt`）  
> - 无 GPU 解锁 → 任意 .pt 文件（`model.track(..., device='cuda:0', half=True)`）

### 8.4 TRT Engine 原理与可移植性

**TRT Engine 是怎么得到的？**

```
原始权重 (.pt)
    ↓ ultralytics model.export(format='engine', half=True)
ONNX 中间格式 (.onnx)
    ↓ TensorRT builder（kernel profiling，~10 分钟）
TRT Engine (.engine)  ← FP16 量化，算子融合，GPU 专属二进制
```

FP16"量化"：把权重从 float32（4 byte）压缩到 float16（2 byte），精度损失可忽略（mAP 几乎不变），计算速度翻倍。

`**.engine` 文件能否直接给别人用？**


| 情况                             | 是否可用    | 原因                        |
| ------------------------------ | ------- | ------------------------- |
| 同型号 Jetson Orin NX + 同 JetPack | ✅ 通常可以  | 架构/版本完全一致                 |
| 不同型号 Jetson（如 Nano/Xavier）     | ❌ 不可用   | GPU 架构不同（Ampere vs Volta） |
| x86 PC（桌面 RTX）                 | ❌ 不可用   | 架构完全不同                    |
| 同机器 JetPack 升级后                | ⚠️ 可能失败 | TRT 版本变化                  |


**结论：`.engine` 文件不应提交到 git。**

### 8.5 git 提交策略（模型文件管理）

```
onboard/perception/camera/models/
├── .gitignore          ← 排除 *.pt / *.engine / *.onnx
├── download_and_export.sh  ← 一键下载 + 导出 TRT（提交此文件）
└── README.md           ← 说明（提交此文件）
```


| 文件类型                     | 大小       | 是否提交  | 理由                   |
| ------------------------ | -------- | ----- | -------------------- |
| `*.pt`（PyTorch 权重）       | 6–39 MB  | ❌ 不提交 | 太大；ultralytics 会自动下载 |
| `*.onnx`（中间格式）           | 12–80 MB | ❌ 不提交 | 太大；export 时自动生成      |
| `*.engine`（TRT 二进制）      | 8–42 MB  | ❌ 不提交 | **硬件绑定，无法跨机器**       |
| `download_and_export.sh` | 2 KB     | ✅ 提交  | 让别人一键复现              |


**别人拿到代码后的操作：**

```bash
# 1. 一键下载 + 导出（首次运行，约 15 分钟）
bash onboard/perception/camera/models/download_and_export.sh

# 2. 正常启动（自动检测 .engine）
bash onboard/perception/camera/run.sh
```

**导出命令（手动版，如需重新导出）：**

```bash
cd RoboMimic_Deploy
LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:$LD_LIBRARY_PATH \
PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH \
python -c "
import sys, numpy as np
if not hasattr(np, 'bool'): np.bool = bool
sys.path.insert(0, '/usr/lib/python3.8/dist-packages')
from ultralytics import YOLO
YOLO('onboard/perception/camera/models/yolo11m.pt').export(
    format='engine', device=0, half=True, imgsz=320, workspace=4)
# 约 10-15 分钟，生成 yolo11m.engine (42 MB)
"
```

**运行（推荐）：**

```bash
bash onboard/perception/camera/run.sh           # yolo11m.engine，~35 FPS
bash onboard/perception/camera/run.sh --show    # + MJPEG 预览
bash onboard/perception/camera/run.sh --model models/yolov8n.pt  # 切回快速模型
```

**TRT 注意事项：**

- `.engine` 绑定硬件，换机器必须重新导出
- 加载需要 `LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat`（`run.sh` 已设置）
- `PYTHONPATH=/usr/lib/python3.8/dist-packages` 让 conda 能 `import tensorrt`

---

## 九、Color→Depth 像素映射详解

### 9.1 问题：D435 色彩相机与深度相机参数不同

D435 内部有两个物理上分离的传感器（实测，640×480 模式）：

```
D435 正面布局：
┌──────────────────────────────────────┐
│  [IR-L]  [IR-R]  [RGB]  [激光投射]   │
│                   ↑                   │
│             Color 光心                │
│       ←14.5mm→                        │
│   Depth 光心（IR 对的中点）            │
└──────────────────────────────────────┘
```


| 参数          | Color 相机     | Depth 相机          |
| ----------- | ------------ | ----------------- |
| 焦距 fx       | **607.5** px | **386.3** px      |
| 焦距 fy       | 607.0 px     | 386.3 px          |
| 主点 ppx      | 317.2 px     | 319.7 px          |
| 主点 ppy      | 251.5 px     | 241.9 px          |
| 水平 FOV      | **55.6°**    | **79.3°** → 视野更宽  |
| 与 Color 的基线 | —            | tx = **-14.5 mm** |


> 以上为实际相机在 640×480 分辨率下的测量值，可在启动时从 `[INFO]` 日志确认。

### 9.2 直接使用 color 坐标采样 depth 的误差

若直接用 `depth_arr[cy, cx]`（把 color 像素当 depth 像素）：


| 误差来源    | 公式                                | 举例（cx=450，距中心 133px）     | 1m 处横向误差   |
| ------- | --------------------------------- | ------------------------ | ---------- |
| FOV 比例差 | `Δ = cx_offset × (1 − fx_d/fx_c)` | `133×(1−386/607) = 48px` | **12 cm**  |
| 基线视差    | `Δ = |tx|/Z × fx_d`               | `0.0145/1.0×386 = 6px`   | **1.5 cm** |
| **合计**  |                                   | **~54 px**               | **~14 cm** |


图像边缘时误差更大（cx=500 时约 70px = 18cm），会导致深度采样到背景而非球。

### 9.3 正确转换方法：三步法

```
Color 像素 (cx, cy)
        │
        │ Step 1 — 去除 Color 内参，得到归一化方向向量
        │   ndcx = (cx − ppx_color) / fx_color
        │   ndcy = (cy − ppy_color) / fy_color
        │   含义：光线方向（与分辨率/FOV 无关的角度）
        ▼
归一化方向 (ndcx, ndcy)
        │
        │ Step 2a — 加入 Depth 内参，投影到深度图像素（修正 FOV）
        │   dx0 = ndcx × fx_depth + ppx_depth
        │   dy0 = ndcy × fy_depth + ppy_depth
        │
        │ Step 2b — 读取粗略深度，修正基线视差
        │   Z_coarse = depth_arr[dy0, dx0] × depth_scale  (或默认 1.0m)
        │   Δx = tx / Z_coarse × fx_depth   (tx = -0.0145m → 向左偏)
        │   dx = dx0 + Δx
        ▼
深度图像素 (dx, dy)
        │
        │ Step 3 — numpy patch 采样（11×11，取中位数）
        │   patch = depth_arr[dy-5:dy+6, dx-5:dx+6] × depth_scale
        │   depth_m = median(patch[有效值])
        ▼
depth_m（球心深度，单位 m）
        │
        │ 反投影到 3D（使用 Color 内参 + color 像素，结果在 Color 坐标系下）
        │   rs2_deproject_pixel_to_point(color_intrin, [cx, cy], depth_m)
        │   X = (cx − ppx_color) / fx_color × depth_m
        │   Y = (cy − ppy_color) / fy_color × depth_m
        │   Z = depth_m
        ▼
p_optical [X, Y, Z]（光学坐标系，Z 朝前）
```

### 9.4 数值示例

球心在 color 图 (cx=450, cy=200)，实际距离 1.2m：

```
Step 1:  ndcx = (450 − 317.2) / 607.5 = 0.2186
         ndcy = (200 − 251.5) / 607.0 = -0.0849

Step 2a: dx0 = 0.2186 × 386.3 + 319.7 = 404
         dy0 = -0.0849 × 386.3 + 241.9 = 209

         若直接用 color 坐标 cx=450，此处差了 450-404 = 46px → 12cm 误差

Step 2b: depth_arr[209, 404] ≈ 1200 raw → Z_coarse = 1.2m
         Δx = -0.0145 / 1.2 × 386.3 = -4.7 ≈ -5px   (基线视差)
         dx = 404 + (-5) = 399
         dy = 209

Step 3:  patch = depth_arr[204:215, 394:405]
         median ≈ 1198 raw → depth_m = 1.198m

反投影: X = 0.2186 × 1.198 = 0.262m
        Y = -0.0849 × 1.198 = -0.102m
        Z = 1.198m
→ 球在 color 相机前方 1.198m，右方 0.262m，上方 0.102m
```

### 9.5 计算开销


| 方法                       | 耗时/次      | 备注                 |
| ------------------------ | --------- | ------------------ |
| 旧：直接 `depth_arr[cy, cx]` | 5.8 μs    | 误差最大 14cm          |
| 新：三步法 + patch            | 16.3 μs   | 额外 10.5 μs         |
| 整帧耗时（TRT）                | ~8,000 μs | 三步法占 **0.13%**，可忽略 |


### 9.6 为什么反投影仍用 Color 内参？

`rs2_deproject_pixel_to_point(color_intrin, [cx, cy], depth_m)` 的作用是：

```
像素坐标 (cx, cy) + 深度 depth_m → 3D 点（在 color 相机坐标系下）
```

- 我们传入的是 **color 像素**，所以必须用 **color 内参**
- 得到的结果天然在 **color 相机坐标系**下，与后续 `optical_to_body()` → `transform_to_base()` 链路对齐
- 若改用 depth 内参 + depth 像素，结果会在 depth 相机坐标系下，还需要再做一次 extrinsics 变换

---

## 十、诊断命令速查

```bash
# ── GPU 状态 ──────────────────────────────────────────────
# 查看当前 GPU 频率（918400000 = 918 MHz = 最大）
cat /sys/class/devfreq/*/cur_freq

# 解锁最大性能（每次重启后需重新执行）
echo "123" | sudo -S nvpmodel -m 0
echo "123" | sudo -S jetson_clocks

# 实时功耗/温度监控
sudo tegrastats

# ── PyTorch CUDA 验证 ────────────────────────────────────
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('设备:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
print('PyTorch:', torch.__version__)
"

# ── YOLO 推理速度基准 ────────────────────────────────────
python -c "
from ultralytics import YOLO; import numpy as np, time
m = YOLO('yolov8n.pt')
d = np.zeros((320,320,3), dtype=np.uint8)
for _ in range(3): m(d, verbose=False, device='cuda:0', half=True)
t = time.perf_counter()
for _ in range(20): m(d, verbose=False, device='cuda:0', half=True)
print(f'avg: {(time.perf_counter()-t)/20*1000:.1f} ms')
"
# ≈23ms 正常；>100ms → GPU 未启用或频率限速

# ── RealSense 各操作耗时 ─────────────────────────────────
python -c "
import pyrealsense2 as rs, time, numpy as np
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
p.start(c)
a = rs.align(rs.stream.color)
for i in range(5):
    f = p.wait_for_frames()
    cf = f.get_color_frame(); df = f.get_depth_frame()
    t0 = time.perf_counter()
    color = np.asanyarray(cf.get_data()).copy()
    t1 = time.perf_counter()
    aligned = a.process(f)
    t2 = time.perf_counter()
    d = df.get_distance(320, 240)
    t3 = time.perf_counter()
    print(f'[{i}] color.copy={( t1-t0)*1e3:.1f}ms  align={( t2-t1)*1e3:.1f}ms  get_distance={(t3-t2)*1e3:.2f}ms')
p.stop()
"
# color.copy ≈ 2ms  align ≈ 100ms（被我们去掉了）  get_distance ≈ 0.01ms

# ── 完整流水线验证 ────────────────────────────────────────
python -u onboard/perception/camera/ball_detector.py --imgsz 320
# 正常输出：[BALL] pelvis=(...) d=1.07m YOLO=25.1fps
```

---

## 十一、ROS2 / DDS 相关问题

### 11.1 进程在 `rclpy.init()` 前被 OOM Killer 杀死（`bad_alloc` + `Killed`）

#### 现象

任何调用 ROS2 的脚本（`run_apriltag_target.sh`、`run_target.sh`、
`run_dual_d435.sh` 等）在 Jetson 上通过 SSH 启动时，Python **在打印第一行之前**
就被杀死，stderr 只显示：

```
bad_alloc caught: std::bad_alloc
bad_alloc caught: std::bad_alloc
...
Killed
```

相机驱动完全没有参与，隔离测试（只运行 `rclpy.init()` + 创建节点，不打开相机）
同样复现。`dmesg` 显示是 OOM Killer 触发：

```
Out of memory: Killed process <PID> (python) total-vm:30862628kB, anon-rss:14476516kB
```

注意：系统有 15 GB RAM，`free -h` 显示 13 GB 可用。**并非真正内存不足。**

#### 根本原因

**FastDDS（Fast RTPS）是 ROS2 Foxy 的默认 RMW**。FastDDS 在初始化时会通过
`mmap` 预分配一个共享内存传输缓冲区，其大小与系统 RAM 成正比。
在 16 GB Jetson Orin NX 上，这个缓冲区约为 **14–15 GB**。

Linux 的内存过度提交（overcommit）允许 `mmap` 成功，但当内核真正尝试分配
物理页时，OOM Killer 立即介入并用 SIGKILL 杀死进程（exit code 137）。

**Unitree G1 设计使用 CycloneDDS**（`rmw_cyclonedds_cpp`），不会有这个问题。
`~/unitree_ros2/setup.sh` 会设置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，
但该脚本只在交互式 shell 中通过 `.bashrc` 加载，**非交互式 SSH 不执行 `.bashrc`**，
因此通过 SSH 启动的脚本默认回退到 FastDDS，触发 OOM。

#### 修复

在所有 `run_*.sh` 脚本中，在 `source /opt/ros/foxy/setup.bash` 之后加入：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

`onboard/perception/camera/` 下的所有 `run_*.sh` 已包含此修复。

#### 快速验证

```bash
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda run -n robomimic python -c "
import rclpy
from rclpy.node import Node
rclpy.init()
class T(Node): pass
t = T('test_node')
print('rclpy OK')
rclpy.shutdown()
"
# 应输出：rclpy OK
```

---

### 11.2 `conda: command not found`（非交互式 SSH）

#### 现象

通过 SSH 远程执行脚本时报：

```
bash: conda: command not found
```

#### 根因

非交互式 SSH 不执行 `.bashrc`，conda 的 shell 函数未被定义。

#### 修复

在所有 `run_*.sh` 脚本的 `conda run` 之前加入：

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
```

---

### 11.3 usbfs_memory_mb 与 librealsense `bad_alloc`

librealsense 通过 USB 内核缓冲区传输帧数据。系统默认值（16 MB）对高分辨率
D455 流不够，会导致 librealsense **帧缓冲池** 耗尽并抛出 `bad_alloc`。
这是**另一个不同的** `bad_alloc`，发生在相机流启动后，而非 DDS 初始化时。

`/etc/rc.local` 在启动时设置：

```sh
echo 256 > /sys/module/usbcore/parameters/usbfs_memory_mb
```

256 MB 对 D455 在 USB 3.2 下以 1280×720@30fps 或 848×480@60fps 运行足够。

> **不要**将此值设为 1000（1 GB）。这会在内核中预留 1 GB 内存，
> 配合 FastDDS 的 14 GB mmap，总计超出可用 RAM，导致系统不稳定。

检查当前值：

```bash
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

---

### 11.4 D455 只有 15 fps（分辨率 1280×720）

#### 根因

相机连接到了 **USB 2.1** 口。D455 在 1280×720@30fps 需要 USB 3.x 带宽。
USB 2.1 上限约 60 MB/s，librealsense 自动降帧率。

#### 修复

将 D455 重新插入 **USB 3.2**（蓝色或标有 "SS"）端口。验证：

```bash
rs-enumerate-devices | grep USB   # 应显示 USB 3.2
```

---

### 11.5 球/目标看起来按 torso yaw，而不是 pelvis frame

#### 现象

机器人姿态为：torso / chest camera 指向球，pelvis 相对 torso 大约左转 90 度。
理论上 `ball_pos_b` / `target_pos_b` 是 pelvis body frame，应该随 pelvis 坐标轴变化；
实际观察却像是按 torso/camera 朝向输出。

典型表现：

- 摄像头正对球时，发布的 `x` 很大、`y` 很小，像 camera/torso frame。
- pelvis 已经转到侧向，但 `rt/ball_state` / `rt/target_state` 没有对应旋转。
- FreeKick 里的 `ball_pos_b` / `target_pos_b` 看起来 yaw 参考了 torso。

#### 运行时证据

在 `apriltag_detector.py` 的 camera-to-pelvis transform 后加日志，复现时看到：

```json
{"waist_q":[0.0,0.0,0.0],"joint_age_s":null,"p_cam_body":[2.58,-0.22,0.14],"p_base_pelvis":[2.64,-0.20,-0.50]}
```

这说明 detector 没有收到 waist joint state，正运动学链退化为：

```text
pelvis -> waist_yaw(0) -> waist_roll(0) -> waist_pitch(0) -> camera
```

因此输出虽然命名为 `p_base_pelvis`，实际方向近似 camera/torso frame。

修复后，同样姿态下日志变为：

```json
{"waist_q":[-1.33,-0.18,-0.23],"joint_age_s":0.0004,"p_cam_body":[2.62,-0.44,0.25],"p_base_pelvis":[0.27,-2.76,0.29]}
```

`waist_q` 非零且 `joint_age_s` 为毫秒以内，`p_base_pelvis` 明显按 pelvis 坐标旋转，符合预期。

#### 根因

旧实现依赖 ROS `/lowstate` 和 `unitree_hg.msg.LowState` 读取腰部关节角。
在当前 onboard 环境中，ROS message workspace 可能不存在或未 source，导致 listener 没有收到任何消息，
`q_wy/q_wr/q_wp` 一直保持默认 0。

另一个坑：直接复用 `deploy_real/config/real.yaml` 的 `net` 也可能失败。
例如 laptop 配置是 `enp5s0f1`，但 Jetson 上可用网卡是 `enP8p1s0`，会报：

```text
ChannelFactoryInitialize: enp5s0f1: does not match an available interface
Exception: channel factory init error.
```

#### 修复

Camera 和 LiDAR detector 都应直接订阅 Unitree DDS `rt/lowstate`：

```text
rt/lowstate -> LowStateHG -> q[12], q[13], q[14]
```

LiDAR 进程同时要跑 ROS2 / Livox / rclpy，因此 lowstate DDS listener 放在隔离子进程中，
主进程只读取最新 waist joints snapshot。这样可以避免 Unitree DDS 和 ROS2/Livox 在同一
Python 进程内互相影响，同时保留 `rt/lowstate` 的实时腰部关节角。

并在启动时做网卡选择：

- `--lowstate-net` 显式传入时优先使用。
- 未传入时先读 `deploy_real/config/real.yaml` 的 `net`。
- 如果该网卡在当前机器不存在，自动选择可用候选，如 `enP8p1s0`、`eth0`、`usb0`。

启动时如果看到类似：

```text
[WARN] configured lowstate net 'enp5s0f1' is unavailable; using 'enP8p1s0'
[INFO] Unitree DDS joint listener started (rt/lowstate)
```

说明进入了正确路径。

LiDAR 端还会在状态行里打印：

```text
[lidar] pelvis=(...) raw=(...) waist_q=(...) joint_age_s=...
```

其中 `rt/lidar_ball_state` 和 fuser 输出的 `rt/ball_state` 都应是 pelvis body frame。
如果 lowstate 丢失或过期，会每 2 秒报警：

```text
lowstate stale/missing (...); MID360->pelvis transform is using default or stale waist joints
```

#### 快速验证

运行 detector 后观察日志或临时 debug 输出：

```text
waist_q != [0,0,0]
joint_age_s < 0.05
```

现场验证过的 LiDAR 修复表现：

```text
[lidar] pelvis=(-0.01,-1.78,-0.71) raw=(...) waist_q=(-1.41,-0.09,-0.01) joint_age_s=0.01~0.03
```

修复前同一姿态近似输出 torso-like 坐标：

```text
[lidar] pelvis=(+1.79,-0.19,-0.76) ... waist_q=(+0.00,+0.00,+0.00) joint_age_s=null
```

修复后 fuser 应优先使用 LiDAR raw observation，并发布：

```text
[fuser] source=lidar -> rt/ball_state
```

若 `joint_age_s` 是 `null` 或 `waist_q` 长时间为 `[0,0,0]`，说明 lowstate 没有进 detector，
此时先检查：

```bash
ip -o link show
python tools/check_dds_connection.py <网卡名>
```

注意：FreeKick 的 `anchor_pos_b` / `anchor_ori_6d` 按训练设置是 torso-link 相关；
不要把它们和 policy observation 里的 `soccer_pos_b` / `target_pos_b` 混淆。
本问题只针对 detector 发布到 `rt/lidar_ball_state`、`rt/ball_state`、`rt/target_state`
的 pelvis-frame 观测。

---

### 11.6 `rt/lidar_ball_state` 在其他 pane 看不到

#### 现象

LiDAR detector 日志显示已经检测并发布 raw ball，但 dashboard / fuser / `check_ball_state.py`
看不到 `rt/lidar_ball_state`，或者 fuser 一直退回 camera / none。

#### 根因

启动环境里若残留：

```bash
ROS_LOCALHOST_ONLY=1
```

ROS2 会把通信限制在 localhost。即使 detector 本身在跑，跨进程 / 跨 pane 的 DDS topic
也可能不可见，现场表现就是 LiDAR raw topic 消失。

#### 修复与验证

LiDAR 启动脚本应显式设置：

```bash
export ROS_LOCALHOST_ONLY=0
bash onboard/perception/lidar/run.sh --show --dds-topic rt/lidar_ball_state
```

验证顺序：

```bash
echo "$ROS_LOCALHOST_ONLY"   # 应为 0 或空；推荐脚本内固定为 0
bash onboard/perception/run_sensor_dashboard.sh
```

dashboard 应能同时看到 `rt/lidar_ball_state` 和 fuser 发布的 `rt/ball_state`。

---

### 11.7 Camera / fuser 输出混在同一个终端，或 Ctrl-C 后残留进程

#### 现象

使用灰度相机 + fuser 时，camera pane 里同时刷 bright-ball、AprilTag 和 fuser 状态，
不容易判断当前帧来自哪个模块；按 Ctrl-C 后有时 fuser 或 bright worker 仍残留。

#### 修复

推荐使用：

```bash
bash onboard/perception/camera/run_gray_perception.sh --with-fuser --show
```

当前脚本行为：

- camera 主循环统一低频打印 AprilTag / bright-ball 状态；bright worker 不再直接向终端打印。
- `--with-fuser` 会把 fuser stdout/stderr 重定向到 `/tmp/ball_fuser.log`，避免 camera pane 混入 fuser 状态。
- camera 和 fuser 都以独立 process group 启动；`INT` / `TERM` / `EXIT` 会清理两组子进程。
- `apriltag_detector.py` 将 `SIGTERM` 转成 `KeyboardInterrupt`，确保 `finally` 路径关闭 camera 和 bright worker。

排查命令：

```bash
tail -f /tmp/ball_fuser.log
pgrep -af "apriltag_detector.py|ball_fuser.py"
```

按 Ctrl-C 后 `pgrep` 不应再看到对应进程；如果要强制重启，可先：

```bash
pkill -f "onboard/perception/camera/apriltag_detector.py" 2>/dev/null || true
pkill -f "onboard/perception/ball_fuser.py" 2>/dev/null || true
```

---

## 12. `--ball` 模式帧率降至 18 fps

### 根因

YOLO 线程在推理期间持有 librealsense `**frames` Python 对象引用**，时间约 25 ms。
librealsense 的 DMA 帧缓冲区在对应 Python 对象引用计数归零之前无法被相机驱动回收。
主线程 `pipeline.wait_for_frames()` 在缓冲池耗尽时阻塞，造成帧率从 30 fps 跌至 ~18 fps。

### 诊断

```python
# 问题代码（错误示范）：YOLO 线程直接存储了 frames 引用
self._yolo_input = (color_small, depth_frame)   # depth_frame 持有 librealsense 内部缓冲区
```

### 修复

在主线程将数据传递给 YOLO 线程之前，先将 numpy 数组**复制**出来：

```python
# apriltag_detector.py 主循环
depth_arr = np.asanyarray(depth_frame.get_data()).copy()   # ← copy() 释放 DMA 引用
color_small = cv2.resize(color_bgr, (imgsz, imgsz))        # 已经是新数组
self._yolo_input = (color_small, depth_arr)                # 传递纯 numpy，无 librealsense 引用
```

YOLO 线程持有的只是普通 numpy 数组，librealsense 可以立即回收帧缓冲区，主线程不再阻塞。

---

## 13. 启用 `--ball` 后帧率降至 15 fps（深度分辨率设为 424×240）

### 根因

D455 的 librealsense 流配置有严格的**有效组合**限制。
`1280×720 color + 424×240 depth @ 30fps` **不是合法组合**；
librealsense 无法匹配硬件模式，回退到最接近的合法帧率，即 15 fps。

### 有效组合（截至 librealsense 2.54）


| Color 分辨率 | Depth 分辨率 | 帧率           |
| --------- | --------- | ------------ |
| 1280×720  | 848×480   | 30 fps ✅     |
| 1280×720  | 640×360   | 30 fps ✅     |
| 1280×720  | 424×240   | 15 fps（降帧）⚠️ |
| 640×480   | 424×240   | 30 fps ✅     |


### 修复

将深度流分辨率改回 `848×480`：

```python
# apriltag_detector.py
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)   # ✅
# cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30) # ❌ 会降帧至 15fps
```

### 备注

`--ball-hsv` 模式完全不开启深度流，因此不受此限制，始终 30 fps。

---

## 14. 灰白足球颜色/形态检测实验总结（结论：当前不采用）

### 背景

`apriltag_detector.py` 的 `--ball` 模式使用 YOLO 识别足球。
本节记录对 **HSV 色块检测** 和 **Hough 圆形检测** 两种轻量级替代方案的调试实验。
调试工具：`debug/hsv_tuner.py --web`（网页交互式参数调整）。

### 结论

**当前实验环境下两种方案均不可靠，相机暂只用于 AprilTag 识别。** 如需球检测，继续使用 YOLO（`--ball` 模式）。

---

### 14.1 HSV 色块检测的根本限制

**问题**：HSV 调试面板的 "Merged" 区域几乎全黑，球根本没有出现在掩码里。

**根因**：该足球为灰白色，饱和度极低（S ≈ 10–30）。原始参数 H=100–135、S_MIN=70 专门针对蓝紫色色块，与球完全不匹配。只有球上少量蓝色斑块可能被捕捉到，但在实验室光线下极不稳定。

**Merged 面板为空** = HSV 检测的直接判定失败依据。

#### 尝试：用 S_MAX 捕捉灰白色

为此在 `_detect()` 增加了 `s_max` 参数（默认 255=关闭），
允许只保留低饱和度区域（灰白物体）：

```python
# 捕捉灰白球的参数组合
H_LOW=0, H_HIGH=179   # 不限颜色
S_MIN=0, S_MAX=40     # 只要低饱和度（灰/白）
V_MIN=130             # 要求一定亮度（排除深色地毯）
```

**问题**：实验室白色墙面同样满足上述条件（S≈5，V≈230），导致掩码被墙面大斑块淹没，无法用形态学方法分离出球的圆形轮廓。

---

### 14.2 Hough 圆形检测的限制

Hough 检测（`_detect_hough()`）不依赖颜色，在灰度图上寻找圆形边缘。

**问题：场景中圆形物体太多**

实验室场景包含：椅子底座（圆形）、机器人腿/轮（圆形局部）、纸板箱圆角、apriltag 板圆形标记等。在 p2 较低时，`cv2.HoughCircles` 会返回数十个候选圆（画面中的大量红/绿圈）。

#### 已实现的过滤策略


| 过滤器   | 参数                   | 原理                                                 |
| ----- | -------------------- | -------------------------------------------------- |
| 局部对比度 | `HOUGH_MIN_CONTRAST` | 圆内平均亮度 − 外圈（1.6×半径）平均亮度；球比地毯亮                      |
| 内部纹理  | `HOUGH_MAX_TEXTURE`  | 圆内像素标准差；球面光滑（std≈15–30），apriltag 格子高纹理（std≈60–120） |
| 排除顶部  | `EXCL_TOP`           | 球在地面，始终出现在画面下半部（cy > 60–80% of height）             |


**颜色编码（Hough 可视化面板）**：

- 🟢 绿色 = 对比度和纹理均通过
- 🟠 橙色 = 对比度通过但纹理超标（疑似 apriltag 板）
- 🔴 红色 = 对比度未通过
- 🔵 青色 = 最终赢家

**仍存在的问题**：椅子白色塑料座面（光滑 + 对比高）得分与球相近；p2 低时误检数量太多，p2 高时球本身也可能漏检。

---

### 14.3 关键参数说明（web 调参界面）


| 参数                   | 说明                   | 灰白球建议值 |
| -------------------- | -------------------- | ------ |
| `S_MAX`              | 饱和度上限，专为灰白物体设计       | 35–45  |
| `EXCL_TOP`           | 排除顶部 N%，球始终在下半画面     | 50–70% |
| `HOUGH_MAX_TEXTURE`  | 圆内纹理上限，排除 apriltag 板 | 40–50  |
| `HOUGH_MIN_CONTRAST` | 圆内外对比度下限             | 5–8    |
| `HOUGH_P2`           | Hough 累加器阈值（越高候选越少）  | 40–50  |


---

### 14.4 现状与后续

- **相机模块现在只用于 AprilTag 识别**（`run_apriltag_target.sh`）
- 球检测继续使用 `--ball` 模式（YOLO）
- `debug/hsv_tuner.py --web` 保留为调试工具，如需重新尝试颜色/形态检测可用
- 如果将来更换颜色鲜明的足球（如橙色、黄色），HSV 方案可快速重启

