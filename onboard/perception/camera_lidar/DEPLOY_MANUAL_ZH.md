# 融合球感知模块部署手册

**模块**：`onboard/perception/ball_detector_fused.py`  
**感知源**：胸部 RealSense D435（YOLO 目标检测）+ 头顶 Livox MID360 激光雷达  
**输出**：`rt/ball_state` DDS 话题（球心在骨盆坐标系中的三维位置，50 Hz）

---

## 1. 系统概述

### 1.1 传感器融合逻辑

```
每帧（50 Hz）：
  ┌─────────────────────────────────────────┐
  │  胸部相机 YOLO 检测到球？                 │
  │  YES → 使用相机深度（D435 + 视差校正）    │
  │  NO  → 使用激光雷达质心（MID360）         │
  │  两者均超时 → 仅卡尔曼预测，valid=False   │
  └─────────────────────────────────────────┘
       ↓
  Kalman 滤波器（三维匀速运动模型）
       ↓
  DDS 发布 rt/ball_state（valid + xyz）
```

### 1.2 头关节角度

激光雷达安装在头部，头关节当前固定为 **2.3°（向下）**，代码中对应：

```python
HEAD_JOINT_ANGLE = np.radians(2.3)   # 位于 ball_detector_fused.py 顶部
```

> **注意**：如果头部角度有变动，修改该常量即可，无需改动运动学链。

---

## 2. 硬件要求

| 硬件 | 规格 | 说明 |
|------|------|------|
| 胸部相机 | RealSense D435 | USB 3.0，胸口位置 |
| 激光雷达 | Livox MID360 | 头顶安装，已完成 URDF 标定 |
| 计算平台 | Unitree G1 板载电脑 (Jetson Orin) | CUDA 推理 |
| YOLO 模型 | yolo11m.pt 或 .engine | 见第 4 节 |

---

## 3. 软件依赖

在板载电脑上确认以下依赖已安装：

```bash
# Python 包
pip install pyrealsense2 ultralytics torch numpy opencv-python

# ROS2（foxy 或 humble）
ls /opt/ros/foxy/setup.bash      # 或 humble

# Livox ROS2 驱动
ls ~/yixuan/yichao-deploy/ws_livox/install/setup.sh

# CycloneDDS Python 绑定（已包含在 common/ball_state_dds.py 中）
python -c "import cyclonedds"
```

---

## 4. 启动前准备

### 4.1 确认相机序列号

```bash
cd /path/to/RoboMimic_Deploy
python onboard/perception/ball_detector_fused.py --list-cameras
```

输出示例：
```
[INFO] Connected RealSense devices:
  [0] serial=123456789  Intel RealSense D435
```

如果只有一个相机，默认使用第 0 个作为胸部相机（无需额外参数）。  
如果有多个相机，记录胸部相机序列号，启动时用 `--chest-serial` 指定。

### 4.2 确认激光雷达 ROS2 话题

在另一个终端检查 `/livox/lidar` 话题：

```bash
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh
ros2 topic hz /livox/lidar
```

正常输出：`average rate: 10.000`  
如果没有输出，参考 `onboard/perception/lidar/LIDAR_FREQ_TROUBLESHOOTING.md`。

### 4.3 确认 /lowstate 话题

```bash
ros2 topic hz /lowstate
```

正常输出：`average rate: 500.000`

### 4.4 导出 TensorRT 引擎（可选但强烈推荐）

```bash
cd /path/to/RoboMimic_Deploy
python -c "
from ultralytics import YOLO
model = YOLO('onboard/perception/camera/models/yolo11m.pt')
model.export(format='engine', imgsz=320, device=0)
"
```

导出完成后，`yolo11m.engine` 会自动被检测并优先使用（推理速度提升约 3-4×）。

---

## 5. 启动步骤

### 5.1 方法一：使用 run_fused.sh（推荐）

```bash
cd /path/to/RoboMimic_Deploy

# 普通启动
bash onboard/perception/run_fused.sh

# 带可视化（浏览器查看胸部相机 + 检测框）
bash onboard/perception/run_fused.sh --show

# 指定胸部相机序列号
bash onboard/perception/run_fused.sh --chest-serial 123456789

# 指定 TRT 引擎
bash onboard/perception/run_fused.sh --model onboard/perception/camera/models/yolo11m.engine
```

### 5.2 方法二：直接运行 Python

```bash
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh
cd /path/to/RoboMimic_Deploy

python onboard/perception/ball_detector_fused.py
```

### 5.3 所有命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `models/yolo11m.pt` | YOLO 模型路径（.pt 或 .engine） |
| `--imgsz` | `320` | YOLO 输入分辨率（像素），越小越快 |
| `--width` | `640` | D435 采集宽度 |
| `--height` | `480` | D435 采集高度 |
| `--chest-serial` | （第一个设备） | 胸部相机序列号 |
| `--list-cameras` | — | 列出连接的相机后退出 |
| `--show` | — | 启动 MJPEG 服务器（端口 8080） |

---

## 6. 运行状态验证

### 6.1 观察终端输出

正常运行时轮流打印以下内容：

```
[chest/BALL ] pelvis=(+0.803,+0.012,-0.714)  d=0.91m  conf=0.82  fps=28.4
[lidar] pelvis=(+0.815,+0.009,-0.708)  cand=18
[chest/BALL ] pelvis=(+0.797,+0.033,-0.705)  d=0.89m  conf=0.79  fps=28.1
```

- `[chest/BALL ]`：相机检测到球，正在发布相机结果
- `[chest/COAST]`：相机正在"惯性"追踪（YOLO 丢失但 bbox 保留 10 帧）
- `[chest/     ]`：相机完全丢失球
- `[lidar]`：激光雷达检测到球，信息实时刷新

### 6.2 查看 DDS 数据

在控制机（或另一个终端）运行：

```bash
python tools/check_ball_state.py
```

正常输出：
```
[ball] valid=True  pos=(+0.801, +0.015, -0.711)  age=8ms
```

### 6.3 浏览器可视化（--show 模式）

在开发机浏览器中打开：
```
http://192.168.123.164:8080/stream
```

图像显示：
- 绿色框 = YOLO 检测到球
- 橙色框 = 惯性追踪（coast）
- 左上角显示帧率
- 左下角显示骨盆坐标系下的球位置

---

## 7. 关键参数调整

所有参数均在 `ball_detector_fused.py` 文件顶部的常量区，修改后重启生效。

### 7.1 头关节角度（影响激光雷达坐标变换）

```python
HEAD_JOINT_ANGLE = np.radians(2.3)   # 修改为实际头部俯仰角（度）
```

### 7.2 融合优先级时间窗口

```python
CAM_STALE_SEC   = 0.40  # 相机结果超过此时间（秒）后切换到激光雷达
LIDAR_STALE_SEC = 0.50  # 激光雷达结果超过此时间后切换到纯预测
```

### 7.3 激光雷达检测阈值

```python
LIDAR_REFLECT_THR = 130   # 反射率阈值（球有高反射贴膜时可调低到 100）
LIDAR_MIN_POINTS  = 4     # 候选点云最少点数
LIDAR_MAX_RANGE   = 1.8   # 最大检测距离（米）
```

### 7.4 卡尔曼滤波器噪声

```python
_R_CAM   = np.diag([0.010, 0.010, 0.010])  # 相机测量噪声协方差
_R_LIDAR = np.diag([0.040, 0.040, 0.040])  # 激光雷达测量噪声协方差
```

值越大，滤波器对该传感器的信任度越低（输出更依赖预测）。

### 7.5 卡尔曼过程噪声

```python
# 在 _FusionKF 类中
_Q0 = np.diag([0.02, 0.02, 0.02, 0.50, 0.50, 0.50])
```

前三项（位置）和后三项（速度）越大，预测越"跟得上"快速运动，但也更抖。

---

## 8. 胸部相机外参标定

如果球位置偏差较大，需要更新胸部相机外参：

打开 `onboard/perception/camera/camera_to_base.py`，修改以下两行：

```python
_CHEST_XYZ = [0.1289635, 0.00, 0.066]   # 相机在 waist_pitch 坐标系中的位置（米）
_CHEST_RPY = (0.00, 0.523599, 0.00)      # 相机姿态（roll, pitch, yaw，弧度）
```

**测量方法**：
1. 将机器人站立于已知位置
2. 放置一个已知坐标的球
3. 运行 `--show` 模式，记录输出的 `pelvis` 坐标
4. 与真实值对比，调整 `_CHEST_XYZ` 直到误差 < 3 cm

---

## 9. 常见问题排查

### Q1：`No RealSense device found`

```bash
# 检查 USB 连接
lsusb | grep Intel

# 尝试拔插后重启
rs-enumerate-devices
```

### Q2：`/livox/lidar` 话题无数据

```bash
# 检查 Livox 驱动是否运行
ros2 node list | grep livox

# 如果没有，手动启动驱动
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

详细排查见：`onboard/perception/lidar/LIDAR_FREQ_TROUBLESHOOTING.md`

### Q3：`KeyError: 'radius'` 崩溃

已修复（`deploy_mujoco/deploy_mujoco.py` 中的可视化 bug）。  
如仍出现，运行最新版本的 `deploy_mujoco.py`。

### Q4：检测不到球（相机一直显示 `no ball`）

- 确认球具有白色或高对比度外观（YOLO 在 class 32 = sports ball 上训练）
- 降低置信度阈值：`CONF_THRESHOLD = 0.20`
- 检查 `--imgsz`：默认 320，可提高到 640 提升精度（但速度更慢）

### Q5：激光雷达候选点少（`cand=0`）

- 确认球上贴有高反射率贴纸（反射率需 ≥ `LIDAR_REFLECT_THR=130`）
- 降低阈值：`LIDAR_REFLECT_THR = 100`
- 检查球是否在检测范围内（默认 0.2–1.8 m，`LIDAR_MAX_RANGE`）

### Q6：`rt/ball_state` 中 `valid=False` 虽然看到球

- 检查 DDS domain ID（控制端和感知端均需为 0）
- 用 `tools/check_ball_state.py` 在机器人本机验证 DDS 是否发出

### Q7：YOLO 首次启动很慢（~30 s）

正常现象：CUDA JIT 编译需要时间。使用 TensorRT 引擎（`.engine`）可将此延迟减少到 5 s。

---

## 10. 文件索引

| 文件 | 说明 |
|------|------|
| `onboard/perception/ball_detector_fused.py` | **本模块主程序** |
| `onboard/perception/run_fused.sh` | 启动脚本 |
| `onboard/perception/camera/camera_to_base.py` | 相机→骨盆坐标变换（含胸部相机外参） |
| `onboard/perception/lidar/mid360_to_base.py` | MID360→骨盆坐标变换 |
| `onboard/perception/lidar/center_kalman_filter.py` | 原独立激光雷达 KF（本模块内置了融合版 KF） |
| `common/ball_state_dds.py` | DDS 发布/订阅封装 |
| `tools/check_ball_state.py` | 验证 DDS 数据工具 |

---

## 11. 附录：坐标系约定

所有输出位置均在**骨盆（pelvis）坐标系**中：

```
X → 机器人正前方
Y → 机器人左侧
Z → 向上

原点 = 骨盆关节中心
```

`rt/ball_state` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `x` | float32 | 球心前后位置（米，正 = 前方） |
| `y` | float32 | 球心左右位置（米，正 = 左侧） |
| `z` | float32 | 球心上下位置（米，正 = 上方） |
| `valid` | bool | 最近窗口内是否有有效检测 |
| `timestamp_us` | uint64 | 发布时的微秒时间戳 |
