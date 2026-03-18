# Camera Ball Detector — RealSense D435 + YOLOv8

`onboard/perception/camera/ball_detector.py`

机载相机感知方案。RealSense D435 提供彩色图像和深度图，YOLOv8 检测足球，深度反投影后经运动学链变换到 pelvis 坐标系，通过 DDS 发布 `rt/ball_state`。

> 与雷达方案（`onboard/perception/lidar/`）发布同一 DDS Topic，`deploy_real.py` 无需修改，两套方案可以互换。

---

## 目录

1. [整体架构](#1-整体架构)
2. [双线程模型](#2-双线程模型)
3. [YOLO 检测与追踪](#3-yolo-检测与追踪)
4. [Coasting（惯性保持）](#4-coasting惯性保持)
5. [深度采样](#5-深度采样)
6. [坐标变换链](#6-坐标变换链)
7. [EMA 位置滤波](#7-ema-位置滤波)
8. [DDS 发布](#8-dds-发布)
9. [关节角来源](#9-关节角来源)
10. [实时性保障](#10-实时性保障)
11. [相机硬件重置](#11-相机硬件重置)
12. [关键参数速查](#12-关键参数速查)
13. [与 catch_ball/camera_ball.py 的差异](#13-与-catch_ballcamera_ballpy-的差异)

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│ 主线程                                                             │
│  pipeline.wait_for_frames()  ← 阻塞，等待下一帧（最多 17ms@60Hz）  │
│  align.process()             ← depth 对齐到 color 视角            │
│  → 写 buf_color / buf_depth                                       │
│  → buf_updated.set()         ← 通知 YOLO 线程                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │ threading.Event
┌──────────────────────────▼───────────────────────────────────────┐
│ YOLO 线程（daemon）                                                │
│  buf_updated.wait()          ← 阻塞等待新帧                       │
│  → 拷贝 buf_color / buf_depth                                     │
│  → model.track()             ← YOLO + ByteTrack 检测             │
│  → 深度 patch 采样                                                 │
│  → rs2_deproject → optical_to_body → EMA → camera_to_base        │
│  → dds.publish("rt/ball_state")                                   │
└──────────────────────────────────────────────────────────────────┘
                                │ DDS（网线同网段）
┌──────────────────────────────▼───────────────────────────────────┐
│ 本地电脑 deploy_real.py                                            │
│  BallStateSubscriber → state_cmd.ball_pos_b → Score._build_obs() │
└──────────────────────────────────────────────────────────────────┘
```

线程间通信：

| 对象 | 方向 | 保护内容 |
|------|------|---------|
| `buf_lock` | 主→YOLO | `buf_color`, `buf_depth` |
| `buf_updated` | 主→YOLO | 新帧到达信号（Event，非 Semaphore） |
| `stop_flag` | 主→YOLO | 退出信号 |

**为什么双线程？** 相机采集（`wait_for_frames`）和 YOLO 推理串行时，DDS 发布帧率 = min(相机帧率, YOLO帧率)。双线程后两者独立运行：主线程以相机全速（60Hz）采集，YOLO 线程以 GPU 算力上限（Orin NX 约 25-40fps）单独推理。

---

## 2. 双线程模型

### 主线程（相机采集）

```python
with buf_lock:
    buf_color = np.asanyarray(cf.get_data()).copy()  # 深拷贝彩色图
    buf_depth = np.asanyarray(df.get_data())         # uint16 array，不拷贝
buf_updated.set()
```

`buf_depth` 不做拷贝是因为 YOLO 线程会在 `with buf_lock` 内部再做 `.copy()`，主线程无需重复拷贝。

### YOLO 线程（消费）

```python
buf_updated.wait(timeout=1.0)   # 等待新帧，1s 超时防止死锁
buf_updated.clear()             # 重置，确保每帧只处理一次

with buf_lock:
    color = buf_color.copy()
    depth = buf_depth.copy()    # uint16 → 之后乘 0.001 转为米
```

`buf_updated` 是 `threading.Event`（而非 Semaphore）：主线程在 YOLO 处理期间可能产生多帧，YOLO 只取**最新的那帧**，旧帧自动丢弃。这正是"最新帧优先"的期望行为。

---

## 3. YOLO 检测与追踪

```python
results = model.track(
    color,
    conf=CONF_THRESHOLD,  # 默认 0.3，略低阈值提升召回率
    persist=True,         # 跨帧保持 ByteTrack 状态（必须设置）
    verbose=False,
    imgsz=args.imgsz,     # 推理输入尺寸，默认 480
)
```

`persist=True` 让 ByteTrack 追踪器在连续帧之间保持状态，支持卡尔曼滤波预测和 ID 关联，减少运动模糊时的漏检。

筛选 COCO class 32（`sports ball`），取置信度最高的一个：

```python
for box in result.boxes:
    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:  # 32
        if float(box.conf[0]) > best_conf:
            best_conf, best_box = float(box.conf[0]), box
```

### imgsz 对速度的影响

相机输出 848×480，YOLO 推理前 resize：

| imgsz | 推理分辨率 | Orin NX 参考时间 | 适用场景 |
|-------|-----------|-----------------|---------|
| 640 | 640×360 | ~30ms | 高精度、球较小 |
| 480 | 480×272 | ~20ms | 默认，平衡（推荐） |
| 320 | 320×192 | ~12ms | 最低延迟、近距离球 |

---

## 4. Coasting（惯性保持）

YOLO 在运动模糊、部分遮挡、光照变化时可能漏检。若漏检时立即清零位置，DDS 输出会出现跳变，影响 Score 策略的观测。

实现：允许连续漏检最多 `COAST_FRAMES=10` 帧，期间继续使用最后一次成功检测的 BBox 位置。

```python
if best_box is not None:
    miss_count = 0
    last_bbox  = tuple(map(int, best_box.xyxy[0]))
else:
    miss_count += 1

if last_bbox is not None and miss_count <= COAST_FRAMES:
    # 正常处理：取深度、坐标变换、DDS 发布
    # valid 字段：True=YOLO当帧检测到，False=Coasting
    dds.publish(x, y, z, valid=(best_box is not None))
else:
    dds.publish(0.0, 0.0, 0.0, valid=False)
```

终端状态：

| 显示 | 含义 |
|------|------|
| `[BALL ]` | 当前帧 YOLO 检测到球，`valid=True` |
| `[COAST]` | YOLO 漏检，沿用最后位置，`valid=False` |
| `[     ]` | 连续漏检超过 10 帧，发布零点，`valid=False` |

Coasting 期间深度仍从 `last_bbox` 的像素位置实时重新采样。如果球已移动到别处，深度会取到错误值，EMA 的跳变门限（0.6m）会拒绝该异常测量，保留上一帧的 EMA 值。

---

## 5. 深度采样

单点深度噪声约 ±3mm（1m 处），且物体边缘存在飞点（depth=0 或中间值）。采用 patch 中位数：

```python
patch   = depth[y0d:y1d+1, x0d:x1d+1].astype(np.float32) * 0.001  # mm→m
valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]          # 0.1~10m
depth_m = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0
```

`DEPTH_SAMPLE_RADIUS=5` → 采样区域 11×11=121 像素，中位数对 30% 以内的异常值免疫。

**注意**：深度操作全程在 numpy array 上进行（不访问 RealSense SDK 帧对象），确保跨线程安全（SDK 帧对象有内部引用计数，不可跨线程访问）。

---

## 6. 坐标变换链

```
RealSense D435
    │ rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
    ▼
光学坐标系（Z前，X右，Y下）
    │ optical_to_body()
    ▼
相机 body 坐标系（X前，Y左，Z上）
    │ EMA 滤波（在 camera body 系做，噪声特性最稳定）
    ▼
p_cam_ema（平滑后）
    │ transform_point_camera_to_base(p_cam, q_wy, q_wr, q_wp, q_head)
    ▼
pelvis/base 坐标系（X前，Y左，Z上）
    │ DDS publish "rt/ball_state"
    ▼
deploy_real.py → state_cmd.ball_pos_b
```

### 运动学链（来自 URDF g1_sysid_23dof.urdf）

```
pelvis
  └─ waist_yaw_joint   Rz(q_wy),   t=[0, 0, 0]
      └─ waist_roll_joint  Rx(q_wr),   t=[-0.0039635, 0, 0.044]
          └─ waist_pitch_joint  Ry(q_wp),   t=[0, 0, 0]
              └─ head_joint   Ry(q_head),  t=[0.0039635, 0, 0.3159]
                  └─ head_camera_joint（固定）
                         xyz=[0.0448, 0.01, 0.1219]
                         rpy=[0.0119, 0.8377, 0.0053]
```

实现为链式齐次变换矩阵乘法（`camera_to_base.py`）：

```python
T = (
    _T(_Rz(q_wy), [0.0, 0.0, 0.0])
    @ _T(_Rx(q_wr), [-0.0039635, 0.0, 0.044])
    @ _T(_Ry(q_wp), [0.0, 0.0, 0.0])
    @ _T(_Ry(q_head), [0.0039635, 0.0, 0.3159])
    @ _T_HEAD_CAMERA
)
```

关节角每帧实时传入，支持机器人运动中的动态更新。

**为什么 EMA 在 camera 系而非 base 系做？** 传感器测量噪声在相机坐标系下各向异性特性最稳定（主要是深度 Z 向误差）。经过含旋转的运动学链变换后，噪声特性在 base 系中变得复杂，不适合直接滤波。

---

## 7. EMA 位置滤波

在相机 body 坐标系对球的 3D 位置做指数滑动平均：

```python
if center_ema is None:
    center_ema = p_cam_arr.copy()                    # 冷启动
elif np.linalg.norm(p_cam_arr - center_ema) < EMA_GATE:
    center_ema = EMA_ALPHA * p_cam_arr + (1 - EMA_ALPHA) * center_ema
# 超过 EMA_GATE 的跳变：拒绝，保持当前 EMA 不变
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `EMA_ALPHA` | 0.6 | 平滑系数；越大跟踪越快，越小越平滑 |
| `EMA_GATE` | 0.6 m | 跳变门限；新测量与当前 EMA 差超过此值则拒绝 |

门限目的：过滤 YOLO 误检、深度无效导致的反投影错误、Coasting 期间背景深度采样错误。正常球速（< 3m/s）不会触发。

---

## 8. DDS 发布

```python
from common.ball_state_dds import BallStatePublisher

dds = BallStatePublisher(domain_id=0)
dds.publish(x, y, z, valid=True)   # valid=False 表示未检测到球
```

消息结构（`common/ball_state_dds.py`）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `timestamp_us` | uint64 | 发布时刻（µs since epoch） |
| `x / y / z` | float32 | 球心在 **pelvis body 系** 的坐标（m） |
| `valid` | uint8 | `1`=检测到；`0`=未检测到（Coasting 或无球） |

- **Topic**：`rt/ball_state`
- **QoS**：BestEffort，KeepLast(1)
- **频率**：与 YOLO 推理帧率相同，约 25~40 Hz（受 Orin NX 算力限制）

> 雷达方案发布频率约 10 Hz（受 MID360 点云帧率限制）。相机方案可以更高频，Score 策略通常以 50Hz 运行，插值逻辑在 `deploy_real.py` 中处理。

`deploy_real.py` 的本地验证：

```bash
python tools/check_ball_state.py
# 正常输出：[OK ]  x=+0.823  y=-0.012  z=-0.673  age=85ms
```

---

## 9. 关节角来源

关节角通过 ROS2 `/lowstate` 话题获取（与雷达方案一致）：

```python
class _JointListener(Node):
    def _cb(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]   # waist_yaw_joint
        self.q_wr = q[13]   # waist_roll_joint
        self.q_wp = q[14]   # waist_pitch_joint
```

`q_head = 0.593412 rad`（≈34°）固定为 URDF 默认值——G1 头部关节在部署时不主动运动，实际保持默认俯仰角。

YOLO 线程每帧读 `joint.q_wy` 等属性时无需加锁：Python GIL 保证 float 赋值是原子操作。

---

## 10. 实时性保障

### SCHED_FIFO 实时调度（需要权限）

```python
param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
os.sched_setscheduler(0, os.SCHED_FIFO, param)
```

使 YOLO 线程不被普通进程抢占。需要 `CAP_SYS_NICE`：

```bash
sudo python onboard/perception/camera/ball_detector.py
# 或一次性授权（不需要每次 sudo）：
sudo setcap cap_sys_nice+ep $(which python)
```

无权限时自动降级为 `os.nice(-10)`（效果有限）。

### Orin NX 上的 GPU 推理

确保 PyTorch 使用 CUDA：

```python
import torch
print(torch.cuda.is_available())   # 应为 True
```

YOLOv8n 在 Orin NX 上约 12-20ms/帧，YOLOv8s 约 20-30ms/帧。选择模型时在精度和延迟间权衡。

---

## 11. 相机硬件重置

进程异常退出后 RealSense 可能处于"流传输中"状态，再次启动时 `wait_for_frames` 超时。自动处理：

```python
def _start_pipeline():
    for attempt in range(2):
        profile = pipeline.start(rs_cfg)
        try:
            pipeline.wait_for_frames(timeout_ms=5000)
            return profile
        except RuntimeError:
            pipeline.stop()
            ctx.query_devices()[0].hardware_reset()  # USB 重新枚举
            time.sleep(3)
    raise RuntimeError("RealSense failed to start after hardware reset.")
```

`hardware_reset()` 等效于物理拔插相机，3 秒后自动恢复，最多重试 2 次。

---

## 12. 关键参数速查

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `CONF_THRESHOLD` | 0.3 | 顶部常量 | YOLO 置信度阈值，越低召回越高但误检增多 |
| `DEPTH_SAMPLE_RADIUS` | 5 px | 顶部常量 | 深度 patch 半径（11×11 共 121 点） |
| `DEPTH_MIN` | 0.1 m | 顶部常量 | 深度有效下限 |
| `DEPTH_MAX` | 10.0 m | 顶部常量 | 深度有效上限 |
| `EMA_ALPHA` | 0.6 | 顶部常量 | EMA 平滑系数 |
| `EMA_GATE` | 0.6 m | 顶部常量 | EMA 跳变拒绝门限 |
| `COAST_FRAMES` | 10 帧 | 顶部常量 | 漏检后保持位置的最大帧数 |
| `--model` | yolov8n.pt | 命令行 | YOLO 模型路径 |
| `--imgsz` | 480 | 命令行 | YOLO 推理输入尺寸（像素） |

---

## 13. 与 catch_ball/camera_ball.py 的差异

本服务直接从 `catch_ball/camera_ball.py` 改造而来，核心逻辑完全复用。差异如下：

| 方面 | catch_ball/camera_ball.py | onboard/camera/ball_detector.py |
|------|--------------------------|--------------------------------|
| **运行位置** | 本地电脑（通过 USB 直连相机） | G1 机载电脑（Orin NX） |
| **结果发布** | LCM `camera_ball_lcmt` | DDS `rt/ball_state`（与雷达方案同接口） |
| **关节角** | ROS2 optional（可退化到固定值/命令行） | ROS2 必须（机载电脑已有 ROS2 Foxy） |
| **可视化** | 可选 OpenCV 窗口 | 去掉（机载无 display） |
| **速度外推** | 有（显示用的像素速度外推） | 去掉（纯感知服务，不显示） |
| **YOLO 可选** | — | — |
| **坐标变换** | `transform/camera_to_base.py` | `onboard/perception/camera/camera_to_base.py`（内容相同） |
| **`q_head`** | 命令行可设置 | 固定 0.593412 rad（URDF 默认） |
