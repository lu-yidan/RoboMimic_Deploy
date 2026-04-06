# Camera + LiDAR 头部融合球检测（双进程架构）

头部 D435 相机 (YOLO) 与头部 MID360 激光雷达在**独立进程**中并行运行，通过 DDS 通信。
轻量级融合节点按**相机优先、LiDAR 补充**的逻辑合并输出 `rt/ball_state`。

---

## 为什么用双进程

| | 多线程（旧方案） | 双进程（当前） |
|---|---|---|
| **GIL 竞争** | 7 线程抢一把 GIL → YOLO 被 lidar 阻塞 | 各进程独立 GIL → 完全并行 |
| **Camera FPS** | ~3-4 FPS | **~20-30 FPS**（和单独 run.sh 一样） |
| **LiDAR FPS** | ~10 FPS | ~10 FPS（不变） |
| **融合延迟** | < 20ms（线程共享内存） | < 25ms（DDS 传输） |

---

## 架构

```
┌──────────────────────────┐     ┌───────────────────────────┐
│  进程 1: camera detector  │     │  进程 2: lidar detector    │
│  camera/ball_detector.py  │     │  lidar/ball_detector.py    │
│                           │     │                            │
│  RealSense D435 capture   │     │  ROS2 /livox/lidar         │
│  → YOLO (TensorRT)        │     │  → reflectivity filter     │
│  → depth + EMA            │     │  → center estimate         │
│  → pelvis transform       │     │  → pelvis transform        │
│                           │     │                            │
│  DDS: rt/cam_ball_state   │     │  DDS: rt/lidar_ball_state  │
│       (~20-30 Hz)         │     │       (~10 Hz)             │
└────────────┬──────────────┘     └──────────────┬─────────────┘
             │                                   │
             │         DDS (进程间通信)            │
             ▼                                   ▼
        ┌─────────────────────────────────────────────┐
        │       进程 3: fusion_node.py                 │
        │                                              │
        │   if cam_age < 200ms:  用相机                │
        │   elif lidar_age < 400ms:  用 LiDAR          │
        │   else:  valid=False                         │
        │                                              │
        │   DDS: rt/ball_state  (~50 Hz)               │
        └──────────────────────────────────────────────┘

        ┌──────────────────────────────────┐
        │  Livox MID360 驱动（后台）         │
        │  ros2 launch livox_ros_driver2   │
        │  → /livox/lidar topic            │
        └──────────────────────────────────┘
```

---

## 文件结构

```
onboard/perception/camera_lidar/
  run_fused.sh        ← 一键启动（Livox + camera + lidar + fusion）
  fusion_node.py      ← 轻量融合进程
  ball_detector_fused.py  ← 旧版单进程方案（保留备用）
  README.md           ← 本文档
```

---

## 快速启动

```bash
cd ~/yixuan/yichao-deploy/RoboMimic_Deploy

# 默认设置
bash onboard/perception/camera_lidar/run_fused.sh

# 开启浏览器可视化（端口 8080，相机画面）
bash onboard/perception/camera_lidar/run_fused.sh --show
```

**Ctrl-C** 退出时自动关闭所有进程（Livox、camera、lidar、fusion）。

---

## DDS Topic 说明

| Topic | 发布者 | 频率 | 用途 |
|-------|--------|------|------|
| `rt/cam_ball_state` | camera/ball_detector.py | ~20-30 Hz | 相机检测结果 |
| `rt/lidar_ball_state` | lidar/ball_detector.py | ~10 Hz | LiDAR 检测结果 |
| `rt/ball_state` | fusion_node.py | ~50 Hz | **融合输出**（policy 层消费） |

---

## 融合逻辑

```
每 1/50 秒：
  读取 rt/cam_ball_state 最新样本
  读取 rt/lidar_ball_state 最新样本

  if 相机样本新鲜 (age < 200ms) 且 valid:
      转发相机位置 → rt/ball_state, valid=True
  elif LiDAR 样本新鲜 (age < 400ms) 且 valid:
      转发 LiDAR 位置 → rt/ball_state, valid=True
  else:
      发布 (0,0,0), valid=False
```

---

## 可调参数

### fusion_node.py 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cam-topic` | `rt/cam_ball_state` | 相机 DDS topic |
| `--lidar-topic` | `rt/lidar_ball_state` | LiDAR DDS topic |
| `--out-topic` | `rt/ball_state` | 融合输出 topic |
| `--fusion-hz` | `50` | 融合循环频率 |
| `--cam-stale-ms` | `200` | 相机数据过期时间 (ms) |
| `--lidar-stale-ms` | `400` | LiDAR 数据过期时间 (ms) |

### camera/ball_detector.py 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dds-topic` | `rt/ball_state` | DDS 发布 topic（融合模式下传 `rt/cam_ball_state`） |

### lidar/ball_detector.py 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dds-topic` | `rt/ball_state` | DDS 发布 topic（融合模式下传 `rt/lidar_ball_state`） |

---

## 单独使用（不融合）

Camera 和 LiDAR 检测器仍然可以单独运行，默认发布到 `rt/ball_state`：

```bash
# 只用相机
bash onboard/perception/camera/run.sh --show

# 只用 LiDAR
python onboard/perception/lidar/ball_detector.py
```

---

## 常见问题

### Q1：Camera FPS 低于 20

- 确认 `.engine` 文件加载成功（看启动日志中的 `TensorRT engine found`）
- 确认 `jetson_clocks` 执行成功
- 降低 `--imgsz`（如 160）进一步提速

### Q2：LiDAR 无输出

- 确认 Livox 驱动已启动：`ros2 topic hz /livox/lidar`
- 检查 `LIDAR_REFLECT_THR` 反射率阈值

### Q3：`failed to set power state`

- D435 USB 状态异常，拔掉 USB 等 10 秒重插
- 确保没有残留进程占用相机：`ps aux | grep ball_detector`

### Q4：如何切回单进程方案

```bash
# 旧版单进程（camera + lidar 在同一个 Python 进程）
python onboard/perception/camera_lidar/ball_detector_fused.py --show
```
