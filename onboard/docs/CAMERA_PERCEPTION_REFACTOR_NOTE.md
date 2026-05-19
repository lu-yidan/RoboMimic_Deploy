# Camera Perception 重构说明

本文记录近期 camera perception 相关重构的变更点、运行方式和现场验证结果。整体运行架构仍以
`CAMERA_PERCEPTION_ARCHITECTURE.md` 为准；本文只说明这次改动的动机和操作注意事项。

## 目标

- 让灰度 AprilTag、亮球检测、lidar raw ball 和最终 policy ball topic 的职责更清楚。
- 减少灰度链路中不必要的 BGR 转换，把 CPU 时间留给 AprilTag / bright-ball 检测。
- 把最终 ball 平滑集中到 `ball_fuser.py`，避免各传感器各自发布 policy topic。
- 提供一组更接近实机操作的启动脚本和 dashboard，方便现场按 pane 检查数据流。

## Topic 与职责

- `rt/target_state`：由 camera AprilTag 服务发布，表示 AprilTag target 在 pelvis body 系的位置。
- `rt/cam_ball_state`：由 camera bright-ball / YOLO 链路发布，只作为 raw camera ball observation。
- `rt/lidar_ball_state`：由 lidar ball detector 发布，只作为 raw lidar ball observation；默认 topic 已改为这个 raw topic。
- `rt/ball_state`：只由 `onboard/perception/ball_fuser.py` 发布，供 policy / bridge 使用。

`ball_fuser.py` 订阅 lidar 和 camera raw topic，按 `lidar > camera` 的优先级选择观测，再经过统一的
`CenterKalmanFilter` 平滑后发布 `rt/ball_state`。这样最终策略输入只有一个 owner。

## 主要改动

- `onboard/perception/camera/apriltag_detector.py`
  - 灰度 V4L2 输入保持单通道处理，只有预览或显示需要时才转 BGR。
  - AprilTag 和 bright-ball 共用同一个 camera owner，避免多个进程抢同一 RealSense / V4L2 节点。
  - bright-ball worker 使用 latest-frame 语义，检测落后时丢弃旧帧。
  - bright-ball worker 不再直接向终端打印；camera 主循环统一低频输出状态，避免 pane 中多路日志互相覆盖。
  - `SIGTERM` 会按 `KeyboardInterrupt` 处理，进入 `finally` 清理 camera 和 bright worker。
  - 增加 `--profile-timing` / `--profile-window`，可输出 capture、bright、apriltag、preview、total 等阶段耗时。

- `onboard/perception/camera/config.py`
  - 集中放置灰度相机、AprilTag board、bright-ball 的默认参数。
  - 默认灰度相机配置为 `/dev/video3`、`GREY`、`30Hz`。

- `onboard/perception/camera/timing.py`
  - 新增轻量 `StageTimer`，用于现场快速看 mean / p95 耗时。

- `onboard/perception/ball_fuser.py`
  - 作为最终 `rt/ball_state` 的唯一发布者。
  - 将最终 ball Kalman smoothing 从单个传感器输出后移到融合层。
  - 支持 `--lidar-topic`、`--cam-topic`、`--output-topic`、`--hz`、`--status-hz` 等参数。

- `onboard/perception/lidar/ball_detector.py`
  - 默认 DDS topic 调整为 `rt/lidar_ball_state`，表示 raw lidar ball。
  - 通过 Unitree DDS `rt/lowstate` 获取 waist joints；listener 运行在隔离子进程中，避免和 ROS2 / Livox / rclpy 主进程冲突。
  - 状态行打印 `waist_q` 和 `joint_age_s`，用于确认 MID360->pelvis transform 是否真的用了实时腰部关节。

- 启动和调试脚本
  - `onboard/perception/camera/run_gray_perception.sh`：默认启动灰度 AprilTag + bright-ball；加 `--with-fuser` 同时启动 fuser。
    - `--with-fuser` 下 fuser 输出重定向到 `/tmp/ball_fuser.log`，camera pane 只保留 camera 统一状态。
    - camera / fuser 以独立 process group 管理；`Ctrl-C`、`TERM`、`EXIT` 会同时清理两边。
  - `onboard/perception/run_ball_fuser.sh`：单独启动 fuser。
  - `onboard/perception/lidar/run.sh`：显式设置 `ROS_LOCALHOST_ONLY=0`，避免 `rt/lidar_ball_state` 只在 localhost 可见或被其他 pane 看不到。
  - `onboard/perception/run_sensor_dashboard.sh`：启动 target / camera ball / lidar ball / fused ball dashboard。
  - `tools/start_tmux_layout.sh`：预填 bridge、policy、灰度感知、lidar、dashboard 的 tmux pane 命令，人工确认后逐 pane 回车启动。

## 常用运行方式

只启动灰度相机感知：

```bash
bash onboard/perception/camera/run_gray_perception.sh
```

灰度相机感知 + final ball fuser：

```bash
bash onboard/perception/camera/run_gray_perception.sh --with-fuser --show
```

此模式下 fuser 日志在：

```bash
tail -f /tmp/ball_fuser.log
```

单独启动 lidar raw ball：

```bash
bash onboard/perception/lidar/run.sh --show --base-y-bias 0.05 --dds-topic rt/lidar_ball_state
```

单独启动 fuser：

```bash
bash onboard/perception/run_ball_fuser.sh
```

启动传感器 dashboard：

```bash
bash onboard/perception/run_sensor_dashboard.sh
```

启动 tmux 现场布局：

```bash
bash tools/start_tmux_layout.sh
```

`start_tmux_layout.sh` 只会预填命令，不会自动启动硬件服务；进入 tmux 后按 pane 检查环境变量、设备和 topic，再逐个回车。

## 操作注意事项

- 灰度链路优先使用 `run_gray_perception.sh --with-fuser`，避免忘记启动最终 `rt/ball_state`。
- 如果要在同一个 pane 看 fuser 状态，查看 `/tmp/ball_fuser.log`；不要让 fuser 直接和 camera 主循环抢终端输出。
- 若要看性能，追加：

```bash
bash onboard/perception/camera/run_gray_perception.sh \
    --with-fuser \
    --profile-timing \
    --profile-window 30 \
    --preview-max-hz 10 \
    --status-hz 4
```

- `--show` 会打开 MJPEG 预览，便于调试，但会增加编码开销；性能排查时可关闭或限制 `--preview-max-hz`。
- policy / bridge 只应消费 `rt/ball_state`；不要让 camera 或 lidar detector 直接覆盖 final topic。
- dashboard 默认观察 `rt/target_state`、`rt/cam_ball_state`、`rt/lidar_ball_state` 和 `rt/ball_state`，用于确认 topic ownership 是否正确。
- 若 dashboard 看不到 `rt/lidar_ball_state`，先确认启动环境没有 `ROS_LOCALHOST_ONLY=1`；当前 LiDAR 脚本会显式设为 `0`。

## 已做验证

- 默认 topic 关系已按 raw sensor + fuser 固化：
  - camera raw ball -> `rt/cam_ball_state`
  - lidar raw ball -> `rt/lidar_ball_state`
  - fuser final ball -> `rt/ball_state`
- 灰度链路保留 `GREY` 单通道输入路径，并将 BGR 转换限制在预览 / 可视化路径。
- tmux 布局已按新 topic ownership 预填命令：右上 camera + fuser，右中 lidar raw，左下 dashboard。
- G1 上 camera pane 重启后只显示统一低频状态；bright worker 不再直接打印，fuser 状态不再混入 camera pane。
- `run_gray_perception.sh --with-fuser` 的 Ctrl-C / TERM / EXIT 清理路径已覆盖 camera、fuser 及其子进程组。
- `ROS_LOCALHOST_ONLY=1` 会导致 `rt/lidar_ball_state` 对其他 pane 不可见；LiDAR 脚本改为 `ROS_LOCALHOST_ONLY=0` 后 dashboard / fuser 能看到 raw LiDAR topic。
- LiDAR lowstate 子进程修复后，状态行从默认 waist 变为实时 waist：

```text
[lidar] pelvis=(-0.01,-1.78,-0.71) raw=(...) waist_q=(-1.41,-0.09,-0.01) joint_age_s=0.01~0.03
```

修复前同一姿态近似 torso-frame：

```text
[lidar] pelvis=(+1.79,-0.19,-0.76) ... waist_q=(+0.00,+0.00,+0.00) joint_age_s=null
```

修复后 fuser 选择 LiDAR raw observation，并发布最终 policy topic：

```text
[fuser] source=lidar -> rt/ball_state
```

## 现场检查命令

```bash
# 灰度 AprilTag + bright ball + final fuser
bash onboard/perception/camera/run_gray_perception.sh --with-fuser --show

# fuser 状态单独看，避免混入 camera pane
tail -f /tmp/ball_fuser.log

# raw LiDAR ball，脚本内会设置 ROS_LOCALHOST_ONLY=0
bash onboard/perception/lidar/run.sh --show --dds-topic rt/lidar_ball_state

# 四路 topic dashboard
bash onboard/perception/run_sensor_dashboard.sh
```

现场重点看：

- camera pane 是否只有统一低频 status。
- `/tmp/ball_fuser.log` 是否显示 `source=lidar` 或按预期切到 camera。
- LiDAR 状态行里的 `waist_q` 是否非零，`joint_age_s` 是否约 `0.01-0.03s`。
- dashboard 是否同时看到 `rt/cam_ball_state`、`rt/lidar_ball_state` 和 `rt/ball_state`。
