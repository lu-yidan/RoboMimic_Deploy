# Onboard — 机载感知服务

本目录包含运行在 **G1 机载电脑**上的感知服务代码。
服务启动后通过 **DDS**（与 Unitree SDK 同网段）实时发布球的位置，
本地电脑连接网线后即可直接订阅，无需 SSH 进机器手动启动额外进程。

---

## 目录结构

```
onboard/
└── perception/
    ├── lidar/
    │   ├── ball_detector.py      ← MID360 点云 → raw lidar 球心 → DDS 发布
    │   ├── rviz_publisher.py     ← RViz2 可视化：点云 + Marker 发布
    │   ├── center_kalman_filter.py ← 卡尔曼滤波平滑球心轨迹
    │   ├── mid360_to_base.py     ← 坐标变换：MID360 系 → pelvis (base) 系
    │   └── README.md             ← Lidar 使用手册（含 RViz2 配置）
    └── camera/
        ├── run_gray.sh              ← 入口：灰度 AprilTag + 亮球（+可选 fuser）
        ├── _launch.sh               ← 内部：环境 + 默认参数，启动 detector
        ├── target_ball_detector.py  ← 干活：AprilTag target + 灰度亮球 → DDS
        ├── camera_to_base.py        ← 坐标变换：相机系 → pelvis 系（含胸部占位外参）
        ├── README.md                ← Camera 使用手册（本地文档）
        └── TROUBLESHOOTING.md       ← 性能优化全记录
```

---

## DDS 消息与 Topic

| 字段 | 类型 | 含义 |
|------|------|------|
| `timestamp_us` | uint64 | 发布时刻（µs since epoch） |
| `x / y / z` | float32 | 球心在 **pelvis body 系** 的坐标（m） |
| `valid` | uint8 | `1` = 本帧检测到球；`0` = 未检测到，位置为上一帧 EMA 值 |

- **Raw lidar topic**：`rt/lidar_ball_state`
- **Raw camera topic**：`rt/cam_ball_state`
- **Final policy topic**：`rt/ball_state`（由 `onboard/perception/ball_fuser.py` 发布）
- **QoS**：BestEffort，KeepLast(1)
- **频率**：约 10 Hz（受 Livox MID360 点云帧率限制）

消息定义和 Publisher / Subscriber 封装见 `common/ball_state_dds.py`。

---

## 新机器环境配置（CycloneDDS Python 绑定）

> 机载电脑已经预装了 Unitree 提供的 CycloneDDS C 库（位于
> `~/unitree_ros2/cyclonedds_ws/install/cyclonedds`）。
> `pip install cyclonedds` **不能直接使用**，因为它会在编译 Python 绑定时拉取
> 系统里其他版本的头文件，导致运行时库不匹配（`DDS_RETCODE_BAD_PARAMETER` 或
> `undefined symbol` 等错误）。
> 必须让 Python 绑定**对准 Unitree 自带的那套 CycloneDDS** 来编译。

### 步骤

```bash
# 1. 激活你的 Python 环境
conda activate robomimic   # 按实际环境名修改

# 2. 指向 Unitree 自带的 CycloneDDS（头文件 + 库）
export CYCLONEDDS_HOME=~/unitree_ros2/cyclonedds_ws/install/cyclonedds
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME:${CMAKE_PREFIX_PATH:-}"
export CPATH="$CYCLONEDDS_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LIBRARY_PATH:-}"

# 3. 编译安装（不使用缓存，强制重新编译）
pip install --no-build-isolation --no-cache-dir "cyclonedds==0.10.5"
```

> **注意**：如果系统 `/usr/local/include/dds/` 里有其他版本的 CycloneDDS 头文件，
> 上面的 `CPATH` 变量优先级高于系统路径，通常不需要手动移走，但如果 pip 编译时
> 仍报 `conflicting types for dds_stream_*`，需要先执行：
> ```bash
> sudo mv /usr/local/include/dds /usr/local/include/dds.bak
> # pip install 成功后恢复
> sudo mv /usr/local/include/dds.bak /usr/local/include/dds
> ```

### 验证安装

```bash
# 确认运行时链接到正确的 libddsc.so
ldd $(python -c "import sysconfig; print(sysconfig.get_path('platlib'))")/cyclonedds/_clayer.cpython-*-linux-aarch64.so | grep ddsc
# 应显示 => ~/unitree_ros2/cyclonedds_ws/install/cyclonedds/lib/libddsc.so.0

# 快速功能测试
python -c "from cyclonedds.domain import DomainParticipant; DomainParticipant(0); print('cyclonedds ok')"
```

### ~/.bashrc 注意事项

Unitree 机器默认的 `~/.bashrc` 里可能有如下几行，会在每个终端自动 `source`
旧版 CycloneDDS RMW 并设置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，
这和 Python DDS 环境冲突，建议**注释掉**：

```bash
# 注释掉 fishros 那一整块（若有）
# echo "ros:foxy(1) noetic(2) ?"
# read choose
# case $choose in
# 1) source /opt/ros/foxy/setup.bash;
# source ~/cyclonedds_ws/install/setup.bash;
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; ...
# esac

# 注释掉以下两行（若有）
# source ~/unitree_ros2/setup.sh
# export CYCLONEDDS_HOME=/usr/local
```

每次新开终端，手动按需 source：

```bash
conda activate robomimic
source /opt/ros/foxy/setup.bash
source ~/yixuan/yichao-deploy/ws_livox/install/setup.sh
```

---

## 快速启动

### 方案 A — Lidar（Livox MID360）

#### 1. 依赖

```bash
# ROS2 Foxy（已预装）
# livox_ros_driver2（MID360 ROS2 驱动，已预装于 ws_livox）
# cyclonedds Python 绑定（见上方步骤，不能直接 pip install cyclonedds）
```

#### 2. 启动 MID360 驱动

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

#### 3. 启动 raw 球检测服务

在 `RoboMimicDeploy_G1` 根目录下运行：

```bash
python onboard/perception/lidar/ball_detector.py
```

服务启动后发布 `rt/lidar_ball_state`。策略使用的最终 `rt/ball_state`
由 `onboard/perception/ball_fuser.py` 发布。

---

新的推荐链路是 raw sensor topic + fuser：lidar 发布 `rt/lidar_ball_state`，
camera 发布 `rt/cam_ball_state`，`ball_fuser.py` 按 `lidar > camera` 选择观测并
统一 Kalman 平滑后发布 `rt/ball_state`。详见
`onboard/docs/CAMERA_PERCEPTION_ARCHITECTURE.md`。

### 方案 B — Grayscale AprilTag + Bright Ball

```bash
bash onboard/perception/camera/run_gray.sh
bash onboard/perception/camera/run_gray.sh --with-fuser
```

灰度链路默认使用 `/dev/video3`、`GREY`、`30Hz`，发布 AprilTag target 到
`rt/target_state`，发布 camera raw ball 到 `rt/cam_ball_state`。使用
`--profile-timing` 可输出 capture / bright / apriltag / preview / total 的
mean/p95 耗时。

---

## 本地验证

本地电脑通过网线连接 G1 后，在 `RoboMimicDeploy_G1` 根目录运行：

```bash
python tools/check_ball_state.py
```

正常输出示例：

```
[OK ]  x=+0.823  y=-0.012  z=-0.673  age=85ms
```

- `age` 为数据距当前时刻的延迟，超过 300ms 则显示 `---`（数据过期）。
- 确认能稳定收到数据后，再启动 `deploy_real/deploy_real.py`。

---

## 感知流程（lidar 方案）

```
Livox MID360（点云，~10 Hz）
        │
        │ ROI 滤波 + 反射率阈值
        ▼
候选点云（高反射率球面点）
        │
        │ 最小二乘球心拟合（已知半径 0.115m）
        ▼
球心（MID360 坐标系）
        │
        │ EMA 时间滤波（α=0.6，跳变门限 0.6m）
        ▼
球心（MID360 坐标系，平滑后）
        │
        │ transform_point_mid360_to_base()
        │ （链式正运动学：pelvis → waist → torso → head → MID360）
        │ 使用实时关节角 q_wy / q_wr / q_wp / q_head
        ▼
球心（pelvis body 系）
        │
        │ DDS publish "rt/ball_state"
        ▼
deploy_real.py → state_cmd.ball_pos_b → FreeKick._build_obs()
```

---

## 添加新的感知方案

如需新增其他感知方式，只需：

1. 在 `onboard/perception/<方案>/` 下新建 `ball_detector.py`
2. 用任意方式获取球在 pelvis 系的坐标
3. 调用相同接口发布：

```python
from common.ball_state_dds import BallStatePublisher
dds = BallStatePublisher(domain_id=0)
dds.publish(x, y, z, valid=True)
```

`deploy_real.py` 和 `FreeKick.py` **无需任何修改**。

---

## 参数调整

### Lidar 方案（`onboard/perception/lidar/ball_detector.py`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `reflect_thr` | 150 | 反射率阈值，球面反射率较高，可从 60 逐步调高 |
| `min_points` | 3 | 候选点数下限，太少则不可靠 |
| `max_range` | 1.8 m | ROI 最大距离 |
| `min_range` | 0.2 m | ROI 最小距离（过滤自身遮挡） |
| `x_low / x_high` | 0~5 m | 仅检测机器人正前方区域 |
| `z_low / z_high` | ±1.5 m | 高度范围 |
| `alpha` | 0.6 | EMA 平滑系数，越大跟踪越灵敏，越小越平滑 |

> Camera 灰度方案的参数详见 `onboard/perception/camera/README.md`。
