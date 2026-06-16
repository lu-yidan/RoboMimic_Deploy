# Camera Ball Detector — 排障与性能调优全记录

> 硬件：**Unitree G1**，机载电脑 NVIDIA Jetson Orin NX 16 GB，JetPack 5.1.2  
> 相机：**Intel RealSense D435**（USB 3.0）

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

灰度/IR 入口：`run_apriltag_gray_ball.sh`，默认 `/dev/video3` + `GREY`。

打开预览：

```text
http://192.168.123.164:8080/stream
```

如果端口被占用：

```bash
./onboard/perception/camera/run_apriltag_target.sh --show --show-port 8081
```

### 0.2 现在这路相机到底是什么模式？

灰度入口看到的是 RealSense 的 **IR/灰度 UVC 流**。

表现：

- 画面是灰度/黑白
- AprilTag 对比度很好，识别稳定
- 白色足球、反光点阵球会很亮

灰度入口当前使用 `--ball-bright`，对白球/反光点阵球更敏感。

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

当前稳定入口：

```bash
# 灰度 / IR，适合 AprilTag + 白球/反光点阵球
./onboard/perception/camera/run_apriltag_gray_ball.sh --show
```

灰度入口默认：

```bash
--camera-profile gray-ir
--color-backend v4l2
--v4l2-device /dev/video3
--v4l2-fourcc GREY
--ball-bright
```

代码里 V4L2 fallback 会按当前 `--v4l2-fourcc` 过滤节点：

- 灰度入口只找支持 `GREY` 的节点，不会跳到彩色节点

灰度入口使用 `--camera-profile gray-ir`：IR/灰度流使用更宽 FOV 的近似内参，并给 chest 外参加 `+0.020m` Y 偏置。这个 profile 更适合 AprilTag 位姿和白球单目深度估计。

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
- 现场稳定方案使用灰度 V4L2 + 单目球半径估计

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
pgrep -af "apriltag_detector.py|sensor_dashboard.py"

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
```

---

## 目录

1. [硬件与环境配置](#一硬件与环境配置)
2. [ROS2 / DDS 相关问题](#十一ros2--dds-相关问题)

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
