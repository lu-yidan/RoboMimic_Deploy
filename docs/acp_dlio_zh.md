# G1 ACP：DLIO 定位与实时打点

已接入 `MID360 PointCloud2 + 雷达内置 IMU → DLIO → 腰部补偿 → 骨盆位姿 → ACP`。
真机系统实际为 Ubuntu 20.04 / ROS2 Foxy，已在 G1 独立的 `~/acp_dlio_ws` 编译通过。
当前阶段是现场定位/策略**双侧只读**验收，已开始现场定位检查，尚未完成定位稳定性与 ACP 真机闭环验收。
`deploy/acp-g1` 已替换现场 `under-the-table` 工作分支；原 SMP 配置修改按用户要求丢弃。

[ACP 论文](https://arxiv.org/html/2610.31608v1) 使用 DLIO 提供根位姿与去畸变点云，
高程图另由建图模块生成。本次第一版沿用**实测平地**，DLIO 不会自动提供完整 ACP 地形网格。
依赖固定为 [DLIO ROS2](https://github.com/vectr-ucla/direct_lidar_inertial_odometry/tree/c8acc37100e349d70a9d8432d656cbce7e5072cd)
与 [Livox driver 1.2.4](https://github.com/Livox-SDK/livox_ros_driver2/tree/b6ff7d1c8b210a96b74e919823771e1e32249758)，
具体补丁与版本见 `onboard/localization/dlio/sources.json`、`patches/dlio.patch`。

## 首次准备

以下在 G1 执行；不通过它们自动启动控制。代码拉取不会下载外置模型。
`policy/acp/models/` 必须是固定的 `bundle_stage3_1000`，连同 `manifest.json`、
`policy.onnx.data`，不能只复制 ONNX 文件。
`policy.onnx` SHA256 为 `2a85ea0312bbdff8b681dabe659c8f9c68b1d55221c4f9ec41284dd674e310e8`。
六腕固定姿态/增益仍需现场核对，不能将 bundle 示例值视为已校准。
本次 G1 的 bundle 与 site 配置均已准备，三份模型/清单文件哈希匹配，
ONNX Runtime 1.19.2 已离线加载并得到三路有限输出。可直接进入下一节的只读窗口；
下面的配置生成命令供首次安装或换机使用，重复运行会拒绝覆盖。

```bash
cd /home/unitree/RoboMimic_Deploy
git pull --ff-only origin deploy/acp-g1
# 已在此 G1 编译过；换机或补丁更新后在干净 ROS/system shell 中重建：
bash onboard/localization/dlio/build.sh
# 仅生成本机配置，不启动驱动，不更改网卡。已有配置时不覆盖。
/usr/bin/python3 onboard/localization/dlio/configure.py \
  --host-ip 192.168.123.164 --lidar-ip 192.168.123.120
```

`.120` 三次 ping 均回应，`.20` 无回应；8 秒被动 UDP 检查未见目标流量。
因此 `.120` 目前为待驱动确认的设备地址。驱动启动后必须收到有效点云和 IMU，才算确认。
配置在 `~/.config/robomimic/acp-dlio/`，现场安装角在 `site.json`。
`head_angle_rad=0.593412`、`lidar_angle_rad=0` 与安装平移来自仓库名义 URDF，
要对照实际安装核查。腰部采用 SDK 电机顺序第 12/13/14 项 yaw/roll/pitch。

## 启动窗口与旧流程

旧 `rb → sr` 别名有兼容入口，ACP 需显式选择新模式和一个新 tmux 会话：

```bash
cd /home/unitree/RoboMimic_Deploy
ACP_DLIO=1 bash tools/deploy/start_tmux_layout.sh acp-dlio-readonly
# 也可使用原别名：rb，然后 ACP_DLIO=1 sr acp-dlio-readonly
```

所有命令只预填，按回车才运行。已有同名会话只会重新连接，不会替换旧命令。
机器人调试模式由操作者在启动这些程序前手动进入，程序不接管这个步骤。

| 窗口 | 内容与操作 |
| --- | --- |
| 左上 | 原 C++ 编译与 bridge，明确 `BRIDGE_READ_ONLY=1`，先启动 |
| 右上 | 只读导出腰部状态，依赖 bridge；不发布电机命令 |
| 右中 | MID360 + DLIO + 骨盆转换，使用独立 ROS domain 37 |
| 左下 | 定位状态、时效、输出是否有效 |
| 左中 | 1k ACP 策略，明确 `--read-only`；核查模型文件后启动 |
| 右下 | 可选 RViz，需 G1 图形桌面或有显示的远程桌面 |
| monitor | 预填当前位置停车目标；也可在此输入新的打点命令 |

第一轮先启动左上、右上、右中、左下。机器人保持静止至少 6 秒完成 IMU 检查和校准，
再观察移动/转向时的位姿。原 `onboard/perception/lidar/run.sh` 不要同时启动：
它也会占用同一个雷达，退出时还会全局清理驱动。
本入口只停止自身启动的进程；发现已有 Livox 驱动会拒绝重复启动。

在**已经 SSH 进入 G1 的终端**检查下一步，先不启动右下 RViz：

```bash
cd /home/unitree/RoboMimic_Deploy
/usr/bin/python3 onboard/localization/dlio/check.py --duration 5
```

`Blocking stage: relay` 表示定位进程已经停止或状态文件过期；
`imu/raw_cloud/registration/pelvis` 分别指向 IMU、原始点云、DLIO 配准、腰部/骨盆转换。
先修复这些状态；`pose_valid=true` 后才做下方实测平地标定。
例如配准点云年龄超过 250 ms 时不会接受当前位姿，不要通过放宽时效阈值跳过。
`--duration 5` 连续观察五秒，分别报告各阻塞阶段的采样次数和新鲜骨盆位姿比例，
最后再打印当前一帧。单帧恢复不代表间歇超时已经解决。
中转状态每 20 ms 更新一次；检查器仍按真实采集时间计算年龄。
`Cloud ages AT ARRIVAL` 区分点云抵达中转时已有的延迟与后续断流，
`max arrival gaps` 是当前中转会话累计的最大间隔，包含初始化阶段。
Python 中转与腰部导出限制 BLAS 为单线程，DLIO 的 OpenMP 设置保持独立。
更新这些 Python 文件不需要重编 C++；需由操作者重启右中与右上对应进程才会生效。

左中打印缺少 `/dev/shm/robomimic-1000/acp-state.npz`，表示定位/标定尚未就绪，
不是缺少模型。Python 可以保持只读等待；只有定位新鲜且本会话标定完成才生成该文件。

先核对相对移动方向、转腰时骨盆位姿是否稳定、静止漂移及时间戳。
DLIO 的 `acp_odom` 是每次启动建立的局部坐标系，原点不在地面，初始 yaw 也不保证为零。
地面 z 可能是负值。雷达内置 IMU 与雷达是刚性关系，不能改接骨盆 IMU 代替。
Livox 原驱动加速度为 g，在独立 `/acp/livox/imu_si` 转成 m/s²，原始话题保持原单位。
MID360 内置 IMU 在雷达系的位置 `[0.011,0.02329,-0.04412] m` 来自
[官方手册 v1.2，IMU Data](https://www.livoxtech.com/mid-360/downloads)。

确认安装链、腰部符号并测量骨盆坐标原点相对平地的高度后，在静止站姿执行：

```bash
# 确认安装链并测量后，输入实际高度，不提供默认值。
read -r -p '实测骨盆原点到平地的高度(m): ' PELVIS_HEIGHT_M
/usr/bin/python3 onboard/localization/dlio/calibrate.py \
  --pelvis-height "$PELVIS_HEIGHT_M" --confirm-mount
/usr/bin/python3 onboard/localization/dlio/check.py
```

标定只对当前 relay 会话有效；重启定位后要重新确认世界系与平地高度。
状态 `pose_valid=true` 表示定位几何链已更新，
`acp_output_valid=true` 还要求本会话标定完成、点云/IMU/腰部新鲜同步。
缺失标定时不会给 ACP 伪造有效状态。定位姿态与 bridge 状态仍要满足 40 ms 同步要求。

左中启动后先设置明确目标，再按 Y：
**START 仍是原 FSM 复位/准备姿态，B 是 LocoMode，Y 从 LocoMode 切 ACP**。
只读模式的 START/B/Y 仅改变推理侧 FSM，不驱动真实机器人复位或行走。
后续闭环保留相同操作顺序；闭环命令在本轮只读日志核查后提供。

具体顺序为：`pose_valid=true` → 实测标定 → `acp_output_valid=true` →
左中只读策略 → `goal.py --stop` → B → Y。仅切入检查先用当前点停车目标，
确认后再发送短距离目标。`Need a fresh DLIO pelvis pose in the current host/session`
表示打点时位姿不新鲜或命令运行在其他主机，目标会被拒绝。

## 进入 ACP 后实时打点

图形方式：右下 RViz 的 Fixed Frame 是 `acp_odom`，选择工具栏 **2D Goal Pose**，
在平面上按下选目标位置、拖动选目标航向，松开发布 `/acp/goal_pose`。
以点云和骨盆箭头确认方向；RViz 的 z=0 网格只供选 XY，不代表真实地面高度。
无图形桌面时，从另一 SSH 终端运行：

```bash
cd /home/unitree/RoboMimic_Deploy
# 按打点瞬间的骨盆 yaw，前方 0.5 m；目标航向保持打点时的朝向。
/usr/bin/python3 onboard/localization/dlio/goal.py --relative 0.5 0
# 左侧 0.5 m，并要求最终朝向向左转 90 度。
/usr/bin/python3 onboard/localization/dlio/goal.py --relative 0 0.5 --heading-deg 90
# 已知 DLIO 世界系的 XY 和绝对航向。
/usr/bin/python3 onboard/localization/dlio/goal.py --world 1.2 -0.3 --heading-deg 45
# 将打点瞬间的当前位置和朝向固定为停车目标。
/usr/bin/python3 onboard/localization/dlio/goal.py --stop
```

本地电脑无需安装 ROS/RViz，可以在**本地 Linux 图形桌面的终端**执行：

```bash
cd "/home/yujy/Documents/ChatGPT/G1 locomotion/RoboMimic_Deploy_acp"
bash onboard/localization/dlio/view_remote.sh g1-wifi
```

这使用 `ssh -Y` 将 G1 上的 RViz 窗口显示到本地；ROS 进程仍在 G1 的 domain 37，
保留 `ROS_LOCALHOST_ONLY=1`。已在本机/G1 实测窗口启动（8 秒），
不等于当时已验证真实点云与打点。请先确认定位时效，再打开图形窗口。
普通 SSH/tmux 未设置 `DISPLAY` 时，右下会提示使用此入口，不再直接启动后 abort。
脚本强制新建 X11 连接，避免复用没有图形转发的旧 SSH master。
如果是直接在本地启动本地安装的 RViz，则还需另行配置跨主机 DDS，当前入口未启用。

只想从本地终端发目标，明确让命令在 G1 上执行：

```bash
ssh g1-wifi 'cd /home/unitree/RoboMimic_Deploy && /usr/bin/python3 onboard/localization/dlio/goal.py --relative 0.5 0'
```

相对目标只在打点时转换一次，随后固定在世界系中，不会跟着机器人向前滑。
更新目标不重置动作历史，也不切换 FSM。目标记录带定位会话 ID；旧会话目标不能复用。
`--stop` 是策略到点停车指令，紧急退出仍用原阻尼抢占操作。
定位重启、跳变、主机时钟跳变、点云/IMU/腰部过期都不能通过重写文件时间绕过。

## 已完成验证与下一份日志

55 项 ACP/DLIO 回归通过，其中针对性检查 24 项，覆盖动态腰部补偿、g/SI、时间映射、重复 tick、定位重启、
目标坐标和策略接收等。G1 Foxy 上的离线整链测试使用真实 DLIO 二进制和合成点云/IMU，
收到 951 帧骨盆输出；末段静止位置范围最大约 `3.1e-7 m`，只证明合成输入接线正确，
不代表真实雷达精度。停止点云后 ACP 文件被撤销。没有运行真实雷达驱动或控制实验。
上述为离线验证范围；用户随后已自行启动真实定位链路，其现场时效和地面标定仍需验收。
结果摘要在 `docs/validation/dlio_foxy_20261124.json`。

另以每帧 20,000 点、10 Hz 做隔离时效测试。合成发布器改用 Foxy 的类型化字节数组，
避免逐字节校验阻塞同进程 IMU。原始点云生成年龄为 100 ms 时，末段 301 次状态采样中，
骨盆位姿新鲜率为 99.3%，ACP 输出有效率为 95.0%；120 ms 压力输入的输出有效率为 84.7%。
两次均验证点云断流后撤销 ACP 输出，但仍有间歇失效，**不视为持续新鲜或现场验收通过**。
状态更新间隔 P99 分别为 25.4/21.5 ms；未放宽任何时效阈值。
合成发布器修正前的失败记录也保留在 `docs/validation/dlio_timing_20261124.json`。

现场结束后保留策略 `logs/` 中的普通日志、`.acp_trace.npz`，
当前 `site.json`、`MID360_config.json`、`acp-dlio-status.json`、标定与目标 JSON，
以及 `logs/acp-dlio/relay-*.jsonl` 和最近一次 `~/.ros/log/` 的启动目录。
新版自动保存各阶段计数、时效和独立错误；ROS 节点 stdout/stderr 同时显示与落盘。
需复现点云问题时，在独立 ROS/system 终端记录：

```bash
source /opt/ros/foxy/setup.bash
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=37 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 bag record -o "$HOME/acp_dlio_bag_$(date +%Y%m%d_%H%M%S)" \
  /livox/lidar /livox/imu /acp/livox/imu_si \
  /dlio/odom_node/odom /dlio/odom_node/pointcloud/deskewed /acp/pelvis/odom /acp/goal_pose
```

只读通过后，才有现场证据决定是否进入已有防跌流程下的 B → Y 短距离闭环。
现有 MuJoCo 动态转向切入失败仍保留，未通过提高倾角阈值掩盖。
