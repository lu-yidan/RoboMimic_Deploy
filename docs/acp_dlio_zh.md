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
原 `head_angle_rad=0.593412` 来自共享变换示例，并非这台原厂固定安装的实测角。
2026-11-24 只读日志核查后，ACP 的原厂安装候选改为 `head_angle_rad=0`、
`lidar_angle_rad=0`；使用实际腰部角重算，重力偏差中位数由约 34° 降至 0.66°。
这是有日志支持的候选修正，尚不是完整外参标定；安装平移、yaw 与运动中的腰部补偿仍需现场核对。
共享 PHP/SMP 变换函数默认值不改。旧 `site.json` 不会随 git pull 自动更新；
本次这台 G1 的配置已单独备份并修正，须由操作者重启右中 DLIO 才生效。
腰部采用 SDK 电机顺序第 12/13/14 项 yaw/roll/pitch。

## 启动窗口与旧流程

旧 `rb → sr` 别名有兼容入口，ACP 需显式选择新模式和一个新 tmux 会话：

```bash
cd /home/unitree/RoboMimic_Deploy
ACP_DLIO=1 bash tools/deploy/start_tmux_layout.sh acp-dlio-readonly
# 也可使用原别名：rb，然后 ACP_DLIO=1 sr acp-dlio-readonly
```

启动右中 DLIO 前，主机的 UDP 接收缓存上限需至少 8 MiB：

```bash
sudo sysctl -w net.core.rmem_max=8388608
```

此设置本次启动有效，重启主机后需重设。脚本只检查，不自动执行 sudo 或写永久系统配置。
DLIO/RViz 通过独立 Cyclone DDS XML 显式请求 8 MiB 接收缓存，仍使用本机 domain 37；
配置使用 G1 ROS Foxy/Cyclone DDS 0.7 支持的 `MinimumSocketReceiveBufferSize` 字段。
只提高内核上限而不重启这些 ROS 进程，不会扩大它们已经打开的 socket。
该 XML 仅用于定位、RViz 和 ROS 录制终端；bridge/腰部 DDS 继续使用原配置。
依据为 [ROS 2 Foxy DDS 调优文档](https://github.com/ros2/ros2_documentation/blob/foxy/source/How-To-Guides/DDS-tuning.rst)
与 [Cyclone DDS 0.7 配置源码](https://github.com/eclipse-cyclonedds/cyclonedds/blob/0.7.0/src/core/ddsi/include/dds/ddsi/ddsi_cfgelems.h)。

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

第一轮先启动左上、右上、右中、左下。机器人保持静止至少 10 秒完成 IMU 检查和校准，
再观察移动/转向时的位姿。原 `onboard/perception/lidar/run.sh` 不要同时启动：
它也会占用同一个雷达，退出时还会全局清理驱动。
本入口只停止自身启动的进程；发现已有 Livox 驱动会拒绝重复启动。

在**已经 SSH 进入 G1 的终端**检查下一步，先不启动右下 RViz：

```bash
cd /home/unitree/RoboMimic_Deploy
/usr/bin/python3 onboard/localization/dlio/check.py --duration 10
```

`Blocking stage: relay` 表示定位进程已经停止或状态文件过期；
`localization_reset` 表示已经锁定定位跳变，需重启右中 DLIO，稳定后重新标定地面和目标；
此时即使点云恢复也不会自动解锁。`First localization fault` 保存首次触发时的时间差、
位移、角度及对应限值，后续回调不会覆盖证据。
`imu/raw_cloud/registration/pelvis` 分别指向 IMU、原始点云、DLIO 配准、腰部/骨盆转换。
先修复这些状态；`pose_valid=true` 后才做下方实测平地标定。
例如配准点云年龄超过 250 ms 时不会接受当前位姿，不要通过放宽时效阈值跳过。
`--duration 10` 连续观察十秒，分别报告各阻塞阶段的采样次数和新鲜骨盆位姿比例，
最后再打印当前一帧。单帧恢复不代表间歇超时已经解决。
中转状态每 20 ms 更新一次；检查器仍按真实采集时间计算年龄。
`Cloud ages AT ARRIVAL` 区分点云抵达中转时已有的延迟与后续断流，
`max arrival gaps` 是当前中转会话累计的最大间隔，包含初始化阶段。
Python 中转与腰部导出限制 BLAS 为单线程，DLIO 的 OpenMP 设置保持独立。
更新这些 Python 文件不需要重编 C++；需由操作者重启右中与右上对应进程才会生效。

默认点云分帧改为 20 Hz，每帧约积累 50 ms 数据，减少等待整帧造成的采集年龄。
单位时间的原始雷达点数不因此增加。可用 `ACP_LIDAR_HZ=10` 做原 10 Hz 对照；
门限仍为配准 250 ms、骨盆 150 ms，没有用放宽门限消除告警。
中转保留每次里程计回调的有效性检查与骨盆输出，ACP 状态文件按策略所需的 50 Hz 写入，
避免每秒约 100 次重复创建 writer/打包 NPZ。失效仍立即撤销文件，保留采集时间戳。

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
标定保留速度 `<=0.05 m/s`、角速度 `<=0.1 rad/s` 的静止要求。
拒绝时打印具体超限数值、数据年龄或跳变原因；不会通过平均掉跳变来强行标定。
状态 `pose_valid=true` 表示定位几何链已更新，
`acp_output_valid=true` 还要求本会话标定完成、点云/IMU/腰部新鲜同步。
缺失标定时不会给 ACP 伪造有效状态。定位姿态与 bridge 状态仍要满足 40 ms 同步要求。
策略保留最近 64 个唯一 bridge tick，按定位时间选择最近的关节角、关节速度和骨盆陀螺仪样本，
用于构造 ACP 观测。定位和匹配样本均须在 150 ms 内，超过 40 ms 偏差则拒绝；
不插值、不外推或改写时间戳。这里 bridge 的时间是本机 DDS 接收时间，不能视为硬件采集时间的精确测量。
遥控器、当前关节状态与 PD 力矩限幅继续使用最新 bridge 帧。
重复 tick 不刷新历史样本时间，tick/接收时间倒退会锁定错误并要求重启策略。
策略还比较同一匹配 tick 的 SDK 骨盆 IMU 与 DLIO 骨盆重力方向，
偏差超过 10° 时拒绝进入 ACP，运行中则退出到阻尼。两路世界坐标系的 yaw 可以不同；
该检查不要求机器人复位，也不能代替完整安装外参验证。
`check.py` 的 ready 只表示定位状态文件就绪，最终重力一致性检查发生在 Python 策略侧。

左中启动后先设置明确目标，再按 Y：
**START 仍是原 FSM 复位/准备姿态，B 是 LocoMode，Y 从 LocoMode 切 ACP**。
只读模式的 START/B/Y 仅改变推理侧 FSM，不驱动真实机器人复位或行走。
后续闭环保留相同操作顺序；闭环命令在本轮只读日志核查后提供。

具体顺序为：`pose_valid=true` → 实测标定 → `acp_output_valid=true` →
左中只读策略 → `goal.py --stop` → B → Y。仅切入检查先用当前点停车目标，
确认后再发送短距离目标。`Need a fresh DLIO pelvis pose in the current host/session`
表示打点时位姿不新鲜或命令运行在其他主机，目标会被拒绝。
新版错误同时显示骨盆/状态年龄及跳变锁定原因，CLI 不自动重试或使用旧位姿替代。
策略读取目标时在每次文件读取后取时间，避免生产者在读取期间更新文件而被误判为未来。
目标缓存的文件标识与内容来自同一个已打开的文件，避免原子替换期间误报重复目标。

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

92 项 ACP/DLIO 与部署桥接回归通过，覆盖动态腰部补偿、g/SI、时间映射、重复 tick、定位重启、
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

随后在隔离 domain 78 上做一组顺序缓存对照：同样每帧 20,000 点、点云年龄 100 ms、
各运行 30 秒。默认缓存新增 41 次主机 UDP 接收缓存丢包，8 MiB 配置为 0 次；
原始点云最大到达间隔为 194/104 ms，ACP 输出有效率为 91.9%/97.4%。
两次点云断流撤销均通过，骨盆位姿末段新鲜率均为 100%。
这是单组合成对照，UDP 计数是主机全局增量；仍有输出失效，现场稳定性尚待复核。
结果保存在 `docs/validation/dlio_udp_20261124.json`。

再以相同每秒 200,000 个合成点比较 10 Hz/20 Hz 分帧（20,000/10,000 点每帧，
首点生成年龄 100/50 ms），两组均使用 50 Hz 状态写入和 8 MiB DDS 缓存。
独立运行与现场相同的 `check.py --duration 10`：10 Hz 为 461/490 次 ready，
仍有 27 次 registration、2 次 acp_state；20 Hz 为 489/489 次 ready。
两组 `goal.py --stop` 均成功，停止点云后的状态撤销均通过。
20 Hz 的生产端末段 1,101 次状态采样也全部有效；这是单组隔离对照，未包含真实雷达或 RViz 负载。
结果在 `docs/validation/dlio_scan_rate_20261124.json`，真实现场仍需按新版入口复核。

2026-11-24 随后取回用户录制的两份真实数据（约 71.4/89.5 秒，部署提交 `6b05522`），
原始文件 SHA256 与 G1 一致，SQLite 检查正常。逐帧运行当前 `PoseGate`，
两份原生里程计和骨盆输出均未触发跳变；重复采集时间戳按现有逻辑忽略。
骨盆相邻帧最大位移分别为 3.89/7.77 mm，XYZ 各轴跨度最大值为 9.44/11.32 mm。
匹配时间戳的 14,076/17,899 帧 IMU 确认加速度恰好乘以 9.80665，角速度保持原值。
在独立回环 domain 78 中，用当前原生 DLIO 二进制各回放一次 checked 点云和 SI IMU，
保留采集时间戳、不运行 relay/IPC 或硬件驱动，两次也未触发跳变，最大位置步长为 4.37/3.91 mm。
best-effort 回放观察器的实际消息计数也保留；该结果不保证估计器接收无丢包。

这些数据**没有复现此前 10 ms 内 15 cm 的跳变，不能认定旧故障已修复**。
毫米级配准修正仍会产生逐帧差分速度尖峰；诊断轨迹中新鲜位姿分别有 81.2%/81.8%
满足当前瞬时标定运动条件。这可能导致偶发标定拒绝，但无法还原旧命令当时的具体拒绝项。
本轮没有修改时效、运动或跳变阈值，也未完成地面标定或 ACP 实时只读验收。
详细结果、输入哈希和回放限制在 `docs/validation/dlio_live_bags_20261124.json`。
下一步由操作者启动现场只读链路，在当前定位会话按实测高度标定；失败时保留新版具体拒绝数值。

同日首次 ACP 只读策略日志为 `20261124_202814_real_policies`。标定与状态输入已就绪，
但 Y 请求被定位/最新关节帧时间差拒绝，1,792 条普通日志中没有 ACP 执行帧，
未生成 `.acp_trace.npz`，不能记为 ACP 只读验收通过。
随后 20 秒被动订阅测量得到最新状态/定位偏差中位数约 52 ms，直接重读文件仍约 43 ms，
说明只减少文件读取延迟不能稳定满足同步要求。
修复改为匹配历史关节/IMU；同一份采样的时序回放中，最初 10 帧后 955/955 帧满足原同步限值，
固定 1k bundle 在本机完成 964 帧 939 维输入离线推理，输出有限且按最新状态限幅。
这些是保存数据的离线核验，现场仍须重新按 B → Y 完成实时只读验收。

ARM ONNX Runtime 1.19.2 的同帧对拍还发现 occupancy 的微小负值 `-5.96e-8`，
动作和接触点在四帧对拍中与本机完全一致。概率范围校验只接受 `4×float32 epsilon`
以内的舍入越界并截到 `[0,1]`，较大越界或非有限值仍拒绝，动作和接触点不变。
新只读 trace 额外保存 `observation_q/dq/gyro`、匹配 tick/时间、Kp/Kd、推理耗时和原始概率范围，
其中 `q/dq` 仍表示最新 PD 状态；重建观测时应使用 `observation_*`。
修复后 G1 对相同保存采样完成 964 帧离线推理，推理耗时中位数/P99/最大值为
4.76/4.92/5.01 ms；本机与 G1 的 964 帧输入、动作、接触点和限幅目标完全一致，
occupancy 最大差约 `4.23e-6`。首个缓存尚未填充的样本仍明确拒绝。
离线首帧目标相对当前吊姿的最大差约 0.79 rad；只读 FSM 不改变真实姿态，
这不能代替真实 B/LocoMode 站姿的切入和六腕固定目标核对。
结果与限制在 `docs/validation/acp_readonly_20261124.json`。

同日第二次只读日志 `20261124_205302_real_policies` 已连续执行 ACP 28.17 秒、1365 帧，
ACP 期间无自动阻尼退出。939 维输入独立重建最大误差为 0，匹配时间差最大 14.84 ms，
推理耗时 P50/P99/最大为 6.36/12.29/18.26 ms。
本轮仍未通过只读验收：骨盆重力与 SDK IMU 约有 34° 固定差，需应用上述安装候选并复测。
当时机器人未复位；只读 START/B/Y 不改变实际吊姿，因此约 0.79 rad 的首帧目标差不能
替代真实 B/Loco 姿态下的切入验证。

1364/1365 帧动作/接触点在 ARM 与 x86 完全相同，第 1185 帧选点相差一个 0.1 m 网格。
离线暴露 planner 分数后复现：两候选分数约为 0.089594，平台间最多约 `3.69e-7` 的差异
翻转了近似并列的 ArgMax；ARM 离线结果与现场动作/接触完全相同。
这是尚未通过的逐帧对拍，不能放宽容差后记为通过；先修正错误重力输入，再看有效输入下是否重现。
实时循环计算有约 6.6% 帧超过 20 ms，整体记录频率约 48.4 Hz，仍有尾延迟待处理。
完整数据摘要见 `docs/validation/acp_readonly_20261124_205302.json`。

这次修正后的复测不需要重编 bridge：保留左上只读 bridge、右上腰部导出，
由操作者在右中 Ctrl+C 后重启原 DLIO 命令。更换安装角会改变推算的骨盆原点，
因此必须为新会话重新测量/标定地面并重新发送目标，不能沿用旧标定或旧目标。
静止定位稳定后，若当前实测骨盆高度仍为 0.78 m，可直接逐条执行：

```bash
/usr/bin/python3 onboard/localization/dlio/calibrate.py --pelvis-height 0.78 --confirm-mount
/usr/bin/python3 onboard/localization/dlio/check.py --duration 10
```

然后启动左中原 `--read-only` 策略命令，在 monitor 执行 `goal.py --stop`，
再依次 START（等待准备插值至少 2 秒）→ B → Y，按键之间松开。
保持双侧只读，记录至少 15 秒后仅在左中 Ctrl+C 保存 trace。
新版 trace 保存匹配 SDK 四元数及重力偏差，避免再用同帧最新 IMU 近似比较。

第三次只读 `20261124_211631_real_policies` 已使用 0° 安装候选和新会话标定，
连续执行 ACP 22.51 秒、1071 帧，无自动阻尼退出。SDK 与 DLIO 重力偏差
中位数 0.618°、最大 0.709°，修正效果在实时链路得到验证。
939 维输入独立重建误差为 0；全部动作/接触点在 G1 与本机 ONNX 逐值相同，
占用概率最大差 `3.16e-6`，本轮未重现上一轮的近似并列 ArgMax 差异。
匹配时间差最大 14.45 ms，位姿年龄最大 74.57 ms，均在原有门限内。

操作者确认全程吊臂下基本静止，因此这里只通过**静止输入与推理对拍**，
不覆盖实际平移、转向、腰部运动、输入失效退出或真实 B/Loco 姿态下的切入。
整体记录频率约 47.54 Hz，运行时计算 P50/P99 为 17.39/25.12 ms，
17.65% 帧计算超过 20 ms，时序验收仍未通过。
运行时保存的全部 29 路 PD 力矩估计在 manifest 限值内（浮点超出最大约 `1.91e-6 Nm`）；
这不是电机实测输出，因为 C++ 与 Python 两侧均只读。

随后只创建订阅器、显式禁止 DDS DataWriter 的 300 帧耗时诊断无输入/推理失败。
cProfile 记录了 900 次重复生成固定网格；现将网格预计算后返回独立可写副本，
保持数值、展开顺序和所有输入检查。该诊断没有覆盖完整现场 FSM/目标读取/普通日志与旧感知发布，
不能据此宣称控制周期已修复；优化后的实时耗时需下一轮只读确认。
详细结果见 `docs/validation/acp_readonly_20261124_211631.json`。

现场结束后保留策略 `logs/` 中的普通日志、`.acp_trace.npz`，
当前 `site.json`、`MID360_config.json`、`acp-dlio-status.json`、标定与目标 JSON，
以及 `logs/acp-dlio/relay-*.jsonl` 和最近一次 `~/.ros/log/` 的启动目录。
新版自动保存各阶段计数、时效和独立错误；ROS 节点 stdout/stderr 同时显示与落盘。
遇到持续新鲜但定位跳变时，先保持策略关闭，做一次 90 秒定位录包。
左上 bridge 和右上腰部导出保持运行，右中 DLIO 先停止。
在另一个 G1 终端先启动录包：

```bash
cd /home/unitree/RoboMimic_Deploy
env -u CONDA_PREFIX -u LD_LIBRARY_PATH -u PYTHONPATH \
  bash onboard/localization/dlio/record.sh 90
```

显示 `Listening for topics` 后，在右中按原命令启动 DLIO，机器人保持吊臂下静止。
本轮不运行标定、目标或 ACP 策略；若发生 `localization_reset`，保留定位进程直到录包自动结束。
录包只创建传感器/里程计订阅，不启动或停止雷达、bridge 或电机程序。
使用 best-effort QoS，避免要求实时发布者等待诊断录制器。
输出目录在 `logs/acp-dlio/record-*/`，包含原始/checked 点云、raw/SI IMU、
DLIO 与骨盆输出的 ROS bag、起止状态、配置、commit 与同会话诊断轨迹。
在 G1 Foxy 的隔离 domain 78 上验证了 5 秒自动录制/关闭，保存并读回 89 条合成 IMU；
此验证仅覆盖录包工具，不表示修复了真实点云的定位跳变。

只读通过后，才有现场证据决定是否进入已有防跌流程下的 B → Y 短距离闭环。
现有 MuJoCo 动态转向切入失败仍保留，未通过提高倾角阈值掩盖。
