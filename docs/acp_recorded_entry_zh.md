# 1k ACP 候选：真实站姿切入仿真

此页保留旧FixedPose入口的历史对照。当前交互入口已按用户要求改为
**B/LocoMode → Y/ACP**，实时打开MuJoCo的命令见[交互说明](acp_interactive_zh.md)。

固定 `bundle_stage3_1000`，`policy.onnx` SHA256 为
`2a85ea0312bbdff8b681dabe659c8f9c68b1d55221c4f9ec41284dd674e310e8`。
本轮不训练，不连接机器人；用户先查看仿真，再安排现场只读实验。

## 结果与适用条件

从 under-the-table `6a25875` 的两份真实日志各取一段稳定 AMP 站姿，
每段选三个未经平均的实测帧，共六个初始姿态。分别测试原地停车、
沿初始朝向前进 1 m、侧向 1 m 并转向 90°。

| 准备阶段 | 进入 ACP | 完整自由运行 15 秒并末段停稳 |
| --- | ---: | ---: |
| 无支撑 | 0/18 | 0/18 |
| 理想骨盆支撑，2 秒时撤除 | 18/18 | 18/18 |

无支撑场景均在 FixedPose 阶段约 1.22–1.30 秒跌倒。当前没有自主切入通过的证据。
有支撑场景首帧 29 路目标的最大跳变为 0.25790 rad，首帧均未触发目标限幅；
ACP 阶段最大骨盆倾角为 0.24274 rad。完整轨迹中三个前进场景各有一帧
目标限幅，其中一个还有一帧 PD 力矩裁剪，均保留在原始记录中。
末一秒每帧需满足距离 <0.2 m、航向误差 <0.25 rad、平面速度 <0.15 m/s，
以及现有评估器的高度、倾角和有限值条件。

支撑的精确定义：准备阶段将骨盆世界位置、姿态固定在拟合后的初始值，
每个物理步将根速度归零；关节由实际 FixedPose 控制，渐变到本 bundle 的默认角。
在 2.0 秒调用实际 FSM 请求 ACP，入口及输出 guard 接受后，**首个 ACP 物理步之前**
撤除支撑，随后完整 15 秒不再固定或扶持骨盆。它不是吊带的柔性或受力模型。

所有受支撑场景在 ACP 首帧双脚法向力均为零；撤除后 20–60 ms 内出现
大于 5 N 的脚底接触。记录中的接触力由 MuJoCo 计算，不能作为真实接触力或
现场释放条件的测量值。现场需能复现并独立验证支撑与释放方式。

## 初始状态来源

原始目录为 `/home/yujy/Downloads/SMP LOG`：

- `1016/20261016_223025_freekick.json`，稳定窗口 `[11978,12103)`。
- `1105/20261105_093832_freekick.json`，稳定窗口 `[16529,16621)`。

日志元数据记录该提交且 `git_dirty=true`，不将其描述为干净提交的现场复现。

窗口要求：AMP 状态、零速度指令、连续记录、倾角 <0.15 rad、
关节速度 RMS <0.08 rad/s、角速度范数 <0.15 rad/s、关节变化 <0.06 rad。
保存实测关节角、速度、骨盆四元数和角速度，初始值不取平均。
根 XY 设为零；缺失的根线速度设为零；根高度由仿真脚碰撞几何最低点拟合到
地面上方 2 mm。这些是明确的仿真假设，日志没有提供真实定位、脚底力或外部支撑力。
`selected_poses.json` 保存原始文件哈希、帧号、时间和实测值。
六腕仍使用 bundle 中的示例固定角与增益，尚未完成现场核验。

## 本机查看与重新计算

当前本机代码目录：
`/home/yujy/Documents/ChatGPT/G1 locomotion/RoboMimic_Deploy_acp`。
证据目录：
`/home/yujy/Documents/ChatGPT/G1 locomotion/data/acp_g1_reports/real_pose_entry_v1`。
245 已同步本轮代码。真实日志衍生的姿态、轨迹与视频保存在本机，
上传到其他主机需用户明确授权。

查看已经计算的前进仿真，使用图形桌面终端：

```bash
cd "/home/yujy/Documents/ChatGPT/G1 locomotion/RoboMimic_Deploy_acp"
/home/yujy/miniconda3/envs/robomimic/bin/python tools/acp/view_recorded_entry.py \
  --trace "../data/acp_g1_reports/real_pose_entry_v1/simulation/supported_pose0_forward.npz" --loop
```

将文件名改成 `supported_pose3_turn.npz` 查看转向，
`supported_pose0_stop.npz` 查看停车，`free_pose0_stop.npz` 查看无支撑失败。
空格暂停，R 重播；蓝色骨盆标记表示准备支撑，橙色表示撤除。
脚边绿/红标记表示该帧**记录的仿真**法向力大于/不大于 5 N；紫色为目标位置。
回放只显示存档轨迹，不重复策略推理或物理积分，也不创建 DDS 接口。

需要重新计算全部 36 个场景时，选择一个新的空输出目录：

```bash
cd "/home/yujy/Documents/ChatGPT/G1 locomotion/RoboMimic_Deploy_acp"
/home/yujy/miniconda3/envs/robomimic/bin/python tools/acp/evaluate_recorded_entry.py \
  --logs-root "/home/yujy/Downloads/SMP LOG" \
  --bundle "../data/acp_g1_reports/stage3_flat_trial_v1/bundle_stage3_1000" \
  --output-dir "../data/acp_g1_reports/real_pose_entry_review/simulation"
```

评估工具复用 `FSM.run()`、`FixedPose`、ACP 与 Python runtime 的输出 guard，
使用仿真真值位置与射线地形。它验证了这一入口组合的仿真行为；DDS、C++ bridge、
实时定位及硬件关节映射仍待现场只读验收。NPZ 含 50 Hz 的 939 维输入、
关节状态、原始及限幅后目标、增益、力矩、接触力和支撑阶段；JSON 单独保存首帧完整 29 路数据。

无显示环境可用 `view_recorded_entry.py --video /absolute/path/new.mp4` 渲染，
需设置 `MUJOCO_GL=egl` 并提供含 libx264 的 ffmpeg；必要时用 `--ffmpeg /path/to/ffmpeg`。
证据目录的 `videos/` 已包含前进、转向与无支撑失败的视频。
本轮新增工具及相关 ACP 回归共 41 项测试通过。

用户查看确认后，再交付现场双侧只读命令；本页不作为真机闭环验收通过记录。
