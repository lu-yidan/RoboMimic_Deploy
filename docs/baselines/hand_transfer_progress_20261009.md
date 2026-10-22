# 钩爪脱挂与换杆实验：2026-10-09

本轮完成了**同杆脱钩—后撤—回挂—试探加载**的名义动力学实验。**向上换杆 25 cm 尚未通过**，两者分别保存，不能合并称为攀爬成功。

## 已通过：同杆脱挂闭环

- 正式结果：`outputs/hand_release_reseat_v2/summary.json`；10 项声明检查全部通过。
- 全过程：`outputs/hand_release_reseat_v2/transfer_full_run.mp4`，18 秒仿真，451 帧、25 fps、1280×720。文件 18.04 秒，包含末帧 0.04 秒停留。
- 手部近景：`recorded_right_hook_closeup_v2.mp4`。从同一动力学记录重新取景，不是另一次实验，不是将 IK 目标当成机器人实际运动。先前近景受前景横杆遮挡，保留但不推荐展示。
- 曲线：`motion_metrics.png`；逐步记录：`control_log.csv`、`contact_log.json`、`trajectory.npz`。
- 独立复跑：`outputs/hand_release_reseat_repeat_v2`。qpos、qvel、关节目标、时间数组最大差值均为 **0**；这只证明本机同配置的确定性复现。

| 项目 | 测量结果 |
|---|---:|
| 仿真时长 | 18.000 s |
| 右钩路径 | 上提约 50 mm、后撤约 60 mm、回原杆 |
| 身体位置最大偏移（每物理步） | 26.25 mm |
| 身体最大倾斜（每物理步） | 0.680° |
| 最后两秒右钩平均竖直载荷 | 14.18 N |
| 结束时右钩竖直载荷 | 16.60 N |
| 最后两秒右钩最低竖直载荷 | 10.39 N |
| 非钩爪/脚部的机器人与环境接触力峰值（每物理步） | 0 N |
| 电机指令超过原模型力矩限制的比例 | 0 |

0–2 秒卸载并稳定；2–4 秒抬钩；4–6 秒后撤；6–10 秒悬空保持；10–12 秒前移；12–14 秒回挂；14–18 秒试探加载。6–10 秒右钩接触力为零，由左钩和两脚支撑。加载目标设为 20 N，但截至 18 秒只达到 16.60 N；**通过的是连续两秒超过 10 N 的加载条件，不能宣称达到并稳定保持 20 N**。

## 控制与平台边界

使用学长仓库原 `deploy_mujoco.py` 入口新增的 `hand_transfer` 实验分支，沿用冻结墙面、钩爪、质量、碰撞和电机力矩限制。没有修改已经通过的原场景站起基线、10 秒支撑结果或冻结场景。

控制器为显式 Cartesian 路径 + 七自由度手臂 IK + 关节 PD。IK 使用独立 scratch data，实时仿真在初始化之后只通过 `mj_step` 推进。没有运动学改写实时 qpos、外力托举、焊接约束或固定浮动基座。手部目标关节速度限为 1 rad/s。

右臂可选 `qfrc_bias` 前馈包含模型重力及速度偏置项，配置中的历史命名为 `arm_gravity`。支撑侧 Kp 为上一阶段 PD1.5 配置的 5 倍，右臂为 3 倍；Kd 按倍数平方根调整。全部力矩仍限于原模型范围。这是本次名义实验的调参结果，不代表实机可用增益。

试探加载直接使用 MuJoCo 右钩接触力真值，积分调整向下参考偏置，增益 0.0002 m/(N·s)，偏置范围 0–25 mm。基础位姿反馈也来自仿真真值，尚未接入估计器、真实力传感器、噪声或时延。

该分支使用完整关节位置参考，**没有复用旧 climb-v1 的小范围归一化动作去冒充新策略，也没有经过 Climb Actor/FSM 的等价性验证**。原有 100D/29D actor 接口保留。本轮没有训练网络，也没有证明连续攀爬或结构强度。

## 未通过：上移 25 cm

正式失败案例：`outputs/hand_transfer_25cm_failure_v2`，含完整录像、曲线、源代码快照和日志。对应第 5 组候选配置；此前 8 组探索结果保留在 `outputs/hand_transfer_trial_v1` 至 `v8`。所有候选均未通过完整验收，不因右钩出现承重就改判成功。

该案例右钩能从 `rung_05` 移到 `rung_06`，最后两秒平均竖直载荷约 80.38 N，身体最大偏移约 27.2 mm。然而：

1. 上方接近阶段 IK 残差最大约 27 mm，超过 5 mm 的路径检查阈值。
2. 最后手部沿横杆方向偏离指定抓取点约 66 mm，未通过 20 mm 对齐条件。
3. 抬手阶段手腕碰杆，结束阶段髋部接触中间杆；每物理步检测的额外接触峰值约 124.59 N。

因此这里记录的是失败轨迹。非零接触力甚至较大的承重，都不足以证明正确挂接。现有证据指向**接近路径、冗余手臂姿态及全身姿态协调**问题；不能据此断言 G1 无法达到该横杆，或简单归结为电机力度不足。

## 复现

在 PowerShell 执行，已有输出目录必须换新名字：

```powershell
cd D:\G1Climbing\external\RoboMimic_Deploy

# 已通过的同杆脱挂，自动保存完整录像和日志
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_hook_release_reseat hydra.job.chdir=false hand_transfer.output_dir=outputs/my_hook_reseat

# 同配置独立复跑，不录视频
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_hook_release_reseat hydra.job.chdir=false hand_transfer.output_dir=outputs/my_hook_reseat_repeat hand_transfer.video=false

# 生成科学曲线；近景是已记录动力学的重取景
.\.venv-cad\Scripts\python.exe tools\plot_hand_transfer.py --run outputs/my_hook_reseat
.\.venv\Scripts\python.exe tools\render_hand_closeup.py --run outputs/my_hook_reseat

# 已知未通过的 25 cm 换杆，用于复现诊断，不是成功演示
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_hand_transfer hydra.job.chdir=false hand_transfer.output_dir=outputs/my_transfer_failure hand_transfer.arm_gravity=true hand_transfer.base_feedback=true hand_transfer.support_gain=5 hand_transfer.arm_gain=3 hand_transfer.initial_clearance=0.08 hand_transfer.clearance=0.04 hand_transfer.retreat=0.075 hand_transfer.load_goal_n=0 hand_transfer.video=true

.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

本轮 18 项单元测试通过，包括 IK 不改写实时仿真状态、同杆路径确实回到原杆、25 cm 路径遵循墙面倾角、阶段边界平滑。完整动力学是否通过以每次 `summary.json` 为准，命令正常结束本身不表示成功。

## 下一步

保留已通过的同杆脱挂作为回归基线。对 25 cm 换杆先生成带关节限位、钩爪开口净空和全身碰撞约束的连续参考，重点处理上方横杆接近阶段；加入合理的支撑姿态调整，并在执行前拒绝不可达的路径。随后用同样的完整录像、额外接触审计及试探加载检查验收。通过之后再做另一只手及脚部换杆，最后形成训练所需的一步攀爬任务。
