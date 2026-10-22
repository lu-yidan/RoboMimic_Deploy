# 原场景成熟策略复现：BeyondMimicMJ 躺下站起

## 验收结论

**已有权重的完整动作部署复现通过。** 在学长 `lu_pub` 对应的原始 `g1_description/g1_liao.xml` 中，完整FSM调用原来的StandUpMJ策略，机器人完成9秒站起片段，随后保持站立约2秒。全过程视频、物理状态、控制日志、文件校验值及复现命令均已保存。

**训练溯源尚有明确缺口，不能声称“原训练实验已完整复现”。** 已核实训练框架声明、权重内的运行名称、参考动作及部署配置；部署仓库没有提供该次训练的源码提交、完整PPO/奖励/随机化配置、随机种子或检查点迭代号。不能把当前mjlab默认参数冒充这次权重的真实训练配置。如果阶段1要求这些训练参数也全部可追溯，则阶段1仍有此项未闭环。

## 来源与版本

- 实际运行仓库：[`lu-yidan/RoboMimic_Deploy`](https://github.com/lu-yidan/RoboMimic_Deploy/tree/lu_pub)，本地工作分支`liuze-climb`，上游基点`0dd6ab4899068a9987ca596a1c32cb1c94f22ec4`。
- 框架上游为[`ccrpRepo/RoboMimic_Deploy`](https://github.com/ccrpRepo/RoboMimic_Deploy)。本实验运行的是学长分支，不将两个仓库的版本混称为同一个版本。
- 策略：[`BeyondMimicMJ.py`](https://github.com/lu-yidan/RoboMimic_Deploy/blob/0dd6ab4899068a9987ca596a1c32cb1c94f22ec4/policy/beyondmimic_mj/BeyondMimicMJ.py)，源文件明确说明训练使用mjlab。该部署动作属于BeyondMimic型参考动作跟踪，不是本项目自行训练的攀爬策略。
- 训练框架官方来源：[`mujocolab/mjlab`](https://github.com/mujocolab/mjlab)，其文档提供G1动作跟踪训练示例；算法相关官方代码为[`HybridRobotics/whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking)。这些公开默认配置仅供后续训练参考，不能据此还原本权重的全部参数。
- 权重：`policy/beyondmimic_mj/model/beyondmimic_mj.onnx`；SHA256：`40a904463849e193af0751080d6148a03acd72a085e4de870dc25e30b2f747ea`。
- ONNX内置`run_path`：`2026-03-10_11-44-32_g1_fallAndGetUp2_subject2_mj`；导出器标注PyTorch 2.9.0。运行名称是导出元数据，不是可下载的训练日志地址。
- 参考动作：`fallAndGetUp2_subject2_mj.npz`，50 Hz，8195帧；StandUpMJ截取原序列10.0–19.0秒，即第500–949帧，共450帧。原始动捕版权、采集来源及重定向过程未随此部署包完整提供。
- 配置：[`standup_mj.yaml`](https://github.com/lu-yidan/RoboMimic_Deploy/blob/0dd6ab4899068a9987ca596a1c32cb1c94f22ec4/policy/beyondmimic_mj/config/standup_mj.yaml)。本地模型XML、该配置、策略类、ONNX和NPZ均经Git对象检查，与上游基点一致。

## 运行配置与初始化

| 项目 | 数值或处理 |
| --- | --- |
| 场景 | 原始`g1_liao.xml`，包含原G1手部、地面与远处可动物箱；未换钩爪 |
| MuJoCo / Python | 3.3.7 / 3.12.10 |
| 物理步长 / 控制周期 | 0.002 s / 0.02 s，10次物理步对应一次控制 |
| 状态 / 执行器 | nq=43、nv=41、29个力矩执行器，包含箱体自由度 |
| 策略输入 / 输出 | 154维观测、29维关节位置动作；MuJoCo DFS顺序 |
| 暖机 | 原配置10个控制步，0.2秒 |
| 动作与PD | 直接使用上游默认角度、动作缩放、Kp/Kd及动作裁剪±10 |
| 力矩限制 | 使用原`mujoco.yaml`的29维限幅，逐物理步检查 |
| 初始化 | 仅t=0设置为站起片段首帧的根位姿和关节角；初始速度为零 |
| FSM | Passive → FixedPose → StandUpMJ；FixedPose仅作为入口转换，不额外运行两秒恢复姿态 |
| 初始状态之后 | 不再写入qpos，不使用外力、基座固定、参考轨迹强制回放或中途复位 |

新增的`baseline`配置分支从学长的原启动入口进入自动测试器，自动安排FSM命令和录像。测试器使用原策略类与完整FSM，复用同样的状态字段、PD公式、控制周期和限幅；并非未经修改的交互式主循环。它在控制区间开始采样状态并计算动作，消除了手动按键时间差。原键盘和手柄模式保持可用。

## 实测结果

正式录像运行：`outputs/standup_baseline_v2`。独立复跑：`outputs/standup_baseline_repeat_v2`。

| 指标 | 实测 | 本次部署验收阈值 |
| --- | --- | --- |
| 物理仿真时长 | 11.28 s，5640个物理步 | 完整动作及结束保持 |
| 参考动作执行 | 450/450帧，逐帧索引0–449 | 全部执行 |
| 初始骨盆高度 | 0.05975 m | <0.20 m |
| 最终骨盆高度 | 0.77029 m | 见保持阶段条件 |
| 结束保持阶段最低 / 平均高度 | 0.76970 / 0.77147 m | >0.65 / >0.70 m |
| 结束保持阶段最大倾斜 | 11.65° | <25° |
| 动作阶段关节RMSE | 0.18129 rad | <0.35 rad |
| MuJoCo数值警告 | 0 | 0 |
| 独立复跑的状态和目标动作最大差值 | 0.0 | <1e-10 |

阈值是本项目的部署检查标准，不是论文官方指标，也不能代表真机成功率。仅检验固定场景、固定初始状态；没有检验跌倒姿态随机化、外部扰动或钩爪改装。

## 视频与日志

- `full_run.mp4`：完整仿真录像，H.264，1280×720，25 fps，283帧，11.32秒。覆盖仿真0–11.28秒；末帧多保持0.04秒，没有删除开头、失败段或跳过中间动作。
- `storyboard.png`：从该视频抽取0、2、4、6、8、11.28秒的画面；不是独立渲染的假动作。
- `control_log.csv`：50 Hz状态、阶段、参考帧、骨盆位姿指标、关节误差、力矩与地面接触体。
- `trajectory.npz`：`physics`每行按`[time(1), qpos(43), qvel(41), torque(29)]`排列，500 Hz；同时含控制频率的qpos/qvel、目标关节位置、推理前154维观测、初始状态和视频对应仿真时刻。非策略阶段观测用NaN标记，不应当作有效推理输入。
- `runtime.log`、`fsm_events.json`：原策略输出和FSM切换记录。
- `metrics.json`：结果和阈值；`audit.json`：独立复跑对比、录像解码帧数与时间核验。
- `provenance.json`、`resolved_config.yaml`：版本、训练运行标识、运行环境、完整部署参数和文件SHA256。
- `source_snapshot/`：此次运行使用的测试器、入口、相关配置、权重和参考动作快照；网格校验值位于provenance中。
- `artifact_sha256.json`：验收产物校验值。

探索性记录单独保留在`outputs/baseline_preflight`：Pinocchio整段动作最终跌倒，未记为通过案例。早期`standup_baseline_v1`及`standup_baseline_repeat_v1`发生结果JSON序列化错误，虽然仿真执行完毕，仍不作为正式通过记录；修正后才运行上述v2及独立复跑。

## PowerShell复现

当前项目环境已安装依赖。以后新建环境时，先安装`requirements-baseline-record.txt`，并按仓库README检出完整网格和策略权重。

```powershell
Set-Location D:\G1Climbing\external\RoboMimic_Deploy

# 完整动作、视频和日志。换一个新的输出目录，避免覆盖已有证据。
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_standup_baseline hydra.job.chdir=false baseline.output_dir=outputs/standup_my_run

# 再跑一次，不录像，用于独立确定性比对。
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_standup_baseline hydra.job.chdir=false baseline.record_video=false baseline.output_dir=outputs/standup_my_repeat

# 解码整段视频、检查日志和两次运行是否一致。
.\.venv\Scripts\python.exe -X utf8 tools\audit_standup_baseline.py --run outputs/standup_my_run --repeat outputs/standup_my_repeat
```

生成器不会覆盖已存在的目录。成功退出会打印`passed: true`；如果站立高度、倾斜、轨迹完整性或数值稳定性不达标，会返回失败，不能将“模型加载成功”算作动作通过。

## 尚需补齐的训练证据

后续公开检索仍未定位到该`run_path`对应的原始训练配置。找到的学长公开仓库[`RoboNaldo-YD`](https://github.com/lu-yidan/RoboNaldo-YD)描述的是Isaac Lab中的足球射门训练，不能据此归因为本次mjlab站起权重的训练来源。该缺口继续保留，不用不匹配的训练仓库替代。

本次没有重新训练权重。要完全闭环“原训练配置可复现”，需要该ONNX对应训练运行的环境配置、PPO配置、奖励与随机化设置、训练代码提交号、随机种子、检查点迭代号，以及参考动作生成记录。现有仓库与权重元数据不足以恢复这些信息。本报告不把这个缺口转化成对用户的阻塞请求；已经完成的部署证据可以先用于组会与下一阶段平台验证。
