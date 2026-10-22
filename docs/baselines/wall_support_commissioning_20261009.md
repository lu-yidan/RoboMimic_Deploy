# 横杆墙的10秒初始支撑验收

## 结论与范围

已将新横杆墙接入现有100维观测、29维位置残差动作和Climb部署适配器。使用固定关节位置目标的PD控制，在标称起始姿态完成10秒支撑，达到原有姿态保持阈值，并记录了双钩、双脚的接触载荷。

这不是学习策略，也不是连续攀爬。没有把平地站起权重直接用于横杆墙；控制器不读取接触力来调节动作，接触力仅供验收。原始训练配置的溯源缺口仍在，不影响本次独立的场景和控制接口验收。

## 实现

- 原入口新增配置`mujoco_wall_support`，运行`deploy_mujoco/run_wall_support.py`。
- `WallSupportEnv`继承既有`ClimbSupportEnv`，复用原来的Gym接口、关节映射、位置PD、力矩限幅和失败条件。
- 直接从声明的支撑姿态激活Climb FSM状态；不经过平地FixedPose姿态插值。注入的`ZeroResidualActor`恒定输出零残差，目标角度就是初始化角度。
- 每步比较部署适配器与环境的100维观测、目标角度、Kp/Kd，全部逐值相等。环境使用同一目标执行40个0.5毫秒物理步，控制周期20毫秒。
- 新场景生成独立合同指纹；旧场景PPO权重被明确拒绝加载，防止用错观测和动作定义。
- 目标点是第5排横杆上的右侧抓点，世界坐标约`[0.385607, -0.20, 1.55] m`。轴向统一为世界+Y。测试发现MuJoCo胶囊的局部Z轴与预想方向相反，早期v1目标点被反转到左侧；已在v2显式声明轴向符号并重跑正式实验。

源场景保持`outputs/climb_wall_2m_v3`原样。仿真没有固定基座、添加托举外力、运行中写入状态、改变横杆位置或添加额外接触点。钩爪的腕部转接件仍是占位，刚性连接是仿真假设，不能用于证明实物连接强度。

## PD参数选择

采用上游攀爬接口原增益为对照，不改变关节力矩限幅。预试验比较Kp倍率1、1.5、2，Kd相应乘以倍率的平方根。1.5是测试的倍率中第一个满足原30 mm姿态保持标准的值，因此正式验收采用Kp×1.5、Kd×sqrt(1.5)，而不是为通过而放宽阈值。

这里的平方根缩放仅用于保持单关节近似阻尼比例，不等价于已证明全身接触系统稳定。增益是在同一初始状态上选出的，本次结果属于确定性开发验证，不是留出测试成功率。

## 10秒正式对照结果

| 指标 | 原PD | Kp×1.5、Kd×sqrt(1.5) |
| --- | --- | --- |
| 完整仿真时长 | 10.00 s | 10.00 s |
| 全程最大骨盆位移 | 43.80 mm | 28.50 mm |
| 最终骨盆位移 | 42.17 mm | 26.85 mm |
| 全程最大骨盆倾角 | 6.03° | 4.04° |
| 全程最大钩爪接触位点位移 | 3.75 mm | 2.13 mm |
| 力矩限幅触发比例 | 0 | 0 |
| 原姿态保持标准 | 未通过 | 通过 |

原标准：完整时长内骨盆位移小于30 mm、倾角小于10°、双钩位点位移小于30 mm。峰值由每个物理步更新，不仅检查最后一帧。

通过条件还包括：最后2秒每个钩爪和足底平均竖直支撑力大于10 N；其他身体部位平均竖直支撑小于1 N；平均竖直接触力总和与模型重量的相对差小于2%；部署与环境接口一致；无力矩饱和。姿态阈值有明确余量限制，28.5 mm距离30 mm较近，不能据此推断扰动鲁棒性。

## 接触载荷

采用最后2秒的50 Hz采样均值，单位N：

| 接触部位 | 竖直支撑力 |
| --- | --- |
| 左钩 / 右钩 | 42.97 / 42.95 |
| 左脚 / 右脚 | 122.31 / 122.36 |
| 其他身体部位 | 0 |
| 合计 / 模型重量 | 330.59 / 330.59 |

双钩约承担26.0%的模型重量，其余由双脚承担。全部接触记录只包含双钩与`rung_05`、双脚与`rung_00`。这不是双钩全重悬挂试验。输出中的继承字段`bar_load_success_verified=false`继续保留，表示没有继承此前研究中的完整挂接/承载成功结论；本次仅新增“该标称姿态下测得接触支撑”的证据。

接触力来自MuJoCo求解器，经接触坐标系转换为世界坐标，按机器人受力方向统一符号。已用1 kg物体落在平面上的独立测试核对总支撑力为9.81 N，并检查多接触点求和。50 Hz载荷日志不是高频冲击峰值，也不是腕部真实传感器读数。

## 产物

- 正式运行：`outputs/wall_support_pd15_10s_v2`。
- 原PD对照：`outputs/wall_support_original_pd_10s_v2`。
- `support_full_run.mp4`：1280×720，25 fps，251帧，覆盖仿真0–10秒；文件时长10.04秒，末帧保持0.04秒。画面明确标注固定目标PD，不将其包装为训练结果。
- `support_metrics.png`：两组姿态对比、四个接触部位的力和总支撑力。
- `control_log.csv`：每个控制步的姿态、奖励、力矩饱和比例与接触载荷。
- `contact_log.json`：每个接触对、间隙及机器人受到的世界坐标力。
- `trajectory.npz`：控制频率的关节/基座状态、观测、目标角度、仿真时刻及录像时刻。
- `summary.json`、`manifest.json`、`contract.json`、`source_snapshot/`、`video_audit.json`、`artifact_sha256.json`：验收结果、参数、来源和校验。

旧v1记录保留用于追踪轴向问题，正式引用以v2为准。15项测试通过，包含旧环境回归、已知重量的接触力检验、新抓点方向和旧模型拒绝加载。

## PowerShell复现

```powershell
Set-Location D:\G1Climbing\external\RoboMimic_Deploy

# 首次生成新合同；本机已存在这两个目录时跳过。
.\.venv\Scripts\python.exe tools\prepare_wall_support.py
.\.venv\Scripts\python.exe tools\prepare_wall_support.py --kp-scale 1.5 --out outputs/wall_support_bundle_pd15_v2

# 10秒支撑录像，输出目录必须尚未存在。
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_wall_support hydra.job.chdir=false wall_support.output_dir=outputs/wall_support_my_run

# 原PD对照。返回summary中的passed=false表示未满足保持阈值，不能算成功。
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_wall_support hydra.job.chdir=false wall_support.bundle=outputs/wall_support_bundle_v2 wall_support.output_dir=outputs/wall_support_my_control wall_support.video=false

.\.venv-cad\Scripts\python.exe tools\plot_wall_support.py --run outputs/wall_support_my_run --comparison outputs/wall_support_my_control
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## 下一步

在此通过的标称支撑基础上，先设计和执行单手卸载、抬钩、移到上方25厘米横杆并重新加载的动作；保持另一只手与双脚提供支撑，逐阶段验收。随后才加入抬脚与身体爬升，形成一排的完整循环。只有这些动作的可达性和动力学通过，才将参考轨迹交给已有动作跟踪训练框架。
