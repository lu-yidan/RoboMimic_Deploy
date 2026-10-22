# 攀爬部署接口的首个训练基线

最新进展（2026-10-10）：完成 17 项横杆位置误差网格，严格验收 3/17；加入接触确认和卸载力矩可行性保护，两个触发案例完成 0.5 s 保持复核；单次右钩换杆残差 PPO 完成 2,048 步，并通过 MuJoCo 部署入口实跑。零修正、训练前、训练后均为 5/5，尚无学习收益证据。新任务采用 77D/7D 接口，区别于下文历史 100D/29D 姿态保持任务。详见[本轮实验报告](../docs/baselines/rung_robustness_training_20261010.md)。


2026-10-09：已经接通实体钩爪场景、Gymnasium、PPO、ONNX和部署适配器。当前任务是机器人已经在横杆前就位后的**4秒初始支撑姿态保持**，不是换杆、抬脚或连续攀爬。

## 环境与控制

- 29关节、100维观测，与`policy/climb/contract.py`共用转换。
- 策略50 Hz；MuJoCo步长0.0005秒，每动作40个物理步。每步计算位置PD力矩并按模型执行器限幅。
- 自由基座，无焊接约束，无外加基座力，无旧QP辅助，无运行中重写状态。
- 场景从旧研究的`normal35`导入，保留实体连墙横杆、钩爪和脚台。资源清单记录SHA，网格沿用Unitree许可。旧控制器的成功结论不随场景导入。
- 起始姿态已经抬臂挂钩，超出原站姿配置的部分动作范围。因此导入器生成独立`contract.json`，默认角度改为场景初始姿态，关节限位取场景。借用的PD增益和动作尺度尚未经过攀爬优化。
- 当前目标为已接触的`right_bar`。位置和轴向由仿真几何转到骨盆坐标系，属于理想观测；没有实际感知误差、延迟或丢失。
- 奖励及终止可以使用仿真真值，actor不输入世界基座位置或腕踝力。重置固定、没有域随机化。

每步奖励为`exp(-(root_error/.03)^2-(tilt/.15)^2-(hand_error/.03)^2)`减去动作幅值、动作变化惩罚以及失败惩罚。根部下降超过0.15米、倾斜超过45度或位移超过0.2米等触发提前终止；4秒为时间截断。姿态保持通过要求全过程根部偏移小于30毫米、倾斜小于10度、双手位置偏移小于30毫米。该标准不证明横杆承载或悬挂成功。

## 已测结果

所有评估来自同一个确定性开发起点，不是独立随机试验，不应计算泛化成功率。

| 策略 | 4秒完整运行 | 最终根部偏移 | 全程最大根部偏移 | 姿态保持通过 |
|---|---|---|---|---|
| 零动作PD | 是 | 98.816 mm | 98.816 mm | 否 |
| 随机小扰动动作，标准差0.03 | 是 | 134.537 mm | 134.537 mm | 否 |
| PPO训练前确定性输出 | 是 | 99.152 mm | 99.152 mm | 否 |
| PPO 4096步后 | 是 | 72.308 mm | 76.956 mm | 否 |
| 同一模型ONNX部署适配器 | 是 | 72.308 mm | 76.956 mm | 否 |

首轮CPU训练31.56秒，约129.8动作步/秒。训练前后单次评估回报8.789和9.948；训练检查点重载动作逐值一致。收益变化仅说明这个短流程可以更新策略，不能证明收敛。

导出验证覆盖402个训练前后记录观测：动作最大误差`5.96e-8`，位置目标最大误差`1.19e-7 rad`。随后实际调用`Climb.run`，通过其ONNX推理、历史动作和输出增益跑200次动作更新；适配器与训练环境观测逐值一致。相对原PyTorch评估的qpos最大差值`1.02e-8`（qpos含不同单位，不作位置精度指标）。

12项单测及Gymnasium接口检查通过。关键动力学测试使用独立40步PD循环，核对qpos/qvel、执行器限幅和无外力；另检验重置、非法动作、时间截断和结束后step拒绝。

## PowerShell复现

```powershell
Set-Location D:\G1Climbing\external\RoboMimic_Deploy
# 首次建立独立环境；已有.venv时跳过下一行。
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-climb-train.txt
```

首次导入资产（已有`assets/climb_mounted`时跳过；导入器拒绝覆盖）：

```powershell
.\.venv\Scripts\python.exe tools\import_climb_scene.py `
  --case D:\G1Climbing\results\g1_unified_climbing_demo_20261009\normal35 `
  --license D:\G1Climbing\models\unitree_g1\LICENSE
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_climb_*.py" -v
.\.venv\Scripts\python.exe tools\check_climb_interface.py
```

重跑必须选择新目录，保留已有结果：

```powershell
.\.venv\Scripts\python.exe tools\check_climb_env.py --out outputs\climb_env_baseline_rerun
.\.venv\Scripts\python.exe training\train_climb.py --steps 4096 --out outputs\climb_ppo_smoke_rerun
.\.venv\Scripts\python.exe tools\export_climb_policy.py --run outputs\climb_ppo_smoke_rerun
```

当前结果：`outputs/climb_env_baseline_v1/summary.json`、`outputs/climb_ppo_smoke_v1/summary.json`、后者的`export/verified_export.json`。导出目录包含`policy.onnx`、`metadata.json`、`contract.json`和适配器仿真轨迹。训练清单保存种子、软件版本、超参数、源码哈希和源码副本。后续修改冻结源码应明确新版本，并保留对应的原始源码与结果。

资产及输出被Git忽略，不能只克隆本分支就获得完整实验资产；当前导入命令依赖本机研究场景。共享实验时需要另行分发带许可和SHA清单的场景与模型包。

## 下一阶段

2026-10-10 更新：电机力矩代价与分阶段腿部反馈已验证，新增独立配置 `mujoco_climb_cycle_optimized`。同参考单周期近限矩记录比例由 23.63% 降为 8.28%；加入严格保留约束的备用 QP 求解后，**无重置连续两周期 218.36 s、累计抬升 49.97 cm 已通过运行及独立验收**。27 项测试通过。完整证据、失败诊断、边界与命令见 [载荷分配与连续周期报告](../docs/baselines/load_allocation_continuation_20261010.md)。原配置与历史参考保留用于对照。

腿部动作优化：默认完整周期入口已切换到通过验收的 `outputs/compact_cycle_plan_v13/plan.npz`，脚尖偏航峰值 15.01°、膝间距峰值 33.55 cm。完整动力学及独立复核通过；腕部力矩和末态精度余量仍小。新旧实测对照、保留失败及复现命令见 [紧凑步态报告](../docs/baselines/compact_gait_20261009.md)。旧参考仍可通过 `-Plan outputs/cycle_contact_plan_v17/plan.npz` 显式运行。

2026-10-09 更新：固定横杆墙上的完整 25 cm 名义周期已通过，含双手、双脚换杆、载荷转移、身体抬升和关节构型闭合。它采用确定性力矩控制，尚未作为学习策略接入上述 Actor/FSM。证据、失败对照和复现命令见 [完整周期验收](../docs/baselines/full_cycle_20261009.md)。PowerShell 可执行 `tools/run_climb_cycle.ps1` 重新运行并录制，默认使用已审核的参考轨迹。后续学习任务应以此完整周期为新任务，不能把旧姿态保持的 PPO smoke 结果当作攀爬成功。

1. 在固定起点上建立达到姿态保持标准的学习基线，先观察更长训练的收敛趋势，保留零动作对照。
2. 再引入声明范围内的起始误差、杆位置误差和测量扰动，分开训练集与留出评估，报告失败类型。
3. 为换杆和载荷转移另建任务及承载验证，不能仅靠双手位置小误差判断挂稳。
4. 真机前补齐起始状态进入、目标感知、时间同步和源tick停更处理；当前没有启用机器人命令发送。
