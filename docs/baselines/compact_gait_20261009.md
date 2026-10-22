# G1 紧凑腿部动作：完整周期优化

## 结果与证据

2026-10-09，`outputs/compact_cycle_dynamic_v13` 从初态连续运行 **109.18 s**，完成双钩、双脚各上移一层以及身体抬升，运行验收 11/11、独立复核 11/11 均通过。22 项测试通过，日志在该目录的 `tests.log`。它是固定场景的轨迹规划与关节力矩控制，尚非训练得到的策略。

原始成功基线 `outputs/cycle_dynamic_v18` 保留不动。下表两侧均取实际 MuJoCo 状态记录，不混用理想参考动画。

| 指标 | 原始周期 | 紧凑周期 |
|---|---:|---:|
| 脚尖相对正前方的最大偏航 | 122.73° | **15.01°** |
| 最大膝间距（世界 Y 方向） | 58.48 cm | **33.55 cm** |
| 髋偏航关节最大绝对角度 | 152.94° | **26.45°** |
| 12 个腿关节累计绝对转动量 | 36.92 rad | **19.69 rad** |
| 完整周期时长 | 106.60 s | 109.18 s |
| 最终骨盆抬升 | 249.53 mm | 249.88 mm |
| 全程基座跟踪误差峰值 | 3.300 mm | 2.793 mm |
| 额外支撑 / 自碰撞 / 净空阶段碰撞力峰值 | 0 / 0 / 0 N | 0 / 0 / 0 N |
| 最后两秒末端对齐误差峰值 | 4.777 mm | 9.822 mm |
| 最后两秒关节构型闭合误差峰值 | 0.01706 rad | 0.03813 rad |
| 最后两秒基座线速度峰值 | 0.219 mm/s | 0.090 mm/s |
| 至少一个电机使用率超过 99% 的记录比例 | 16.89% | 23.63% |

脚尖偏航降低约 87.8%，膝间距降低约 42.6%，腿关节累计转动量降低约 46.7%。累计转动量是路径长度指标，不能当作能耗。偏航取足部纵轴在水平面的投影；所有风格指标在 50 Hz 的真实状态记录上计算，定义见 `tools/measure_cycle_style.py`。

证据文件：`summary.json`、`independent_audit.json`、`style.json`、`baseline_style.json`、`trajectory.npz`、`control_log.csv`、`contact_log.json`、`phase_events.json`、`manifest.json`、`reference.npz`、`reference_audit.json`、`source_snapshot/`。模型包与旧周期相同，均为 `outputs/wall_support_bundle_pd15_v2`。

录像：`cycle_recorded_1x.mp4` 是完整实际记录，`cycle_recorded_4x.mp4` 是明确标速的快放。`comparison_4x.mp4` 左侧为旧周期、右侧为紧凑周期，按相同仿真时间对照，较早结束侧保持末帧；它不是按阶段对齐的视频。曲线为 `cycle_metrics.png/pdf`。

## 具体修改

1. 足部上杆目标偏航从左右 ±120° 改为 0°。空中关节路径搜索同时约束实际脚尖偏航不超过 20°；真实执行峰值为 15.01°。
2. 上脚前，骨盆后移并下蹲，腰部先前倾 0.4 rad。右脚上杆后，四点接触下将骨盆向右侧移 4 cm、再降低 4 cm，腰前倾到 0.5 rad（28.6°），再卸载左脚。
3. 左脚卸载时，骨盆高度从 0.99 m 升到 1.23 m；前后方向按 PCHIP 路径通过 `(z,x)=(0.99,-0.10)、(1.04,-0.10)、(1.10,-0.065)、(1.16,-0.06)、(1.23,-0.02)`。该路径针对当前固定几何调定，不能直接推广到另一种墙或初态。过渡延长到 18 s，降低支撑脚踝的快速转动。
4. 两脚到位后，身体抬升并恢复腰部中立；取消旧方案左右脚分别卸载、转正、重落脚的 22 s 流程。末态闭合后保持 8 s。
5. 原关节加速度反馈 100/20、基座反馈 80/18、原电机限矩、接触模型及墙面尺寸不变。QP 达到迭代上限或仅给出低精度解时，以原精度和约束重试到 50000 次；仍不可行就失败。新增完整线性约束残差检查。确认残差和力矩偏差均在 1e-3 内后，只裁掉力矩解的数值越界，发送值严格落在原电机范围内。本轮 1165 次重试，不能据此声称实时计算已经满足真机需求。

仿真仅初始化时设置机器人状态，此后由关节电机和 MuJoCo 接触推进。没有根部驱动、额外外力、焊接约束或执行中覆盖状态。录像渲染从真实轨迹读回状态，仅用于显示。

## 尚未改善的指标

这次优化改善了腿部外展和扭转，周期时长增加 2.58 s，末态对齐和关节闭合精度下降，腕部接近限矩的比例增加。末端误差 9.822 mm 距 10 mm 验收线很近，关节闭合误差 0.03813 rad 也接近 0.04 rad 上限；参考最小关节限位裕量仅 0.02 rad。验收标准没有放宽，但当前余量不足以证明鲁棒性。

下一步应优先改善支撑脚落点/足部俯仰与手臂冗余构型，降低腕部载荷和末态偏差，再做误差扰动、无重置连续两周期及感知接入。当前只证实固定初态的一层 25 cm 周期；没有验证连续爬升两米、训练策略、真机或钩爪安装结构强度。

## 保留的失败

- `compact_cycle_dynamic_v4/v8`：60° 脚偏航方案，左脚卸载时分别因迭代上限 / 力矩分配不可行停止。
- `compact_cycle_dynamic_v5`：脚尖朝前但不预先侧移，48.18 s 左脚卸载入口不可行。
- `compact_cycle_dynamic_v6`：加了侧移，55.06 s 抬升途中仍不可行。
- `compact_cycle_plan_v9/v10`：身体前移过早，参考足部约束/碰撞检查不通过，未作为正式动力学成功。
- `compact_cycle_dynamic_v11`：几何和抽样静态分配可行，但过渡过快，54.68 s 动力学分配不可行。
- `compact_cycle_dynamic_v12`：首次完成 105.18 s 全过程，但最后两秒关节误差 0.04016 rad 超线，记录力矩有最大约 4.67e-5 Nm 的数值越界，因此判为失败。
- `compact_cycle_dynamic_v13`：保持时长和数值边界处理修正后，从头连续运行，全部验收通过。

## PowerShell 复现

在已有本机依赖和冻结模型包的基础上运行。每次输出目录必须不存在。

```powershell
cd D:\G1Climbing\external\RoboMimic_Deploy
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'

# 默认使用这次通过的紧凑参考；运行、独立审核并录制
.\tools\run_climb_cycle.ps1 -OutputDir outputs/my_compact_cycle

# 显式运行保留的旧参考
.\tools\run_climb_cycle.ps1 -Plan outputs/cycle_contact_plan_v17/plan.npz -OutputDir outputs/my_original_cycle

# 从当前规划代码重新生成同配置参考
.\.venv\Scripts\python.exe -X utf8 tools\plan_climb_cycle.py `
  --out outputs/my_compact_reference --yaw-deg 0 `
  --prep-x -0.1 --prep-z 1.03 --unload-x -0.02 --waist-pitch 0.4 `
  --shift-y -0.04 --shift-z 0.99 --shift-waist 0.5 `
  --curved-transfer --transfer-seconds 18 --closed-hold-seconds 8
.\tools\run_climb_cycle.ps1 -Plan outputs/my_compact_reference/plan.npz -OutputDir outputs/my_regenerated_compact

# 量化实际轨迹
.\.venv\Scripts\python.exe tools\measure_cycle_style.py --states outputs/my_compact_cycle/trajectory.npz --out outputs/my_compact_cycle/style.json
.\.venv-cad\Scripts\python.exe tools\plot_climb_cycle.py --run outputs/my_compact_cycle
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`outputs` 中的模型、参考、记录和视频为本机资产，不随 Git 自动分发。运行源码在启动时冻结；参考源码在规划时冻结。报告与结果的 SHA-256 清单见输出目录的 `delivery_manifest.json`。
