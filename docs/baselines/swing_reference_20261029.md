# 摆动抬升参考正式化（2026-10-29）

采用决定：用户已确认直接采用 v2。本项目后续实验以 `research_baseline.json` 声明的已审计参考及冻结策略为基线，以物理 v2 与末端姿态检查作为主要验收口径。关节闭合作为质量指标，35 mrad 为预警线，原 40 mrad 严格门只保留历史对照。当前训练/运行环境的旧终止标志不改写；判定以独立 `physical_v2_audit.json` 为准。该记录限定单周期正式链。后续已有四个开发工况通过连续八周期、约两米的物理/姿态及冻结策略审计，详见 [连续周期记录](v2_multicycle_20261029.md)；仍未重训。

状态：**完整审计链通过**。24 工况全周期、策略/控制器独立审计、新参考 24/24 前缀精确重放和来源链核对全部完成。未重训，未覆盖历史实验。该参考已是正式的单周期产出，不再只是 `.dsh-watch` 诊断候选。

最终校验：`outputs/swing_lift_phase_sine_policy_20261029_v2/reference_chain_audit.json`，其中 `passed=true`、`eligible_single_cycle_reference=true`。

参考 SHA-256：`84af42f4af0452de2888f0669fdecc71a3644c0421676a8d78b86ed13b656e95`。

冻结策略 SHA-256：`799d4d7b76ea2581b4fed3b000c8dafc6323227656f6bbf12ae35bbf601d8167`，与父实验完全一致。

## 24 工况结果

每个工况均从物理时间零连续运行 113.66 s。物理 v2 加末端姿态检查 **24/24 通过**；原严格门 **18/24**（原参考为 17/24）。新结果包含筛选时已经完成并按哈希保留的 T00、T12 两条记录，其他 22 条在包络选定后运行。全部使用同一份冻结 PPO 和固定控制器。

| 指标 | 结果 |
|---|---:|
| 新参考 24 工况闭合均值 / P95 / 最大值 | 38.115 / 39.852 / 40.092 mrad |
| 原参考共同完成的 22 工况，旧 / 新闭合均值 | 38.431 / 37.986 mrad |
| 共同完成的 22 工况中改善 / 变差 | 20 / 2 |
| 两个变差工况 | interior_04 +0.044 mrad；boundary_04 +0.058 mrad |
| 非预期 / 自碰撞 / 摆动接触力峰值 | 0 / 0 / 0 N |
| 全程最小原生关节限位裕量 | 2.987 mrad |
| 末段最小组法向支撑力 / 竖直支撑力 | 43.730 / 36.555 N |

原参考 T00、T01 在中途结束，不能拿它们的缺失终局数据计算旧均值。因此同时报告新 24 工况分布和相同 22 工况配对均值，不混用分母。

原诊断候选同为 24 工况，其均值 38.134 mrad、最大值 40.322 mrad。本次均值只再降低约 0.019 mrad；主要收益是消除 T12 回退、降低分布尾部，并补齐正式来源与审计链。不能将其描述为大幅训练收益。

旧严格门的 6 个未通过工况：T00、T01、T02、T09、T15 的抬升最大值超过旧 260 mm 上限约 0.012–0.051 mm；T14 的闭合为 40.092 mrad。它们均完成完整周期，保留旧失败标志，在物理 v2 的明确死区及质量指标分层下单独报告。

## 实现

`training/swing_foot_lift.py` 提供确定性的左脚 Cartesian 抬升，入口为 `tools/plan_clearance_cycle.py --lift-existing`。`--swing-foot-lift` 是摆动段额外抬升，原 `--foot-lift` 仍是最初卸载目标抬升，`--swing-clearance` 仍是原关节路径搜索净空，不能混用。

从最终 `support_retarget_plan_v1` 重建，保留既有高杆终点修正和右脚支撑调整。仅改变左腿六个关节；时间、接触计划、其余自由度和首尾构型保持不变。位置 IK 残差要求小于 20 μm，名义几何审计采样间隔为 5 ms。在线控制器、PPO 权重、动作校准与奖励函数不变。

正式生成的全间隙正弦参考与 `.dsh-watch/rebuild/plan_swing_open_v3/plan.npz` 六个数组完全相等。其窗口实际为 53.7–82.26 s，覆盖了卸载、摆动和落地，并非纯摆动段。

## T12 排查

T12 最大终局关节误差由左踝俯仰主导。原诊断包络在 80.66 s 还有约 1.4 mm 抬升；在卸载末端 70.2 s，位置型 IK 同时改变了参考脚部俯仰约 13°。这是参考变化，不等于实际机器人发生同幅度旋转。

| 候选 | T12 闭合误差（mrad） |
|---|---:|
| 原参考 | 38.331384 |
| 53.7–82.26 s 正弦 | 39.492318 |
| 上述包络在 80.66 s 前平滑收回 | 39.647986 |
| 上述包络在 81.46 s 前平滑收回 | 39.647309 |
| 70.2–82.26 s 正弦（选定） | **37.945750** |
| 摆动阶段正弦在 80.66 s 前收回 | 38.346877 |

选定峰值 8 mm、窗口 70.2–82.26 s 的正弦包络。保留原卸载动作，T12 比原参考改善 0.385635 mrad，比诊断候选改善 1.546569 mrad。筛选仅用 T12 和 T00；T00 完整运行且未撞杆，但旧严格终点门未通过。

结果不支持“只需提前收回落地尾段”的解释。限定到真正摆动阶段后改善，但有效抬升曲线也同时改变，尚未独立分离早期姿态、局部净空和接触历史的因果贡献。

## 证据与边界

计划：`outputs/swing_lift_phase_sine_plan_20261029_v2/`。

完整策略评估：`outputs/swing_lift_phase_sine_policy_20261029_v2/`。

诊断与筛选：`outputs/swing_lift_diagnosis_20261029_v1/`。预先声明的筛选协议在 `D:/G1Climbing/collab/logs/swing_reference_*20261029*.json`。

完整链包含：重建计划逐数组一致、名义碰撞审计、新参考独立前缀重放、动作及观测/奖励重建、控制器参考重建、源文件和模型哈希绑定。前缀须从时间零重放 3510 步到 70.2 s；不使用旧参考的审计代替。旧参考与新参考的物理前缀一致，但边界奖励因浮点累计时间略超过切换点有约 8e-15 的差异；新参考上的独立重放仍要求包括奖励在内完全一致。

额外记录末段双手/双脚的压缩法向力、竖直支撑力和承载杆位，结合原运行时 2000 Hz 接触/限位/数值检查与末端姿态检查。闭合单独作为质量指标报告；保留旧严格结果。物理 v2 抬升明确采用交接规范的 250 ± 10 mm 加 1 mm 死区，这是验收约定，不宣称已证明误差来源是数值噪声。

24 工况是开发/防回归集合；T12 和 T00 用过筛选。本次不访问保留集、不重训，不声称随机墙面泛化或连续八周期两米验收。

## 复现

在具备父实验及场景资源的仓库根目录运行，输出目录须不存在：

```powershell
.venv\Scripts\python.exe -m tools.plan_clearance_cycle --lift-existing outputs/support_retarget_plan_v1/plan.npz --swing-foot-lift 0.008 --swing-lift-start 70.2 --swing-lift-end 82.26 --out outputs/swing_reference_repro_plan
.venv\Scripts\python.exe -m tools.validate_swing_reference prepare --parent outputs/calibrated_geometry_training_baseline_20261027_v1 --plan outputs/swing_reference_repro_plan/plan.npz --out outputs/swing_reference_repro_policy
.venv\Scripts\python.exe -m tools.validate_swing_reference run --out outputs/swing_reference_repro_policy --workers 4
.venv\Scripts\python.exe -m tools.validate_swing_reference audit --out outputs/swing_reference_repro_policy
.venv\Scripts\python.exe -m tools.validate_swing_reference run --out outputs/swing_reference_repro_policy --kind prefix --workers 4
.venv\Scripts\python.exe -m tools.audit_swing_reference_chain outputs/swing_reference_repro_policy
```

测试：`.venv\Scripts\python.exe -m unittest discover -s tests -p test_swing_foot_lift.py`。覆盖零抬升恒等、非法输入、落地段保持、关键净空段一致及独立 Cartesian 目标检查。
