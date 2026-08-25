# SMP V3.3 Recovery 部署与整机测试说明

## 结论与适用范围

当前分支集成的是 SMP `baseline/v33-escape-model-95000-balanced` recovery 策略，支持在 MuJoCo 和实机部署程序中按一次 **Y** 启动 recovery。策略已经通过目标 MuJoCo 场景验证，可以进入有吊架、急停和监护人员的整机首轮测试。

该模型是“仰卧并受板覆盖后脱困、起身”的专项模型，不是任意倒地姿态的通用起身模型。当前结论不包含无保护实机部署，也不保证俯卧或训练分布之外的姿态能够恢复。

## 模型来源与完整性

- SMP tag：`baseline/v33-escape-model-95000-balanced`
- checkpoint：`smp_getup_escape_plate_v33_g1/wandb_checkpoints/owzoec67/model_95000.pt`
- checkpoint SHA-256：`38063879c144bd29af8e792bb7547b0e6c99e4043ba9b4c3c08219dab16ef81a`
- 部署模型：`policy/smp_recovery/model/smp_v33_escape_model_95000.onnx`
- ONNX SHA-256：`e7a31e831f394de06e69e143c57a2e984cd598003d17cd33c346d8a5ea7a7d0f`

ONNX 包含训练时的 observation normalizer。actor 输入为 96 维，顺序为：

1. base linear velocity（3 维，部署时统一写为 `0`）
2. base angular velocity（3 维）
3. projected gravity（3 维）
4. relative joint position（29 维）
5. joint velocity（29 维）
6. previous action（29 维）

输出为 29 维关节动作。部署端使用训练配置中的逐关节 default pose、action scale、Kp/Kd，并根据关节扭矩限制约束目标位置。切入 recovery 后先执行 10 个控制周期（约 0.2 秒）的平滑过渡。

## 控制方式

- **Y（上升沿）**：从任意非阻尼状态切入 SMP recovery；按住不会重复触发。
- **F1**：立即切入 damping。F1 的优先级高于 Y。
- **B**：切回 locomotion。
- MuJoCo 中原来的 Y 状态打印改为 **L1 + Y** 松开时触发。

无线手柄输入做了上升沿锁存，避免低层状态回调频率高于 50 Hz 控制循环时漏掉短按 Y。

## MuJoCo 验证结果

使用当前 SMP V3.3 官方 MjLab 评估场景，对 actor observation 的前三维统一置零，验证 checkpoint `model_95000.pt`：

- 设备：RTX 4090
- 随机种子：`20260817`
- 环境数：128
- 时长：1000 step / 20 秒
- 初始姿态：V3.3 legacy `supine`
- 覆盖物：8 kg，0.90 m × 0.64 m × 0.07 m
- 完成脱困并稳定站立：119 / 128（92.97%）
- 无效初始化：9 / 128（7.03%）
- 有效样本脱困并稳定站立：119 / 119（100%）
- 脱困时间中位数 / P90：1.76 秒 / 2.08 秒
- 稳定站立时间中位数：5.34 秒
- 接触力中位数 / P99 / 最大值：1110 N / 2186 N / 7084 N
- 最大关节扭矩均值：104.45 Nm
- 最大关节速度均值 / P95：14.82 / 16.34 rad/s

另外完成了以下部署侧检查：

- PyTorch checkpoint 与导出 ONNX 的最大动作误差：`2.29e-5`
- 96 维 observation、29 维 action、线速度固定为零和扭矩约束测试通过
- MuJoCo 部署闭环站立 smoke test：500 simulation steps 通过
- 目标专项姿态闭环恢复测试通过；训练分布外的相反俯仰姿态未通过，因此不得把该模型当作通用 recovery

## 整机首轮测试清单

整机测试必须先在无人员接触机器人、无刚性覆盖物的受保护条件下进行。

1. 确认使用本分支和上述 ONNX SHA-256；部署环境可正常加载 ONNX Runtime。
2. 使用承重吊架或可靠保护绳，地面铺缓冲垫，清空机器人动作范围。
3. 一名操作员只负责手柄和急停，测试前先确认 **F1 damping** 能即时生效。
4. 先在悬空或轻载条件确认关节方向、默认位姿、Kp/Kd 和动作幅值无异常。
5. 再在无覆盖物的目标仰卧姿态短按一次 **Y**，观察完整恢复过程；禁止人员扶持运动中的机器人。
6. 检查关节过流、温度、速度、落足和自碰撞记录。出现剧烈冲击、异常关节方向、打滑或失稳时立即按 F1/急停。
7. 无载重复测试稳定后，才逐级增加柔性、轻量负载。不要直接以 8 kg 刚性板开始实机测试。

首轮通过标准建议为：连续多次无载目标姿态恢复成功、无保护系统介入、无过流/过温/自碰撞、动作与 MuJoCo 趋势一致。满足这些条件后，再单独评审负载测试方案。

## 已知限制

- 尚未完成真实整机测试；本文的“准备完成”仅表示代码、模型和 MuJoCo 验证足以支持受保护首轮测试。
- base linear velocity 始终为零，这是当前 checkpoint 的部署输入约定。
- 模型只针对 V3.3 目标场景；俯卧、侧卧、悬空或不同覆盖物均属于分布外条件。
- MuJoCo 正式评估存在 7.03% 无效初始化，且瞬时接触力较高，实机必须从无载、低风险条件逐级推进。
