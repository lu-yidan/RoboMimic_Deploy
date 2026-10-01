# A6_no_alpha@9999 部署与对照记录

2026-10-01。此次迁移的是此前左右对照视频**右侧的 A6_no_alpha@9999**，不是后续 α 扫描中的 AC3_a100，也不是历史真机 A6@9999。定位为优先验证的压板恢复候选，尚无此 checkpoint 的真机结果。

## 一行切换

`policy/smp_recovery/config/smp_recovery.yaml` 默认已改为：

```yaml
profile: a6_no_alpha_9999
```

比较历史真机 A6 可改为 `path_a6_9999`；回到原默认为 `ec7_9000`。只修改 profile，不需要改顶层 model_path、hash 或控制参数。重启部署进程后生效，启动输出应显示 `profile=a6_no_alpha_9999`。环境变量 `SMP_RECOVERY_PROFILE` 优先于 YAML；使用 YAML 选择时先执行 `unset SMP_RECOVERY_PROFILE`。

沿用现有 SMP_RECOVERY 的启动和手柄操作。本次未启动 DDS 或真机控制，未修改 FSM、PD 增益、力矩限幅、动作缩放或日志逻辑。

## 训练身份与改动

- run：`logs/rsl_rl/a6_paper/formal_20260929_seed20261013/A6_no_alpha/model_9999.pt`。
- 训练初始化：R2@9000 的 actor 与观测归一化；critic 和 optimizer 重新初始化。并非从历史 A6@9999 继续训练。
- 4096 环境、10000 PPO 更新、seed 20261013、训练 episode 10s。
- checkpoint SHA-256：`24f1a4513cf5cabb9bd0b9ef54e90a3d50e625a87774878379af5da5dcf8fbd1`。
- 导出的确定性 ONNX 为 `[1,93] → [1,29]`，内嵌该 checkpoint 的归一化。无需 base 线速度。

相对首轮 A6_full 对照，唯一设计变量是受阻时任务系数固定为 **α=1**：不再压低受阻阶段的起身/安静支撑任务奖励。**SMP 奖励、几何脱困奖励均保留**；不是无 SMP 的模型。

几何进展、稠密净空、与板分离三项权重分别为 0.45、0.08、0.01；脱困完成 0.6、过大板接触力 −0.03、Q −0.03、L −0.20，其他 V33/R2 任务与执行代价沿用。它没有后续实验新增的“首次脱困后关闭向外奖励”或位置锚定机制。α 是训练奖励设置，不是部署时的控制开关。

场景为平地 50%、上下滑动压板 25%、自由板 25%。平地低/中/后段为 40/40/20，板下全部低位，总体约 70/20/10。低位四方向均衡，自然 LAFAN/程序化为 75/25。保留分部件质量/惯量、电机 Kp/Kd、0–10ms 延迟、push、观测噪声等随机化。使用 f2s2 prior、ws=6；不因站起或低 SMP 提前终止。此组没有完全固定顶板或台阶，板尺寸也不是后续尺寸随机化版本。

## 它是否最好

它是首轮七组消融中**压板 SR10 最高**的 checkpoint，可优先验证；不能据此称为跨训练重复、跨场景或真机安全性全面最优。后续 AC3 虽也使用 α=1，是另一次训练，不能混用结果。

首轮同协议训练端评测，每组上下板 48 次、自由板 48 次，四方向均衡，低位自然/程序化 75/25；每次 20s。SR10 要求脱困后满足完整站立判据且连续保持 10s。

| 指标 | 首轮 A6_full@9999 | 本次 A6_no_alpha@9999 |
|---|---:|---:|
| 压板 SR10 | 45/96（46.9%） | 70/96（72.9%） |
| 脱困后 3s 最大 XY 位移 P95 | 2.470 m | 0.486 m |
| 位移统计有效轨迹数 | 61 | 71 |
| 关节力矩峰值 P95 | 139.0 Nm | 139.0 Nm |
| 关节速度峰值 P95 | 15.18 rad/s | 15.05 rad/s |
| 关节机械功率峰值 P95 | 1157.8 W | 1010.1 W |

位移是相对**首次脱困位置**，不是相对初始躺姿；仅统计完成脱困后完整 3s 窗口的轨迹，两个有效子集不同。视频展示的是少量样例，不能替代总体统计。表中的 A6_full 是本轮重训完整配置，**不是下表的历史 A6**。

## 本次部署端检查

512 条输入与原生 PyTorch 对照，最大绝对误差 `8.1062e-6`，通过既有误差门槛。实际 SmpRecovery 的 hash 校验、enter/run、有限性和力矩投影检查通过。逐项核对源 run 的 env.yaml：93D 观测顺序、29 个关节顺序、默认姿态、action scale、Kp/Kd、力矩限幅和 20ms 控制周期一致。

使用实际部署控制器、CPU MuJoCo 3.2.3，物理步长 2ms，每条 20s。平地四方向各 16 条程序化有效接触姿态；扰动测试取各方向前 4 条。历史 A6 使用完全相同初态重新评测。此次没有板、push 或随机观测噪声。

| 条件 | 本次 A6_no_alpha | 历史 A6@9999 |
|---|---:|---:|
| 标称平地 SR10 | 62/64 | 64/64 |
| 上半身质量与惯量 ×1.3 | 15/16 | 16/16 |
| 物理电机 Kp/Kd ×0.8 | 16/16 | 15/16 |
| 物理电机 Kp/Kd ×1.2 | 16/16 | 16/16 |
| 命令延迟 10ms | 16/16 | 16/16 |
| 上半身 ×1.3 + 电机 ×0.8 + 延迟 10ms | 14/16 | 13/16 |

所有 288 条轨迹均无数值不稳定或直立后再摔倒。新模型标称平地两条未满足 SR10，连续站稳最长分别为 6.98s 和 9.86s，最后 10s 的未通过项是脚间距；并非完全没有起身。

| 标称平地指标 | 本次 A6_no_alpha | 历史 A6@9999 |
|---|---:|---:|
| 关节力矩峰值 P95 | 139.00 Nm | 139.00 Nm |
| 关节速度峰值 P95 | 13.68 rad/s | 13.32 rad/s |
| 关节机械功率峰值 P95 | 1042.77 W | 914.41 W |
| 最后 10s base XY 累计路程均值 | 0.0041 m | 0.0057 m |

峰值 P95 的口径：每条轨迹先取所有关节、所有时刻的最大绝对值，再跨轨迹取 P95；功率为 `abs(tau*dq)`。力矩和功率使用 500Hz 物理执行值。新模型标称平地最大单次关节速度达到 **29.57 rad/s（左腕 pitch）**，电机增益 ×1.2 时功率峰值 P95 为 **1316.69 W**。因此“小位移”不等于低负载；本次结果不支持直接宣称它比历史真机 A6 更安全。

建议先按同姿态、同板布置对照历史 A6，再依据真机起身率、关节负载及脱困后运动决定论文最终方法。若最终选此策略，必须标明 α=1 的实际配置；历史 A6 的真机证据不能自动归给新 checkpoint。

## 复查与证据

模型目录同名 `.manifest.json` 记录源模型、导出 hash、接口误差、控制参数核对，以及本次两模型全部条件的汇总。完整本地轨迹为 `outputs/recovery_candidates/a6_no_alpha_deploy_20261001/`（不入 Git）。

```bash
cd /home/luyd/workspace/RoboMimic_Deploy
/home/luyd/miniconda3/envs/robomimic/bin/python tools/check_recovery_candidates.py \
  --profiles a6_no_alpha_9999 --report a6_no_alpha_interface_validation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/luyd/miniconda3/envs/robomimic/bin/python tools/validate_recovery_stress.py \
  --cases outputs/recovery_candidates/paired_flat64/paired_cases.json \
  --out outputs/recovery_candidates/a6_no_alpha_repeat \
  --profiles a6_no_alpha_9999 path_a6_9999 --skip-amp --workers 8
```

复查需保留本机的 parity fixture 与冻结 cases，输出目录须为新目录。在其他机器可使用原 checkpoint 和 `tools/export_recovery_candidate.py` 重新生成 fixture；不要用另一个 α=1 run 代替。部署 ONNX 本身已随本次提交跟踪。
