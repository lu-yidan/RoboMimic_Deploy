# EC7@9000 部署说明

2026-09-22，已接入现有 SMP 93D 部署接口。当前部署的是 EC7 的 `model_9000.pt`，不是预检模型，也不是尚未出现的9999版本。没有启动真机控制。

## 一行切换

`policy/smp_recovery/config/smp_recovery.yaml` 现默认：

```yaml
profile: ec7_9000
```

回到此前选中的模型可改为 `path_a6_5000`；其他可用比较项包括 `path_a6_9999`、`ft_r2_9000`。无需修改顶层model_path/hash或控制参数，实际使用model_profiles中选中的条目。修改后重启部署进程，切换FSM状态不会重新读取配置。`SMP_RECOVERY_PROFILE` 环境变量优先于YAML；如使用YAML选择，启动前执行 `unset SMP_RECOVERY_PROFILE`。

沿用原有手柄和真机启动方式；启动输出/运行日志应显示 `profile=ec7_9000`。本次没有改FSM、控制参数、限幅或日志机制。

## 模型与实验含义

- 源工作区：`smp-a6-egress`，分支 `codex/a6-egress-convergence`。
- 源run：`logs/rsl_rl/egress_convergence/formal_20260921_a6_egress_v1/EC7/model_9000.pt`。
- actor从A6@9999续训；EC7在首次脱困后永久关闭向外进展奖励，加入首次脱困位置附近0.30m软范围，并将板场景脱困后的向上速度目标/阈值增加50%。
- 训练场景：平地50%、完全固定顶板25%、自由板25%；不是旧的上下滑动板。总体低/中/后段约70/20/10，低位四方向均衡，低位自然/程序化75/25。
- 保留分部件质量/惯量、Kp/Kd、0–10ms延迟、push、观测噪声及板尺寸/自由板质量随机化。
- 位置范围、脱困检测和速度门控属于训练奖励，不是部署时额外运行的定位控制器或硬边界。
- ONNX内嵌该checkpoint的观测归一化，确定性 `[1,93] → [1,29]`。模型及checkpoint SHA见同目录manifest。

## 本次验证

1. 512条输入对照原生PyTorch，ONNX最大绝对误差 `5.7220459e-6`，通过既有误差门槛。
2. 实际SmpRecovery加载、hash校验、enter/run、目标有限性和力矩投影检查通过。
3. 对照原run的env.yaml：观测顺序、29关节顺序、default pose、action scale、Kp/Kd、effort limit及20ms控制周期一致。
4. 部署端CPU MuJoCo平地四方向各4条固定test-bank姿态，每次20s：16/16连续站稳10s，无再次摔倒。

| 指标 | 本次16条轨迹 |
|---|---:|
| 最大关节力矩峰值P95 | 139.00 Nm |
| 最大关节速度峰值P95 | 11.28 rad/s |
| 最大关节机械功率峰值P95 | 883.21 W |
| 首次直立时间中位数 | 2.52 s |
| 最后10s base XY累计路程均值 | 0.277 m |

P95按每条轨迹的所有关节、所有时刻最大绝对值再跨轨迹取分位，不是单关节时间P95。力矩/功率峰值仍高，不能由16/16推断更安全。此为接口与平地回归检查，尚未在部署端验证板下脱困或真机。

原始结果、完整轨迹：`outputs/recovery_candidates/ec7_9000_flat16/`；接口报告：`outputs/recovery_candidates/ec7_interface_validation.json`。输出目录仅保存在本地，关键结果写入模型manifest。

复查：

```bash
cd /home/luyd/workspace/RoboMimic_Deploy
/home/luyd/miniconda3/envs/robomimic/bin/python tools/check_recovery_candidates.py \
  --profiles ec7_9000 --report ec7_interface_validation.json
```

复查依赖本机保留的 `outputs/recovery_candidates/ec7_9000_parity.npz`；在其他机器需用原checkpoint运行 `tools/export_recovery_candidate.py` 重新生成fixture。
