# A3 / A6 部署候选与固定顶板下一步方案

2026-09-21。已接入现有SMP 93D部署链路，没有启动真机控制程序。

## 一行切换

编辑 `policy/smp_recovery/config/smp_recovery.yaml`，只改 `profile`：

```yaml
profile: path_a6_5000
```

可选：

| profile | checkpoint | 特点 |
|---|---|---|
| `path_a6_5000` | A6 model_5000.pt | G + Q + L′，中期少接触切换、活动板候选 |
| `path_a6_9999` | A6 model_9999.pt | 相同奖励，训练更久；普通板改善但不是全面更好 |
| `path_a3_9999` | A3 model_9999.pt | 低位几何进展，不硬性要求手支撑；成功率较均衡但负载较高 |
| `ft_r2_9000` | 原R2 model_9000.pt | 已有真机参考策略 |

为遵守此前默认配置要求，文件当前仍选择 `ft_r0_9500`；测试新候选请明确修改上述一行。
不需要改顶层model_path、hash或控制参数，运行时按model_profiles自动解析。

修改后必须重启部署进程；退出/重新进入SMP状态不重新加载配置。
环境变量`SMP_RECOVERY_PROFILE`优先级更高，用YAML切换前执行：

```bash
unset SMP_RECOVERY_PROFILE
cd /home/luyd/workspace/RoboMimic_Deploy
/home/luyd/miniconda3/envs/robomimic/bin/python deploy_mujoco/deploy_mujoco.py
```

沿用现有手柄/FSM操作（Y进入SMP、F1阻尼、B回locomotion）。真机使用原有启动命令和网络配置，不需换控制参数。启动输出与日志会记录实际profile和模型SHA。
这些新候选只完成导出与部署仿真验证，未完成硬件验证；首先用于平地对比，不以训练端成功率批准板下真机动作。

## 导出与验证

源：`smp-r2-v33-path/logs/rsl_rl/v33_path_ablation/formal_20260920_r2_v33path_v1`。
三者均从原始R2开始finetune，非原96D V33 actor。
ONNX内含各自观测normalizer，确定性actor `[1,93] → [1,29]`。
关节顺序、default pose、action scale、Kp/Kd、控制步、力矩投影和原有完整日志机制沿用现有部署合同。
每模型512条输入与原生PyTorch对照，最大绝对误差≤2.4e-6；实际SmpRecovery enter/run及hash校验通过。
模型旁manifest保存源checkpoint、ONNX SHA及验证结果。

复查命令：

```bash
/home/luyd/miniconda3/envs/robomimic/bin/python tools/check_recovery_candidates.py \
  --profiles path_a6_5000 path_a6_9999 path_a3_9999 \
  --report path_interface_validation.json
```

部署CPU MuJoCo配对平地检查输出：`outputs/recovery_candidates/path_flat16/`，四方向各4姿态、每次20秒，包含原R2参考。
这16个样例是部署回归检查，不是正式训练validation总体成功率。

## 固定顶板建议（仅方案，尚未实现/启动训练）

高度指板底面到地面的净空，不是板中心高度。初期0.55–0.65m，能力达标后逐步扩展到0.40–0.60m。
目标分布可设0.40–0.45m占25%、0.45–0.55m占50%、0.55–0.60m占25%。暂不默认加入低于0.40m的困难组。
依据：已有配对reset的身体顶部加2mm约为伏卧0.20–0.33m、两侧卧约0.36–0.56m、仰卧约0.32–0.51m；这只是当前样本范围，不能当作机器人理论最小通行高度。
固定净空后在同方向/来源池内筛选和重采样，保证机器人初始碰撞体不穿板、至少有开放水平出口；不能为容纳任意高姿态每次自动抬高顶板，破坏规定高度分布。
先用无桌腿固定顶板；它没有可学习的质量或惯量响应。活动板保留质量随机化，尺寸变化时同步更新惯量、几何覆盖计算和reset筛选。
长宽厚初步建议0.70–1.10m × 0.50–0.80m × 0.04–0.08m，独立于净空采样，具体可行范围待几何筛选确认。

场景：C0平地50%+固定顶板50%；C1平地50%+活动板50%；C2平地50%+固定顶板25%+活动板25%。
各组相同reset：平地内部低40%/中40%/后20%，受限场景全部低位；总体低70%/中20%/后10%。
低位四方向各25%，每方向自然LAFAN75%+程序化25%；中后段用自然LAFAN。
受限低位再按约50%较深覆盖、50%靠近边缘分层，给横向脱困提供课程；保持方向、来源配额，不能让可行性筛选把侧卧全部筛掉。
保留机器人Kp/Kd、质量/惯量、延迟、摩擦及训练扰动随机化。当前本次工作只部署A系列，并未启动C系列。

## 本次部署平地检查结果

四方向各4个固定姿态，20秒，所有策略16/16连续稳定站立10秒，均无再次跌倒。以下P95按每条轨迹的最大关节/最大时刻峰值统计，不等于单关节时间P95：

| profile | 力矩峰值P95 Nm | 关节速度峰值P95 rad/s | 功率峰值P95 W |
|---|---:|---:|---:|
| ft_r2_9000 | 97.85 | 11.36 | 478.87 |
| path_a6_5000 | 129.01 | 13.98 | 707.59 |
| path_a6_9999 | 139.00 | 11.66 | 865.03 |
| path_a3_9999 | 139.00 | 11.82 | 902.80 |

检查结果支持模型导出与部署控制链正常工作，不支持将新策略认定为比R2更安全。各模型峰值不一定来自同一关节或同一时刻；原始逐关节数据保存在对应case JSON中。
