# balanced2 @5999 部署候选

在 `policy/smp_recovery/config/smp_recovery.yaml` 改一行：

```yaml
profile: v34_balanced2_5999
```

重启进程。环境变量 `SMP_RECOVERY_PROFILE` 优先于文件；若使用文件设置，先取消环境变量覆盖。默认仍为 `a6_no_alpha_9999`。

## 来源与训练

4090-bj：`/root/workplace/smp-v34-plate-transfer/logs/V34ER_balanced2_6000_20261010/model_5999.pt`。父策略V34 model-only@5999，追加6000次PPO更新，4096环境、固定学习率3e-5。
场景：平地30%、上下活动压板30%、固定四腿桌20%、固定无腿顶板20%。本版本不是原3322或2233。

相对原V34桌子续训：取消1500N接触力硬终止，保留穿透/仿真异常；受阻任务缩放α=1；四项起身任务×2，几何/分离/稠密脱困项×2。几何清空后关闭进展奖励，距reset位置1.2–1.5m渐退、超过1.5m本回合锁定脱困奖励为零；完成奖励与高度及直立度挂钩。低/中/高阶段向上目标速度.20/.30/.40m/s，接近目标高度衰减；其他动作平滑、关节速度/功率限制保留。

## 部署与验证

93维输入、29维输出，包含原checkpoint归一化。源robot/actions配置与已部署V34 model-only父策略逐项一致，沿用90%标称力矩上限、Kp/Kd、默认关节姿态、action scale。未提高电机能力。

512条随机输入ONNX/PyTorch核对通过，最大绝对误差1.52588e-5；实际SmpRecovery加载/hash/enter/run/限幅检查通过。

CPU MuJoCo3.2.3实际部署控制器，23秒平地冒烟（仰卧2、伏卧2），SR3为4/4，最大头高约1.170–1.177m，无仿真异常。仅4条接口测试，不能当作总体成功率或真机验证。SR3：头高≥1.10m、base直立度≥.90、双足合力各>20N，20秒内开始连续3秒，观察23秒。脚力合力与训练max-contact口径不同。

部署保留10步进入过渡、action clip10及位置目标力矩投影；训练没有这些处理。部署测试物理步长2ms、训练5ms；控制周期均20ms。没有连接真机/DDS。本轮没有在部署端验证桌子，桌下视频来自训练仿真。

W&B：https://wandb.ai/tabletennis/smp/runs/pc5jlbd0 。模型同名manifest记录hash、来源和检查结果。
