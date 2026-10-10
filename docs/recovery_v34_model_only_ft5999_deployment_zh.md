# V34 model-only @5999 部署

已加入策略 `v34_model_only_ft5999`。它是完成6000次训练的上下板 model-only 版本，不是正在训练的桌子/顶板微调。默认策略保持 a6_no_alpha_9999。

## 一行切换

编辑 `policy/smp_recovery/config/smp_recovery.yaml`：

```yaml
profile: v34_model_only_ft5999
```

重启部署进程生效。环境变量 SMP_RECOVERY_PROFILE 优先于文件，使用文件选择时先 unset SMP_RECOVERY_PROFILE。旧的 v34_93d_gate6000 是训练父策略，与本次不同。

## 来源与控制

来源：4090-bj `/root/workplace/smp-v34-plate-transfer/logs/V34_model_only_deploy_6000_20261010/model_5999.pt`。由 V34 93D gate6000 延续，使用迁移的部署机器人XML和原V34控制、奖励与prior。模型包含checkpoint观测归一化，输入93D、输出29D，不含线速度。

核对source env.yaml，默认关节姿态、action scale、Kp/Kd与现有部署配置一致。训练执行力矩上限是公共部署上限的90%，本profile单独设置tau_limit：髋pitch/yaw及腰yaw79.2Nm，髋roll/膝125.1Nm，踝及腰pitch/roll45Nm，肩/肘/腕roll22.5Nm，腕pitch/yaw4.5Nm。切换回旧profile自动恢复其公共限幅。日志新增effective_tau_limit记录实际值。

部署保留10步进入过渡、动作clip10及位置目标力矩投影；原model-only训练没有这些部署处理。训练物理步长5ms，本次部署仿真2ms，控制均20ms。因此不能把训练端视频当作完全一致的部署效果。

## 验证

512条输入ONNX/PyTorch最大绝对误差7.6294e-6；实际SmpRecovery加载、hash、enter/run与力矩投影检查通过。新增3项限幅测试覆盖本profile降额、旧profile不变、拒绝越过公共上限及非法值。

CPU MuJoCo3.2.3实际部署控制器、23秒平地冒烟：仰卧2条、伏卧2条，新SR3均通过（4/4）。头高1.10m、基座直立度0.90、双足各>20N、20秒内开始连续3秒，观察23秒；部署脚力为接触合力，而训练指标为max-contact，故这不是严格的训练端复测。该小样本不代表总体成功率。源训练站姿偏宽，桌下zero-shot仍有困难；此checkpoint尚无真机验证。

本次没有连接真机或启动DDS。详细来源hash、控制核对和检查结果保存在模型同名manifest.json。
