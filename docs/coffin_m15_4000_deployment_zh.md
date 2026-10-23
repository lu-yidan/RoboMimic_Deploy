# M15@4000 箱内恢复候选

部署分支：`codex/table-config-balanced4`。在 `policy/smp_recovery/config/smp_recovery.yaml` 改一行后重启：

```yaml
profile: a6_coffin_m15_4000
```

该名称已注册到 `model_profiles`。环境变量 `SMP_RECOVERY_PROFILE` 优先于 YAML；若设置过，需要同步更改或取消。此次提交不改变默认 profile。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_coffin
```

箱体配置在 `deploy_mujoco/config/mujoco_coffin.yaml`：内长140 cm、内宽60 cm、总高40 cm（包含4 cm底板，不包含3 cm盖板）；默认自由箱体15 kg、自由盖板3 kg，程序化仰卧。Y启动SMP，R重置，START为FixedPose。

## 来源与训练

原始 A6@9999 → C1@4500 → M15@4000。源 checkpoint：4090-nblt `/root/workplace/smp-coffin-eval-20261021/logs/Coffin140_C14500_M15_6000_20261022_r1/model_4000.pt`。

上身（torso及手臂后代）的质量和惯量：25%环境固定1.0倍、25%固定1.5倍、50%均匀1.0–1.5倍。不是整机增重。保留Kp/Kd、延迟及其他动力学随机化。低位/过渡/蹲起向上速度目标0.10/0.15/0.20 m/s；此策略不是高速M15V2。

reset：50%平地（总环境20%自然低位、20%程序化低位、10%自然过渡），40%有盖箱、10%无盖箱；箱内程序化仰卧/伏卧各半。固定140×60×40 cm。继承C1奖励与控制配置。

93D观测、29D动作，归一化内嵌ONNX。沿用A6标称力矩上限，不应用balanced4的90%覆盖。物理步长2 ms，控制周期20 ms。质量随机化属于训练设置，切换profile不会把部署机器人模型自动加重。

## 验证范围

固定上身质量/惯量1.3倍的256次仿真验证：有盖仰卧50/51（98.0%）、无盖仰卧13/13；有盖伏卧0/51。SR3为连续站稳3秒，允许在箱内站立，不要求移盖或跨出箱体。此checkpoint根据验证集选择，结果不是独立测试集或真机成功率。原始统计见 `docs/evaluations/coffin_m15_4000_upper13_256.json`。平地姿态标签还包含自然过渡初态。

导出一致性与部署仿真冒烟检查见 `policy/smp_recovery/model/smp_a6_coffin_m15_4000_93d.manifest.json`。尚无此checkpoint的真机验证。
