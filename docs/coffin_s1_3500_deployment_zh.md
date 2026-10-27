# S1@3500 部署候选

分支：`codex/table-config-balanced4`。在 `policy/smp_recovery/config/smp_recovery.yaml` 修改一行并重启：

```yaml
profile: a6_coffin_s1_3500
```

已添加至model_profiles；SMP_RECOVERY_PROFILE环境变量优先于YAML。默认策略未改，S1@500及其他候选仍可选择。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_coffin
```

场景默认内长140 cm、内宽60 cm、总高40 cm（含4 cm底板、不含3 cm盖板），自由箱体15 kg、盖板3 kg。场景参数在 `deploy_mujoco/config/mujoco_coffin.yaml`。Y进入SMP、R重置、Start为FixedPose。

来源：原始A6 → C1@4500 → S1@3500。服务器4090-nblt，checkpoint为 `/root/workplace/smp-coffin-eval-20261021/logs/Coffin140_C14500_S1_6000_20261023/model_3500.pt`。

训练修改与S1@500相同，见 `docs/coffin_s1_s2_500_deployment_zh.md`：强化低位躯干坐直进展，未坐直时保留0.15 m/s向上目标下限，减轻俯仰角速度惩罚，缩短低位阶段切换停顿，约10%环境使用现有自然库中的半坐附近reset。上身质量/惯量覆盖1.0–1.5倍，保留其他动力学、Kp/Kd、延迟随机化。40%有盖箱、10%无盖箱、50%平地。它不是S2高速站起版本。

93D观测、29D动作、内嵌归一化，A6标称控制参数和力矩上限。物理2ms、控制20ms。切换profile不会自动更改部署机器人质量。

## 验证范围

固定上身1.3倍的256环境验证：有盖仰卧49/51、无盖仰卧12/13、有盖伏卧0/51；原始统计见 `docs/evaluations/coffin_s1_3500_upper13_256.json`。SR3要求连续站稳3秒，允许箱内站立，不要求几何移盖或跨出。3500按历史有盖仰卧验证成功率选择；不是独立测试或真机成功率。S1后期存在性能退化，因此不使用最终权重替代此候选。

导出512样本一致性及23秒部署控制器冒烟结果见模型manifest。尚未验证该checkpoint能解决真机半坐卡点；冒烟只检查数值、控制器加载和reset，不代表成功率或真机安全验证。
