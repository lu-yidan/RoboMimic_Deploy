# C1@4500 固定棺材恢复策略

分支：`codex/table-config-balanced4`。在 `policy/smp_recovery/config/smp_recovery.yaml` 的顶层选择：

```yaml
profile: a6_coffin_c1_4500
```

已添加到 `model_profiles`。默认选择保持不变；`SMP_RECOVERY_PROFILE` 环境变量优先于 YAML，切换后重启。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_coffin
```

场景尺寸等在 `deploy_mujoco/config/mujoco_coffin.yaml` 修改。默认140×60×40 cm（长宽为内尺寸，高含4 cm底板，不含盖板），自由箱体15 kg、自由盖板3 kg，程序化仰卧。Y启动SMP，R重置，START仍为FixedPose。

来源：4090-nblt `/root/workplace/smp-coffin-eval-20261021/logs/Coffin140_A6orig_C1_6000_20261021/model_4500.pt`。父策略为原始A6@9999，不是A6 no-alpha或V2@500。93D观测、29D动作，归一化内嵌，继承A6控制参数及共享标称力矩上限，不使用balanced4的90%限幅覆盖。物理2ms、控制20ms。

训练reset为平地50%（自然低位20%、程序化低位20%、自然过渡10%）、有盖箱40%、无盖箱10%；箱内仅程序化仰卧/伏卧各半。固定尺寸、箱体13.5–16.5kg、盖板2.5–3.5kg，保留机器人动力学随机化。修复箱底支撑阶段判断、取消箱内位移惩罚，移盖后起身四项奖励逐步放大至1.5倍；保留负载和平滑代价。

## 验证范围

训练侧验证bank中，有盖箱仰卧连续站稳3秒为51/51；允许箱内站立，不要求几何移盖或跨出。checkpoint按验证集选择，此51次结果不是独立测试集或真机成功率。当前重点只看仰卧；不能据此声称伏卧或平地能力全面保留。附原始统计 `docs/evaluations/coffin_c1_4500_256.json`（含其他姿态，平地supine/prone标签还包含过渡组，不用于宣称纯姿态成功率）。

这是仿真候选，尚无该checkpoint的真机验证。部署转换一致性与本地控制器冒烟结果记录在模型manifest。
