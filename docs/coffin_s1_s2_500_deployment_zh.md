# S1/S2@500 快速坐起候选

分支 `codex/table-config-balanced4`。在 `policy/smp_recovery/config/smp_recovery.yaml` 修改一行并重启：

```yaml
profile: a6_coffin_s1_500
```

或 `profile: a6_coffin_s2_500`。两项均在model_profiles中。SMP_RECOVERY_PROFILE环境变量优先，切换时同步修改或取消。默认profile未改。

| 策略 | 坐起干预 | 过渡/蹲起头部速度目标 |
|---|---|---|
| S1@500 | 躯干直立进展、低位目标下限0.15 m/s、减轻俯仰角速度惩罚、缩短阶段切换停顿 | 0.15/0.20 m/s |
| S2@500 | 同S1，低位目标/下限0.225 m/s，放宽起身超速阈值 | 0.3375/0.45 m/s |

共同来源：原始A6 → C1@4500 → 对应S1或S2续训500次。远端checkpoint为4090-nblt `/root/workplace/smp-coffin-eval-20261021/logs/Coffin140_C14500_S1_6000_20261023/model_500.pt` 和对应S2路径。

训练上身质量/惯量25%固定1.0、25%固定1.5、50%均匀1.0–1.5，保留其他动力学、Kp/Kd和延迟随机化。40%有盖箱、10%无盖箱、50%平地；平地自然低位仰卧配额约占总环境10%，收窄到425个半坐附近有效bank姿态。未直接将真机日志作为reset。箱内仍程序化仰卧/伏卧各半。

两者均93D观测、29D动作、归一化内嵌。沿用A6控制参数和标称力矩上限，物理2ms、控制20ms；切换profile不修改物理模型质量。Y进入SMP；Start仍为FixedPose，不自动启动SMP。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_coffin
```

默认箱体内长140 cm、内宽60 cm、总高40 cm（含4 cm底板，不含3 cm盖板），自由箱体15 kg、盖板3 kg；场景设置见 `deploy_mujoco/config/mujoco_coffin.yaml`。

## 验证范围

统一上身1.3倍、256环境验证：S1有盖仰卧44/51、无盖仰卧13/13；S2有盖仰卧45/51、无盖仰卧7/13。两者有盖伏卧0/51。SR3是连续站稳3秒，允许箱内站立，不要求移盖/跨出。均为早期候选，不代表已解决真机半坐卡点，且没有此checkpoint真机结果。

512组导出推理一致性与23秒部署控制器冒烟结果记录于各自manifest；冒烟检查只验证有限状态、数值与reset，不是成功率评估。
