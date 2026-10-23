# MV@4000 箱内恢复候选

部署分支：`codex/table-config-balanced4`。在 `policy/smp_recovery/config/smp_recovery.yaml` 修改并重启：

```yaml
profile: a6_coffin_mv_4000
```

已注册到 `model_profiles`；`SMP_RECOVERY_PROFILE` 环境变量优先于 YAML。此次不修改默认profile。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_coffin
```

箱体配置：`deploy_mujoco/config/mujoco_coffin.yaml`。默认内长140 cm、内宽60 cm、总高40 cm（含4 cm底板、不含3 cm盖板），自由箱体15 kg、自由盖板3 kg，程序化仰卧。Y启动SMP，R重置，START为FixedPose。

## 来源与差异

原始A6@9999 → C1@4500 → MV@4000。源checkpoint：4090-nblt `/root/workplace/smp-coffin-eval-20261021/logs/Coffin140_C14500_MV_6000_20261022_r1/model_4000.pt`。

MV的低位/过渡/蹲起头部向上目标为0.15/0.225/0.30 m/s，是标准目标的1.5倍；奖励速度上界0.375/0.45/0.45 m/s，独立向上超速阈值0.45 m/s。站立目标为0，目标接近阶段高度时衰减。它与M15不是只差速度：MV上身质量和惯量为25%固定1.0倍、25%固定1.3倍、50%均匀1.0–1.4倍；M15覆盖到1.5倍。

保留Kp/Kd、延迟和其他动力学随机化、力矩/功率/平滑成本。reset为50%平地（总环境20%自然低位、20%程序化低位、10%自然过渡）、40%有盖箱、10%无盖箱；箱内程序化仰卧/伏卧各半。箱体固定140×60×40 cm。

93D观测、29D动作，ONNX内嵌归一化；A6标称力矩上限，物理2 ms、控制20 ms。切换profile不会自动增加部署模型的质量。

## 验证范围

固定上身质量/惯量1.3倍的256次仿真验证，有盖仰卧47/51（92.2%）、无盖仰卧12/13，有盖伏卧0/51。SR3要求连续站稳3秒，允许箱内站立，不要求移盖或跨出。checkpoint按验证集选择，不是独立测试或真机成功率。原始统计见 `docs/evaluations/coffin_mv_4000_upper13_256.json`；平地姿态标签包含自然过渡初态。

导出一致性、部署仿真冒烟检查记录在对应模型manifest；尚无此checkpoint真机验证。
