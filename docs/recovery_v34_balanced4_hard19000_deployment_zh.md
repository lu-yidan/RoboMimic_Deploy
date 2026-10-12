# balanced4＋困难重采样 @19000 部署候选

分支：`codex/table-config-balanced4`（基于`codex/recovery-r012-d4-deploy`）。在 `policy/smp_recovery/config/smp_recovery.yaml` 修改：

```yaml
profile: v34_balanced4_hard19000
```

或在原桌子启动命令前添加 `SMP_RECOVERY_PROFILE=v34_balanced4_hard19000`。环境变量优先于YAML；切换后重启部署进程。仓库默认仍为 `a6_no_alpha_9999`。

源checkpoint：4090-bj 的 `/root/workplace/smp-v34-plate-transfer/logs/V34ER_balanced4_hard20_2ms_continue20k_20261011/model_19000.pt`。从balanced4@12000续训；物理2ms、控制20ms。平地/上下板/四腿桌/无腿顶板30/30/20/20%；桌子与顶板各约20%的reset重采样有效失败episode对应的原始初始状态，保持姿态及来源类别，不使用失败末态。

93D输入、29D动作，归一化内嵌。机器人与动作配置核对和balanced2一致；沿用Kp/Kd、默认关节姿态、action scale、90%标称力矩上限。部署保留10步进入过渡、动作clip10及位置目标力矩投影；这些与训练侧不完全相同。

验证：512条输入ONNX/PyTorch最大误差7.63e-6；hash、实际SmpRecovery加载/enter/run及限幅检查通过。CPU MuJoCo部署控制器平地23秒冒烟测试，仰卧2次、伏卧2次均达到SR3（4/4）。这不是总体成功率，也不是真机验证。未运行桌子场景的实际部署控制器评测。

训练侧512次验证：平地100%、上下板96.8%、四腿桌77.5%、无腿顶板96.1%。此前桌子录像使用训练侧控制，不能代替部署端验证。

桌子配置及按键见 `docs/table_simulation_config_zh.md`。本分支使用完整尺寸，height为桌面上表面高度；不接受under-the-table分支的top_size/leg_size半尺寸参数。默认40cm总高、36cm净空。程序化default-arms仰卧与自然切片不同，复现实验应同时对齐完整初始姿态、位置和几何。
