# E1 部署记录（2026-09-17）

默认配置 `policy/smp_recovery/config/smp_recovery.yaml` 的 `profile` 为 `e1_prone_9999`。回退 FT12k 时改为 `ft12k_sim`。环境变量 `SMP_RECOVERY_PROFILE` 若已设置，会覆盖 YAML 的 profile。

E1 的来源为早期 L4 → V33 风格奖励 FT12k → 无压板、增加准备伏卧的 E1，再训练 10000 次 PPO 更新。使用 E1_prone/model_9999.pt；不是原始 V33 权重，也不是 E3。E1 训练为 20 秒 episode，约 20% 后段、20% 中段、60% 低位（包含 50% 准备伏卧），保留 push、随机化与 actor 噪声。

93D 观测与内嵌归一化、29D 动作、控制周期 0.02 秒、PD 参数、动作尺度及力矩裁剪均沿用 FT12k 部署契约。

源 checkpoint SHA256：a356adb7f7b7495cf71cf5700155703d0b8fdb469e7d28bfa807221e6f703709。

ONNX SHA256：d6e29c32d257e2abc982f8be28f187e0e08d8fc8575c1696f55b46e069010fe7。

## 部署端 MuJoCo 验证

使用实际 SmpRecovery 控制器，MuJoCo 3.2.3，物理步长 0.002 秒，单次评估 20 秒，成功要求连续稳定站立 10 秒。

|初态|成功数|
|---|---:|
|程序化仰卧|14/16|
|程序化伏卧|1/16|
|左侧卧|16/16|
|右侧卧|16/16|
|准备姿态伏卧（单独补测）|16/16|

四姿态 64 次的单次关节速度峰值 P95 为 12.85 rad/s、功率峰值 P95 为 440.21 W；准备伏卧分别为 8.74 rad/s、393.63 W。P95 是跨 episode 的峰值分位数，不能解释成硬件允许上限。E1 对初态敏感，不是 FT12k 的全面提升；以上为仿真结果，尚未进行 E1 真机验证。

随机输入 512 条的 PyTorch/ONNX 最大绝对误差为 8.11e-6。完整结果、导出脚本与源 checkpoint 位于本地 `outputs/e1_deploy/`。
