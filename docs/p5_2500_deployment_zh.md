# P5–2500 部署（2026-09-17）

默认 profile 为 `p5_all_low_2500`，源自 `smp-prior-replay-ft/outputs/p_series_analysis/P5_all_low/model_2500.pt`，并非 3000/3500。

P5 从 FT12k 迁移 actor，100% 低位 reset：50% 压板准备伏卧＋50% 平地四方向低位；f2s2 prior、ws6、无在线失败回放。训练 episode 10 秒，保留 push、随机化与噪声。低 SMP 终止配置保留，但低位 episode 豁免。训练时的压板环境不需要在部署端观测中添加压板信息。

部署保留原 93D 观测、29D 动作、内嵌归一化、0.02 秒控制周期、PD、动作尺度及力矩限制。schema 3 日志继续有效。

导出 PyTorch/ONNX 512 条随机输入最大绝对误差 5.72e-6。

部署端 MuJoCo 3.2.3 初检使用真实 SmpRecovery 控制器，20 秒回合内要求连续稳定站立 10 秒：四种程序化姿态各 4/4，共 16/16；准备伏卧 16/16。四姿态单回合速度峰值 P95 11.83 rad/s、单关节功率峰值 P95 490.32 W；准备伏卧为 10.61 rad/s、469.54 W。这是小样本仿真检查，不代表真机结果；没有在部署端复测压板。

源文件和导出 SHA256、完整摘要保存在模型旁的 manifest；本地逐回合结果位于 `outputs/p5_deploy/`。

切换策略：编辑 `policy/smp_recovery/config/smp_recovery.yaml` 的 `profile`。回退可用 `e1_prone_9999` 或 `ft12k_sim`。环境变量 `SMP_RECOVERY_PROFILE` 若存在，会覆盖 YAML，测试 P5 时应清除或改为 `p5_all_low_2500`。
