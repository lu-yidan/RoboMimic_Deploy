# SMP V34 gate-6000 93D 部署候选说明

该分支在 A11/A13 的严格 93D 部署栈上增加 V34 plate-escape gate-6000。
它是显式 canary：默认 profile 仍为 A11，只有设置
`SMP_RECOVERY_PROFILE=v34_93d_gate6000` 才会加载 V34；ONNX 文件、输入维度或
SHA-256 不一致时初始化会直接失败。

## 模型与结论边界

- SMP W&B：`tabletennis/smp/6qzo23hw`
- task：`Smp-Getup-Escape-Plate-V34-93D-G1`
- checkpoint：`model_6000.pt`
- checkpoint SHA-256：`471fb41444a5d8f2bce1f93b0e2d65a613e86c1d417ef4cba36ba48b9779474b`
- ONNX SHA-256：`f6dcd529f5036de4b6060426238db4aa26e635f63c822a0968c77463d0758c55`
- actor：单帧 93D，输出 29 个动作；observation normalizer 已嵌入 ONNX

冻结的 8 kg 压板 prone/supine 仿真矩阵中，gate-6000 的 macro/worst 恢复成功率为
94.34%/93.95%，原 V34 96D 对照为 89.06%/86.52%。但 gate-6000 的平均峰值力矩
和功率为 90.70 Nm/365.28 W，高于对照的 82.15 Nm/338.20 W；而且它只有一个
projected warm-start seed，没有 action-rate envelope。因此这是 MuJoCo 对比候选，
不是已批准真机模型。

## 在另一台电脑比较

```bash
git fetch origin
git switch codex/smp-v34-93d-deploy-canary
git pull --ff-only
python -m pip install -r requirements.txt
python -m unittest tests.test_smp_v34_93d_deployment

SMP_RECOVERY_PROFILE=v34_93d_gate6000 python deploy_mujoco/deploy_mujoco.py
```

默认 A11 对照：

```bash
SMP_RECOVERY_PROFILE=a11 python deploy_mujoco/deploy_mujoco.py
```

用相同的 grounded prone、supine 和拖动后静置姿态按 `Y` 激活。比较恢复时间、
蹲下转起身时的关节冲击、脚瞬间收拢、足滑、站稳后小碎步以及扰动后的恢复。

## 真机边界

若之后进行工程 canary，必须使用吊架、软垫、现场急停和低风险初始姿态，并从低限幅、
短时激活开始。当前 manifest 明确标记 `NOT_HARDWARE_APPROVED`；本提交不构成真机
安全验证或 RA-L 性能证据。
