# SMP V36 SD seed-20261902 gate-5999 93D 部署候选

本分支以部署仓库 `codex/smp-v35-rd-93d-deploy-canary` commit
`f63ba4cb13cc8fed53fa572c15ac6415f123dcbb` 为基线，加入 V36 `SD`
（stairs-only safe terrain + finite post-stand wrench）seed 20261902 的 gate-5999
ONNX。V35 RD 仍是 YAML 默认 profile；V36 SD 必须通过
`SMP_RECOVERY_PROFILE=v36_sd_seed20261902_gate5999` 显式选择，防止误用到真机。

## 不可变来源

- SMP task：`Smp-Getup-V36-93D-Safe-Stairs-Wrench-G1`
- SMP commit：`0e145e8f57c3634dfddc043f2c4f2d3e991cccaa`
- W&B：`tabletennis/smp/2yehj243`
- policy/environment seed：`20261902`
- checkpoint：`model_5999.pt`
- checkpoint SHA-256：`5a2c09641ec3389b699f344b50e846a248309ba4777994611838b0b149ba9311`
- ONNX SHA-256：`392dfda5ad441532d3e7ddc1142cee867643823bfc8acd271e454b3f9004fd4d`
- actor：单帧 93D，输出 29 个动作，observation normalizer 已嵌入 ONNX
- 导出一致性：256 个固定随机输入，最大绝对误差 `5.340576171875e-5`，输出全有限

该模型从 V35 RD seed 20261802 gate-5999 继续训练；加载 actor、critic 和 observation
normalizers，重置 optimizer、iteration 和 environment steps。训练仅包含安全台阶 level 0--2，
level 3 留出；reset 经地形接触校验，并有 2.4 m landing island 与站稳后有限时长物理
wrench。它没有压板训练、reset replay bank 或部署侧 action-rate envelope。

冻结工程诊断覆盖 4 个地形、4 个姿态、16 cells、4096 rollouts：macro strict success
`98.63%`，worst cell/pose `96.48%`（left），prone `99.61%`，supine `99.22%`，
secondary fall、invalid dynamics、terrain exit 均为 `0`。不过只有 warm-start seed，评测中
未启用自动 wrench，也没有三组 from-scratch seed 或正式真机 safety trial，因此这不是上真机授权。

## 在另一台电脑进行 MuJoCo 对比

```bash
git fetch origin
git switch codex/smp-v36-sd-93d-deploy-canary
git pull --ff-only
python -m pip install -r requirements.txt
python -m unittest \
  tests.test_smp_a11_deployment \
  tests.test_smp_a13_deployment \
  tests.test_smp_v34_93d_deployment \
  tests.test_smp_v35_rd_93d_deployment \
  tests.test_smp_v36_sd_93d_deployment

# V36 SD 必须显式选择。
SMP_RECOVERY_PROFILE=v36_sd_seed20261902_gate5999 \
  python deploy_mujoco/deploy_mujoco.py

# 不设置环境变量时仍运行 V35 RD 默认策略。
python deploy_mujoco/deploy_mujoco.py
```

请用相同 prone、supine、侧卧、交叉腿、半蹲和台阶接触状态，对比恢复时间、首次稳定站立、
单脚支撑、蹲起瞬间关节速度和力矩、脚是否突然收向 pelvis、足滑、双脚间距、小碎步、
二次跌倒以及外力后的恢复。不要根据单次成功或 W&B reward 决定真机部署。
