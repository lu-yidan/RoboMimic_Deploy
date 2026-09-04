# SMP V35 RD gate-5999 93D 部署候选

本分支以部署仓库最新的 `codex/smp-v34-93d-deploy-canary` commit
`087bc00a50edb259485baa14c92f4a1eed02f763` 为基线，加入 V35 `RD`
（expanded grounded reset + finite post-stand wrench）seed 20261801 的 gate-5999
ONNX。按当前工程测试需求，YAML 默认 profile 已切换为 `v35_rd_gate5999`；仍可通过
`SMP_RECOVERY_PROFILE` 显式切换回 V34、A13 或 A11。

## 不可变来源

- SMP task：`Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Wrench-G1`
- SMP commit：`81b031c7907dd009e4ed686eaec9963a54ced447`
- W&B：`tabletennis/smp/b7n8ih2u`
- policy/environment seed：`20261801`
- checkpoint：`model_5999.pt`
- checkpoint SHA-256：`9f31070bd5f22cfa5f6bf0c9ab13755f8dc4d20b4c88ba2a4bb4c8a6fa92c044`
- ONNX SHA-256：`9a446a1e299eaf1af5bf1f260e8c544cf8ed31b6a41742cb190a43512f1a6a9c`
- actor：单帧 93D，输出 29 个动作，observation normalizer 已嵌入 ONNX
- 导出一致性：256 个固定随机输入，最大绝对误差 `3.0517578125e-5`，输出全有限

RD 在平地 V34 93D 上继续训练，加入更广的接地 reset、2 秒站立保持和有限时长的
站立后物理 wrench。它没有复杂地形训练，也没有 action-target rate envelope。
当前只有定性目视比较，没有完成冻结正式评测，因此不能从训练曲线推断恢复成功率或真机安全性。

## 在另一台电脑进行 MuJoCo 对比

```bash
git fetch origin
git switch codex/smp-v35-rd-93d-deploy-canary
git pull --ff-only
python -m pip install -r requirements.txt
python -m unittest \
  tests.test_smp_a11_deployment \
  tests.test_smp_a13_deployment \
  tests.test_smp_v34_93d_deployment \
  tests.test_smp_v35_rd_93d_deployment

# RD 是该分支默认 profile。
python deploy_mujoco/deploy_mujoco.py

# 显式回到 V34 对照。
SMP_RECOVERY_PROFILE=v34_93d_gate6000 python deploy_mujoco/deploy_mujoco.py
```

请在相同的 prone、supine、侧卧、交叉腿和半蹲状态比较：恢复时间、首次稳定站立、
蹲起瞬间关节速度与力矩、脚是否突然收向 pelvis、足滑、双脚间距、小碎步、二次跌倒以及
扰动后的恢复。不要根据单次成功或 W&B reward 选择模型。
