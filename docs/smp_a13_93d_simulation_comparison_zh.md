# SMP A13 93D 部署栈仿真对比说明

## 为什么暂不建议直接上真机

A13 gate-4999 在冻结平地五姿态评测中达到 100% strict recovery，恢复时间、足滑、
动作二阶变化和峰值力矩也优于其 A12 源模型；但它仍只有一个 warm-start policy seed，
不是三个独立 from-scratch seed。更重要的是，恢复后连续稳定站立 3 秒的 fixed-pose
macro 只有 8.154%，说明“能起来”和“真机上已安全收敛”不是一回事。当前也没有完成
逐关节真机 safety-limit 审计、部署闭环原始日志以及预注册 80-trial 计划。因此本分支只把
A13 暴露为需要显式选择的 MuJoCo 对比 canary，默认策略仍为 A11。

## 模型身份

- SMP 训练 commit：`45f7d650bc39d4549d721a201a181dc54a26c63f`
- SMP 冻结评测 commit：`e30a71bd98e3ad789587f449e2deefa51f9a4372`
- task：`Smp-Getup-Scratch-A13-F2S2-Continuous-Reset-G1`
- run：`a13_continuous_reset_5k_seed20261501_offline_retry_after_auth`
- checkpoint：`model_4999.pt`
- checkpoint SHA-256：`c4d05589645aa6b993e220c0cf3a9533fa83d50f5b9ed6764bb724b643cf1f94`
- ONNX：`policy/smp_recovery/model/smp_a13_continuous_reset_model_4999_93d.onnx`
- ONNX SHA-256：`4e1ddcce7fcf6596846d293fe6381490d78fb643f60903873e6ebd03e9db2255`

ONNX 内嵌 observation normalizer，输入为严格 `[1,93]`，输出为 `[1,29]`。256 个
固定随机输入的 PyTorch/ONNX 最大绝对误差为 `7.62939453125e-06`，输出全部有限。

## 同一部署仿真中的 A11/A13 对比

安装依赖并先跑测试：

```bash
python -m pip install -r requirements.txt
python -m unittest tests.test_smp_a11_deployment tests.test_smp_a13_deployment tests.test_logger_schema
```

A11 是默认 profile：

```bash
SMP_RECOVERY_PROFILE=a11 python deploy_mujoco/deploy_mujoco.py
```

A13 必须显式选择：

```bash
SMP_RECOVERY_PROFILE=a13 python deploy_mujoco/deploy_mujoco.py
```

两次都在接地且近零速度的 prone、supine、left-side、right-side 姿态下按一次 `Y`，
用 `F1` 随时切 damping。请比较首次站起时间、站稳后小碎步、脚底滑动、躯干摆动、
明显关节冲击和扰动后恢复，不要只比较最终是否站起来。A13 激活后不要直接拖动机器人；
拖动应在 recovery 未激活时完成，释放并静置后再按 `Y`。

本分支未把 A13 设为默认，也不构成真机授权。真机前至少还要完成 ONNX 部署闭环日志、
逐关节限值审计以及带吊架、软垫、急停的分级 canary。
