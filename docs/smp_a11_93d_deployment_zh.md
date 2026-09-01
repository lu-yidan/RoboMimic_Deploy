# SMP A11 93D Recovery 部署说明

## 适用范围

本分支部署 A11 grounded-safety 的 gate-1000 工程候选。它是 93D 单帧
actor，不读取也不伪造 base linear velocity。当前证据仅覆盖平地、无板、
仰卧/俯卧/左右侧卧的 MuJoCo 冻结评测；不得据此声称复杂地形、压板脱困、
无保护真机安全或 RA-L criterion 已满足。

## 不可变模型身份

- SMP commit：b999375607d05120a8f73680b9d438723d6186e0
- W&B：tabletennis/smp/slbh9cnq
- checkpoint：model_1000.pt
- checkpoint SHA-256：5fecaee243ac8fea0798a4adf9de4b4801414203ce66a33a72b59876f3314bfb
- ONNX：policy/smp_recovery/model/smp_a11_grounded_safety_gate1000_93d.onnx
- ONNX SHA-256：d7dbf7824c0582866b12ab5987747e91271ef16f6de60587656a4a545c1a795e

ONNX 输入严格为 [1,93]：body-frame angular velocity 3、projected gravity 3、
relative joint position 29、joint velocity 29、last action 29；输出为 [1,29]。
normalizer 已内嵌。256 个固定随机输入的 PyTorch/ONNX 最大绝对误差为
5.7220458984375e-06，输出全部有限。

## 另一台电脑的 MuJoCo 验证

    git fetch origin codex/smp-a11-93d-deploy-canary
    git switch --track origin/codex/smp-a11-93d-deploy-canary
    sha256sum policy/smp_recovery/model/smp_a11_grounded_safety_gate1000_93d.onnx
    python -m unittest tests.test_smp_a11_deployment tests.test_logger_schema
    python deploy_mujoco/deploy_mujoco.py

启动后先确认打印 93 -> 29 和上述 ONNX SHA。Y 仍以按键上升沿进入 recovery，
F1 仍具有最高优先级。不要在 recovery 激活后拖动机器人；应先把机器人以近零
速度放置到接地的 prone/supine/left/right 姿态，再按 Y。

## 真机工程 canary

只允许吊架、软垫、急停和独立监护下进行，先验证 F1 damping。首次只做平地无板、
静止接地姿态；每次都保留 logger schema 2 binary、metadata 和同步视频。出现 NaN/Inf、
关节方向异常、持续碰撞、速度/扭矩/功率超限或无法及时恢复时立即 damping，并保留失败。
这些 canary 不是正式 80-trial 性能证据。
