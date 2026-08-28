# SMP Recovery 新电脑 MuJoCo 配置与验证

本文用于在一台新的 Ubuntu/NVIDIA 电脑上复现
`deploy/smp-v33-recovery-95000` 分支的 MuJoCo 部署测试。环境文件只覆盖
MuJoCo 验证；真实机器人部署还需要 Unitree SDK、CycloneDDS 和对应网络配置。

## 1. 拉取分支和模型

```bash
git clone git@github.com:lu-yidan/RoboMimic_Deploy.git
cd RoboMimic_Deploy
git fetch origin
git switch --track origin/deploy/smp-v33-recovery-95000
```

如果仓库已经存在：

```bash
cd RoboMimic_Deploy
git fetch origin
git switch deploy/smp-v33-recovery-95000
git pull --ff-only
```

Recovery ONNX 是普通 Git blob，已随分支提交，不依赖 Git LFS。拉取后验证：

```bash
test -f policy/smp_recovery/model/smp_v33_escape_model_95000.onnx
sha256sum policy/smp_recovery/model/smp_v33_escape_model_95000.onnx
```

期望文件大小为 `872587` bytes，SHA-256 为：

```text
e7a31e831f394de06e69e143c57a2e984cd598003d17cd33c346d8a5ea7a7d0f
```

## 2. 创建已验证环境

推荐安装 Miniconda。NVIDIA 驱动需能支持 CUDA 12.1；先检查：

```bash
nvidia-smi
```

从仓库内的环境文件创建独立环境：

```bash
conda env create -f environment/smp_recovery_mujoco.yml
conda activate robomimic-smp-mujoco
```

不要再执行 `pip install -r requirements.txt` 覆盖该环境；此验证以
`environment/smp_recovery_mujoco.yml` 的固定版本为准。

本次已验证的软件版本：

| 软件 | 版本 |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.3.1 + CUDA 12.1 |
| NumPy | 1.24.4 |
| MuJoCo | 3.2.3 |
| ONNX | 1.17.0 |
| ONNX Runtime | 1.19.2 |
| Hydra Core | 1.3.4 |
| OmegaConf | 2.3.1 |
| pygame | 2.6.1 |
| SciPy | 1.10.1 |
| PyYAML | 6.0.3 |

Recovery 推理使用 ONNX Runtime CPU provider；GPU/CUDA 主要用于保持项目其他
策略依赖与已验证电脑一致。

## 3. 环境与模型预检

在仓库根目录运行：

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import mujoco
import numpy as np
import onnxruntime as ort

model_path = Path("policy/smp_recovery/model/smp_v33_escape_model_95000.onnx")
expected_sha = "e7a31e831f394de06e69e143c57a2e984cd598003d17cd33c346d8a5ea7a7d0f"
actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
assert actual_sha == expected_sha, (actual_sha, expected_sha)

mj_model = mujoco.MjModel.from_xml_path("g1_description/g1_liao.xml")
assert mj_model.nu == 29, mj_model.nu

session = ort.InferenceSession(
    str(model_path), providers=["CPUExecutionProvider"]
)
assert session.get_inputs()[0].shape == ["batch", 96]
assert session.get_outputs()[0].shape == ["batch", 29]
dummy = np.zeros((1, 96), dtype=np.float32)
action = session.run(None, {session.get_inputs()[0].name: dummy})[0]
assert action.shape == (1, 29), action.shape

print("PASS: XML=29 DOF, ONNX=96->29, SHA-256 verified")
PY
```

预期输出：

```text
PASS: XML=29 DOF, ONNX=96->29, SHA-256 verified
```

## 4. 连接并检查手柄

当前 MuJoCo 启动程序要求手柄，未连接时会报
`RuntimeError: No joystick connected!`。

```bash
ls -l /dev/input/js*
python -c "import pygame; pygame.joystick.init(); print('joysticks:', pygame.joystick.get_count())"
```

第二条命令应输出 `joysticks: 1` 或更大。如果手柄按键序号不同，可以用环境
变量覆盖，例如 `JOYSTICK_Y=3`。

## 5. 启动 MuJoCo

```bash
conda activate robomimic-smp-mujoco
cd RoboMimic_Deploy
python deploy_mujoco/deploy_mujoco.py
```

默认配置位于 `deploy_mujoco/config/mujoco.yaml`，加载
`g1_description/g1_liao.xml`，仿真步长 0.002 秒、控制 decimation 10，
即 recovery 按 50 Hz 运行。

主要操作：

- 短按一次 **Y**：从当前状态切入 SMP recovery。
- **L2 按下再松开**：MuJoCo 阻尼保护。
- **START**：切到 FixedPose。
- **B**：切到 locomotion。
- **L1 + Y**：只打印关节位置，不启动 recovery。
- **SELECT** 或关闭窗口：退出。

注意：MuJoCo 使用 L2 松开触发保护；实机无线遥控器使用 F1。Y 是上升沿触发，
按住不会重复启动。Recovery actor 的前三维 base linear velocity 在部署代码中
始终为零。

## 6. 新电脑验收顺序

1. 先完成 SHA-256、XML 和 ONNX 预检。
2. 启动窗口，确认终端出现
   `[SMP_RECOVERY] Initialized: ... (96 -> 29)`。
3. 先确认 L2 阻尼保护有效，再测试 Y。
4. 当前部署 XML 没有 8 kg 覆盖板；这里只验证无载 recovery、按键和部署链路。
5. 尽量使用目标仰卧姿态。该 checkpoint 是 V3.3 覆盖板仰卧脱困专项模型，
   不是任意姿态的通用起身模型。

正式 V3.3 MjLab 目标场景的既有验证结果为：128 个环境中 119 个脱困并稳定
站立；9 个为无效初始化，故有效样本为 119/119。详细数据和整机安全边界见
`docs/smp_recovery_deployment_zh.md`。

## 7. 常见问题

- `No joystick connected!`：确认 `/dev/input/js0` 存在并检查当前用户权限。
- MuJoCo 窗口打不开：确认本地桌面会话中 `echo $DISPLAY` 非空；SSH 使用
  X11 转发或在物理桌面运行。
- ONNX opset 错误：确认使用 `onnxruntime==1.19.2`，不要装旧版 1.10。
- NumPy 类型导入错误：确认使用 `numpy==1.24.4`，不要降到 1.20。
- Y 无响应：先确认 pygame 的 Y 物理编号；必要时通过
  `JOYSTICK_Y=<编号>` 覆盖。
