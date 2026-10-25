# lyd-deploy

宇树 G1 29 自由度统一部署分支。整合 balanced4 桌子/箱体场景、可选 ScaleBFM 和 AMP 策略切换；SMP 只保留三个部署模型。基于 [RoboNaldo Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy)，保留上游许可证及其他原有动作功能。

## 下载与环境

```bash
git clone --single-branch --branch lyd-deploy --depth 1 git@github.com:lu-yidan/RoboMimic_Deploy.git
cd RoboMimic_Deploy
conda activate robomimic
pip install -r requirements.txt
```

已有部署环境可直接使用。新环境参考 `environment/smp_recovery_mujoco.yml`；真机 SDK、DDS 与网络配置见 [机载安装指南](onboard/docs/INSTALL_AGENT_GUIDE.md)。

该分支从整合后的快照开始，不继承旧实验历史。旧 clone 切换分支不会自动回收旧对象；需要小体积部署副本时，使用上面的单分支 clone。

## 唯一策略配置入口

编辑 [`config/deploy.yaml`](config/deploy.yaml)，重启进程生效。MuJoCo、Python 真机入口和 C++ bridge 的 Python 策略进程共用此文件。

```yaml
scalebfm:
  enabled: false
  # 其他模型目录、动作文件和 CUDA 选项见文件内配置。
amp:
  profile: amp_0309_1
smp:
  profile: v34_balanced4_hard19000
```

| 选项 | 可选值 |
|---|---|
| `scalebfm.enabled` | `false` / `true` |
| `amp.profile` | `amp_0309_1` / `stratified_guidance_60000` / `stratified_single_60000` |
| `smp.profile` | `path_a6_9999` / `a6_no_alpha_9999` / `v34_balanced4_hard19000` |

AMP profile 选择对应 ONNX 和参数覆盖项；当前三个导出使用相同部署观测和动作接口，模型加载时校验 SHA-256。不要额外改变动作缩放或关节顺序。

SMP 每个模型保留自己的哈希、元数据和力矩限制。balanced4 使用额定限制的 90%，两个 A6 使用各自原有公共限制。`policy/*/config` 是控制合约，不是日常策略选择入口。

离线验证可以用 `LYD_DEPLOY_CONFIG=/绝对路径/另一个配置.yaml` 指定完整配置副本。兼容旧评估流程的 `SMP_RECOVERY_PROFILE` 环境变量仍优先于文件；日常部署应清除该变量，避免覆盖所选策略。

## 仿真与场景

```bash
# 普通场景
python deploy_mujoco/deploy_mujoco.py
# 桌子
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table
# 箱体和盖子
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin
```

场景参数在 `deploy_mujoco/config/`，与策略选择独立。尺寸、质量、盖子开关说明见 [桌子](docs/table_simulation_config_zh.md) 和 [箱体](docs/coffin_simulation_config_zh.md)。场景可用不代表任意策略都能完成该场景任务。

## ScaleBFM

默认关闭；关闭时不会导入外部 `scalebridge`、加载 BFM 权重或启动 BFM 推理。

使用时将 `config/deploy.yaml` 的 `scalebfm.enabled` 改为 `true`，并设置 `root` 为本机外部 ScaleBFM 仓库目录。该目录需包含 `ScaleBridge`、`ScaleTrack` 及 `artifacts/scalebfm/humanoid_transformer_m` 的 checkpoint、metadata 和 mode table；模型不随本部署仓库分发。`checkpoint: null` 和 `motion: null` 使用适配器的官方 M 模型与 squat 动作默认路径。

`motion` 可指定 50 Hz、匹配 checkpoint 关节/身体顺序的参考 `.npz`；`motion_globs` 是动作列表。真机保持 `reference_forcing: true`，因为 LowState 没有根部定位。启用时需要匹配的 CUDA/PyTorch 环境。本分支用官方导出器定义加载原始 PyTorch 权重，不依赖特定 GPU 编译的 TensorRT 引擎；已在 Python 3.8 / PyTorch 2.3.1 / RTX 4070 Ti SUPER 验证。

```bash
# 固定官方源码版本，并校验 M 模型、metadata、mode table 的 SHA-256
python tools/setup_scalebfm.py
pip install -r environment/scalebfm-requirements.txt
```

官方权重来自 [Hugging Face](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main)。默认外部安装目录为部署仓库旁的 `ScaleBFM`，请使配置中的 `root` 与下载脚本输出一致。

```bash
# 已配置外部模型后，离线检查；不连接 DDS
python tools/check_scalebfm_onboard.py --steps 100
```

## 操作与真机入口

| 操作 | 仿真 / 真机 |
|---|---|
| START | FixedPose |
| A | 当前选择的 AMP |
| B | LocoMode |
| Y | 当前选择的 SMP，可抢占其他策略 |
| R1 | ScaleBFM 启用时，从 FixedPose/Loco/AMP 播放选中动作 |
| 左 / 右 | ScaleBFM 动作选择 |
| SELECT | 阻尼；真机也支持 F1 |

ScaleBFM 关闭时 R1 不会进入该策略。真机阻尼指令优先级最高。仿真 reset 操作不能用于真机。现有两个入口沿用 ScaleBFM 分支的 FreeKick 禁用行为。

真机通信与启动拓扑见 [机载说明](onboard/docs/README.md) 和 [bridge 说明](bridge/README.md)；`tools/start_tmux_layout.sh` 提供多窗格启动命令。模型验证不等于真机运动验证。

## 验证和来源

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/test_scalebfm_real.py
```

测试覆盖策略模型接口与 AMP 训练轨迹一致性、SMP 力矩限制、日志、场景、ScaleBFM 开关及 FSM 切换。本机已完成官方 M 模型的 100 步 GPU 推理与 CUDA Graph 一致性检查，以及 squat 的 MuJoCo 闭环和策略切换测试；没有连接真机。结果见 [本次验证记录](docs/validation/lyd_deploy_20261025.json)。

- [AMP guidance 模型来源和评估](docs/amp_guidance_60000_deployment_zh.md)
- [AMP single 模型来源和评估](docs/amp_stratified_single_60000_deployment_zh.md)
- [日志格式](docs/logging.md)
- 三个 SMP 模型的来源与哈希：`policy/smp_recovery/model/*.manifest.json`
