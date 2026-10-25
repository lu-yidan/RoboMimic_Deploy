# ScaleBFM 部署与动作数据

[返回部署手册](../README_zh.md#6-scalebfm-模型与动作)

本文说明 `lyd-deploy` 的 ScaleBFM 接口；通用安装、手柄和真机启动见主 README。模型切换仍需重启，动作库中已预加载的 motion 可通过左右选择、R1 播放。

## 1. 外部目录与来源

运行 `python tools/motion/setup_scalebfm.py`，默认在部署仓库旁创建：

```text
ScaleBFM/
├── ScaleBridge/                         # 参考动作、观测与环境代码
├── ScaleTrack/                          # 官方网络/导出器定义、训练机器人 XML
└── artifacts/scalebfm/humanoid_transformer_m/
    ├── model_22200.pt                   # 原始 M checkpoint，约 69.5 MiB
    ├── model_22200_tensorrt_metadata.json
    ├── mode_table.pt
    └── download_manifest.json
```

metadata 文件名沿用官方命名，不表示本分支使用 TensorRT。`policy/scalebfm/torch_agent.py` 读取官方导出器的网络及包装定义，严格加载 actor 权重；不执行训练脚本、命令行入口或 TensorRT 编译。源码版本和下载哈希由 setup 脚本固定，导出器定义还会在加载时核对哈希。

外部来源：[官方源码](https://github.com/zengweishuai/ScaleBFM)、[官方权重](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main)。不要将权重目录复制进本部署仓库。

## 2. 共享配置

配置位于 `config/deploy.yaml` 的 `scalebfm` 段：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `false` | 全局开关；仿真、真机和 bridge 共用 |
| `root` | `~/workspace/ScaleBFM` | 外部源码根目录，按本机下载位置调整 |
| `checkpoint` | `null` | 默认官方 M checkpoint；非空为另一个兼容 checkpoint 路径 |
| `motion` | `null` | 默认官方 squat；可指定自己的 NPZ |
| `reference_forcing` | `true` | 使用参考根位置构造观测；真机需保持开启 |
| `cuda_graph` | `true` | 使用 CUDA Graph，启动时预热/捕获 |
| `control_mode` | 未写出时为 `7` | whole-body 模式；可在 YAML 显式添加 |
| `motion_globs` | 见配置 | 动作文件 glob 列表，启动时收集 |

MuJoCo 场景 YAML/命令行仅能覆盖 `scalebfm.motion`，其余选项仍在共享配置中。场景 motion 为 null 时沿用共享选择。路径支持 `~`；motion 相对路径按部署仓库根目录解析。checkpoint 建议使用绝对路径，且必须匹配现有 M 型 metadata、mode table 和接口。

当前普通 `scene.xml` 的 squat 跟踪通过了闭环检查；箱体使用的 `g1_liao.xml` 不属于该验证范围。开启 ScaleBFM 不会自动进入该状态，仍需先 FixedPose 再 R1。

## 3. 参考动作格式

默认 squat 为 417 帧、50 Hz，约 8.34 秒，到结尾保持最后一帧。它是官方仓库自带动作，不是 DuoBox 实验数据。

自己的动作需包含以下字段。T 至少为 2；当前 M 模型有 29 个关节、30 个 body。实际数量和顺序以 metadata 为准。

| 字段 | 形状 | 单位/约定 |
|---|---|---|
| `fps` | 标量 | 必须为 50，不能只改标签而不重采样 |
| `joint_pos` | `[T,29]` | rad，metadata `joint_names` 顺序 |
| `joint_vel` | `[T,29]` | rad/s，同上 |
| `body_pos_w` | `[T,30,3]` | 世界系，m，metadata `body_names` 顺序 |
| `body_quat_w` | `[T,30,4]` | 归一化 wxyz 四元数 |
| `body_lin_vel_w` | `[T,30,3]` | 世界系，m/s |
| `body_ang_vel_w` | `[T,30,3]` | 世界系角速度，rad/s |
| `joint_names` / `body_names` | 可选字符串数组 | 提供时会校验完整顺序 |

适配器检查维度、有限值、50 Hz 和四元数范数。没有名称数组时，只能检查数值结构，无法自动证明排列正确；不要把 MuJoCo 深度优先关节顺序当成训练顺序。

常见准备流程：选定单人片段 → retarget 到 G1 → 重采样 50 Hz → 按 metadata 顺序做 FK 和速度计算 → 保存 NPZ。SMPL、原始 DuoBox、只有 root/dof 的 GMR qpos 都不是这个接口的直接输入。官方打包脚本位于外部 `ScaleTrack/scripts/pretrain/data_process/package_motions.py`，其 IsaacLab 环境独立于轻量部署环境。

## 4. 动作库、预设与切换

`motion` 和 `motion_globs` 共同决定动作列表。左右只改变待播选择，不打断当前动作；R1 从第 0 帧开始所选参考，并按照当前机器人朝向/水平位置对齐。切换不会重置真实或仿真的物理状态，也不重新加载 checkpoint。

真机在控制开始前预加载动作；运行中只允许选择缓存文件。新增动作后重启。

三个 `duobox_*` YAML 仅提供路径预设：

- `duobox_straight`：`outputs/duobox_scalebfm/motions/straight_003_ma.npz`
- `duobox_guard`：`outputs/duobox_scalebfm/motions/guard_dodge_004_liu.npz`
- `duobox_combo`：`outputs/duobox_scalebfm/motions/moving_combo_027_ma.npz`

这些数据不随仓库分发。历史工作站生成的 batch15/comparison 输出不能当成新 clone 已经具备的数据。路径不存在时，提供实际 motion 或先准备数据。

## 5. 检查与进阶工具

先在共享配置中开启 ScaleBFM，并安装可选依赖：

```bash
pip install -r environment/scalebfm-requirements.txt
python tools/diagnostics/check_scalebfm_onboard.py --steps 100
python tools/validation/smoke_scalebfm_fsm.py
```

第一项不创建 DDS，检查真实 FSM 模型初始化、动作预加载、CUDA Graph/普通 PyTorch 输出一致性及静态状态推理时延。第二项运行实际 MuJoCo/PD/FSM，包括 10 秒 squat、倒地仿真重置、动作重入、AMP/Loco 往返切换、无效文件保护和 Passive 退出。

| 工具 | 用途与前提 |
|---|---|
| `tools/evaluation/evaluate_scalebfm_motion.py` | 单个兼容 NPZ 的 MuJoCo 评估；`--motion`、`--out` 必填，`--no-video` 关闭录像 |
| `tools/motion/prepare_duobox_scalebfm.py` | 单段 DuoBox/GMR 预处理；需要原始 pair、actor、外部 GMR 和 ScaleBFM 路径 |
| `tools/evaluation/batch_duobox_scalebfm.py` | 批量数据准备；需自行提供数据根目录、GMR 和 ScaleBFM 路径 |
| `tools/motion/reactive_stream_tracker.py` / `reactive_realtime.py` | 实验用增量参考管线；依赖外部生成器/GMR 和工作目录，不是普通部署启动入口 |
| `tools/validation/validate_scalebfm.py` | 保留的独立上游 ScaleBridge 验证脚本；直接实例化上游 agent，不走本分支的 PyTorch 适配器，需要另行匹配上游后端 |

使用进阶工具前查看 `--help`，显式提供本机路径；部分实验脚本仍保留旧工作站路径默认值。普通部署应优先使用已验证的 `deploy_mujoco.py`、`check_scalebfm_onboard.py` 和 `smoke_scalebfm_fsm.py`。

## 6. 适用范围

`reference_forcing=true` 不提供绝对位置跟踪证明，也不等于悬挂或支撑。机器人仍由 MuJoCo 物理或真实电机反馈驱动。仿真 Backspace/R 是测试便利功能，真机不能复用。当前验证不覆盖每个动作、每个初态、每种机器人资产或机载 GPU；具体结果见 [验证记录](validation/lyd_deploy_20261025.json)。
