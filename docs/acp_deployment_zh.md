# ACP G1：平地到点基线

起点为 245 上干净的 `lyd-php-parkour@8ccde58`。部署分支是
`deploy/acp-g1`，远端工作区 `/root/RoboMimic_Deploy_acp`；原 PHP
工作区保留。训练分支 `train/acp-g1` 从 `acp@6a483e6` 建立，工作区
`/root/InstinctMJ_acp_g1`。

当前代码具备部署接口和离线验证工具。没有随代码提供训练权重；只有通过
原生/ONNX 对拍及平地到点、转向、停车评估的 Stage 3 bundle 才是有效基线。
合成 ONNX 单元测试不能证明策略会走路。

## 训练数据与环境

本轮按用户指定使用
[PHP training-motion-examples-v1](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1)。
首版只转换 locomotion 包中 terrain 数组为空的平地动作。
原始 50 Hz、骨盆根、具名关节转换为 ACP 的 23 关节，六腕焊死；接触标签
必须在 ACP 碰撞模型上重新生成。转换器及 provenance 在训练分支的
`tools/acp_g1/`，原始源文件哈希、帧范围和坐标变换随数据保存。

隔离环境使用 `/root/InstinctMJ_acp_g1/.venv`、官方 mjlab v1.6.0
`0fb8a681136be94ffc636a3dd423cabb97d91f10` 与训练仓 `uv.lock`。
训练缓存、安装日志和训练环境设置放在 `/root/autodl-tmp/acp-g1/`。
ACP runner 不支持 DDP：先用一张 5090 训练，另一张供评估。

顺序是 Stage 1 tracker、Stage 2 planner、Stage 3 goal reaching。
首版 Stage 3 使用独立的 `Instinct-ACP-Stage3-Flat-G1-v1`：只在平地训练，
起始航向覆盖整圈，站立奖励半径 0.2 m、航向容差 0.25 rad；目标指令的
归一化半径仍是 0.5 m。原 17 类地形任务保留。
每阶段保留 `model_*.pt` 与同迭代的 `acp_state_*.pt`，后者包含世界模型、
critic 和优化器续训状态。先验证冒烟、显存和续训，再进行长训练；阶段间
需要实际评估，不以保存了 checkpoint 作为收敛依据。

## 导出与逐帧重建

在匹配的训练环境运行部署仓的导出工具。需要显式提供六腕固定参数；
`fixed_joints.example.json` 是部署建议，未冒充训练出的腕部增益。

```bash
cd /root/RoboMimic_Deploy_acp
source /root/autodl-tmp/acp-g1/environment.sh
/root/InstinctMJ_acp_g1/.venv/bin/python tools/acp/export_bundle.py \
  --instinct-mj /root/InstinctMJ_acp_g1 \
  --checkpoint /absolute/path/to/stage3/model_ITER.pt \
  --dataset /root/autodl-tmp/acp-g1/data/php_flat \
  --task Instinct-ACP-Stage3-Flat-G1-Play-v1 \
  --fixed-joints tools/acp/fixed_joints.example.json \
  --output policy/acp/models
/root/InstinctMJ_acp_g1/.venv/bin/python tools/acp/replay_bundle.py policy/acp/models
```

`policy.onnx` 包含 planner、tracker 裁剪、观测归一化和 actor。
导出工具会把新版 Torch 拆出的外部权重合并回这个文件，使其 SHA256 同时
覆盖图和参数。运行时与重建工具拒绝仍依赖外部权重的 ONNX，并加载已经
校验过的模型字节；单独匹配图文件哈希不足以验证外部参数。
`manifest.json` 保存实际关节序、默认角、增益、力矩限值、目标半径、
源码/权重哈希。只有 checkpoint 的 `infos.stage == 3` 能证明 Stage 3；
Stage 2 不会被误标。`parity.npz` 保存原生及 ONNX 三输出、完整输入、
具名关节状态、骨盆状态、目标与独立射线命中数据。重建工具用这些原始量
复原历史、地形和目标，比较全部 939 维输入及三输出。

`--dataset` 也可通过 `ACP_DATASET` 指定；导出与训练必须使用同一数据与配置。

## MuJoCo

```bash
cd /root/RoboMimic_Deploy_acp
LYD_DEPLOY_CONFIG=config/deploy_acp.yaml python deploy_mujoco/deploy_mujoco.py \
  --config-name mujoco_acp
LYD_DEPLOY_CONFIG=config/deploy_acp.yaml python tools/acp/evaluate_mujoco.py
```

GUI 先 START 进入 FixedPose，等待两秒站姿渐变完成，再按 C 进入 ACP。
FixedPose 使用本次 bundle 的默认角和 PD 增益，入口同时检查实际关节是否
接近目标。策略 50 Hz、PD 200 Hz；力矩限值使用 bundle。射线只接受焊接到
世界的静态地形，跳过机器人本体；地图为骨盆 yaw 系、y 外层 x 内层的
17×11 点，每点 `(x,y,pelvis_z-hit_z-0.5)`。

无头部署评估分别对前方目标、侧方转向目标和原地停车完整运行 15 秒；
末 1 秒每帧须同时满足距离 <0.2 m、航向误差 <0.25 rad、平面速度
<0.15 m/s、骨盆高度 0.78±0.1 m、倾角 <0.7 rad 和有限值要求，
全程跌倒或推理失效均失败。原生 Stage 3 评估采用 episode 语义：
在 episode 内首次连续满足上述条件 1 秒即记成功。
地图或定位过期、不同步、
非有限值、倾斜/高度越界时同周期进入阻尼；重新有效不会自动重入。

## Bridge 只读验证

`acp.enabled` 与 `acp.real.enabled` 独立；默认配置不会启用 ACP 真机路径。
先在单独的部署配置中启用 real，并配置已验证的定位/地图生产进程。
两侧只读开关必须同时设置：

```bash
BRIDGE_READ_ONLY=1 /absolute/path/to/sdk2_bridge
LYD_DEPLOY_CONFIG=/absolute/path/to/acp-real-config.yaml \
  python bridge/python/deploy_policy.py --read-only
```

C++ 只读模式不创建 LowCmd publisher；Python 只读模式不创建 BridgeCmd
publisher，正常退出和异常退出也不发布退出电机命令。源 C++ 已在 x86_64、
官方 Unitree SDK2 `b7fec602` 下编译和链接通过；arm64 机器人目标仍需用对应 SDK 构建。

定位/地图由 `policy.acp.state_ipc.StateWriter` 原子写入，默认
`/dev/shm/robomimic-<uid>/acp-state.npz`。协议含 boot_id、frame_id、严格
递增 sequence、同一主机 monotonic 采样时间、骨盆世界位姿及完整网格。
平地可显式提供实测 `ground_height`，由实时骨盆高度生成网格。没有定位时
保持无效，不使用全零位姿。跨机器时钟需在生产进程中正确转换时间域。

`target_position` 和 `target_heading` 使用与定位相同的世界坐标系。
骨盆角速度直接使用经过校验的 pelvis IMU gyro；定位四元数必须表示骨盆，
需要在实机只读阶段核对其与 IMU 的轴和外参。桥接状态、定位和地图各自
检查年龄，默认 150 ms、同步偏差 40 ms；重复或停滞状态不能刷新有效期。

遥控器 START 站姿，L1+X 边沿请求 ACP，F1/SELECT 阻尼抢占。
定位/地图失效、非有限输出、bridge tick 停滞会退出；ACP 和其 FixedPose
阶段都按 manifest 限制计算出的 PD 力矩。它们不代替物理防护和实机验证。
启用现有 logging 后，退出时会额外保存 `.acp_trace.npz`，默认最多 3000 帧，
含 939 输入、三输出、最终 PD 目标与各采样时间。

## 离线检查

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_acp_contract.py tests/test_acp_state_ipc.py tests/test_acp_bridge.py \
  tests/test_scalebfm_bridge.py tests/test_unified_deployment.py tests/test_sim_reset_camera.py
```

需要部署 Python 环境中的 MuJoCo、ONNX/ORT、PyTorch、CycloneDDS 和 pytest。
测试不打开机器人接口。原 PHP 真机离线测试另需外置 PHP backbone/student
权重；缺少这些文件时该组测试不能完成，不能据此宣称全套回归通过。
