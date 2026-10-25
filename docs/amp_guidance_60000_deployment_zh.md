# AMP 分层恢复策略部署（60k）

基于 `lyd/scalebfm-table-config-balanced4` 的 `4a6e822`，分支
`lyd-amp-stratified-guidance-deploy`。策略是 AMP 训练中的第五组：分层 reset、
双判别器、额外恢复引导奖励，checkpoint 为 `model_60000.pt`。
部署只运行 actor；判别器、奖励和训练 reset 数据不参与部署推理。

## 训练来源与发布记录

- 训练仓库：`lu-yidan/AMP_mjlab`，工作分支 `lyd-multi-discriminator-amp`。
- 训练分支基线：`6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`。
- 截至 2026-10-18，本轮多判别器、分层 reset 和恢复奖励实现仍是该训练工作分支上的未提交改动；上述基线提交本身不包含这些新增功能，不能仅 checkout 基线来复现本模型。
- 训练任务：`Unitree-G1-MultiAMP-Stratified-Guidance-Flat`，4096 个环境 × 24 steps，计划 100k iterations，本次发布取 60k checkpoint。
- W&B：[wwwrf04a](https://forge.coreweave.com/wandb/tabletennis/lyd-multi-amp/runs/wwwrf04a)。
- Checkpoint：`logs/rsl_rl/g1_multi_amp_stratified_guidance/2026-10-17_12-54-34_stratified-dual-guidance-4096x24-100k/model_60000.pt`。
- ONNX 首次发布提交：部署仓库 `496310913d8d7658d3613a99965f1d3878979420`，已推送到 `lyd-amp-stratified-guidance-deploy`，模型二进制随 Git 提交分发。
- 模型路径：`policy/amp/model/amp_stratified_guidance_60000.onnx`。
- ONNX SHA-256：`dc28ed74ae86ab197ab18dd43f49dc90f812971e40c19e29fffcecaad7e5e97c`。

导出时关键训练源文件及运行时观测补丁的 SHA-256 已记录在同目录 manifest 的
`source_sha256` 字段中；这些校验值用于核对导出来源，不代替完整训练代码的版本归档。

## 选择模型

编辑 `config/deploy.yaml` 后重启：

```yaml
amp:
  profile: stratified_guidance_60000
```

其他选择：`amp_0309_1`（默认）、`stratified_single_60000`。

沿用原有 AMP 模式和进入方式。普通 MuJoCo 部署使用 A 键进入 AMP。
无需改变 FSM、控制频率、观测设置、关节映射或 PD 参数。策略选择与哈希检查由 `Amp.py` 完成。
`safe_tilt_threshold` 是现有未使用字段；当前代码不会因倾斜自动退出 AMP，
因此可以持续执行恢复。原有手动 PASSIVE、START、LOCO 切换继续有效。

## 原始 actor 并非直接兼容，转换已封装在 ONNX 内

| 项目 | 训练端 | 现有 AMP 部署接口 | 新模型的处理 |
|---|---|---|---|
| 每帧观测 | 96 维 | 96 维 | 相同内容 |
| 帧内顺序 | 角速度、投影重力、速度命令、相对关节位置、关节速度、上一动作 | 相同 | 保留 |
| 历史 | 4 帧，旧到新，启动时重复首帧 | 相同 | 保留 |
| 角速度／重力 | pelvis 坐标系 | pelvis IMU／MuJoCo 浮动基座 | 相同 |
| 关节顺序 | 电机顺序 | `joint2motor_idx` 定义的左右交错顺序 | 对每帧关节位置、速度、上一动作重排 |
| 输出顺序 | 电机顺序 | 左右交错顺序 | 重排为旧接口 |
| 动作缩放 | 各关节约 0.2903～0.5475 | 所有关节 0.25 | 输出乘以 `训练缩放 / 0.25` |
| 上一动作 | 训练 actor 的原始输出 | 上一次 ONNX 输出 | 输入端逆变换，避免反馈单位错误 |
| 默认关节位置 | 弯膝姿态 | 相同 | 无需偏移补偿 |
| 观测归一化 | checkpoint 中统计量，epsilon=0.01 | 模型内处理 | 已嵌入 ONNX |
| 控制周期 | 0.02 s | 0.02 s | 相同 |
| PD 参数 | 电机组参数 | 相同参数的舍入值 | 保留 YAML，误差小于 0.1% |

因此，只替换模型路径的是**本仓库内带适配的新 ONNX**，不能直接换成训练工具导出的
未经适配的 actor。以后导入的新 checkpoint 也需经过同样的导出和验证。

模型输入 `obs: float32[1,384]`，输出 `actions: float32[1,29]`。
训练顺序、部署顺序、完整映射、动作尺度、checkpoint/ONNX SHA-256 和验证结果均记录于
`policy/amp/model/amp_stratified_guidance_60000.manifest.json`。

## 验证结果

1. 72 个环境、151 个连续控制步骤，包含自然／程序生成的四方向低姿态和站立状态。
   独立从物理状态重建部署观测，并检查全部历史帧。站立环境还覆盖非零前进、侧向、转向命令。
2. 144 个抽样观测对照训练 runner：最大观测误差 `2.8610e-6`，最大 ONNX 动作误差
   `5.7221e-6`，最终关节位置目标最大误差 `1.4306e-6 rad`。
   测试轨迹中输出幅度最大约 19.50，未触发旧接口的 ±100 裁剪。
3. 用真实 `Amp.py.run()` 重放上述观测，验证电机顺序下的最终关节目标；同时验证新旧模型
   都可加载、历史初始化和滚动，以及恢复状态／手动退出。5 项 unittest 通过。
4. 部署端 `g1_description/scene.xml`、原 PD 和力矩上限、500 Hz 物理／50 Hz 控制，
   每个样例连续 20 s，无重置和外力帮助：

| 固定样例 | 连续站稳 1 s 的首次时刻 | 最后 5 s 站立比例 |
|---|---:|---:|
| 仰卧 | 3.52 s | 100% |
| 俯卧 | 2.12 s | 100% |
| 左侧卧 | 2.36 s | 100% |
| 右侧卧 | 2.12 s | 100% |
| 站立至变速行走 | 1.00 s | 100% |

四个恢复样例取验证 bank 各程序化方向的第一帧，不按成功结果挑选。
行走命令每 5 s 为 0、0.5、1、2 m/s，保留旧接口的命令平滑。
对应速度平均绝对误差约 0.014、0.401、0.134、0.669 m/s；站立成功不等于速度精确跟踪。
左侧卧初始碰撞穿透约 1.16 mm，其余样例初始无穿透；未修饰这些初始姿态。
详细结果：`docs/validation/amp_guidance_60000_scene.json`。

这些是五个固定名义条件样例，不是恢复成功率的统计评测。训练端还有力矩-转速／摩擦模型，
部署端使用原有 PD 和力矩裁剪，动力学并非完全相同。未执行真机验证。

## 复现

在部署环境中（需要 numpy、PyYAML、onnxruntime、mujoco）：

```bash
python -m unittest discover -s tests -p test_amp_guidance_deployment.py -v
python tools/evaluation/evaluate_amp_deployment.py --output outputs/amp_guidance_60000/scene_report.json
```

独立测试旧模型可使用 `--model amp_0309_1.onnx`；这个测试参数不会改写 AMP 配置。

在拥有训练 checkpoint 和验证 bank 的 AMP 训练仓库及其 Python 环境中重新导出：

```bash
PYTHONPATH=runtime:. python /path/to/RoboMimic_Deploy/tools/export/export_amp_guidance.py \
  --deploy-root /path/to/RoboMimic_Deploy \
  --checkpoint logs/rsl_rl/g1_multi_amp_stratified_guidance/2026-10-17_12-54-34_stratified-dual-guidance-4096x24-100k/model_60000.pt \
  --bank records/eval60k/validation.npz \
  --output /path/to/RoboMimic_Deploy/policy/amp/model/amp_stratified_guidance_60000.onnx
```

导出脚本会验证现有配置、关节映射、默认姿态和 PD 参数，生成 ONNX、manifest 和回归样本。
它依赖训练仓库及其 mjlab 历史观测补丁；普通部署不依赖这些训练组件。

## 2026-10-18 补充：速度跟随限制

原先的五场景检查确认恢复和站立稳定，未覆盖纯侧移、原地旋转或全部速度阶跃。
不能把这些检查理解为完整速度跟随验证。后续在原部署 MuJoCo 场景补测发现：

- 原 AMP 配置侧向命令只有 ±0.01 m/s，因此摇杆打满也几乎不侧走。
- 绕过限幅和平滑，直接输入 ±0.5 m/s 侧移或 ±1 rad/s 原地旋转，发布模型依然几乎静止。
- 指令 -0.5 m/s 后退时实际约 -0.008 m/s；+1 m/s 前进时实际约 +0.97 m/s。
- 命令平滑的 alpha=0.6，在 50 Hz 下约 0.1 s 达到阶跃目标的 90%，不能解释持续不响应。
- 训练最终课程范围为 vx=[-1,2]、vy=[-1,1]、wz=[-1,1]；部署 fast 模式的 vx=[-1.5,3]、wz=[-1.57,1.57] 超出该范围。

上述数值来自每种命令一个固定初态：先站立 2 s，再保持命令 10 s，统计最后 5 s；
同时测试原摇杆处理和直接输入命令，不修改配置。机器人采用 torso yaw 坐标系的平面速度及 torso 世界系 z 角速度。
原始数据：`docs/validation/amp_guidance_60000_commands.json`。复现：

```bash
python tools/evaluation/evaluate_amp_command_tracking.py --output outputs/amp_guidance_60000/commands.json
```

训练仓库还对五组 60k 模型进行了统一原生接口测试（每种命令 8 个相同初态扰动）：
原始单判别器也几乎不响应纯侧移和原地旋转，说明这不是新增恢复奖励独有的问题。
全部模型正常站立／行走的末 5 s 恢复引导高度门控均为零，但不能排除恢复奖励在训练阶段
对共享 actor 的间接影响。引导奖励组直接从静止接收 2 m/s 指令时可停在原地，
与之前渐进变速的结果不同，存在明显的指令转换路径依赖。

代码层面的候选原因包括：角速度奖励 std=3.14，使 1 rad/s 指令下静止仍能得到约 0.904；
纯侧移／原地旋转没有作为独立模式采样；双判别器的行走条件只使用平移速率大小，
不区分前后、侧向或旋转。没有多随机种子的受控消融，不能把所有差异归因于恢复奖励或判别器数量。
