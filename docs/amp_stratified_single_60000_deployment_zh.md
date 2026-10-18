# 分层 reset＋单判别器 AMP（60k）

这是训练对照的第三组，未添加第五组的 recovery guidance 奖励。
仍保留原基线自身的根部高度等任务奖励，并非完全没有恢复相关奖励。
它与 `strat_single.mp4` 中使用的 `model_60000.pt` 对应。

## 来源

- 训练仓库／工作分支：`lu-yidan/AMP_mjlab` / `lyd-multi-discriminator-amp`，基线 `6c7a294`。
- 截至 2026-10-18，训练功能改动仍未提交；基线提交本身不包含分层 reset。关键导出源文件哈希记录在模型 manifest 中。
- 任务：`Unitree-G1-AMP-Stratified-Flat`。
- W&B：[xoyhv8th](https://forge.coreweave.com/wandb/tabletennis/lyd-multi-amp/runs/xoyhv8th)。
- 4096 环境 × 24 steps，计划 100k iterations；本次取 60k checkpoint。
- Checkpoint：`logs/rsl_rl/g1_amp_locomotion_stratified/2026-10-17_12-46-19_stratified-single-4096x24-100k/model_60000.pt`。
- 部署分支：`lyd-amp-stratified-guidance-deploy`。

## 切换

在 `policy/amp/config/amp.yaml` 只保留一条生效的 `model_path`，然后重启：

```yaml
model_path: "amp_stratified_single_60000.onnx"
```

原模型为 `amp_0309_1.onnx`；双判别器＋恢复奖励为 `amp_stratified_guidance_60000.onnx`。
配置默认仍是原模型。三者使用原有 AMP 模式；`Amp.py`、其他观测配置和 PD 参数不改。
新单判别器 ONNX 同样内置观测归一化、关节顺序转换、动作尺度转换和上一动作逆转换。
输入 `obs[1,384]`，输出 `actions[1,29]`；不要再次修改 `action_scale` 或关节映射。
完整接口解释见 [共同部署说明](amp_guidance_60000_deployment_zh.md)。

## 验证

- 72 个环境、151 步的训练端连续轨迹，144 个回归样本；包含恢复状态和非零速度命令。
- 最大历史观测误差 `2.8611e-6`，最大 ONNX 动作误差 `5.2453e-6`，最终关节目标误差 `1.3114e-6 rad`。
- 两份新模型共 10 项接口、历史、模型哈希、真实 Amp 控制器目标和手动退出测试全部通过。
- 部署 `g1_description/scene.xml`，原 PD／力矩限制，四个相同倒地初态和变速行走各 20 s，无重置：

| 样例 | 首次连续站稳 1 s 时刻 | 最后 5 s 站立比例 |
|---|---:|---:|
| 仰卧 | 2.06 s | 100% |
| 俯卧 | 1.88 s | 100% |
| 左侧卧 | 2.26 s | 100% |
| 右侧卧 | 1.92 s | 100% |
| 站立至变速行走 | 1.00 s | 100% |

部分恢复后的最终根部高度约 0.67～0.69 m，高于判定阈值但姿态较低。
行走 0／0.5／1／2 m/s 各 5 s，速度 MAE 约 0.014／0.313／0.145／0.670 m/s。
这些是固定样例，不代表统计成功率，也不证明完整速度跟随已达标。
之前原生策略补测表明该单判别器的纯侧移与原地转向也较弱，旧部署配置侧向还限制为 ±0.01 m/s。
未进行真机验证。

详细结果：`docs/validation/amp_stratified_single_60000_scene.json`。
模型、checkpoint、回归样本和源文件的校验值：`policy/amp/model/amp_stratified_single_60000.manifest.json`。

## 复现

```bash
python -m unittest discover -s tests -p test_amp_guidance_deployment.py -v
python tools/evaluate_amp_deployment.py --model amp_stratified_single_60000.onnx \
  --output outputs/amp_stratified_single_60000/scene_report.json
```

在 AMP 训练仓库及其运行环境中导出时，使用 `tools/export_amp_guidance.py`，指定
`--task Unitree-G1-AMP-Stratified-Flat`，以及上面的 checkpoint、验证 bank 和新 ONNX 输出路径。
默认 task 仍是双判别器＋引导奖励；单判别器必须显式选择正确 task。
