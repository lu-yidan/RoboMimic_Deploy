# PHP Parkour：统一 MuJoCo / FSM 部署

分支：`lyd-php-parkour`。PHP 位于 `policy/php_parkour/`，与 AMP、SMP、ScaleBFM、robonaldo 并列。使用相同的 `StateAndCmd`、`PolicyOutput`、FSM 和 500 Hz PD / 50 Hz 策略循环。深度图由同一个 MuJoCo 进程采集，不再运行 `deploy_php/` 或 Holosoma 双进程启动器。

## 模型与配置

外部来源为 `~/workspace/php_parkour`；模型使用其已下载的发布版 D435i student：

```yaml
# config/deploy.yaml
php:
  enabled: true
  model_dir: ~/.cache/php-parkour/student
```

模型目录需有 `depth_backbone.onnx` 和 `student.onnx`。本分支校验发布模型 SHA-256，模型不进入 Git。缺失时用 PHP 仓库提供的下载器：

```bash
cd ~/workspace/php_parkour
python scripts/download_assets.py student
```

仍使用本仓库 `robomimic` 环境，无需切换 `hsmujoco` / `hsinference`，也无需启动深度共享内存服务。`php.enabled: false` 可关闭 PHP 模型加载和相机创建，继续使用其他策略。支持 `LYD_DEPLOY_CONFIG` 指定完整配置。

`policy/php_parkour/config/php.yaml` 保存关节名称、默认姿态、增益和模型校验值。其增益与默认姿态对应 PHP 当前 Holosoma inference 的 distillation preset；观测按训练时的实际顺序排列，不使用 ONNX 元数据中具有误导性的声明顺序。

## 启动与场景

```bash
conda activate robomimic
cd ~/workspace/RoboMimic_Deploy
git switch lyd-php-parkour

# 台阶场景（启动后为 FixedPose，等姿态稳定再切 PHP）
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php

# 同一套控制器，切换为平地
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php scene.name=none

# 调整台阶：4 级，每级高 0.12 m，长 0.6 m，宽 2 m，起点 x=1.5 m
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php \
  scene.steps.count=4 scene.steps.height=0.12 scene.steps.length=0.6 \
  scene.steps.width=2.0 scene.steps.start_x=1.5
```

台阶是本仓库可配置的简单静态阶梯，不等同于外部 PHP 的完整训练地形。也可启动已有 `mujoco`、`mujoco_table` 等场景后切 PHP；模型在这些特殊场景的行为并不因接口兼容而得到保证。

## 按键切换

仍需连接原有部署手柄。窗口 `P` 仅是进入 PHP 的快捷键，不是完整的纯键盘驾驶模式。

| 操作 | 效果 |
|---|---|
| `L1+A`（按住 L1 再按 A）或 MuJoCo 窗口 `P` | 从 FixedPose / Loco / AMP / ScaleBFM 进入 PHP |
| 左摇杆 | 前后、侧向和斜向离散方向命令；回中为 stand |
| `R3` 释放 | PHP 高低速切换；每次进入默认低速 |
| `A` 释放 | PHP → AMP |
| `B` 释放 | PHP → Loco |
| `START` 释放 | PHP → FixedPose |
| `Y` 按下（不按 L1） | 立即抢占到 SMP recovery |
| `L2` 释放 | 优先转阻尼 |
| `R1` 释放 | 启用 ScaleBFM 时沿用原有切换流程 |
| 窗口 `R` / `Backspace` | 原有仿真重置 / 站立重置 |

L1+A 的 A 释放不会误触 AMP，即使先松开 L1。L3 仍是 ghost 开关。被动状态不能直接进入 PHP：先 START 或 Backspace 回到站立状态。缺失、非有限或超过 150 ms 的深度输入会拒绝进入 PHP，运行中发生则输出阻尼并退出。

## 实现与验证范围

- 相机：torso_link，位置 `[0.01, 0.01, 0.44]`，与官方 D435i preset 相同朝向及视场角，原始深度 `60×106`。
- 处理：裁剪 `[2:,4:-4]`，0.3–3 m 截断，带抗锯齿 bicubic resize 到 `58×87`，按官方远端/空像素规则归一化。MuJoCo 深度已经是从上到下，无须上下翻转。
- student 输入：`actions(29), base_ang_vel(3), dof_pos(29), dof_vel(29), torso gravity(3), direction(15), depth latent(32)`，共 140 维；输出按名称映射回 29 关节。
- 已验证实际 EGL 深度渲染、ONNX 推理、关节映射、FSM 切换、深度超时、组合键释放；平地和 4×0.12 m 台阶分别完成 10 秒闭环（前 2 秒 stand，后 8 秒前进），未触发跌倒判据（pelvis 高度 <0.3 m）。这只是有限时长的仿真验证。
- 本次接入范围是 MuJoCo。实机和 C++ bridge 尚未接入 PHP 深度传感器，因此不启用 PHP；没有将仿真验证描述成实机验证。

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -p test_php_policy.py
MUJOCO_GL=egl python tools/smoke_php_mujoco.py
```

旧启动器报错原因：重复传入 terrain 子命令导致 Holosoma 仿真未启动，继而无法创建深度共享内存。现已移除该启动链路，`lyd-deploy` 分支保留原有统一部署，PHP 开发独立在本分支。
