# PHP Parkour：统一 MuJoCo / FSM 部署

分支：`lyd-php-parkour`。PHP 位于 `policy/php_parkour/`，与 AMP、SMP、ScaleBFM、robonaldo 并列。使用相同的 `StateAndCmd`、`PolicyOutput`、FSM 和 500 Hz PD / 50 Hz 策略循环。深度图由同一个 MuJoCo 进程采集，不再运行 `deploy_php/` 或 Holosoma 双进程启动器。

## 模型与配置

外部来源为 `~/workspace/php_parkour`；模型使用其已下载的发布版 D435i student：

```yaml
# config/deploy.yaml
php:
  enabled: true
  root: ~/workspace/php_parkour
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

# 复用 PHP 仓库自带障碍场（启动保持 Passive，不自动执行 START）
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php

# 同一套控制器，切换为平地
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php scene.name=none

# 使用该仓库中另一个 OBJ（相对 php.root，或填写绝对路径）
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php \
  scene.php_terrain.obj_path=/absolute/path/to/course.obj
```

默认资源直接来自 `php.root/thirdparty/holosoma/src/holosoma/holosoma/data/terrains/terrain.obj`，对应外部部署脚本的 `terrain-load-step`，包含地板和 12 个独立障碍块。它是训练仓库所带的部署/评估障碍场，不是对全部随机训练地形的重建。资源保持外置，无需复制到本仓库。

加载器按连通组件分别生成碰撞几何，使用原始顶点、面、位置和朝向；移除原场景重叠的平面，避免地面闪烁及重复接触。自定义 OBJ 需由闭合凸组件组成；遇到凹组件会报错，不能将整个凹场景用一个凸包代替。依赖 `trimesh==4.11.3`（已写入 requirements.txt，当前环境已安装）。

## 按键切换

仍需连接原有部署手柄。窗口 `P` 仅是进入 PHP 的快捷键，不是完整的纯键盘驾驶模式。

| 操作 | 效果 |
|---|---|
| `L1+A`（两键先后顺序均可）或 MuJoCo 窗口 `P` | 从 Passive / FixedPose / Loco / AMP / ScaleBFM 显式进入 PHP |
| 左摇杆 | 前后、侧向和斜向离散方向命令；回中为 stand |
| `R3` 释放 | PHP 高低速切换；每次进入默认低速 |
| `A` 释放 | PHP → AMP |
| `B` 释放 | PHP → Loco |
| `START` 释放 | PHP → FixedPose |
| `Y` 按下（不按 L1） | 立即抢占到 SMP recovery |
| `L2` 释放 | 优先转阻尼 |
| `R1` 释放 | 启用 ScaleBFM 时沿用原有切换流程 |
| 窗口 `R` | 配置初态重置并切到 Passive |
| 窗口 `Backspace` | 站立物理复位，保留当前控制器及历史/进度/速度档位 |
| `L1+START` | 站立物理复位并切到 FixedPose |

L1+A 的 A 释放不会误触 AMP，即使先松开 L1。L3 仍是 ghost 开关。Passive 可以通过 L1+A/P 显式进入 PHP，不会自动进入 FixedPose。缺失、非有限或超过 150 ms 的深度输入会拒绝进入 PHP，运行中发生则输出阻尼并退出。

## 实现与验证范围

- 相机：torso_link，位置 `[0.01, 0.01, 0.44]`，与官方 D435i preset 相同朝向及视场角，原始深度 `60×106`，近裁剪面为官方配置的 0.1 m。
- 处理：裁剪 `[2:,4:-4]`，0.3–3 m 截断，带抗锯齿 bicubic resize 到 `58×87`，按官方远端/空像素规则归一化。MuJoCo 深度已经是从上到下，无须上下翻转。
- student 输入：`actions(29), base_ang_vel(3), dof_pos(29), dof_vel(29), torso gravity(3), direction(15), depth latent(32)`，共 140 维；输出按名称映射回 29 关节。
- 已验证实际 EGL 深度渲染、ONNX 推理、关节映射、FSM 切换、深度超时、组合键释放；平地和外部 PHP 障碍场分别完成 10 秒闭环（前 2 秒 stand，后 8 秒前进），未触发跌倒判据（pelvis 高度 <0.3 m）。这只是有限时长的仿真验证。
- 本次接入范围是 MuJoCo。实机和 C++ bridge 尚未接入 PHP 深度传感器，因此不启用 PHP；没有将仿真验证描述成实机验证。

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -p test_php_policy.py
MUJOCO_GL=egl python tools/smoke_php_mujoco.py
```

旧启动器报错原因：重复传入 terrain 子命令导致 Holosoma 仿真未启动，继而无法创建深度共享内存。现已移除该启动链路，`lyd-deploy` 分支保留原有统一部署，PHP 开发独立在本分支。

## 跟随视角与进入提示

默认恢复自由视角，不跟随 G1；鼠标仍可旋转、缩放。需要跟随时显式启用：

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php camera.follow=true
```

`[PHP] Cannot enter: ...` 表示切换被拒绝，不是 PHP 已经运行。提示会明确说明未启用、当前 FSM 状态不允许、深度缺失、形状/数值异常或深度超时。Passive 可直接按 L1+A/P；若为深度错误，查看前面的 `[PHP camera]` 日志。

Backspace 保留当前策略及内部状态，只复位物理姿态/速度并刷新观测。轨迹策略会从原进度继续；需要重新初始化控制器时使用 L1+START 后重新选择策略。

## 实时深度窗口

PHP 启用时默认打开 `PHP depth: raw camera / policy input` 窗口，即使当前是 Passive 也持续更新。上图是原始 106×60 米制深度，下图是实际输入 backbone 的 87×58 归一化图；近黄远蓝，上方显示当前策略、采集延迟和原始深度范围。显示运行于独立进程，不与 MuJoCo 共享 SDL 事件循环或图形上下文。远处或无命中的像素显示为蓝色，不能仅凭颜色判断相机故障；采集延迟持续增长才表示数据没有更新。关闭该窗口只停止预览，不关闭 MuJoCo 或 PHP 推理。

不需要窗口时加 `php_depth_preview=false`。成功切换会打印 `[Input] CMD_PHP_PARKOUR` 和 `[PHP] Entering from ...; depth ready.`，窗口状态同步变为 `SKILL_PHP_PARKOUR`。未出现 Input 日志时检查手柄按键编号，可用 `JOYSTICK_L1` / `JOYSTICK_A` 环境变量调整。
