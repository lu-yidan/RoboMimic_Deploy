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

使用独立的 `robomimic-mj310` 环境（Python 3.10 / 官方 MuJoCo 3.10.0），安装见[迁移说明](native_depth_overlay.md)。无需切换 `hsmujoco` / `hsinference`，也无需启动深度共享内存服务。`php.enabled: false` 可关闭 PHP 模型加载和相机创建，继续使用其他策略。支持 `LYD_DEPLOY_CONFIG` 指定完整配置。

`policy/php_parkour/config/php.yaml` 保存关节名称、默认姿态、增益和模型校验值。其增益与默认姿态对应 PHP 当前 Holosoma inference 的 distillation preset；观测按训练时的实际顺序排列，不使用 ONNX 元数据中具有误导性的声明顺序。

## 启动与场景

```bash
conda activate robomimic-mj310
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

加载器按连通组件分别生成碰撞几何；长方体按原 OBJ 的位置、朝向和尺寸转换为 MuJoCo box（允许 OBJ 坐标舍入带来的 2 微米误差），其他闭合凸组件保留网格。这样避免 MuJoCo 3.2 在大网格地板上倒地恢复时出现穿透和接触失稳；移除原场景重叠的平面，避免地面闪烁及重复接触。自定义 OBJ 需由闭合凸组件组成；遇到凹组件会报错，不能将整个凹场景用一个凸包代替。依赖 `trimesh==4.11.3`（已写入 requirements.txt，当前环境已安装）。

## 按键切换

仍需连接原有部署手柄。窗口 `P` 仅是进入 PHP 的快捷键，不是完整的纯键盘驾驶模式。

| 操作 | 效果 |
|---|---|
| `L1+A`（两键先后顺序均可）或 MuJoCo 窗口 `P` | 从 Passive / FixedPose / Loco / AMP / ScaleBFM / SMP 显式进入 PHP |
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
- 实机深度与 C++ bridge 接入已实现，默认关闭，目前只完成离线验证。安装、本体 tmux 启动、外参和断流处理见 [实机部署](php_real_bridge_zh.md)。

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -p test_php_policy.py
MUJOCO_GL=egl python tools/validation/smoke_php_mujoco.py
```

旧启动器报错原因：重复传入 terrain 子命令导致 Holosoma 仿真未启动，继而无法创建深度共享内存。现已移除该启动链路，`lyd-deploy` 分支保留原有统一部署，PHP 开发独立在本分支。

## 跟随视角与进入提示

默认恢复自由视角，不跟随 G1；鼠标仍可旋转、缩放。需要跟随时显式启用：

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php camera.follow=true
```

`[PHP] Cannot enter: ...` 表示切换被拒绝，不是 PHP 已经运行。提示会明确说明未启用、当前 FSM 状态不允许、深度缺失、形状/数值异常或深度超时。Passive 可直接按 L1+A/P；若为深度错误，查看前面的 `[PHP camera]` 日志。

Backspace 保留当前策略及内部状态，只复位物理姿态/速度并刷新观测。轨迹策略会从原进度继续；需要重新初始化控制器时使用 L1+START 后重新选择策略。

## MuJoCo 右上角深度图

参考 `wmp_deploy` 的同窗布局，在 MuJoCo 三维视窗右上角显示实际输入 backbone 的 87×58 图像，近黄远蓝，并显示采集延迟；深度图标题显示当前 FSM 状态。Passive 下也更新图像。没有额外的深度窗口或显示进程，模型输入本身不受着色影响。

窗口缩放后深度图仍固定在右上角；左键拖动旋转、右键拖动平移、滚轮缩放，Shift 可切换水平/垂直操作。P/R/Backspace 与手柄逻辑不变；关闭窗口退出仿真。

始终使用原生 MuJoCo viewer，保留参数侧栏与 Rendering → Camera → Tracking。先双击机器人选中 body，再选择 Tracking。深度图通过原生 viewer 扩展绘制，不会重设相机；`php_depth_preview=false` 仅关闭深度图。使用官方 MuJoCo 3.10.0 的公开图像接口，不需要源码补丁，详见[安装与迁移说明](native_depth_overlay.md)。

已删除输入层的 `[Input] CMD_*` 日志。Passive 下 A 不允许直接切 AMP，按 START 进入 FixedPose 后再按 A；没有实际切换时不打印命令。PHP 切换成功时仍显示 `[PHP] Entering from ...; depth ready.`，当前状态也随之更新。深度无效等真正错误仍保留具体诊断。

支持 **Y → SMP → L1+A → PHP** 双向切换，无需先经过 START/FixedPose；返回 PHP 时仍要求有效的新深度图。

## Passive 倒地后按 Y 的稳定性修复

曾复现三角网格地面在 SMP 起身时产生巨大速度、QACC 警告或穿地。修改接触软硬参数不足以解决问题；源资源中的长方体改用等形 box 碰撞后，6 个切换时刻（0.02/1/3/5/10/11 秒）各完成 30 秒仿真，未出现警告、自动复位或穿地。仍保持 Passive 启动和显式 Y 切换。

```bash
python tools/validation/validate_php_recovery.py --out /tmp/php_smp_switch.json
```

该检查调用实际 FSM、Passive 和 SMP 模型。没有用 MuJoCo 自动复位来掩盖失败，也不等价于真机验证。
