# PHP Parkour 部署

本仓库通过 `deploy_php/deploy_php.py` 调用外部 `~/workspace/php_parkour` 的官方启动脚本。PHP 使用 Holosoma 的 D435i 深度仿真和推理环境；AMP、SMP、ScaleBFM 仍使用本仓库原来的部署入口。两套部署可以安装在同一机器上，但 PHP 尚未接入本仓库 FSM，不能通过 AMP/SMP 手柄按键热切换。这里提供的是 PHP **MuJoCo sim-to-sim** 入口，不是已经验证的实机部署。

## 配置与依赖

`config/deploy.yaml`：

```yaml
php:
  root: ~/workspace/php_parkour
  model_dir: ~/.cache/php-parkour/student
  scene: steps  # steps 或 flat
```

相对路径相对于 RoboMimic_Deploy 根目录；支持 `LYD_DEPLOY_CONFIG` 选择另一份完整配置。模型保留在外部，不提交到本仓库。模型目录必须包含 `depth_backbone.onnx` 和 `student.onnx`，不能用任意旧版 `policy.onnx` 替代。

PHP 外部仓库必须提供 `run_php_sim.sh`、`run_php_inference.sh` 和初始化完成的 `thirdparty/holosoma`。如果尚未初始化或下载模型：

```bash
cd ~/workspace/php_parkour
git submodule update --init thirdparty/holosoma
python scripts/download_assets.py student
```

按外部仓库 `wbt_training/DEPLOY.md` 的安装步骤准备 `~/.holosoma_deps/miniconda3` 下的 `hsmujoco` 与 `hsinference` 环境。启动器自动使用这两个环境；不向 `robomimic` 环境安装训练依赖。当前机器已有这两个环境及模型。

## 启动台阶或平地

以下命令从 RoboMimic_Deploy 根目录执行。先做离线检查（不连接机器人，也不创建相机或 DDS）：

```bash
conda activate robomimic
python deploy_php/deploy_php.py check
```

终端一：

```bash
# 默认台阶场景，或读取 php.scene
python deploy_php/deploy_php.py sim
# 平地场景，覆盖配置
python deploy_php/deploy_php.py sim --scene flat
```

等日志出现 `depth_img_shm`、`shape=(1, 1, 58, 87)` 后，在终端二启动：

```bash
conda activate robomimic
cd ~/workspace/RoboMimic_Deploy
python deploy_php/deploy_php.py policy
```

推理进程应报告已连接深度共享内存。必须先启动仿真，保留默认深度输入要求。官方仿真脚本启动时会重新创建 `/dev/shm/depth_img_shm`；同一机器只运行一套使用该名称的 PHP 深度仿真，不要在另一套正在运行时启动它。

MuJoCo 窗口按 `8` 降低吊架到脚接触地面，按 `9` 移除吊架；推理终端按 `]` 开始策略。`o` 阻尼，`i` 回到初始保持姿态。MuJoCo 窗口 `Backspace` 重置环境。

| 推理终端按键 | 命令 |
|---|---|
| `w` / `s` | 前进 / 后退 |
| `a` / `d` | 左 / 右 45° |
| `q` / `e` | 左 / 右 90° |
| `=` | 切换速度档位 |

这些是 PHP 原生键盘控制，和本仓库 AMP/SMP 的手柄映射不同。两个终端分别用 `Ctrl+C` 结束进程。

## 查看或传递参数

```bash
python deploy_php/deploy_php.py sim --scene steps --dry-run
python deploy_php/deploy_php.py policy --dry-run
# -- 后的参数原样交给官方脚本，例如查看原生参数帮助
python deploy_php/deploy_php.py policy -- --help
```

`--dry-run` 只显示工作目录、命令和关键环境变量，不启动进程。默认推理接口由官方脚本设为本机 `lo`。更换场景仍使用 Holosoma 的地形配置；本仓库的 `mujoco_table`、`mujoco_coffin` YAML 不能直接传给 PHP。

## 验证范围

`check` 验证实际 ONNX 深度编码器和 student 的输入输出，运行一次 CPU 推理，检查动作有限值，并输出模型 SHA-256。当前发布模型接口是深度 `[1,58,87]` → 32 维特征；93 维本体状态 + 15 维离散方向命令 + 32 维深度特征 → 29 维动作。零状态离线检查不代表闭环运动性能；本次接入没有宣称完成实机或完整台阶闭环验证。
