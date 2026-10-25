# 官方 MuJoCo 环境安装与迁移

`lyd-php-parkour` 使用 **Python 3.10 + 官方 MuJoCo 3.10.0**。深度图通过公开的 `Handle.viewport`、`set_images()` 和 `set_texts()` 显示，始终使用 `mujoco.viewer.launch_passive`，保留左右侧栏、Track/Free 切换、ghost 和调试几何体。仓库不再携带 MuJoCo 源码补丁或编译安装脚本，不修改 `site-packages` 中的库文件。

本次固定 3.10.0 以复现验证结果，并与外部 PHP 环境版本一致；安装命令不自动追随最新版。官方接口参见 [viewer 源码](https://github.com/google-deepmind/mujoco/blob/3.10.0/python/mujoco/viewer.py) 和 [Python 文档](https://mujoco.readthedocs.io/en/3.10.0/python.html)。

## 1. 独立创建环境

已验证：Ubuntu 24.04 x86_64、Python 3.10、MuJoCo 3.10.0。已有 `robomimic` 环境保持原样；不要在其中运行下面的安装命令。若新环境名已经存在，请直接激活已建好的环境，或换一个新名字。

```bash
# 新机器首次获取仓库；已有仓库无需重新 clone
cd ~/workspace
git clone --branch lyd-php-parkour git@github.com:lu-yidan/RoboMimic_Deploy.git
cd RoboMimic_Deploy

conda create -n robomimic-mj310 python=3.10 pip -y
conda activate robomimic-mj310

# NVIDIA 工作站；保留 ScaleBFM 所需的 CUDA 推理能力
python -m pip install 'torch==2.3.1+cu121' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environment/mujoco-official-requirements.txt
python -m pip check
python -c "import sys,mujoco,torch; print(sys.executable); print(mujoco.__version__,torch.__version__,torch.cuda.is_available())"
```

本次创建的环境位于 `/home/ydlu/miniconda3/envs/robomimic-mj310`。新机器按自己的 Conda 安装位置即可，不要复制本机 `.so` 或 Python 环境目录。

仅运行 PHP/AMP/SMP 的无 CUDA 机器，可将上面的 PyTorch 安装命令替换为：

```bash
python -m pip install 'torch==2.3.1+cpu' --index-url https://download.pytorch.org/whl/cpu
```

CPU 版本不支持当前 ScaleBFM 的 CUDA 路径。CUDA wheel 自带所需运行库；NVIDIA 驱动仍由系统提供。图形窗口需要正常的桌面显示和 OpenGL 驱动；不需要为 MuJoCo 安装 Clang、CMake 或编译 C++。本环境用于仿真，实机 SDK/Jetson 环境继续按原部署文档单独配置。

`requirements.txt` 与 `environment/smp_recovery_mujoco.yml` 是旧 Python 3.8 部署组合；新环境使用上面的专用依赖文件，不要再混装旧清单。

## 2. 外部 PHP 资源

本机已有 `~/workspace/php_parkour` 和 `~/.cache/php-parkour/student`，直接复用。新机器需要：

```bash
cd ~/workspace
git clone git@github.com:lu-yidan/php_parkour.git
cd php_parkour
git submodule update --init thirdparty/holosoma
python scripts/download_assets.py student
```

无需安装 PHP 训练环境或启动 Holosoma 服务。检查本仓库 `config/deploy.yaml`：

```yaml
php:
  enabled: true
  root: ~/workspace/php_parkour
  model_dir: ~/.cache/php-parkour/student
```

`root` 下需要 `thirdparty/holosoma/src/holosoma/holosoma/data/terrains/terrain.obj`；`model_dir` 下需要 `depth_backbone.onnx` 和 `student.onnx`，部署加载时会校验模型哈希。自定义位置请修改上述两个字段。ScaleBFM 外部权重和动作仍按 [README 第 6 节](../README_zh.md#6-scalebfm-模型与动作) 配置。

## 3. 启动与操作

```bash
conda activate robomimic-mj310
cd ~/workspace/RoboMimic_Deploy
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php

# 平地
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php scene.name=none
```

实际部署入口仍需手柄。启动保持 Passive；START、Y、L1+A、P 和 Backspace 的策略语义不变。双击机器人躯干选中 body，再在左侧 **Rendering → Camera → Tracking** 开启跟随；选 Free 恢复自由视角。未选中 body 时，原生 viewer 会拒绝 Tracking。`camera.follow=true` 仍可用于启动时开启跟随。

深度图固定在三维视区右上角，避开侧栏；窗口缩放或侧栏开关后会重新定位。显示的是模型使用的 87×58 深度输入，近黄远蓝，标题显示帧龄和当前策略。更新深度图不会修改相机。极小视区会暂时隐藏图像，放大后恢复。`php_depth_preview=false` 只关闭预览，原生窗口始终保留。

## 4. 验证

交互显示测试使用 Xvfb，不要求插入实际手柄。Ubuntu 缺少测试工具时安装：

```bash
sudo apt-get install xvfb xauth x11-utils ffmpeg
```

完整测试包含 bridge 消息结构的离线测试，需额外安装 CycloneDDS 的官方 wheel；这些测试不会连接机器人或启动 DDS 通信：

```bash
python -m pip install --only-binary=cyclonedds 'cyclonedds==0.10.2'
MUJOCO_GL=egl python -m unittest discover -s tests -v
MUJOCO_GL=egl python tools/smoke_php_mujoco.py
MUJOCO_GL=egl python tools/validate_php_recovery.py --out /tmp/php-smp-mj310.json
```

`MUJOCO_GL=egl` 用于无窗口深度渲染测试；图形部署不需要设置它。无法使用 EGL 的机器可在 Xvfb 下使用 GLFW，例如：

```bash
xvfb-run -a env MUJOCO_GL=glfw python tools/smoke_php_mujoco.py
```

本次 49 项测试全部通过；平地和外部障碍场各完成 10 秒 PHP 闭环，6 组不同倒地时长的 Passive→SMP 测试各运行满 30 秒，无状态警告或自动重置。ScaleBFM 额外完成 squat 参考动作的 100 次 CUDA 静态状态推理及 CUDA graph 一致性检查，未超过 20 ms 预算；这项检查不包含闭环跟踪。结果见 [迁移验证记录](validation/mujoco_official_20261025.json)。验证覆盖原生侧栏、Track/Free 保持、深度图方向及窗口缩放、策略输入与切换、Backspace 状态保持、两种场景 PHP 闭环和六组 Passive→SMP 恢复。它不等于所有场景或实机行为均已验证。

## 5. 原环境与回退

本次没有升级、卸载或还原旧 `robomimic` 环境中的任何包；其中原有 viewer 补丁也保持不动。新代码不再依赖该补丁，开启预览时若激活旧环境，会提示使用新环境，不会自动修改环境或切换成简化窗口。

如需完整复现旧环境及旧代码，可另建 Git 工作目录，不覆盖当前代码：

```bash
cd ~/workspace/RoboMimic_Deploy
git worktree add --detach ../RoboMimic_Deploy-mj323 b70141e
conda activate robomimic
cd ../RoboMimic_Deploy-mj323
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php
```

回退目录只在需要时创建。新机器无需安装旧版补丁。
