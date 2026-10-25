# 原生 MuJoCo viewer 深度图补丁

部署始终使用 `mujoco.viewer.launch_passive`，保留左右侧栏、Rendering → Camera、鼠标操作、ghost 和调试几何体。深度图绘制在三维视区右上角，窗口或侧栏大小改变时自动重新定位。更新深度图不修改相机。

## 迁移到新机器

当前补丁支持 **Linux、MuJoCo 3.2.3**，已在 Ubuntu 24.04、Clang/libc++/libunwind 18、Python 3.8 的 `robomimic` 环境验证。它为原生 `_simulate` viewer 扩展增加 RGB 上传和绘制接口，需要使用与官方 wheel 相同的 Clang/libc++ ABI 编译 C++（不能用 GCC 替代）；不修改 `libmujoco.so.3.2.3` 物理引擎，也不要求更换 Python 环境。其他 MuJoCo 版本不会被脚本自动降级或修改。

Ubuntu 24.04 构建依赖（缺少时安装）：

```bash
sudo apt-get install build-essential clang-18 libc++-18-dev libc++abi-18-dev libunwind-18-dev cmake patch xvfb xauth libeigen3-dev libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

在运行部署的同一个 Python 环境、仓库根目录执行：

```bash
conda activate robomimic
python -m pip install 'mujoco==3.2.3'
python tools/install_native_depth_overlay.py
```

第一次安装需要联网下载官方 MuJoCo 源码及固定版本的编译依赖。脚本校验下载 SHA-256，静态链接 C++ 运行库和 LLVM 异常展开库，仅构建并替换 `_simulate`，保留原始扩展备份；实际创建原生 viewer 并上传图像验证失败时自动恢复。源码、构建产物放在 `~/.cache/robomimic-native-viewer`，不提交到仓库。安装后重启仿真。重复运行已安装的同一补丁会直接返回。

仓库需要携带安装脚本和 `tools/native_viewer/mujoco-3.2.3-depth.patch`，无需复制本机 `.so` 文件。Python ABI 或架构不同的机器应各自编译。重新安装/升级 MuJoCo 会覆盖补丁，需要重新安装兼容版本和补丁。

## Track 操作

双击机器人躯干选中 body，再在左侧 **Rendering → Camera → Tracking** 切换跟随；选择 Free 回到自由视角。原生 viewer 要求先选中 body，否则 Tracking 会退回 Free。深度图不会覆盖此选择。`camera.follow=true` 仍可用于启动时开启跟随。

## 关闭与恢复

`php_depth_preview=false` 仅关闭深度预览，始终保留原生窗口。未安装补丁且开启预览时，启动会给出安装提示，不会替换为简化窗口。

恢复安装前的原生 viewer 扩展：

```bash
python tools/install_native_depth_overlay.py --restore
```

恢复后重启仿真，并关闭 `php_depth_preview`。此操作不影响物理引擎和策略模型。
