# PHP 实机：本体运行、C++ bridge 与深度断流

本分支已接入 RealSense 深度采集和现有 C++ bridge，**当前只完成离线验证，未连接相机、未发送电机命令、未验证本体运行速度或实机稳定性**。默认 `php.real.enabled: false`。原 Python 直连入口不支持 PHP；使用 `bridge/python/deploy_policy.py`。

## 运行位置

全部运行在机器人本体计算机：D435i USB → 右上角独立相机进程 → 外参投影/预处理 → 本机深度 IPC → 左中 PHP ONNX/FSM（50 Hz）→ 本机 bridge DDS → C++ bridge → 电机。电脑只用 SSH 操作 tmux，不转发图像、不推理。SSH 断开不停止 tmux 中的控制程序。

相机进程独占 USB，左中策略进程的后台读线程只读深度 IPC，控制线程只读内存快照。启动或停止左中策略不会启动或停止相机。无需训练仓库、Holosoma 服务或 MuJoCo 窗口；PHP 权重仍需放在本体。原 bridge IDL、C++ 程序和电机通信方式未改。

## 本体准备

1. 将本分支仓库及 `depth_backbone.onnx`、`student.onnx` 复制到本体；默认权重目录为运行用户的 `~/.cache/php-parkour/student/`，可修改 `php.model_dir`。加载时检查训练版本的 SHA256，不能混用另一版模型。
2. 复用现有 C++ Unitree SDK2、CycloneDDS 和 `~/robomimic-env.sh`。建议克隆现有可用的本体环境后安装 PHP 依赖，保留旧环境：

   ```bash
   conda create -n robomimic-php --clone robomimic
   conda activate robomimic-php
   python -m pip install Pillow pyrealsense2
   python -m pip check
   python -c 'import torch, onnxruntime, scipy, yaml, cyclonedds, pyrealsense2; print("imports OK")'
   ```

   PHP 复用现有 NumPy/PyTorch/ONNX Runtime/SciPy/PyYAML；需要支持 opset 18 的 ORT、支持 antialias bicubic 的 PyTorch。上述导入不打开设备。本次离线测试版本为 Python 3.10、torch 2.3.1、ORT 1.19.2，**未验证机器人旧 Python 3.8 环境**。Jetson/aarch64 若无兼容 `pyrealsense2` wheel，应按本体系统/JetPack 安装 librealsense 的 Python binding 及 USB udev 规则，不能复制工作站 x86_64 的库或 CUDA wheel。相机 SDK 参考 [官方仓库](https://github.com/realsenseai/librealsense)。
3. 将 D435i 接入本体 USB 3；记录设备序列号。不要同时运行占用该相机的其他采集程序。PHP 不需要原六屏中的踢球灰度相机、雷达或 ball fuser。

## 配置相机与实机开关

先保留关闭状态做离线与相机检查；最后才开启实机 PHP。复制配置而不改默认配置：

```bash
cd ~/RoboMimic_Deploy  # 或实际仓库路径
mkdir -p ~/.config/robomimic
cp config/deploy.yaml ~/.config/robomimic/php-deploy.yaml
cp policy/php_parkour/config/real_depth.yaml ~/.config/robomimic/php-camera.yaml
```

在复制的 `php-deploy.yaml` 中保留 AMP/SMP/ScaleBFM 等节，并设置：

```yaml
php:
  enabled: true
  model_dir: ~/.cache/php-parkour/student
  real:
    enabled: false  # 离线及相机检查完成后，准备实机试验时手动改 true
    camera_config: ~/.config/robomimic/php-camera.yaml
```

相机文件的默认外参：

```yaml
mount:
  position_m: [0.01, 0.01, 0.44]
  rpy_deg: [1.0, 27.0, 1.0]
serial: ""  # 建议填设备序列号；留空只接受恰好一个设备
width: 848
height: 480
fps: 30
max_age_s: 0.15
min_valid_fraction: 0.90
```

`position_m` 是相机**深度光心相对 torso_link** 的位置；躯干坐标 X 前、Y 左、Z 上。RPY 按 `Rz(yaw) Ry(pitch) Rx(roll)`，正 pitch 让前视光轴向下。默认是训练的 **27° 向下**，不是相机铰链刻度，更不是发送给头部的角度命令。实测外参应填写实际安装值；不要凭之前 Robonaldo 的约 37° 直接认为相同。

程序用设备实际内参/米制 optical-Z 深度，按外参投影到固定的训练视角，然后执行训练的裁剪、58×87 bicubic 和归一化。外参配置能修正几何，不能补出相机看不到的区域；有效覆盖不足 90% 会拒绝帧。保留完整相机配置中的 `effort_limits`，不能只用上面片段覆盖文件。使用原生 rectified depth，不对齐 RGB。深度含义参考 [RealSense 投影文档](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。

## 分阶段检查

先在本体跑离线测试，**不启动 C++ bridge**：

```bash
export LYD_DEPLOY_CONFIG="$HOME/.config/robomimic/php-deploy.yaml"
python tools/diagnostics/check_php_real.py --camera-config ~/.config/robomimic/php-camera.yaml --out /tmp/php-real-offline.json
python -m unittest discover -s tests -p test_php_real.py -v
```

离线工具仅在进程内启用 PHP，使用合成状态、深度和实际 ONNX；禁止创建 DDS Participant，不修改配置、不采集相机、不发电机命令。它不是实机闭环测试。

随后可单独做 USB 采集（下面命令会打开相机，但不会启动任何控制器）：

```bash
python tools/capture/preview_php_depth.py --camera-config ~/.config/robomimic/php-camera.yaml --seconds 15 --out /tmp/php-depth.npz
python tools/diagnostics/check_php_real.py --camera-config ~/.config/robomimic/php-camera.yaml --recording /tmp/php-depth.npz
```

查看 `/tmp/php-depth.png`（近黄远蓝）和 coverage/age，检查是否对应实际地面与障碍。无有效帧则报错，不把黑屏当成自由空间。NPZ 保存原始米制深度、内参、外参和采集年龄；回放仍使用合成机器人状态。预览退出后才能启动正式策略，避免争抢设备。本次开发没有执行上述 USB 命令。

## 复用 tmux 启动本体

唯一 tmux 入口为 `tools/deploy/start_tmux_layout.sh`，保留原六屏和额外的 monitor 窗口。已删除 PHP/ScaleBFM 的重复入口，不再通过环境变量选择布局；PHP 仅由 `php.enabled` + `php.real.enabled` 控制。

| 左侧 | 右侧 |
| --- | --- |
| C++ bridge | 相机：灰度或 PHP 深度（按配置） |
| Python 策略（仅读取 PHP 深度） | 雷达检测 |
| Sensor Dashboard（包含 PHP 深度） | 球位置融合 |

右上角统一运行 `tools/capture/run_camera.py`，读取同一个部署 YAML：

```yaml
camera:
  mode: php_depth  # gray：原灰度相机；php_depth：独立 PHP 深度进程
```

可先运行 `python tools/capture/run_camera.py --check` 检查选项，此命令不打开 USB 或 DDS。修改 `camera.mode` 后，在右上停止旧进程再重新运行预填命令即可；无需重建 tmux 或重启左中策略。

默认 `gray`，保留原相机行为。相机选择和 `php.real.enabled` 独立：即使不开启策略 PHP，也可以启动 PHP 相机进行网页预览。选 PHP 时，右中雷达、右下球融合可不启动。相机、Dashboard 与策略使用相同的 Python 环境和 `LYD_DEPLOY_CONFIG`。

网页入口：**PHP 深度图 `http://机器人IP:8091/`**（右上相机 + 左下 Dashboard 即可，左中策略和左上 bridge 均无需启动）；**灰度相机 `http://机器人IP:8080/`**（右上 `gray` 模式自动带 `--show`）。8091 仍保留原传感器 Dashboard。PHP 网页显示处理后的 87×58 深度，不是 RGB 视频。

本体检查完成、保护措施就绪后，将专用配置的 `php.real.enabled` 改成 `true`。以下在机器人本体 SSH 终端执行；环境路径按自己的安装位置调整：

```bash
cd ~/RoboMimic_Deploy
export LYD_DEPLOY_CONFIG="$HOME/.config/robomimic/php-deploy.yaml"
export PYTHON_BIN="$HOME/miniconda3/envs/robomimic-php/bin/python"
export CONDA_ENV_LIB="$HOME/miniconda3/envs/robomimic-php/lib"
export ROBOT_IFACE=eth0  # 本体连接机器人 DDS 的真实网卡名
bash tools/deploy/start_tmux_layout.sh robomimic-php
```

脚本**只预填命令，不发送 Enter**。先启动右上相机和左下 Dashboard 检查图像；左中面板按 Enter 启动 Python，完成模型加载和预热，等待 bridge 状态；左上面板再按 Enter 编译/启动 C++ bridge。C++ 一旦启动就进入电机通信，不要将它当成无副作用的离线检查。复用已有会话不会重建命令；修改配置路径/环境变量后，应在旧控制程序已退出的前提下使用一个新会话名。

启动状态是 Passive；START 才进入 FixedPose。深度有效时 `L1+A` 显式进入 PHP，左摇杆给方向；`F2` 切换高低速（Unitree 包没有仿真的 R3 字段）。`Y`（不按 L1）进入 SMP；单独 A **释放**切 AMP、B 切 Loco、START 切 FixedPose。F1 阻尼优先，select 退出。按住组合键后松 L1 不会误切 AMP。

不要同时启动 Python 直连控制器或其他 LowCmd 发布者。首次硬件验证使用吊挂/人工保护，先确认静态、方向、阻尼及断流行为，再考虑行走。机器人上的推理和投影时延需要实测，工作站离线速度不能代表 Jetson。

## Dashboard 深度图

左下面板按 Enter 启动后访问 `http://机器人IP:8091/`。右侧顶部 PHP 卡片显示模型实际使用的 **87×58** 深度图、帧龄、覆盖率和帧号。原 target/ball 地图及数值卡片保留。相机由右上角独立进程提供，不依赖策略；不要同时启动 `preview_php_depth.py` 抢占 USB。

相机进程的独立后台线程以约 10 Hz 将 PNG 和时间戳原子写入本体共享内存文件，Dashboard 只读该文件。控制线程不写文件、不编码图片、不等待浏览器；预览服务关闭或文件写失败不会改变控制。默认文件是 `/dev/shm/robomimic-<uid>/php-depth.json`，两个进程应使用相同用户；需要自定义时，在 `php.real.dashboard_file` 写同一个绝对路径。`LYD_DEPLOY_CONFIG` 会同时传给相机、策略和 Dashboard。

控制深度单独通过 `/dev/shm/robomimic-<uid>/php-depth.npz` 原子传输，可用 `php.real.depth_file` 配置。它保留采集时间戳、帧号、生产进程标识、启动标识和相机配置哈希；读取不会刷新旧帧时间，配置不一致会拒绝进入 PHP。右上进程退出会写入无效状态，进程被强制终止则由帧龄超时检测。预览 JSON/PNG 不作为控制输入。

深度过期、断流、生产进程停止时，图片会隐藏并显示原因，避免把旧图当成实时图。浏览器连接中断也会在本地超时隐藏图像；页面的有效状态只是监视信息，控制仍由 bridge/FSM 自己判断。PHP 未启用时显示 unavailable。

仅离线检查 Dashboard 页面（不订阅 DDS、不打开相机）可运行：

```bash
python onboard/perception/debug/sensor_dashboard.py --offline --host 127.0.0.1 --port 8091
```

此时没有生产进程，PHP 卡片显示 unavailable；自动测试会注入合成深度验证图片及失效行为。

## 不启动相机时

可以只用 AMP、SMP、Loco、ScaleBFM 等策略。无需 PHP 时将 `php.real.enabled: false`，策略进程不会加载 PHP；相机是否运行由右上面板单独控制。即使 PHP 已启用但相机缺失，**也不因相机缺失让其他策略 damping**：PHP 入口被拒绝，当前策略继续运行。不能用伪造的默认深度进入 PHP。下面的断流阻尼仅针对 PHP 已经运行后失去有效深度的情况。

## 深度掉线会发生什么

| 情况 | 当前实现 |
| --- | --- |
| 启动没有深度、相机打不开、有效覆盖不足 | 留在当前策略，拒绝进入 PHP，打印具体原因 |
| PHP 运行中无深度、时间戳异常、重复帧、采集异常 | 下一个控制周期检查到后退出 PHP，输出 `request_damping`，进入 Passive |
| 静默停帧 | 从最后一帧**采集时间**起超过 150 ms 即失效；控制周期约 20 ms，受实际调度影响 |
| Python 卡死/退出，不能发新命令 | 原 C++ bridge 命令 watchdog 默认超过 100 ms 后阻尼（需 C++ bridge 本身仍正常运行） |
| 深度恢复 | 保持 Passive，不自动恢复 PHP；需放开并重新按 L1+A |
| 当前是 AMP/SMP 等其他策略时相机掉线 | 不强制打断这些不依赖深度的策略；PHP 入口不可用 |

阻尼为 **kp=0、kd=8**：停止主动追踪 PHP 目标，抵抗关节速度，**不是定姿、不是平衡站立，也不是断电**。机器人可能屈膝或倒下。不会自动切 AMP、SMP 或 FixedPose；自动接管未经实机验证。PD 目标按训练 effort envelope 限制，只是基于状态的估算，不等于硬件力矩安全认证。

掉线后先人工保护机器人并按 F1；检查右上相机面板及左中策略面板 `[PHP camera]` 的时间戳、覆盖率或设备错误，检查 USB 供电、线缆、序列号和其他占用进程。短暂丢帧可能自行恢复有效图像；物理拔插后**当前实现不自动重建 pipeline**，应保持 bridge 阻尼，在右上 Ctrl-C 退出相机，排查后重新启动右上相机进程（左中策略无需重启），确认深度正常后手动进入。不要通过放大 max_age 或重复使用最后一张图绕过故障。

退出时先 select 或 Ctrl-C 停 Python，让它发送退出/阻尼；确认机器人受保护后再停止 C++ bridge。`Ctrl-b d` 仅分离 tmux，不停止控制。

新建 tmux 会话的每个面板顶部显示位置和用途；在空闲 shell 输入 `tmux_help` 可查看六屏职责、网页端口与操作提示。已有会话不会被脚本重建，新布局请使用一个新的会话名。
