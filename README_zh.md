<div align="center">
  <h1 align="center">RoboNaldo Deploy</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>RoboNaldo Deploy 是一个基于状态切换机制的机器人多策略部署框架，目前包含的策略适用于宇树G1机器人(29dof)</strong> 🚪🎮
</p>

> 🤖 **本仓库是 [RoboNaldo](https://opendrivelab.com/RoboNaldo/) 的部署仓库。**
> 策略的**训练**代码在配套仓库
> [**opendrivelab/RoboNaldo**](https://github.com/opendrivelab/RoboNaldo)；本仓库负责把训练好的策略
> 部署到真机与仿真的宇树 G1 上。

<table align="center" width="100%">
  <tr>
    <td width="50%" align="center"><b>仿真 Simulation</b></td>
    <td width="50%" align="center"><b>真机 Real robot</b></td>
  </tr>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/f1365c83-3764-43b7-9ecf-57ba89854cf1" controls muted width="100%"></video></td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/45fbf473-060e-49b1-9204-c43bbcc89a3c" controls muted width="100%"></video></td>
  </tr>
</table>

<p align="center"><em>任意球 demo —— RoboNaldo 策略部署在宇树 G1 上</em></p>

## 写在前面

- **本部署框架仅适用于具有三自由度腰部的G1机器人，如果装有腰部固定件的话需要按照官网教程解锁，然后才能正常使用该部署框架。**

- **建议拆下手掌，舞蹈动作会存在干涉**

- **实际机器人部署中出现的问题，十有八九是策略适应性不足所致，大家不必过度怀疑硬件层面的缺陷。**

- **[视频教程](https://www.youtube.com/watch?v=BuHNzqebIqc&feature=youtu.be)**

## 安装配置

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。

### 1.1 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n robomimic python=3.8
```

### 1.2 激活虚拟环境

```bash
conda activate robomimic
```

---

## 2. 安装依赖

### 2.1 安装 PyTorch

PyTorch 是一个神经网络计算框架，用于模型训练和推理。使用以下命令安装：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

> **GPU 说明：** 该 CUDA 12.1 版本可配合较新的 NVIDIA 驱动使用（已在 RTX 4090 + 驱动 580 上验证）。安装后可验证：
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"   # 期望输出 True
> ```
> **无 GPU 的 CPU 备选方案**（仅跑 MuJoCo 仿真已足够）：
> ```bash
> pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
> ```

> **国内下载慢？** 先把 pip 指向国内镜像，再照常安装：
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 2.2 安装 RoboNaldo_Deploy

#### 2.2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/OpenDriveLab/RoboNaldo_Deploy.git
```

#### 2.2.2 安装组件

进入目录并安装：

```bash
cd RoboNaldo_Deploy
pip install -r requirements.txt
```

> **依赖版本说明**（`requirements.txt` 已处理，无需手动改）：
> - `numpy==1.24.4` —— 不要降到 1.20.x，旧版缺少 `numpy.typing.NDArray`，会导致 `import torchvision` 失败。
> - `onnxruntime>=1.16` —— 仓库内策略模型以 ONNX opset 18 导出。旧版本（如 1.10.0）只支持到 opset 15，加载模型时会报 `Opset 18 ... is till opset 15`。Python 3.8 下最后可用版本为 `onnxruntime==1.19.2`。
#### 2.2.3 安装unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

## 文档地图

真机机载部署请先看 `onboard/docs/README.md`。它说明运行时架构、topic 归属，
并给出新人阅读路径。

- 环境配置：`onboard/docs/INSTALL_AGENT_GUIDE.md`
- 真机启动拓扑：`tools/start_tmux_layout.sh`
- 相机模块手册：`onboard/perception/camera/README.md`
- LiDAR 模块手册：`onboard/perception/lidar/README.md`
- 感知架构：`onboard/docs/PERCEPTION_ARCHITECTURE.md`

---
## 运行代码

## 1. 运行Mujoco仿真代码

标准场景：
```bash
python deploy_mujoco/deploy_mujoco.py
```

FreeKick 场景（含球，运行 FreeKick 策略时使用）：
```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_freekick
```

---
## 2. Policy 说明

| 模式名称            | 触发按键                       | 描述                                                                 |
|--------------------|-------------------------------|----------------------------------------------------------------------|
| **PassiveMode**    | F1（真机）/ 松开 L2（仿真）    | 阻尼保护模式                                                         |
| **FixedPose**      | Start                         | 位控恢复至默认关节值                                                 |
| **LocoMode**       | B                             | 用于稳定行走的控制模式                                               |
| **AMP**            | A                             | AMP 运动策略（默认进入 run 模式）                                    |
| **FreeKick**          | R1                            | 踢球得分策略（547维obs，5帧历史，需配合机载雷达感知）               |
| **BeyondMimicMJ**  | D-pad DOWN                    | 站立躺下模仿策略（MuJoCo 训练版，含参考动作跟踪）                   |
| **StandUpMJ**      | D-pad UP                      | 躺下站起模仿策略（MuJoCo 训练版，含参考动作跟踪）                       |
| **Pinocchio1.6MJ** | R2                            | `g1_result_pinocchio_1_6_mj.yaml` 对应的 MuJoCo 模仿策略            |
| **BeyondMimic**    | —                             | 仍保留在仓库中，但当前默认不分配手柄按键                             |

### 手动偏置调整

仿真和真机均支持在不重启的情况下通过手柄实时调整感知 Y 轴偏置：

| 操作 | 按键 | 范围 | 备注 |
|------|------|------|------|
| 目标 Y +5 cm（左移） | 按住 **L1** + 点按 **D-pad Left** | ±1.5 m | 仿真 + 真机 |
| 目标 Y −5 cm（右移） | 按住 **L1** + 点按 **D-pad Right** | ±1.5 m | 仿真 + 真机 |
| 球 Y +5 cm（左移） | 按住 **L2** + 点按 **D-pad Left** | ±1.5 m | 仅真机 |
| 球 Y −5 cm（右移） | 按住 **L2** + 点按 **D-pad Right** | ±1.5 m | 仅真机 |

偏置以 pelvis body 坐标系为基准（+Y = 机器人左侧）。每次调整时终端会打印当前值，**Sensor Dashboard** 侧边栏橙色"Current Bias"卡片也会实时显示。

---
## 3. 仿真操作说明

1. 连接Xbox手柄

2. 运行仿真程序：
```bash
python deploy_mujoco/deploy_mujoco.py
```
3. Start键进入位控模式

4. 按 `B` 进入 LocoMode，并按下 `BACKSPACE` 在仿真中使机器人站立，之后即可通过摇杆控制机器人行走

5. 仿真中保留的单键切换如下：
   - `A` -> AMP（run 模式）
   - `B` -> LocoMode
   - `D-pad DOWN` -> BeyondMimicMJ
   - `D-pad UP` -> StandUpMJ
   - `R2` -> Pinocchio1.6MJ
   - `R1` -> FreeKick

6. `FreeKick` 需使用 `--config-name mujoco_freekick` 加载含球的仿真场景；`BeyondMimicMJ / StandUpMJ / Pinocchio1.6MJ` 之间支持直接互切

7. 任意时刻可**松开 L2** 进入阻尼保护模式（PassiveMode），按 `Start` 返回 FixedPose

8. `BeyondMimic` 仍保留在仓库中，但默认不再绑定手柄按键；`Dance / Kick / KungFu / ASAP / HOST` 等旧策略已经从仓库中删除

---
## 4. 真机操作说明

1. 开机后将机器人吊起来。先按 **L2+B** 进入阻尼保护模式（头部紫灯变橙），再按 **L2+R2** 进入调试模式（头部橙灯变黄）。

2. **推荐方式（机载 Orin）**：使用 tmux 启动脚本，一键启动 C++ bridge、policy 推理、感知服务和 Sensor Dashboard：
   ```bash
   bash tools/start_tmux_layout.sh
   ```
   在 `deploy_policy` 窗格中按 **Start** 进入位控模式。

   也可分窗口手动启动：
   ```bash
   BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main   # 终端 1
   python bridge/python/deploy_policy.py                          # 终端 2
   ```

   过渡验证版（无需 C++ bridge）：
   ```bash
   python bridge/python/deploy_bridge_py.py   # 终端 1
   python bridge/python/deploy_policy.py       # 终端 2
   ```

   **简单模式（直连 以太网，无 Orin）**：`python deploy_real/deploy_real.py`

3. 后续单键切换与仿真保持一致：`A / B / D-pad DOWN / D-pad UP / R2 / R1`。
   任意时刻可按 **F1** 进入阻尼保护模式（PassiveMode）。

4. **FreeKick 踢球策略（R1）真机额外步骤**：FreeKick 策略依赖机载球感知，需在启动前先运行感知服务；LiDAR / camera raw 观测由 `onboard/perception/ball_fuser.py` 融合，并发布最终 DDS topic `rt/ball_state`。未启动感知服务时请勿激活 FreeKick 策略。

5. **Sensor Dashboard**：启动 `tools/start_tmux_layout.sh` 后，在浏览器中打开 `http://<机器人IP>:8091/` 即可查看所有传感器位置（目标、各球感知源、偏置后坐标）及橙色"Current Bias"卡片中的当前偏置值。手动启动：
   ```bash
   python onboard/perception/debug/sensor_dashboard.py
   ```

   完整 bridge 配置说明见 `bridge/README_zh.md` 和 `bridge/VALIDATION.md`。

---
## 注意事项
### 1. 框架兼容性说明
当前框架暂不支持在搭载Orin NX平台的G1机器人上直接部署。初步分析可能是由于`unitree_python_sdk`在Orin平台上的兼容性问题。针对机载Orin平台的部署需求，建议采用以下替代方案：

- 使用[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)替代原Python SDK
- 基于ROS构建双节点架构：
  - **C++节点**：负责机器人与遥控器之间的数据收发
  - **Python节点**：专用于策略推理
- 本仓库已提供一个最小桥接原型，集中在 `bridge/` 目录下，可作为上述拆分方案的参考实现。

### 2. Mimic策略可靠性警告
Mimic策略不保证100%成功率，特别是在湿滑/沙地等复杂地面上。若出现机器人失控情况：
- 按下`F1`键激活**阻尼保护模式**(PassiveMode)
- 按下`Select`键立即终止控制程序

### 3. 当前保留策略的真机提示
当前部署入口已简化为 `LocoMode / AMP / FreeKick / BeyondMimicMJ / StandUpMJ / Pinocchio1.6MJ`。

⚠️ **重要注意事项**：
- **建议拆除手掌**：部分策略的原始训练未考虑手掌碰撞（作者的G1初始无手掌）
- **启动与切换阶段**：策略切换前后建议优先回到 **FixedPose** 或 **PassiveMode**
- **人工保护**：首次在真机验证某个保留策略时，建议全程吊挂并保留人工保护

### 4. 其他动作建议
仓库中现在只保留当前部署会用到的策略实现，其余历史策略已删除，以减少误用和维护成本。

### 5. 强烈建议
**务必**先在仿真环境中熟练操作，再尝试真机部署。

## 致谢

本部署框架基于 [**ccrpRepo/RoboMimic_Deploy**](https://github.com/ccrpRepo/RoboMimic_Deploy)
构建。感谢原作者开源了这套基于状态切换的多策略部署框架，本仓库在其基础上扩展而来。

本仓库是 [**RoboNaldo**](https://opendrivelab.com/RoboNaldo/) 的部署仓库 ——
训练代码见 [opendrivelab/RoboNaldo](https://github.com/opendrivelab/RoboNaldo)。

## ScaleBFM：MuJoCo 手柄切换（lyd）

本分支 MuJoCo 与真机的按键以本节为准，覆盖前文上游 FreeKick 的 R1 映射。

分支 `lyd/scalebfm-table-config-balanced4` 基于 `codex/table-config-balanced4`（`4678455`）；工作目录为
`/home/d080/workspace/RoboMimic_Deploy-scalebfm`。使用 Conda `scalebfm-native`，
相邻 `/home/d080/workspace/ScaleBFM` 中已下载的官方 M 型 `model_22200.pt`、
mode table、metadata 和本地支持原生 PyTorch 的 ScaleBridge。
部署入口所需的 `onnxruntime`、`pygame` 已装入该 Conda 环境。

```bash
conda activate scalebfm-native
cd /home/d080/workspace/RoboMimic_Deploy-scalebfm
python deploy_mujoco/deploy_mujoco.py xml_path=g1_description/scene.xml
```

手柄操作（按下后松开触发）：

- L1＋START（按住 L1，按下并松开 START）或键盘 Backspace：真正重置仿真，
  恢复站立关节姿态、清零速度，立即进入 FixedPose；随后按 R1。
- 单独 START：只切到 FixedPose，约 2 秒插值关节目标，不重置根位置/速度，也不保证倒地起身。
- 方向键左 / 右：循环选择上一段 / 下一段 motion，终端显示序号和文件名；选择时不打断当前动作。
- R1：从 FixedPose、AMP、Loco 或 ScaleBFM 直接播放所选 motion，第 0 帧重新开始；保留仿真位置和速度。模型 checkpoint 保持不变。
- START：退出 ScaleBFM，回 FixedPose；A / B 可切回 AMP / Loco；再按 R1 可重新播放。
- L2 松开：回 Passive；SELECT：退出程序。
- MuJoCo 与真机入口均不再加载 FreeKick，R1 已用于 ScaleBFM。
- 动作列表由 `scalebfm.motion` 和 `scalebfm.motion_globs` 在启动时收集，默认包含 batch15 与 comparison 目录中的动作；新增文件后重启以刷新列表。
- L1＋方向键左 / 右仍调整感知偏置，不改变所选动作。

没有虚拟悬挂或持续支撑力。失衡后使用 L1＋START / Backspace 重置再启动。
重置只是仿真测试便利功能，不代表策略具备起身恢复能力；真机已接入同一策略接口，但没有仿真重置功能。
若只使用原有策略，可传 `scalebfm.enabled=false`，不加载 ScaleBFM。

### 默认动作与官方基线的区别

默认参考来自 ScaleBFM 仓库自带的
`ScaleBridge/scalebridge/data/motion/g1_29dof/squat.npz`，417 帧、50 Hz，约 8.34 秒。
它不是 DuoBox，也不是本次新生成的数据；文件本身没有标注更早的采集来源。
参考到末尾后保持最后一帧，不循环拼接。

使用官方 M 型权重、mode 7、观测函数和策略包装；本地原生 PyTorch 推理，
由 RoboMimic MuJoCo 循环执行 PD。机器人资产、力矩限制、起始姿态以及切换流程
属于本地部署适配，因此不是官方完整基线复现，也不是重新训练后的模型。
`reference_forcing=true` 与官方本地跟踪设置一致：用参考根位置构造观测，
不是把仿真机器人根位置强行写成参考位置；不据此验证绝对位置跟踪。
策略 50 Hz；当前部署物理步长 0.002 秒，即 500 Hz。
本分支默认 `scene.xml`；`g1_liao.xml` 在下蹲测试中根高度降至约 0.19 米，
未通过当前检测，不视为兼容资产。

### 选择自己的动作

```bash
python deploy_mujoco/deploy_mujoco.py \
  xml_path=g1_description/scene.xml \
  scalebfm.motion=/absolute/path/to/your_motion.npz
```

路径在启动时选定；当前不支持运行中换文件。文件必须是已打包的 **G1 29 DOF、
50 Hz** 参考，不能直接输入 SMPL、原始 DuoBox，或只有 root/dof 的 GMR 文件。
需要以下数组（T 是帧数；四元数 wxyz；位置米、关节角弧度）：

| 字段 | 形状 |
| --- | --- |
| fps | 标量 50 |
| joint_pos / joint_vel | `[T, 29]` |
| body_pos_w / body_lin_vel_w / body_ang_vel_w | `[T, 30, 3]` |
| body_quat_w | `[T, 30, 4]` |

**关节和 body 顺序必须与 checkpoint metadata 的 `joint_names` / `body_names` 一致**，
不能直接使用 MuJoCo XML 的深度优先顺序。可附带两个同名字符串数组以在加载时校验顺序。
无名称数组的上游文件只能检查形状、有限值及四元数归一化，不能自动证明排列正确。

自己的动作准备流程是：选定单人片段 → retarget 为 G1 → 重采样到 50 Hz →
按 metadata 顺序做 FK，生成世界系 body 位姿和速度 → 保存 NPZ → 在这里运行 tracker。
官方打包入口为相邻 ScaleBFM 的
`ScaleTrack/scripts/pretrain/data_process/package_motions.py`，依赖 IsaacLab，
不属于当前轻量部署环境。已有 15 段 DuoBox/GMR 单人片段完成批量测试，共 266.88 秒；默认动作列表再加官方 squat，共 16 项。UMR 比较尚未完成。

### 验证与离线录像

`tools/smoke_scalebfm_fsm.py` 无需手柄，运行实际 FSM、PD 与 MuJoCo，检查
重置至 FixedPose → ScaleBFM 的 10 秒跟踪、倒地状态重置、重新进入后 2 秒跟踪、
同一 checkpoint 切换两段 motion、AMP / Loco 双向切换、无效文件保留当前动作和 Passive 退出。
直接切换检查 qpos/qvel 没有被重置；这次短时测试不代表任意运动状态下切换都能稳定。
全程检查外部支撑力为零；它不是机载时延或真机验证。

```bash
python tools/smoke_scalebfm_fsm.py
python tools/validate_scalebfm.py --seconds 10 --video \
  --robot-xml g1_description/scene.xml \
  --out "outputs/scalebfm/squat_$(date +%Y%m%d_%H%M%S)"
```

离线脚本输出 `report.json`、`trace.npz`、`recording.mp4`，需要新输出目录。
它采用独立 ScaleBridge 仿真流程，不经过上述手柄 FSM；可以加 `--motion /path/to/motion.npz`。


### DuoBox 首批单 G1 测试

在 `lyd/scalebfm-table-config-balanced4` 分支运行，继续使用 Conda `scalebfm-native`。
选取 `fbx_export_joints` 中 004 的 liu（闪避）、003 的 ma（后手直拳）、027 的 ma（移动组合拳）。
前两段复用 GMR v10 缓存；027 使用同一个 `--body-anchor` v10 流程新生成。
这些是人类原始数据的重定向动作，不是 Ready-to-React 生成的动作。

转换只做名称重排、30 → 50 Hz 重采样（根四元数 SLERP）、FK 和速度计算，
没有额外减速或平滑；保留上游 v10 的 8 rad/s 连续性约束及几何修正。
源路径、actor、SHA256、重定向配置均记录在每个动作同名 JSON 中。
在原始/重采样时间重合的帧上对照 GMR FK，位置误差小于 1e-7 米，
验证关节顺序与模型坐标转换；这不等同于验证重定向对人体动作的忠实度。

| 预设 | 来源 | 完成时长 | 平均相对身体位置误差 | 跌倒检查 |
| --- | --- | --- | --- | --- |
| `duobox_guard` | 护架闪避：004 / liu | 13.28 s | 3.88 cm | 未触发 |
| `duobox_straight` | 后手直拳：003 / ma | 8.20 s | 3.26 cm | 未触发 |
| `duobox_combo` | 移动组合拳：027 / ma | 37.30 s | 4.53 cm | 未触发 |

每段仅做一次 rollout，采用实际部署 FSM、50 Hz 策略、500 Hz PD 物理循环、无外部支撑。
初始化为站姿 reset，静置 0.2 秒后从动作第 0 帧开始；跌倒判据为根高度 < 0.30 米或
骨盆倾角 > 70 度。`reference_forcing=true`，误差是去根平移后的身体位置误差，
不表示绝对路径、拳速、接触或对打效果都正确。参考末尾保持末帧。

```bash
conda activate scalebfm-native
cd /home/d080/workspace/RoboMimic_Deploy-scalebfm
python deploy_mujoco/deploy_mujoco.py --config-name duobox_guard
# 换为 duobox_straight 或 duobox_combo，分别启动另外两段。
```

进入窗口后 Backspace（或 L1＋START）重置，然后按 R1 开始跟踪。
终端显示进度条、已处理/总帧数和参考动作时间，结束后显示“保持末帧”；
重新启动从 0 开始。交互终端最多每秒刷新 10 次，重定向日志最多每秒一条。
这里的时间是动作进度，不是实际墙钟时间。
配置文件中的动作路径指向当前工作目录的 `outputs/duobox_scalebfm/motions/`；
迁移机器时需复制数据并调整路径。

每段输出保存在 `outputs/duobox_scalebfm/tests/<动作名>/`：

- `tracking.mp4`：实际物理 rollout，绿色点为按机器人根位置对齐的参考；非绝对路径对比。
- `reference.mp4`：完整运动学参考，不能当作物理跟踪成功证据。
- `report.json`、`trace.npz`：结果及逐帧数据；总表为 `outputs/duobox_scalebfm/summary.json`。

转换和评测可复用：

```bash
python tools/prepare_duobox_scalebfm.py \
  --pair /path/to/pair_motion.npz --actor ma --out /path/to/new_motion.npz
python tools/evaluate_scalebfm_motion.py \
  --motion /path/to/new_motion.npz --out /path/to/new_result_directory
```

两条命令均保留已有结果，不覆盖同名输出；评测发现跌倒会提前停止物理 rollout，
仍输出报告及完整运动学参考视频。


### ScaleBFM 部署输入：文件、参考目标与机器人状态

当前链路：DuoBox 人体骨架 → GMR v10 → G1 根位姿和 29 维关节参考角 →
FK → 机器人身体参考位姿 → ScaleBFM → 29 维关节控制目标 → PD → 物理仿真。
GMR 在离线准备时运行，播放时不需要每步重新做 GMR IK。

磁盘 NPZ 同时保存 `joint_pos` / `joint_vel` 和 30 个身体部位的世界系位姿/速度。
当前 FSM 不把文件中的参考关节角直接送到电机，也不把它作为 actor 的目标关节输入；
关节参考用于生成 FK 位姿、文件完整性及运动学参考回放。
策略的任务目标选取其中 14 个 body：骨盆、躯干、双侧肩/肘/腕/髋/膝/踝。
这些目标包含位置和朝向，不是只有 XYZ 的无朝向关键点。

| 策略包装输入 | 单机器人形状 | 来源 |
| --- | --- | --- |
| 根四元数、根角速度历史 | `[1,3,4]`、`[1,3,3]` | 当前机器人状态，真机对应 IMU/状态估计 |
| 关节角/角速度历史 | 各 `[1,3,29]` | 当前机器人编码器状态；不是参考关节角 |
| 上次策略 action 历史 | `[1,3,29]` | 控制器自身缓存 |
| 14 个参考 body 位置/朝向 | `[1,6,14,3]`、`[1,6,14,4]` | 当前帧及未来 5 帧参考，50 Hz 下覆盖 0.1 秒 |
| mode、时间偏移 | mode 7、`[1,6,1]` | 全身跟踪配置与帧偏移 |

文件位姿先做初始朝向对齐，再转到机器人根坐标系；`reference_forcing=true`
时根位置用参考根位置构造目标，这只影响观测，不给机器人施加外力或传送根位置。
包装器还根据机器人当前关节角做 FK，构造身体位姿误差；网络输出 action 经
`default_dof_pos + action_scale * action` 得到关节目标角。实际关节角由物理闭环产生。
所以“文件存关节角”和“策略跟踪身体位姿”同时成立。

后续可以直接提供经过部位映射、比例/坐标轴对齐的身体目标位姿，跳过离线关节 IK；
官方 Xsens 路径就是相关参考。但只有 XYZ 的输入不满足当前 mode 7 的完整位姿接口。
若目标来自在线生成器，还需要提供未来 0.1 秒的目标片段；离线文件已天然具备这些帧。

重定向选择：本轮固定现有 GMR v10，先量化它与 tracker 的组合表现。
ScaleRetarget 当前 `config/retargeter/gmr.yaml` 装配的是 `GmrRetargeter`，
所以它与 GMR 是相关方法，但配置/数据适配与本地 v10 修正并不相同。
UMR 基于学习的人体/机器人表面对应关系，需要对应的人体表面模型与动作适配；
应在相同困难片段、相同 tracker 和初始化下比较，不能仅凭方法更新就认定它更适合拳击。


### DuoBox 15 个单人动作扩大测试

结果位于 `outputs/duobox_scalebfm/batch15_v1/summary.json`，输入溯源在 `motions/*.json`，
实验设置/代码摘要在 `manifest.json`。来自 10 个双人源文件、15 条单人轨迹，每条测试一次。
全部使用同一 GMR v10 + 官方 M checkpoint、FixedPose 重置、0.2 秒等待、无支撑。
本轮不录像，保存每段逐帧轨迹和报告；首批三段录像仍可查看。

15/15 完成，全长合计 266.88 秒；按帧数加权平均相对身体位置误差 4.18 cm。
“完成”只表示未触发既定跌倒/数值异常检查，不代表接触、出拳或地面路径已保真。
速度指标排除首秒，分别取左右手腕相对根部速度的 P95，再计算实际/参考比值；
它是动作速度分布指标，不是严格配对的每一拳峰值速度。
脚滑指标使用法向力 > 1 N 的脚/世界接触点，通过点 Jacobian 计算平面速度，
每帧取所有有效接触点的最大值；表中给出这些接触帧上的均值/P95，不设通过阈值。

| 单人动作 | 时长 s | 相对身体误差 cm | 左/右腕速度比 | 接触点滑速均值/P95 m/s |
| --- | ---: | ---: | --- | --- |
| `000_ma_jab` | 7.32 | 2.73 | 0.98 / 1.01 | 0.110 / 0.450 |
| `002_ma_rear_uppercut` | 8.20 | 3.12 | 0.98 / 0.79 | 0.098 / 0.408 |
| `003_ma_cross` | 8.20 | 3.26 | 0.79 / 0.85 | 0.093 / 0.378 |
| `006_ma_lead_hook` | 14.28 | 4.12 | 0.77 / 0.81 | 0.136 / 0.561 |
| `007_ma_rear_hook` | 12.32 | 5.01 | 0.90 / 0.64 | 0.104 / 0.434 |
| `004_liu_jab_dodge` | 13.28 | 3.88 | 0.84 / 0.90 | 0.074 / 0.318 |
| `006_liu_hook_dodge` | 14.28 | 3.82 | 0.88 / 0.92 | 0.052 / 0.173 |
| `007_liu_rear_hook_dodge` | 12.32 | 4.44 | 0.79 / 0.89 | 0.086 / 0.379 |
| `000_liu_guard` | 7.32 | 2.73 | 0.89 / 1.02 | 0.018 / 0.042 |
| `017_ma_jab_cross` | 16.48 | 3.45 | 0.76 / 0.80 | 0.145 / 0.555 |
| `017_liu_combo_defend` | 16.48 | 4.29 | 0.92 / 0.83 | 0.093 / 0.370 |
| `021_ma_circle` | 32.22 | 5.50 | 1.01 / 1.03 | 0.226 / 1.218 |
| `022_ma_pivot` | 29.58 | 4.03 | 0.74 / 0.82 | 0.125 / 0.478 |
| `027_ma_moving_combo` | 37.30 | 4.53 | 0.78 / 0.78 | 0.171 / 0.705 |
| `027_liu_moving_combo` | 37.30 | 4.02 | 0.83 / 0.86 | 0.144 / 0.544 |

代表问题：007 后手勾拳的右腕速度 P95 约为参考的 0.64 倍；021 转圈出现较大的
承重接触点滑动。这些现象不能单独归因于 GMR；需要把参考运动学质量与 tracker
跟踪损失分开检查。后续 UMR 对照应固定相同片段、机器人、checkpoint 与初始化。

```bash
# 选择本轮任意一个 NPZ，注意使用绝对路径
python deploy_mujoco/deploy_mujoco.py \
  scalebfm.motion=/home/d080/workspace/RoboMimic_Deploy-scalebfm/outputs/duobox_scalebfm/batch15_v1/motions/007_ma_rear_hook.npz

# 重新跑整套评测：使用新的输出目录
python tools/batch_duobox_scalebfm.py --out outputs/duobox_scalebfm/batch15_new
```


### ScaleBFM 真机 C++ bridge 接口

`sr` 使用原六屏布局：左上 C++ 电机通信、左中 Python 策略、左下传感器看板；右侧依次为相机、雷达、球位置融合。命令只预填，按 Enter 才启动。`Ctrl-b` 加方向键切换分屏，`Ctrl-b d` 分离，`sr` 重新连接；分离不停止程序，`rb` 只切换项目目录。

C++ bridge 通过 Unitree C++ SDK 接收 LowState、发送 LowCmd。Python 入口是 `bridge/python/deploy_policy.py`，只通过自定义 DDS bridge 消息接收关节状态、IMU 和手柄，返回 29 维关节目标及 kp/kd，不使用 Unitree Python SDK 驱动电机。不要同时启动旧 `deploy_real/deploy_real.py` 直连入口。C++ 保留命令超时阻尼，Python 保留状态断流阻尼。

真机缺少根位置估计，必须保持 `reference_forcing: true`；控制周期为 0.02 秒，关节数为 29，手部电机不由此策略控制。模型和全部 motion 在策略创建 bridge 订阅之前加载及预热，运行中只允许选择缓存动作。

- 左右选择待播动作；L1/L2 加方向键保留感知偏置功能。
- R1 按下沿启动/重播，允许从 FixedPose、AMP、Loco 或 ScaleBFM 进入。
- A/B：AMP/Loco；START：FixedPose；Y：SMP 恢复；F1：阻尼优先；SELECT：阻尼并退出。
- 初始 Passive 下拒绝直接进入 ScaleBFM，需先进入 FixedPose。

机载目录 `/home/unitree/RoboMimic_Deploy`，分支 `lyd/scalebfm-table-config-balanced4`；ScaleBFM 在 `/home/unitree/ScaleBFM`。沿用 Conda `robomimic`（Python 3.8、NVIDIA PyTorch 2.0 / CUDA 11.4），机载配置保留 `net: eth0` 和对应 ScaleBFM 根路径。`cuda_graph: true`、`torch_threads: 1` 用于预热和推理。

```bash
# 六屏启动入口（在各分屏确认命令后按 Enter）
sr
# 离线检查：不创建 DDS 或发送电机指令
source ~/robomimic-env.sh
python tools/check_scalebfm_onboard.py --steps 100
```

离线检查覆盖 bridge 入口导入、FSM 初始化、16 段动作预加载、CUDA graph/native 输出对比及推理耗时，不包含真实 DDS 收发、日志开销、电机通信或闭环跟踪。

当前分支基于 `codex/table-config-balanced4`（`4678455`）迁入 ScaleBFM；保留 SMP Recovery、Y 键恢复、桌面场景与原日志功能。SMP 状态/指令编号保持 18/20，ScaleBFM 使用 19/21，避免协议编号冲突。

### MuJoCo 实时播放速度

默认物理步长为 2 ms，策略每 10 步运行一次（50 Hz），画面同步独立限制为 60 Hz（`render_hz`）。累计时钟允许推理耗时在后续物理步中补偿；滞后超过 100 ms 时重新对齐，避免长时间卡顿后连续追帧。默认不打印周期性实时倍率信息。ScaleBFM 默认启用 CUDA graph，并将 PyTorch CPU 线程设为 1。修改后需要重启仿真进程。

### ScaleBFM 真机使用 C++ bridge

真机默认入口恢复为 `sr` 的六屏布局：左上 C++ bridge、左中 `bridge/python/deploy_policy.py`、左下看板；右侧为相机、雷达和融合。命令仅预填，按 Enter 才执行。C++ 独占 Unitree LowCmd/LowState 通信，Python 通过自定义 bridge DDS 消息交换状态和关节目标，不使用 Unitree Python SDK 驱动电机。不要启动旧 `deploy_real/deploy_real.py` 直连入口。

策略启动时先预加载 ScaleBFM 与动作库，再等待 bridge 状态。R1 从 FixedPose/AMP/Loco 启动或重播，左右选择待播动作，L1/L2 加方向键仍用于感知偏置，Y 恢复、F1 阻尼保持原有作用。`tools/check_scalebfm_onboard.py` 只做离线加载和推理检查，不创建通信或发送电机命令。

### 浏览器因果响应仿真

`tools/reactive_stream_tracker.py --job <job>` 由 GMR 浏览器 pipeline 启动，只运行 MuJoCo，无 DDS/真机接口。消费增量 GMR qpos 前缀，复用 `prepare_duobox_scalebfm.convert` 生成 50 Hz 机器人位姿参考；保持同一个 ScaleBFM 实例、动作历史和物理状态，仅更新参考缓冲。未生成的未来参考由 MotionTrackingEnv clamp 保持末帧。首帧初始化使用参考姿态，不适用于直接真机启动。物理轨迹输出 `physical_50hz.npz`，页面输出 `pair_motion.npz`，语义和时序限制见 GMR README。

### 响应链路延迟实验（2026-10-13，工作站）

数据与复测统计见 `outputs/reactive_latency_modes/summary.json`。003/004 前 120 人体帧，Ready-to-React 每次生成 4 帧；剔除前两次预测，27 个样本/序列，计时包含 CUDA synchronize 和 agent 更新，不含启动、IPC、感知。串行复测 1 CPU 线程的中位数 57.72/61.98 ms，P95 67.80/75.27 ms；4 线程中位数 76.40/75.70 ms。早期与 CPU retarget 并行的测量波动更大，不作为串行基线。流式 worker 默认改为 1 线程。

单人内存输入的 GMR v10：003 中位 20.22 ms/帧，P95 26.48 ms；004 中位 19.45 ms、P95 38.38 ms（112 样本，排除前 8 帧）。解算后直接提取 14 部位位姿约 0.013 ms，与另一个官方 XML FK 对应位置一致、朝向误差小于 1.3e-7 rad。原型已修正每关节重复解压 NPZ 的问题。

`tools/benchmark_scalebfm_modes.py` 测试全部 8 种掩码，CUDA graph 暖机后各 100 次；推理中位约 0.98–1.08 ms。掩码不缩小网络张量，减少目标部位并不明显加速网络。mode 2/4/7 在相同 003 参考的各一次约 4 秒仿真均完成，目标分别为骨盆+双手、骨盆+双手双脚、全身。

GMR `benchmark_direct_targets.py` 是无 IK 实验：仅做人体骨盆/手部比例缩放和机器人坐标轴偏置，以 mode 2 输入。向量化映射约 0.10 ms/帧，数值对齐完整 GMR 目标预处理。003/004 短仿真均完成，004 最大躯干倾角约 27.5 度。未选部位和关节参考是占位值，输出 NPZ 只能用于此 mode 2 实验；不能当作全身参考使用。它没有 v10 的几何可执行性修正，不是已批准的真机替代方案。不同参考方案的跟踪误差不可直接作为同一目标下的优劣比较。

以上是分段计算与短仿真测试，不是端到端实时截止期、机载性能或长期稳定性验证。部署默认 mode 7 不变，实验通过 `evaluate_scalebfm_motion.py --control-mode` 显式选择。


### 无 IK 的 3 / 5 部位对照（2026-10-13）

`outputs/sparse_comparison_v1/summary.json` 保存 7 段 × 3 方案的实测结果：原始 DuoBox 的 000 护架、003 直拳、007 勾拳、021 转圈（32.22s）、027 移动组合（37.30s），以及已有 R2R 003/004 各 3.98s 输出。同一官方 M checkpoint、FixedPose 起步、50Hz 控制 / 500Hz 物理、无支撑，21 次均完成且未触发跌倒阈值。每段每方案仅一次固定初态测试，不是统计成功率，也没有测实时生成时的缓冲欠载。

直接目标使用 GMR 的人体缩放与坐标轴偏移，但不调用 IK；5 部位额外把脚趾目标按模型固定变换转换成踝部目标。Mode 2 只启用骨盆/双手，mode 4 再启用双脚；二者读取同一个目标文件。文件其余部位和关节角为占位值，禁止按 full-body reference 使用。基线复用现有 GMR v10 + mode 7。没有改变真机默认模式。

相对于统一的人体映射目标，去除首秒、在骨盆 heading 坐标系比较手脚位置；7 段等权平均：3 部位手/脚误差 3.65/20.34cm，5 部位 5.52/13.77cm，GMR14 为 7.43/13.55cm。该指标衡量原动作保留，不是各自 reference 跟踪误差，也不评估肘/胸/头与自碰撞。平均接触脚滑分别 0.084/0.107/0.101m/s。5 部位改善脚步位置但不保证更小脚滑；027 的根部水平路径 RMSE 从 3 部位的 1.21m 降至 5 部位的 0.57m，仍有明显路径漂移。007 的 5 部位路径误差反而更大。视频左/中/右为 direct3/direct5/GMR14，摄像机跟随各自机器人，不能直接从画面判断绝对路径误差。

复现（输出目录必须新建；批量执行可跳过已有完整 report）：
```bash
cd /home/d080/workspace/GMR
.venv/bin/python scripts/prepare_sparse_comparison.py --out /path/to/new_comparison
cd /home/d080/workspace/RoboMimic_Deploy-scalebfm
conda activate scalebfm-native
python tools/run_sparse_comparison.py --manifest /path/to/new_comparison/manifest.json
python tools/summarize_sparse_comparison.py --manifest /path/to/new_comparison/manifest.json --video
```

本轮 checkpoint SHA256：`88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`。三组并排视频为输出目录下 `007_ma_rear_hook_comparison.mp4`、`021_ma_circle_comparison.mp4`、`reactive_004_comparison.mp4`。


### 3 部位墙钟实时仿真（2026-10-13）

入口 `tools/reactive_realtime.py`：预录对手按时间释放历史 → 常驻 Ready-to-React（每次 4 帧，30Hz）→ 人体骨盆/双手的缩放及坐标转换，无 IK → 50Hz ScaleBFM mode 2 → 500Hz MuJoCo。两个常驻进程用本地 Unix socket 传递新块；控制期间不写逐块 NPZ、不等待生成。初始化后按墙钟调度，报告实际实时倍率和每个 20ms 截止时间的超时量；不是硬实时保证。上层仍使用自己生成的身体历史，没有接 G1 实际状态反馈；下层保留 `reference_forcing` 默认配置。

默认参考播放延迟 300ms，未来六步预览不足时钳制到最后可用目标。当前目标不足时继续物理仿真，使用最后目标；缺帧期间新块到达或恢复正常时做 100ms smoothstep 位置/旋转过渡。报告把 current target hold 与 preview shortage 分开；末尾正常 0.5s 保持不计入欠载。`simulation_completed` 与完整生成 `completed` 分开，避免把生成未完成的运行标成通过。

```bash
conda activate scalebfm-native
cd /home/d080/workspace/RoboMimic_Deploy-scalebfm
python tools/reactive_realtime.py --out outputs/my_realtime3 --clip 21 --frames 600
python tools/validate_realtime_run.py outputs/my_realtime3
# 注入一次 400ms 生成延迟：
python tools/reactive_realtime.py --out outputs/my_realtime3_stall --clip 21 --frames 600 --stall-at 120 --stall-ms 400
```

输出目录必须不存在，所有原实验保留。`report.json` 包含逐块生成/首次使用时间、逐控制周期欠载/超时/目标变化；`trace.npz` 是真实 MuJoCo qpos，`human.npz` 是本轮生成及固定对手。延迟统计为对手观测时间戳到目标收到/首次进入策略当前或未来预览的指令时间，不是机器人肢体实际开始响应的延迟。

RTX4090 本地测试：`realtime3_circle200` 为 20s、200ms 缓冲，当前目标欠载 30/999；`realtime3_circle300` 为 300ms 缓冲，1/999；`realtime3_combo300` 为 20s 移动组合，0/999。三者控制超时均为 0，实时倍率约 1.001，均完成且未跌倒。300ms 两个正常运行生成块耗时中位数约 77–85ms，P95 约 140–180ms，长尾不能用平均值代替。

压力测试保留两个结果：初版 `realtime3_circle_stall` 仿真走完但 producer 未结束（旧 report 的 completed 字段未区分此情况），后半段持续落后；不能算生成成功。修正缺帧连续新块的过渡后，`realtime3_stall_fixed` 收齐 600 帧并完成，当前目标保持 63/999、一次控制截止超时约 2.15ms，最大骨盆倾斜 24.6°。调度抖动使两次运行不能视为严格等时 A/B；缓冲并不解决持续吞吐不足，尚无任意长时稳定保证。

浏览器 `http://127.0.0.1:19093/?id=21&mode=reactive_rt3` 新增“实时响应 → 3 部位（无 IK）→ ScaleBFM”。021/027 缓存为上述 20s 正常运行，其他动作点击生成默认最多 10s。页面是已完成实时调度实验的回放，并非浏览器实时推流。橙色 GMR 仅在实验后离线生成/读取用于显示，不进入控制链；无双方物理碰撞。


### 实时四视图网页与对手输入对照

打开 `http://127.0.0.1:19093/live`，选择动作、正常/冻结/延迟对手、4–60s 时长，点击“开始实时运行”。模型加载预热之后边生成边显示，不等待整段结束。四个视图分别是：对手 21 关节骨架（输入时钟）、生成自身 21 关节骨架（带 300ms 延迟的参考时钟）、骨盆/双手目标的位置朝向及实际部位误差、实际 MuJoCo G1。人体视图用原始人体世界坐标，后两者用校准后的机器人坐标；摄像机跟随各自骨盆，可旋转缩放。手部指标同时显示骨盆相对误差和世界误差，不能把整体位置漂移解释成局部姿态误差。

控制进程通过非阻塞 localhost UDP 最多 25Hz 发出最新状态，浏览器服务通过 SSE 推送；慢浏览器只丢中间显示帧，不让控制循环等待。控制截止时间统计包含 telemetry 打包时间。四个视图共享 WebGL renderer，以减少资源竞争。服务器实现在 GMR 的 `scripts/reactive_live.py`，由原 19093 服务加载。网页的停止只终止对应本地实验进程组，不连接 DDS。输出保存在 `outputs/live_runs/<run>/`，日志位于同级 `.log`；完整实验结果重启服务后仍会列出。初次及每次重新开始都需要加载模型，运行期间两模型常驻。

对照模式保持种子 42、初始姿态及前 2s 输入相同。freeze 从 2s 起保持对手姿态；delay 在 2s 起停顿 500ms 再续播，位置和旋转同步变换，并重新计算速度等输入特征。生成模型不接触变换后的未来对手帧。网页表格显示干预前生成骨架最大差异和干预后生成双手世界位置相对正常输入的平均差异（共同可比较时段，排除干预后首个 4 帧块）。这是输入敏感性实验，不是躲闪或反击成功率。

本轮 004 各 8s 测试：normal、freeze、delay 均完成且控制超时为 0；目标欠载分别 0、0、24 帧。干预前最大差异为 0，freeze/delay 干预后手部差异分别 11.41/10.53cm。两组干预的输入位置/旋转变换、相同前缀、因果时间戳和增量重采样检查通过，可用 `tools/validate_realtime_run.py <variant_job> --baseline <normal_job>` 重验。初版四 renderer 的 021 20s 测试出现明显生成欠载；共享 renderer 后的一次 20s 测试欠载/控制超时均为 0，实时倍率 1.001、生成中位数 76.2ms / P95 129.4ms。这是单次本机负载测试，不保证任意浏览器/GPU负载下性能。

### 当前保存节点（2026-10-14）

300ms 是实现中选择的参考缓冲，可通过 `reactive_realtime.py --delay-ms` 调整；网页目前使用默认值。降低它可以减小参考滞后，但可能增加保持末目标及未来预览不足。每块 4 帧覆盖约 133ms，策略还使用 0–100ms 的未来预览，因此参考播放延迟、生成耗时、首次进入策略的延迟和实际肢体响应延迟是不同指标，不应直接相加。上面的 200/300ms 单次对照是选用默认值的依据，尚非最优延迟结论。

本次保存包括实时仿真、3/5/14 部位对照、逐块因果校验与网页遥测。ScaleBFM 的 control_mode 可配置，原默认 mode 7 保持不变；本轮实时实验显式使用 mode 2。没有改变真机部署配置。当前网页尚未叠加录制的原始自身骨架。三个仓库的相关 Python 文件已做语法检查，既有仿真和对照结果见前述记录；保存节点不重复运行整批实验。

### 原始自身叠加（2026-10-14）

实时网页生成自身视图新增默认开启的“叠加灰色原始自身骨架”开关。遥测只为显示读取原始响应演员的位置，按与生成骨架完全相同的 30Hz reference_frame 取样（含源数据首 4 帧偏移）；缺帧和末帧保持时也使用相同索引。保留人体世界位置，不额外平移对齐骨盆；灰色骨架不进入生成或跟踪策略。旧遥测没有该字段时提示重新开始实验。此功能替代上一保存节点中“尚未叠加”的状态。

验证：004 正常输入 8s 实时仿真完成；遥测中 recorded_self_frame 与 reference_frame 相等，原始演员位置与源数据相应帧在五位小数序列化精度内一致。浏览器实际检查开关能显示/隐藏灰色骨架。实验输出为部署仓库 outputs/live_runs/1791947599742892391_004_normal。

本次运行当前目标欠载 38 个控制周期、控制截止超时 0；因果时间戳、物理数值、控制连续性及增量映射/重采样校验通过。不能将本次结果解读为无缓冲欠载。

### AMP 分层恢复策略（60k）

原 AMP 模式支持在 [`policy/amp/config/amp.yaml`](policy/amp/config/amp.yaml) 中仅切换 `model_path`，选择旧模型或新模型 `amp_stratified_guidance_60000.onnx`。新模型内置关节顺序、动作尺度转换和观测归一化；其余配置复用。默认仍为旧模型。详见[部署说明与验证](docs/amp_guidance_60000_deployment_zh.md)。
