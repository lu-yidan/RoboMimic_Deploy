> **当前研究基线（2026-10-29）**：已采用 v2 物理/姿态验收及摆动阶段 8 mm 抬升参考。24 个开发工况的单周期、独立前缀、连续双周期和连续八周期全部通过；八周期共 192 次末段验收通过。每条八周期记录运行 909.28 s，实际累计升高 1.998528–1.998536 m，仅开始重置一次。冻结 PPO，本轮未重训；不代表随机墙面、视觉闭环或真机能力。依据为 [research_baseline.json](research_baseline.json)，详见[正式参考链](docs/baselines/swing_reference_20261029.md)及[连续周期记录](docs/baselines/v2_multicycle_20261029.md)。
>
> **打开当前 v2 在线仿真**：在仓库根目录运行 `.\tools\run_adopted_climb.ps1`。它核对已采用参考、策略及已完成的搬迁复现证据，然后在独立包内运行 T12 八周期的新动力学；加 `-Headless` 无窗口运行，加 `-VerifyOnly` 只检查资产。需要本地仿真包，普通源码克隆不包含全部 `outputs/` 文件。
>
> **只看已经记录的动作**：`.venv\Scripts\python.exe -m deploy_mujoco.replay_verified_climb --run outputs/v2_swing_eight_suite_20261029_v2/training_12 --speed 8`。这是姿态回放，不做在线策略推理。
>
> 当前攀爬实验使用 Python 3.12、MuJoCo 3.3.7，精确依赖以冻结仿真包为准；下方上游安装步骤适用于原部署框架。新增的接触确认加载仍是独立开发候选，尚未替换正式基线。
>
> 可分享在线仿真包：`outputs/portable_v2_live_T12_eight_verified_20261029_v1.zip`。已在本机另一个目录完整重跑 T12 八周期，12 组记录数组逐元素一致。解压后先读外层 `README_先读.txt`；尚未验证全新机器安装或 Linux。该包是本地产出，未声称已经上传到 GitHub。

> **输入通路限制**：当前冻结 PPO 的两组手杆几何输入权重均为零，手杆位置特征尚不能直接改变策略动作。已有物理验证仍有效，但不能视作感知驱动的手杆适应。已取消没有区分能力的“仅遮掉手杆输入”双周期实验；详见[输入通路审计](outputs/rung06_geometry_input_path_20261029_v1/summary.json)。

> 以下为 **2026-10-24 历史阶段和旧入口**：当时使用 148 维观测、26 维残差动作与 QP 控制器组合，单周期约抬升 0.25 m。固定参考修正候选通过原五工况和旧十二项网格；两轮 PPO 未增加通过项；随后固定前后/腰部参考调整已将完整八工况从 5/8 提升到 7/8，旧十二项网格完整回归 12/12，四项审计通过。当时尚未验证连续两米攀爬或真机部署。早期 100 维观测、29 维位置动作的站立接口是另一阶段，不代表当前组合策略。详见[历史实验记录](docs/baselines/closure_waist_body_20261024.md)。
>
> 新 7/8 实验候选可执行 `.\tools\run_reference_closure_sim.ps1`；加 `-Headless` 可无窗口验证。新入口已在最高杆完整复现十组数组，原发布包未替换。
>
> 在仓库根目录执行 `.\tools\run_closure_sim.ps1` 打开当前已验证候选的 MuJoCo 演示；加 `-Headless` 可无窗口记录并核对复现。依赖本地 `.venv` 和 `outputs/` 实验资产，普通源码克隆不包含全部实验文件。该入口只运行仿真。

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

`liuze-climb` 分支新增Windows键盘模式。它使用上述原入口、原场景与完整策略FSM；默认手柄模式不变。需要先检出上游网格和策略权重：

```powershell
git sparse-checkout add '/g1_description/meshes/' '/policy/*/model/'
.\.venv\Scripts\python.exe -m pip install -r requirements-mujoco-desktop.txt
.\.venv\Scripts\python.exe -X utf8 deploy_mujoco\deploy_mujoco.py +keyboard=true hydra.job.chdir=false
```

键盘模式初始暂停。空格运行/暂停；`1`或Enter请求FixedPose；`2`切换Loco；`3`为AMP；`4/5/6`分别请求BeyondMimicMJ、StandUpMJ和Pinocchio；`P`或F1请求Passive；Esc退出。策略切换仍受原FSM规则约束，先从Passive进入FixedPose再选择技能。`W/S、A/D、Q/E`每次调整对应速度指令0.1，`0`清零；`G`切换参考动作显示，`J`打印关节角。`7`请求FreeKick，只适用于含球场景。Python的`-X utf8`用于正确读取带中文的上游配置。

2026-10-09已验证原默认场景与完整策略加载、窗口打开以及键盘事件转换；未据此宣称所有上游技能的运动性能均已验证。该入口不连接机器人，默认场景不是钩爪攀爬场景。

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

## G1 钩爪攀爬：148 维策略仿真入口

`deploy_mujoco.run_geometry_full_cycle` 在本仓库内运行已审计的完整周期策略，支持 blind/oracle 两种额外横杆位置输入。入口校验策略、参考轨迹、源码与场景哈希，并记录全过程状态和验收结果。此入口用于 MuJoCo 仿真实验；oracle 使用仿真横杆位置真值，当前仍依赖预设攀爬轨迹和接触控制器。

以下已验证工况为脚杆升高 7.5 mm、完整周期 113.66 s。独立入口的七组记录数组与原策略评估精确一致，验证文件位于 `outputs/geometry_policy_deploy_replay_v1/deployment_reproduction_audit.json`。

在 PowerShell 中运行：

```powershell
Set-Location D:\G1Climbing\external\RoboMimic_Deploy
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$climbRun = 'outputs/geometry_viewer_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m deploy_mujoco.run_geometry_full_cycle --experiment outputs/highrod_existing_oracle_policy_v1 --case foot_z_plus7.5 --output $climbRun --viewer
```

去掉 `--viewer` 可运行无界面实验。每次使用新输出目录；完成后检查其中的 `summary.json` 和 `trajectory.npz`。本例使用已审计策略，不自动替换为正在训练的新检查点。当前高度误差基准为五个离散工况中三个通过，完整进度见 `docs/baselines/full_cycle_training_20261018.md`。

### 已验证的支撑参考改进（+12.5 mm）

`support_retarget_plan_v1` 调整卸载阶段的支撑参考，五个已知杆高工况的独立验收从 3/5 改善到 4/5；+15 mm 仍终点失败。已有策略迁移后同为 4/5，没有追加训练或额外成功项。新参考下的 +12.5 mm 工况已经在独立部署入口完整执行 113.66 s，七组数组与原评估逐元素一致，见 `outputs/support_policy_deploy_replay_v1/deployment_reproduction_audit.json`。

```powershell
Set-Location D:\G1Climbing\external\RoboMimic_Deploy
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$supportRun = 'outputs/support_viewer_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m deploy_mujoco.run_geometry_full_cycle --experiment outputs/support_reference_policy_transfer_v1 --case foot_z_plus12.5 --output $supportRun --viewer
```

前后对比全过程录像：`outputs/support_retarget_calibration_v1/media12.5/reference_repair_8x.mp4`。这次改善来自参考规划；不代表自主视觉攀爬或真机验证。终点接触位置调整等后续候选只有通过完整验收后才替换此版本。

## 致谢

本部署框架基于 [**ccrpRepo/RoboMimic_Deploy**](https://github.com/ccrpRepo/RoboMimic_Deploy)
构建。感谢原作者开源了这套基于状态切换的多策略部署框架，本仓库在其基础上扩展而来。

本仓库是 [**RoboNaldo**](https://opendrivelab.com/RoboNaldo/) 的部署仓库 ——
训练代码见 [opendrivelab/RoboNaldo](https://github.com/opendrivelab/RoboNaldo)。




### 已审计策略的仿真候选包（2026-10-22）

`outputs/closure_control_simulation_package_v1`包含零附加收尾惩罚续训策略、黑色复合钩爪横杆场景、参考轨迹与所需源码。该策略五个已知工况4/5、十二项网格11/12，网格属于模型选择验证集。两组续训同分，已按冻结规则选定本策略，详见`outputs/closure_policy_selection_v1/selection.json`。包只开放已审计的`foot_x+5_z13.75`演示工况；包内完整113.66秒复现七数组精确一致。

可搬移ZIP为`outputs/closure_control_simulation_candidate_v1.zip`（约16.4MB）。归档保留原候选标签与哈希，选模接纳记录单独保存在`outputs/closure_control_package_selection_receipt_v1.json`。全部99个ZIP文件已解码核对，解压后的场景加载也已验证。

在本机 PowerShell 使用现有虚拟环境：

```powershell
Set-Location D:\G1Climbing\external\RoboMimic_Deploy
$ClimbPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location .\outputs\closure_control_simulation_package_v1
& $ClimbPython -m deploy_mujoco.run_packaged_geometry --verify-only
& $ClimbPython -m deploy_mujoco.run_packaged_geometry --output user_viewer_replay --viewer
```

输出目录须不存在；需要再次运行时换一个目录名。去掉`--viewer`可进行无界面完整复现。运行入口核对包内文件、主要依赖版本，并将完整状态、动作、观测、力矩和奖励与审计轨迹精确比较。窗口提前关闭不算完整通过。

这是MuJoCo仿真包，使用参考轨迹、真实接触反馈与真实杆偏移输入；不连接真机，也不代表视觉自主攀爬。依赖文件记录当前环境，尚未验证另一台全新机器的联网安装。


### GitHub 下载后的仿真复现

选定策略、场景、运行源码及单案例完整审计已打包至 [仿真快照与复现说明](artifacts/climb_simulation_20261022/README.md)。首次克隆请按该说明解压并安装运行依赖；历史报告中的 `outputs/` 路径并未全部上传。
