<div align="center">
  <h1 align="center">RoboMimic Deploy</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>RoboMimic Deploy 是一个基于状态切换机制的机器人多策略部署框架，目前包含的策略适用于宇树G1机器人(29dof)</strong> 🚪🎮
</p>

## 写在前面

- **本部署框架仅适用于具有三自由度腰部的G1机器人，如果装有腰部固定件的话需要按照官网教程解锁，然后才能正常使用该部署框架。**

- **建议拆下手掌，舞蹈动作会存在干涉**

- **实际机器人部署中出现的问题，十有八九是策略适应性不足所致，大家不必过度怀疑硬件层面的缺陷。**

- **[视频教程](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)**

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

### 2.2 安装 RoboMimic_Deploy

#### 2.2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/ccrpRepo/RoboMimic_Deploy.git
```

#### 2.2.2 安装组件

进入目录并安装：

```bash
cd RoboMimic_Deploy
pip install -r requirements.txt
```
#### 2.2.3 安装unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```
---
## 运行代码

## 1. 运行Mujoco仿真代码

标准场景：
```bash
python deploy_mujoco/deploy_mujoco.py
```

Score 场景（含球，运行 Score 策略时使用）：
```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_score
```

---
## 2. Policy 说明

| 模式名称            | 触发按键          | 描述                                                                 |
|--------------------|------------------|----------------------------------------------------------------------|
| **PassiveMode**    | L1               | 阻尼保护模式                                                         |
| **FixedPose**      | Start            | 位控恢复至默认关节值                                                 |
| **LocoMode**       | B                | 用于稳定行走的控制模式                                               |
| **AMP**            | A                | AMP 运动策略（默认进入 run 模式）                                    |
| **Score**          | R1               | 踢球得分策略（547维obs，5帧历史，需配合机载雷达感知）               |
| **BeyondMimicMJ**  | D-pad DOWN       | 摔倒爬起模仿策略（MuJoCo 训练版，含参考动作跟踪）                   |
| **StandUpMJ**      | D-pad UP         | 站起模仿策略（MuJoCo 训练版，含参考动作跟踪）                       |
| **Pinocchio1.6MJ** | R2               | `g1_result_pinocchio_1_6_mj.yaml` 对应的 MuJoCo 模仿策略            |
| **BeyondMimic**    | —                | 仍保留在仓库中，但当前默认不分配手柄按键                             |

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
   - `R1` -> Score

6. `Score` 需使用 `--config-name mujoco_score` 加载含球的仿真场景；`BeyondMimicMJ / StandUpMJ / Pinocchio1.6MJ` 之间支持直接互切

7. 任意时刻可按 `L1` 进入阻尼保护模式，按 `Start` 返回 FixedPose

8. `BeyondMimic` 仍保留在仓库中，但默认不再绑定手柄按键；`Dance / Kick / KungFu / ASAP / HOST` 等旧策略已经从仓库中删除

---
## 4. 真机操作说明
1. 开机后将机器人吊起来，按L2+R2进入调试模式

2. 运行deploy_real程序：
```bash
python deploy_real/deploy_real.py
```
3. Start键进入位控模式

4. 后续单键切换与仿真保持一致：`A / B / D-pad DOWN / D-pad UP / R2 / R1`

5. **Score踢球策略（R1）真机部署额外步骤**：Score策略依赖机载雷达对球的实时感知，需在运行deploy_real之前启动感知服务，并通过DDS topic `rt/ball_state` 发布球的位置信息。在不具备感知服务的情况下，请勿在真机上启动Score策略。

6. 如果机器人搭载的是 Orin/G1 机载平台，建议优先使用 `bridge/` 目录下的桥接部署方案，而不是直接运行 `deploy_real.py`。桥接方案说明见：
   - `bridge/README_zh.md`
   - `bridge/VALIDATION.md`

   过渡验证版：
   ```bash
   python bridge/python/deploy_bridge_py.py
   python bridge/python/deploy_policy.py
   ```

   C++ bridge 版：
   ```bash
   cmake -S bridge -B bridge/build
   cmake --build bridge/build -j2
   BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main
   python bridge/python/deploy_policy.py
   ```

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
当前部署入口已简化为 `LocoMode / AMP / Score / BeyondMimicMJ / StandUpMJ / Pinocchio1.6MJ`。

⚠️ **重要注意事项**：
- **建议拆除手掌**：部分策略的原始训练未考虑手掌碰撞（作者的G1初始无手掌）
- **启动与切换阶段**：策略切换前后建议优先回到 **FixedPose** 或 **PassiveMode**
- **人工保护**：首次在真机验证某个保留策略时，建议全程吊挂并保留人工保护

### 4. 其他动作建议
仓库中现在只保留当前部署会用到的策略实现，其余历史策略已删除，以减少误用和维护成本。

### 5. 强烈建议
**务必**先在仿真环境中熟练操作，再尝试真机部署。



