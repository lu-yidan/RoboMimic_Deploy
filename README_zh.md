# RoboMimic Deploy · lyd-deploy

[English](README.md) | 中文

宇树 G1（29 自由度）的多策略部署仓库，支持 MuJoCo 仿真、Python 直连和 C++ SDK2 bridge 真机部署。`lyd-deploy` 整合了桌下/箱内恢复、AMP 策略切换和可选 ScaleBFM 跟踪；三种入口共用策略选择配置。

## 目录

- [1. 安装与更新](#1-安装与更新)
- [2. 配置文件分工](#2-配置文件分工)
- [3. MuJoCo 场景与参数](#3-mujoco-场景与参数)
- [4. 策略选择](#4-策略选择)
- [5. 手柄、键盘与重置](#5-手柄键盘与重置)
- [6. ScaleBFM 模型与动作](#6-scalebfm-模型与动作)
- [7. 真机部署](#7-真机部署)
- [8. 日志与离线验证](#8-日志与离线验证)
- [9. 常见问题](#9-常见问题)
- [10. 文档索引与来源](#10-文档索引与来源)

## 1. 安装与更新

### 1.1 获取仓库

```bash
git clone --branch lyd-deploy git@github.com:lu-yidan/RoboMimic_Deploy.git
cd RoboMimic_Deploy
```

上面保留所有远端分支的跟踪。只为部署创建小副本时，可自行添加 `--single-branch --depth 1`。本分支从整合后的快照开始；旧 clone 切换分支不会自动释放旧 Git 对象。

已有仓库更新：

```bash
git fetch origin --prune
git switch lyd-deploy
git pull --ff-only
```

以下命令均从仓库根目录运行。先处理自己的未提交修改，再切换分支。

### 1.2 工作站环境

已有 `robomimic` 环境可直接激活。新建环境可按仓库原有依赖组合安装：

```bash
conda create -n robomimic python=3.8.20 -y
conda activate robomimic
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

- `numpy==1.24.4` 与仓库现有接口匹配。
- ONNX 模型要求支持 opset 18 的运行时；Python 3.8 可使用 `onnxruntime==1.19.2`。
- 交互仿真需要图形桌面及 pygame 可识别的手柄；无需手柄的检查见第 8 节。
- `requirements.txt` 包含 Python DDS 依赖。CycloneDDS 系统库、SDK 和机载安装按 [安装指南](onboard/docs/INSTALL_AGENT_GUIDE.md) 配置。
- [环境清单](environment/smp_recovery_mujoco.yml) 记录了 SMP/MuJoCo 依赖；ScaleBFM 另需第 6 节的外部源码、权重和可选依赖。

机载 Jetson 的 PyTorch/CUDA 必须匹配其 JetPack，不要直接照搬工作站的 CUDA 安装命令。当前 ScaleBFM 集成在 Python 3.8 / PyTorch 2.3.1 / RTX 4070 Ti SUPER 上验证过；机载环境需分别检查。

## 2. 配置文件分工

| 文件 | 控制内容 | 适用入口 |
|---|---|---|
| [`config/deploy.yaml`](config/deploy.yaml) | ScaleBFM 总开关及模型目录、AMP/SMP profile | 仿真、Python 真机、bridge 策略进程 |
| [`deploy_mujoco/config/mujoco.yaml`](deploy_mujoco/config/mujoco.yaml) | 基础 XML、物理/控制周期、渲染、日志、ghost | MuJoCo |
| `deploy_mujoco/config/mujoco_table.yaml` | 桌子几何及仰卧初态 | MuJoCo 桌子场景 |
| `deploy_mujoco/config/mujoco_coffin.yaml` | 箱体、盖子、质量、初态、相机 | MuJoCo 箱体场景 |
| `deploy_mujoco/config/mujoco_freekick.yaml` | 含球 XML、球重置位置/速度 | MuJoCo 含球场景 |
| [`deploy_real/config/real.yaml`](deploy_real/config/real.yaml) | 网卡、关节顺序、控制周期、真机日志 | Python 真机、bridge 策略进程 |
| `policy/<策略>/config/*.yaml` | 模型映射、观测、PD、动作缩放、力矩限制 | 对应策略 |

**日常选择策略改 `config/deploy.yaml`；选择场景改 `--config-name`；改变场景尺寸可覆盖命令行参数。** 修改配置后重启进程，当前不支持运行中重新加载 YAML。

配置优先级：

1. `LYD_DEPLOY_CONFIG` 指定的完整文件替代默认 `config/deploy.yaml`。
2. SMP 的兼容环境变量 `SMP_RECOVERY_PROFILE` 优先于文件中的 `smp.profile`。
3. MuJoCo 的 `scalebfm.motion` 可覆盖共享配置中的动作路径；`null` 表示沿用共享配置。它不会打开 ScaleBFM。
4. 场景参数使用 Hydra 合成：基础配置 → 场景 YAML → 命令行覆盖。已有字段用 `key=value`，新增字段用 `+key=value`。

`scalebfm.enabled`、`root` 等选项放在共享配置中，不作为 MuJoCo 命令行参数传入。

## 3. MuJoCo 场景与参数

### 3.1 场景选择速查

| 场景 | 启动参数 | 说明 |
|---|---|---|
| 普通场地 | 不加参数，或 `--config-name mujoco` | `g1_description/scene.xml`；推荐的 ScaleBFM 测试资产 |
| 桌下恢复 | `--config-name mujoco_table` | 添加静态四腿桌；默认仰卧、Passive 启动 |
| 箱体和盖子 | `--config-name mujoco_coffin` | `g1_liao.xml`；默认自由箱体和独立自由盖子 |
| 含球场地 | `--config-name mujoco_freekick` | 加载球与球重置参数；不会重新启用 FreeKick 策略 |
| DuoBox 动作预设 | `--config-name duobox_straight` / `duobox_guard` / `duobox_combo` | 普通场地上的参考动作选择；需要另外生成动作文件 |
| 压板 / 桌子加压板 | `scene.name=plate` / `table_and_plate` | 构建器支持；没有独立命名 YAML，见 3.4 |

```bash
# 普通场地
python deploy_mujoco/deploy_mujoco.py

# 桌下仰卧；按 Y 才开始所选 SMP 策略
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table

# 箱体 + 盖子
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin

# 含球场地
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_freekick
```

只查看 Hydra 合成结果，不打开窗口、不加载策略：

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin --cfg job
```

该输出只显示 MuJoCo 配置；共享策略选择仍以 `config/deploy.yaml` 为准。

### 3.2 桌子：尺寸、位置与初态

默认桌子长 **1.6 m**、宽 **0.6 m**、桌面上表面高 **0.4 m**、板厚 **0.04 m**，因此桌面下净空为 **0.36 m**。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  scene.table.length=1.8 scene.table.width=0.8 \
  scene.table.height=0.5 scene.table.thickness=0.04 \
  'scene.table.pos=[0.2,0.0,0.0]' \
  'initial_state.preset_root_pos=[-0.1,0.1,0.12]' \
  initial_state.preset_yaw_deg=30
```

| 字段（`scene.table.*`） | 默认值 | 含义 |
|---|---|---|
| `pos` | `[0,0,0]` | 桌子水平中心和底部基准的世界坐标，m |
| `length` / `width` | `1.6` / `0.6` | 完整长宽，m，不是半尺寸 |
| `height` | `0.4` | 桌面上表面相对 `pos.z` 的高度 |
| `thickness` | `0.04` | 桌板厚度；净空为 `height - thickness` |
| `leg_width` / `leg_inset` | `0.05` / `0.0` | 桌腿宽度 / 从桌边向内收的距离 |
| `friction` | `[1.2,0.01,0.001]` | MuJoCo 滑动、扭转、滚动摩擦参数 |
| `rgba` | `[0.53,0.32,0.15,1]` | 颜色与不透明度 |

桌子固定在世界系。`height` 必须大于板厚，桌腿不能互相重叠。旧的 `top_size`、`leg_size` 半尺寸配置不再接受。更完整说明见 [桌子配置](docs/table_simulation_config_zh.md)。

### 3.3 箱体：固定/自由、盖子、质量、透明度

```bash
# 固定箱体，无盖
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin \
  scene.coffin.free=false scene.coffin.lid_enabled=false

# 自由箱体，独立盖子；调轻并半透明显示
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin \
  scene.coffin.mass=10 scene.coffin.lid_mass=2 \
  'scene.coffin.rgba=[0.24,0.10,0.05,0.3]' \
  'scene.coffin.lid_rgba=[0.32,0.12,0.045,0.3]'
```

| 字段（`scene.coffin.*`） | 默认值 | 含义 |
|---|---|---|
| `length` / `width` | `1.4` / `0.6` | 箱内完整长宽，m |
| `height` | `0.4` | 从箱底基准到侧壁顶的高度，含底板、不含盖子 |
| `base_thickness` / `wall_thickness` | `0.04` / `0.035` | 底板 / 侧壁厚度，m |
| `pos` / `yaw_deg` | `[0.14373179,0.000005,0]` / `0` | 箱底基准位置 / 绕竖直轴旋转角度 |
| `free` | `true` | 箱体是否拥有自由关节；`false` 时固定 |
| `mass` | `15.0` | 箱体质量，kg，不含盖子 |
| `lid_enabled` | `true` | 是否生成盖子 |
| `lid_thickness` / `lid_mass` | `0.03` / `3.0` | 盖子厚度（m）/ 质量（kg） |
| `friction` | `[1,0.01,0.001]` | 箱体和盖子的摩擦参数 |
| `rgba` / `lid_rgba` | 见 YAML | 箱体 / 盖子颜色；第四项为不透明度 |

盖子是独立自由物体，不是铰接门；`free=false` 只固定箱体，不固定盖子。箱内竖直空间为 `height - base_thickness`。箱体场景使用的 `g1_liao.xml` 没有通过当前 ScaleBFM squat 兼容性验证，优先用于 SMP 实验。参见 [箱体配置](docs/coffin_simulation_config_zh.md)。

### 3.4 压板及桌子加压板

构建器支持沿 Z 轴滑动的压板，以下命令使用现有桌子 YAML 的初态：

```bash
# 只有压板，不生成桌子
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  scene.name=plate '+scene.plate.mass=8.0' \
  '+scene.plate.pos=[0,0,0.8]' '+scene.plate.size=[0.45,0.32,0.035]' \
  '+initial_state.plate_surface_top_z=0.4'

# 桌子和压板同时存在
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  scene.name=table_and_plate '+scene.plate.mass=8.0'
```

`scene.plate.size` 是 MuJoCo 的**半尺寸**，与桌子完整长宽的约定不同。默认 `slide_range=[-1.2,0]`，相对 `plate.pos` 沿 Z 轴运动。`initial_state.plate_slide_qpos` 可直接设置滑动关节位置；若同时设置 `plate_surface_top_z`，后者最终调整板的上表面高度。几何可生成不代表初态无接触穿透，也不代表恢复策略已在该组合上验证。

### 3.5 初态、启动策略、相机和运行频率

| 配置 | 含义 |
|---|---|
| `xml_path` | 相对仓库根目录的机器人/场景 XML；默认 `g1_description/scene.xml` |
| `initial_state.preset` | `none`、`plate_supine_default_arms`、`plate_supine_crossed_arms` |
| `initial_state.preset_root_pos` | 预设根位置 `[x,y,z]`，m |
| `initial_state.preset_yaw_deg` | 预设 yaw，度 |
| `initial_state.qpos` / `qvel` | 可选完整状态数组，长度必须分别等于模型 `nq` / `nv`；覆盖预设，可包含球/箱/盖子自由度 |
| `startup_policy` | `passive`（默认）、`fixed_pose`、`smp_recovery`；只决定启动策略 |
| `camera.lookat` / `distance` / `azimuth` / `elevation` | 视线目标、距离（m）、方位角和仰角（度） |
| `simulation_dt` | 物理步长，默认 `0.002` s，即 500 Hz |
| `control_decimation` | 每多少个物理步运行策略，默认 `10`，即 50 Hz |
| `render_hz` | 画面同步频率，默认 60；不改变控制周期 |
| `torch_threads` | PyTorch CPU 线程数，默认 1 |
| `ghost_enabled` / `ghost_rgba` | 有参考 ghost 输出的策略的显示开关与颜色；不是物理支撑 |

```bash
# 箱体相机与显示参数
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin \
  camera.distance=4 camera.azimuth=90 camera.elevation=-30 \
  render_hz=30 ghost_enabled=false

# 桌子场景启动即进入 SMP（会立即运行恢复策略）
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  startup_policy=smp_recovery

# 普通场地新增相机配置；无程序化障碍
python deploy_mujoco/deploy_mujoco.py \
  '+camera.lookat=[0,0,0.5]' +camera.distance=3
```

这些部署策略按 50 Hz 配置。不要仅为加快画面而改变物理/控制周期；可以单独调整 `render_hz`。修改初态时注意箱体/盖子与机器人的相对位置。

### 3.6 含球场景与自定义配置

`mujoco_freekick` 包含 `ball_reset_pos_w`、`ball_reset_vel_w`、`ball_reset_quat_w`（wxyz）和 `ball_reset_ang_vel_w`。按 X 松开后按这些值重置球；默认位置 `[3,0,0.115]`、速度 `[-2.5,0,0]`。`ball_sensor_hz=10` 控制仿真球观测采样率。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_freekick \
  'ball_reset_pos_w=[2,0,0.115]' 'ball_reset_vel_w=[-1,0,0]'
```

当前入口使用 `enable_freekick=False`，所以含球场景不等于 FreeKick 已启用，R1 仍用于 ScaleBFM。

长期使用的一组参数可保存为 `deploy_mujoco/config/mujoco_my_table.yaml`：

```yaml
defaults:
  - mujoco_table
  - _self_
scene:
  table:
    height: 0.5
    width: 0.8
initial_state:
  preset_yaw_deg: 30
```

然后使用 `--config-name mujoco_my_table`。`_self_` 放在最后，使当前文件覆盖基础配置。

## 4. 策略选择

编辑 `config/deploy.yaml` 的相应段，不要把下列片段覆盖成整个文件：

```yaml
amp:
  profile: amp_0309_1
smp:
  profile: v34_balanced4_hard19000
```

| 类型 | profile | 内容 |
|---|---|---|
| AMP | `amp_0309_1` | 原部署 AMP，默认选择 |
| AMP | `stratified_guidance_60000` | 分层 reset、双判别器、额外恢复引导奖励，60k |
| AMP | `stratified_single_60000` | 分层 reset、单判别器，60k |
| SMP | `path_a6_9999` | A6 9999，保留其原始控制合约 |
| SMP | `a6_no_alpha_9999` | A6 no-alpha 9999 |
| SMP | `v34_balanced4_hard19000` | balanced4 hard-reset 19000，默认选择 |

AMP 导出模型已封装观测归一化、关节重排和动作尺度转换，三个 profile 使用相同的 384 维输入和 29 维输出接口。无需再改 `action_scale` 或关节映射。SMP 使用模型内归一化的 93 维输入，balanced4 保留额定 90% 力矩限制，两个 A6 保留原限制。

其他可用模式仍包括 Passive、FixedPose、LocoMode、BeyondMimicMJ、StandUpMJ 和 Pinocchio1.6MJ；按键见第 5 节。profile 只选择对应模式加载哪个模型，**不会自动进入该模式**。

临时选择 SMP（不改文件）：

```bash
SMP_RECOVERY_PROFILE=path_a6_9999 \
  python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table
```

若文件中的 SMP 选择似乎不生效，执行 `unset SMP_RECOVERY_PROFILE`，再查看启动日志中的 profile 和模型哈希。

## 5. 手柄、键盘与重置

### 5.1 策略和辅助操作

| 操作 | MuJoCo 手柄 | 真机遥控器 |
|---|---|---|
| FixedPose | START 松开 | START |
| LocoMode / AMP | B / A 松开 | B / A |
| SMP 恢复 | Y 按下，不按 L1 | Y 按下沿 |
| ScaleBFM 播放/重播 | R1 松开 | R1 按下沿 |
| 选择上一段/下一段 motion | 左 / 右 | 左 / 右；不按 L1/L2 |
| BeyondMimicMJ / StandUpMJ | 下 / 上 | 下 / 上 |
| Pinocchio1.6MJ | R2 按下 | R2 |
| 阻尼 Passive | L2 按下后松开 | F1 |
| 退出程序 | SELECT | SELECT：阻尼并退出 |
| 参考 ghost 开关 | L3 松开 | 不适用 |
| 重置球 | X 松开，场景必须有球 | 不适用 |
| 打印关节位置 | L1 + Y 松开 | 不适用 |

ScaleBFM 关闭时 R1 不启动它。开启时允许从 FixedPose、Loco、AMP 或 ScaleBFM 自身进入，Passive 下直接进入会被拒绝。Y 可抢占其他策略，阻尼命令优先于恢复。

仿真摇杆输入：左摇杆控制前后/左右命令，右摇杆水平轴控制转向；实际速度范围由当前策略定义。AMP 默认进入 fast/run 命令范围。

### 5.2 三种姿态操作不能混用

| 操作 | 对物理状态的影响 | 结束状态 |
|---|---|---|
| 单独 START | 不修改根位置或速度，仅插值到默认关节目标；不保证倒地起身 | FixedPose |
| 窗口键盘 R | 恢复场景 YAML 指定初态，并重置球/箱/盖子等模型状态 | Passive；按 Y 再恢复 |
| Backspace | 仿真重置站立姿态、清零速度，保留控制器历史和进度 | 保持当前策略 |
| L1 + START | 仿真重置站立姿态并切换默认姿态控制 | FixedPose，适合随后测试 ScaleBFM |

键盘操作要求 MuJoCo 窗口获得焦点。后两种是仿真便利操作，真机没有“瞬移复位”。普通场景测试 ScaleBFM 可依次使用 **L1+START → R1**；桌下/箱内恢复可使用 **R → Y**。

### 5.3 手柄映射和感知偏置

```bash
python tools/joystick_test.py
# 例：覆盖 START 的物理按钮索引
JOYSTICK_START=7 python deploy_mujoco/deploy_mujoco.py
```

按钮索引可用 `JOYSTICK_<按钮名>` 环境变量覆盖；扳机轴与方向帽的处理见 `common/joystick.py` 和仿真入口。

| 组合键 | 效果 |
|---|---|
| L1 + 左 / 右 | 目标 Y 偏置 ±0.05 m，仿真和真机 bridge 支持 |
| L2 + 左 / 右 | 球 Y 偏置 ±0.05 m，真机 bridge 支持 |

偏置限制 ±1.5 m，pelvis 坐标系 +Y 指向机器人左侧；组合键不会在真机切换 motion。Sensor Dashboard 可显示偏置值。

## 6. ScaleBFM 模型与动作

### 6.1 下载与开关

```bash
python tools/setup_scalebfm.py
pip install -r environment/scalebfm-requirements.txt
```

脚本下载固定版本官方源码、M 型原始权重、metadata 和 mode table，校验 SHA-256。默认安装在仓库旁的 `ScaleBFM/`，不会写入部署仓库。已有目录若版本或哈希不匹配，脚本会报错，不会覆盖。

编辑 `config/deploy.yaml`：

```yaml
scalebfm:
  enabled: true
  root: ~/workspace/ScaleBFM       # 改为 setup 脚本输出的实际目录
  checkpoint: null                # 官方 M model_22200.pt
  motion: null                    # 官方 squat.npz
  reference_forcing: true
  cuda_graph: true
  motion_globs: []                # 或列出自己的动作 glob
```

`enabled: false` 时不导入外部 ScaleBridge、不加载 BFM 权重。当前后端用官方导出器定义运行 PyTorch，不要求 TensorRT。自定义 checkpoint 不能随意换成其他模型尺寸：当前 metadata、mode table 和验证范围对应官方 M 模型。

`reference_forcing=true` 使用参考根位置构造观测，不是给 MuJoCo 施加支撑力或强行移动机器人。真机没有根位置估计，因此保持 `true`；不能据此声称绝对位置跟踪已验证。

### 6.2 选择动作与 DuoBox

默认 `squat.npz` 为 417 帧、50 Hz，约 8.34 秒；到末尾保持最后一帧。`motion_globs` 与当前 motion 组成启动时收集的动作列表，新增文件后重启。真机只切换预加载缓存动作。

```bash
# 只覆盖本次仿真的 motion；开关仍由 config/deploy.yaml 控制
python deploy_mujoco/deploy_mujoco.py \
  scalebfm.motion=/absolute/path/to/motion.npz

# 已另外生成 DuoBox 文件时，选择预设
python deploy_mujoco/deploy_mujoco.py --config-name duobox_straight
```

| 预设 | 相对仓库根目录的 motion 文件 |
|---|---|
| `duobox_straight` | `outputs/duobox_scalebfm/motions/straight_003_ma.npz` |
| `duobox_guard` | `outputs/duobox_scalebfm/motions/guard_dodge_004_liu.npz` |
| `duobox_combo` | `outputs/duobox_scalebfm/motions/moving_combo_027_ma.npz` |

这些 DuoBox 输出**不随 Git 分发**，不能把预设名称理解为动作数据已下载。也可在预设后追加 `scalebfm.motion=/实际文件.npz`。共享 `motion` 与场景 motion 中的相对路径均按仓库根目录解释。

自己的动作必须是匹配 checkpoint 顺序的 G1 29 DOF、50 Hz NPZ，不能直接用 SMPL 或原始 GMR qpos。数据字段、准备流程、模型布局和进阶工具说明见 [ScaleBFM 使用手册](docs/scalebfm_deployment_zh.md)。

## 7. 真机部署

### 7.1 推荐：C++ bridge + Python 策略

在机载 [安装指南](onboard/docs/INSTALL_AGENT_GUIDE.md) 完成 SDK、DDS、网络和环境配置后：

```bash
cmake -S bridge -B bridge/build
cmake --build bridge/build -j2

# 终端 1：eth0 替换为实际机器人网卡
BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main

# 终端 2
python bridge/python/deploy_policy.py
```

C++ 独占 LowState/LowCmd；Python 经 bridge DDS 收发状态、关节目标及 PD 参数。不要同时启动直连入口或其他 LowCmd 控制器。首次实机测试沿用机载文档的吊挂、调试模式和人工保护流程，F1 阻尼优先。

`deploy_real/config/real.yaml` 中配置 `net`、`control_dt=0.02`、`num_joints=29` 和日志。网卡选择优先级为 `BRIDGE_NETWORK_INTERFACE` → `ROBOT_IFACE` → YAML `net`。bridge 默认状态过期门限 200 ms、命令超时 100 ms；实际通信配置与验证步骤见 [bridge 说明](bridge/README_zh.md) 和 [验证指南](bridge/VALIDATION.md)。

ScaleBFM 启用时，先执行 `python tools/check_scalebfm_onboard.py --steps 100` 做无 DDS 检查，再按机载流程验证；这项检查不会发送电机命令。

### 7.2 tmux、直连入口与感知

```bash
bash tools/start_tmux_layout.sh
```

六窗格预填 C++ bridge、Python 策略、Dashboard、相机、雷达、融合服务命令；逐个确认后按 Enter 启动。`Ctrl-b` 加方向键切换窗格，`Ctrl-b d` 分离，分离不停止进程。`sr`/`rb` 仅在已配置相应 shell 别名时可用。

Python 直连入口仍保留：`python deploy_real/deploy_real.py`；它需要 Unitree Python SDK，不与 C++ bridge 同时运行。另有 `bridge/python/deploy_bridge_py.py` 作为过渡 bridge 原型。

感知和看板沿用 [机载说明](onboard/docs/README.md)。最终 `rt/ball_state` 应由球融合器发布，避免多个发布者；Dashboard 默认地址为 `http://<机器人IP>:8091/`。当前 FreeKick 未启用，启动这些服务不会改变 R1 的 ScaleBFM 映射。

## 8. 日志与离线验证

### 8.1 记录和查看日志

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  logging.enabled=true logging.tag=table_a6 \
  'logging.states=[SKILL_SMP_RECOVERY,SKILL_AMP]'
```

`logging.log_dir` 默认 `logs`；`states` 按 FSM 状态过滤，`tag` 只是文件标签，不选择策略。真机修改 `deploy_real/config/real.yaml` 的 `logging` 段。SMP 日志包含实际 profile、模型哈希、93 维观测、原始/限幅动作、关节目标及力矩限制等。

参见 [日志字段与审计](docs/logging.md)。`tools/playback_log.py`、`tools/audit_recovery_log.py`、`tools/render_recovery_trace.py` 提供回放/审计功能，具体输入参数用各脚本 `--help` 查看。

### 8.2 无手柄、无电机通信的检查

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/test_scalebfm_real.py

# 下列两项需要外部 ScaleBFM 已安装且共享配置 enabled=true
python tools/check_scalebfm_onboard.py --steps 100
python tools/smoke_scalebfm_fsm.py
```

测试覆盖 AMP 训练轨迹一致性、SMP 力矩限制、场景几何、配置选择、日志及状态切换。ScaleBFM smoke 使用实际 MuJoCo/PD/FSM 验证 squat、重置、重入和 AMP/Loco 往返切换。测试结果及限制见 [整合验证记录](docs/validation/lyd_deploy_20261025.json)。这些结果不等于真机验证，也不保证所有策略都能完成桌下或箱内任务。

## 9. 常见问题

| 现象 | 检查方法 |
|---|---|
| 修改共享策略配置后没变化 | 重启进程；核对 `LYD_DEPLOY_CONFIG`、`SMP_RECOVERY_PROFILE` 和启动日志 |
| `Could not override ...` | 确认选对 `--config-name`；已有字段不用 `+`，该配置中不存在的字段用 `+` |
| ScaleBFM 开启后找不到源码/模型 | 运行 setup 脚本；核对 `root`、固定源码版本和文件哈希 |
| DuoBox preset 报文件不存在 | 输出数据未随仓库分发，生成文件或覆盖 `scalebfm.motion` |
| R1 没反应 | 核对 `enabled`；先 START 进入 FixedPose，仿真可用 Backspace 复位站立 |
| START 后机器人没有起身 | START 只插值关节姿态；恢复用 Y；仿真站立重置用 Backspace |
| 键盘 R 后又躺回原位 | 正常：R 恢复配置初态；它与 Backspace 的站立复位不同 |
| 场景显示慢 | 降低 `render_hz`；保持 50 Hz 策略周期；检查 GPU/CPU 推理耗时 |
| `ghost_enabled=false` 后还有彩色点 | ghost 模型与策略的调试标记是不同显示，开关只控制 ghost |
| 手柄按钮错位或无响应 | 先运行 `tools/joystick_test.py`，核对按钮索引、扳机轴和方向帽 |
| 含球场景不能用 R1 踢球 | 当前入口禁用 FreeKick，R1 是 ScaleBFM |

## 10. 文档索引与来源

| 内容 | 文档 |
|---|---|
| 桌子 / 箱体参数 | [桌子](docs/table_simulation_config_zh.md)、[箱体](docs/coffin_simulation_config_zh.md) |
| ScaleBFM 数据接口与工具 | [ScaleBFM 使用手册](docs/scalebfm_deployment_zh.md) |
| AMP 模型来源与结果 | [guidance 60k](docs/amp_guidance_60000_deployment_zh.md)、[single 60k](docs/amp_stratified_single_60000_deployment_zh.md) |
| SMP 模型元数据 | [三个模型及 manifest](policy/smp_recovery/model) |
| 机载安装 / 运行架构 | [安装指南](onboard/docs/INSTALL_AGENT_GUIDE.md)、[运行说明](onboard/docs/README.md) |
| C++ bridge | [说明](bridge/README_zh.md)、[验证](bridge/VALIDATION.md)、[Wi-Fi 排障](bridge/WIFI_zh.md) |
| 相机 / LiDAR / 感知架构 | [相机](onboard/perception/camera/README.md)、[LiDAR](onboard/perception/lidar/README.md)、[架构](onboard/docs/PERCEPTION_ARCHITECTURE.md) |
| 日志 | [schema 与审计](docs/logging.md) |

本框架源自 [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy) 与 [OpenDriveLab/RoboNaldo_Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy)。RoboNaldo 训练代码见 [OpenDriveLab/RoboNaldo](https://github.com/OpenDriveLab/RoboNaldo)；ScaleBFM 源码见 [zengweishuai/ScaleBFM](https://github.com/zengweishuai/ScaleBFM)，权重见 [Hugging Face](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main)。上游许可证及归属信息保留在对应源码中。

## PHP Parkour（lyd-php-parkour 分支）

PHP 与其他策略并列放在 `policy/php_parkour/`，使用统一 MuJoCo / FSM，配置在 `config/deploy.yaml` 的 `php` 节。外部模型保留在 `~/.cache/php-parkour/student/`。详见 [PHP 部署与按键说明](docs/php_deployment_zh.md)。

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php
# 平地
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php scene.name=none
```

启动保持 Passive，只有按 START 才进入 FixedPose；`L1+A` 或窗口 `P` 显式进入 PHP（允许从 Passive 进入）；左摇杆控制方向，`R3` 切换速度，`A/B/START` 返回 AMP/Loco/FixedPose，`Y` 进入 SMP，`L2` 阻尼。本次支持 MuJoCo，实机深度接入尚未完成。

默认恢复自由视角，可继续用鼠标旋转和缩放；`camera.follow=true` 可主动开启跟随。Backspace 不调用控制器的 exit/enter；PHP 速度档位、动作历史、参考轨迹进度均保留，复位前的深度图则会重新采集。

PHP 启用时默认在 MuJoCo 三维视窗右上角显示模型实际使用的深度图，深度图标题显示当前策略。原生参数侧栏和 Track 模式始终保留；`php_depth_preview=false` 仅关闭预览。MuJoCo 3.2.3 首次需执行 `python tools/install_native_depth_overlay.py`，详见[安装与迁移](docs/native_depth_overlay.md)。

PHP 默认场景现直接复用外部 `php.root` 中的 Holosoma `terrain.obj`；深度图集成在同一窗口右上角，显示模型输入和采集延迟。详见 [PHP 场景配置](docs/php_deployment_zh.md)。
