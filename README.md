# RoboMimic Deploy · lyd-deploy

English | [中文完整使用手册](README_zh.md)

Multi-policy deployment for the Unitree G1 with 29 body joints. This branch combines MuJoCo table/box recovery scenes, selectable AMP and SMP policies, and optional ScaleBFM tracking. Simulation, direct Python deployment, and the C++ bridge policy process share policy selections.

## Contents

- [Installation and updates](#installation-and-updates)
- [Configuration layout](#configuration-layout)
- [MuJoCo scenes](#mujoco-scenes)
- [Policy selection](#policy-selection)
- [Controls and resets](#controls-and-resets)
- [ScaleBFM](#scalebfm)
- [Real-robot deployment](#real-robot-deployment)
- [Logging and validation](#logging-and-validation)
- [Documentation and credits](#documentation-and-credits)

## Installation and updates

```bash
git clone --branch lyd-deploy git@github.com:lu-yidan/RoboMimic_Deploy.git
cd RoboMimic_Deploy
conda create -n robomimic python=3.8.20 -y
conda activate robomimic
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Run commands from the repository root. Existing installations can reuse their environment. The clone command tracks all remote branches; add `--single-branch --depth 1` only if you want a smaller deployment copy. Switching an existing clone does not remove old Git objects.

```bash
git fetch origin --prune
git switch lyd-deploy
git pull --ff-only
```

Python 3.8 deployments use NumPy 1.24.4 and an ONNX runtime supporting opset 18, such as ONNX Runtime 1.19.2. Interactive simulation requires a graphical session and a supported gamepad. DDS system libraries and robot SDK installation are covered by the [onboard installation guide](onboard/docs/INSTALL_AGENT_GUIDE.md). Jetson PyTorch/CUDA packages must match JetPack; the workstation installation above is not a Jetson recipe.

## Configuration layout

| File | Purpose |
|---|---|
| [`config/deploy.yaml`](config/deploy.yaml) | Shared ScaleBFM switch/paths and AMP/SMP profiles |
| [`deploy_mujoco/config/mujoco.yaml`](deploy_mujoco/config/mujoco.yaml) | Base XML, physics/control timing, rendering and simulation logging |
| `deploy_mujoco/config/mujoco_table.yaml` | Table geometry and initial pose |
| `deploy_mujoco/config/mujoco_coffin.yaml` | Box, lid, masses, initial pose and camera |
| `deploy_mujoco/config/mujoco_freekick.yaml` | Ball scene and ball reset parameters |
| [`deploy_real/config/real.yaml`](deploy_real/config/real.yaml) | Network interface, joint order, control period and robot logging |
| `policy/<name>/config/*.yaml` | Model registry and observation/actuation contracts |

Restart after configuration changes. `LYD_DEPLOY_CONFIG` can select an alternate complete shared configuration file. The legacy `SMP_RECOVERY_PROFILE` environment variable overrides the SMP selection in that file.

MuJoCo uses Hydra: select a preset with `--config-name`, override existing fields using `key=value`, and add absent fields using `+key=value`. Only `scalebfm.motion` can override shared ScaleBFM settings in a scene/CLI; enablement and model settings remain in `config/deploy.yaml`.

## MuJoCo scenes

| Preset | Behavior |
|---|---|
| `mujoco` (default) | Standard `scene.xml`; the validated ScaleBFM squat asset |
| `mujoco_table` | Static table, supine initial pose, passive startup |
| `mujoco_coffin` | Free box and independent free lid, using `g1_liao.xml` |
| `mujoco_freekick` | Ball scene; does not enable the disabled FreeKick policy |
| `duobox_straight`, `duobox_guard`, `duobox_combo` | Standard scene with a reference-motion preset; requires separately generated data |

```bash
python deploy_mujoco/deploy_mujoco.py
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_freekick

# Inspect composed simulation config without starting the viewer or policies
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table --cfg job
```

### Table geometry

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  scene.table.length=1.8 scene.table.width=0.8 \
  scene.table.height=0.5 scene.table.thickness=0.04 \
  'scene.table.pos=[0.2,0,0]' initial_state.preset_yaw_deg=30
```

`length` and `width` are full dimensions in metres. `height` is the top surface height relative to `pos.z`; clearance is `height - thickness`. Defaults are 1.6 × 0.6 m, 0.4 m high, and 0.04 m thick. `leg_width`, `leg_inset`, `friction` and `rgba` configure the legs and surfaces. The table is fixed. See the [parameter reference](README_zh.md#32-桌子尺寸位置与初态).

### Box and lid

```bash
# Fixed open box
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin \
  scene.coffin.free=false scene.coffin.lid_enabled=false

# Lighter box and lid, translucent walls
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin \
  scene.coffin.mass=10 scene.coffin.lid_mass=2 \
  'scene.coffin.rgba=[0.24,0.10,0.05,0.3]'
```

Box `length`/`width` are inner dimensions; `height` includes the base but excludes the lid. Defaults: 1.4 × 0.6 × 0.4 m, 0.04 m base, 0.035 m walls, 15 kg box, 0.03 m / 3 kg lid. `free=false` fixes the box only: the lid remains a separate free body, not a hinged door. `pos`, `yaw_deg`, `friction`, `rgba` and `lid_rgba` are configurable.

### Plate, initial state, camera and timing

```bash
# Vertical sliding plate, using the table preset's initial robot pose
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  scene.name=plate '+scene.plate.mass=8' \
  '+initial_state.plate_surface_top_z=0.4'

# Table plus plate
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  scene.name=table_and_plate '+scene.plate.mass=8'

# Camera and rendering
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_coffin \
  camera.distance=4 camera.azimuth=90 render_hz=30 ghost_enabled=false
```

Unlike table dimensions, `scene.plate.size` uses MuJoCo **half-extents**. Plate defaults are `[0.45,0.32,0.035]`, mass 8 kg and Z slide range `[-1.2,0]`.

`initial_state.preset` accepts `none`, `plate_supine_default_arms` or `plate_supine_crossed_arms`. Configure position/yaw with `preset_root_pos` and `preset_yaw_deg`. Complete `qpos`/`qvel` arrays override the preset and must match the model's `nq`/`nv`, including obstacle joints.

`startup_policy` accepts `passive`, `fixed_pose` and `smp_recovery`. Selecting recovery here runs it immediately at startup. Default physics timestep is 0.002 s with `control_decimation=10` (50 Hz policies); rendering defaults to 60 Hz. Adjust `render_hz` independently instead of changing the policy timing.

Full parameter tables, custom YAML examples and ball reset settings are in the [Chinese scene guide](README_zh.md#3-mujoco-场景与参数). Generating a scene does not establish recovery success in it.

## Policy selection

Edit the corresponding sections in `config/deploy.yaml` and restart:

```yaml
amp:
  profile: amp_0309_1
smp:
  profile: v34_balanced4_hard19000
```

- AMP: `amp_0309_1`, `stratified_guidance_60000`, `stratified_single_60000`.
- SMP: `path_a6_9999`, `a6_no_alpha_9999`, `v34_balanced4_hard19000`.

AMP exports share the deployment interface and embed normalization, joint remapping and action conversion. Do not apply those conversions again. SMP profiles retain their model hashes and control limits; balanced4 uses 90% of the nominal torque limits. Selecting a profile does not enter the policy: use A for AMP or Y for SMP.

## Controls and resets

| Action | Simulation | Robot |
|---|---|---|
| FixedPose / Loco / AMP | START / B / A released | START / B / A |
| SMP recovery | Y pressed without L1 | Y rising edge |
| ScaleBFM play/restart | R1 released | R1 rising edge |
| Previous/next motion | Left/right, without L1 | Left/right, without L1/L2 |
| BeyondMimicMJ / StandUpMJ | Down/up | Down/up |
| Pinocchio1.6MJ | R2 pressed | R2 |
| Damping | Release L2 after pressing it | F1 |
| Exit | SELECT | SELECT, with damping on exit |
| Toggle ghost / reset ball | L3 / X released | Not applicable |

ScaleBFM must be enabled; enter it from FixedPose, Loco, AMP or ScaleBFM itself. FreeKick is disabled in the current entries, so R1 is not a kick command. Y preempts other policies; damping takes priority.

**Resets differ:** START changes joint targets without resetting physical position/velocity. Keyboard R restores the configured scene initial state and selects Passive. Backspace restores standing physics and clears velocity while preserving the current controller, its history and phase. L1+START resets standing physics and selects FixedPose. These simulation resets do not exist on the robot. Use **R → Y** for recovery experiments and **L1+START → R1** for standing ScaleBFM tests.

The default viewer uses a free camera; set `camera.follow=true` to track G1.

Gamepad inspection: `python tools/joystick_test.py`. Button indices can be overridden with environment variables such as `JOYSTICK_START=7`. L1+left/right adjusts target Y bias by ±0.05 m; the robot bridge also supports L2+left/right for ball Y bias, bounded to ±1.5 m.

## ScaleBFM

```bash
python tools/setup_scalebfm.py
pip install -r environment/scalebfm-requirements.txt
```

The setup script downloads pinned official source and checksum-verified M weights/metadata/mode table outside this repository. Set `scalebfm.root` to the printed location, then set `scalebfm.enabled: true` in `config/deploy.yaml`. Default is `false`.

`checkpoint: null` uses the official M checkpoint; `motion: null` uses the 417-frame / 50 Hz squat reference. `cuda_graph` enables graph replay. On the robot, keep `reference_forcing: true`; LowState does not supply root localization. This setting changes observations, not physical support forces.

```bash
python deploy_mujoco/deploy_mujoco.py \
  scalebfm.motion=/absolute/path/to/motion.npz
python tools/check_scalebfm_onboard.py --steps 100
python tools/smoke_scalebfm_fsm.py
```

Motion paths are resolved relative to the repository root. `motion_globs` builds the selectable library at startup. DuoBox presets point to generated files under `outputs/duobox_scalebfm/motions/`; those files are not included in Git. See the [ScaleBFM guide](docs/scalebfm_deployment_zh.md) for array shapes and data preparation.

The current PyTorch backend uses upstream export definitions and does not require TensorRT. Validation covered Python 3.8 / PyTorch 2.3.1 on RTX 4070 Ti SUPER with `scene.xml`; `g1_liao.xml` is not considered a validated ScaleBFM squat asset.

## Real-robot deployment

Follow the [onboard setup](onboard/docs/INSTALL_AGENT_GUIDE.md) before starting motor communication:

```bash
cmake -S bridge -B bridge/build
cmake --build bridge/build -j2
# Terminal 1; use the actual robot-facing interface
BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main
# Terminal 2
python bridge/python/deploy_policy.py
```

C++ owns LowState/LowCmd; Python sends joint targets over bridge DDS. Do not run `deploy_real/deploy_real.py` or another LowCmd controller at the same time. The direct Python entry remains available for installations with Unitree Python SDK.

`deploy_real/config/real.yaml` controls network/timing/logging. The interface preference is `BRIDGE_NETWORK_INTERFACE`, then `ROBOT_IFACE`, then YAML `net`. Keep 29 joints and 0.02 s control timing for ScaleBFM.

`bash tools/start_tmux_layout.sh` opens the bridge, policy, dashboard and perception panes with commands prefilled. Press Enter in each pane to start; detaching tmux does not stop processes. See the [bridge guide](bridge/README_zh.md), [validation procedure](bridge/VALIDATION.md) and [perception guide](onboard/docs/README.md).

## Logging and validation

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_table \
  logging.enabled=true logging.tag=table_a6 \
  'logging.states=[SKILL_SMP_RECOVERY,SKILL_AMP]'

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/test_scalebfm_real.py
```

Robot logging is configured in `real.yaml`. `logging.tag` names the log; it does not select a policy. See [log schema and audit](docs/logging.md).

The [integration report](docs/validation/lyd_deploy_20261025.json) records model contracts, effort limits, scene smoke checks, GPU inference and ScaleBFM MuJoCo/FSM checks. No robot control was performed during those checks; they do not establish task success for every scene or motion.

## Documentation and credits

- [Complete Chinese guide](README_zh.md), including troubleshooting and parameter tables.
- [Table configuration](docs/table_simulation_config_zh.md) and [box/lid configuration](docs/coffin_simulation_config_zh.md).
- [AMP guidance](docs/amp_guidance_60000_deployment_zh.md) and [AMP single](docs/amp_stratified_single_60000_deployment_zh.md) provenance and evaluation.
- [Camera](onboard/perception/camera/README.md), [LiDAR](onboard/perception/lidar/README.md), and [perception architecture](onboard/docs/PERCEPTION_ARCHITECTURE.md).

Based on [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy) and [OpenDriveLab/RoboNaldo_Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy). Training: [RoboNaldo](https://github.com/opendrivelab/RoboNaldo). ScaleBFM: [official source](https://github.com/zengweishuai/ScaleBFM) and [weights](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main). Upstream licenses and attribution are retained.

## PHP Parkour (lyd-php-parkour branch)

PHP now lives alongside the other policies in `policy/php_parkour/` and uses the same MuJoCo process and FSM. Configure `php.enabled` and the external `php.model_dir` in `config/deploy.yaml`. No separate Holosoma launcher or depth shared-memory server is needed.

```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php
# Flat ground:
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php scene.name=none
```

Startup stays Passive. START explicitly selects FixedPose; L1+A or P explicitly selects PHP, including from Passive. The exact backbone input appears in the top-right of the MuJoCo viewport. Left stick selects direction, R3 toggles speed, A/B/START return to AMP/Loco/FixedPose, Y preempts with SMP and L2 enters damping. This integration is for MuJoCo; hardware depth input is not connected. See the [PHP guide](docs/php_deployment_zh.md).

The PHP preset now reuses `terrain.obj` from the external PHP checkout (`php.root`). The depth inset shares the MuJoCo window and shows capture age. On MuJoCo 3.2 this uses a compact GLFW viewer; `php_depth_preview=false` restores the native parameter-sidebar viewer.
