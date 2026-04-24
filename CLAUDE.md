# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run commands

Env: conda `robomimic` (Python 3.8), deps in `requirements.txt`, plus `unitree_sdk2_python` installed from git. PyTorch 2.3.1 via conda.

MuJoCo sim (Hydra config in `deploy_mujoco/config/`):
```bash
python deploy_mujoco/deploy_mujoco.py                              # default: mujoco.yaml, scene without ball
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_score   # scene_with_ball.xml; required for Score
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_php     # g1_with_terrain.xml; for PHPParkour
```

Real robot (direct Python path):
```bash
python deploy_real/deploy_real.py    # reads deploy_real/config/real.yaml
```

Onboard-Orin bridge flow (C++ owns LowCmd; Python runs policy only):
```bash
cmake -S bridge -B bridge/build && cmake --build bridge/build -j2
BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main
python bridge/python/deploy_policy.py
# or full-Python prototype:
python bridge/python/deploy_bridge_py.py
```

There is no test suite. Validation is manual — see `bridge/VALIDATION.md` for the staged procedure and the policy tables in `README.md` for joystick bindings.

Useful utilities in `tools/`: `check_ball_state.py`, `check_dds_connection.py`, `joystick_test.py`, `playback_log.py` (replays logs in MuJoCo), `frame_ball_target_distance.py`.

## Architecture

### FSM + shared I/O buffers

A single `FSM` object owns one instance of every policy. Each control tick, the host loop (`deploy_mujoco.py` / `deploy_real.py`) fills `state_cmd` with sensor readings, then calls `FSM.run()`. The active policy writes into `policy_output`; the host applies the result (PD control → torque, or direct torque — see below).

- `common/ctrlcomp.py` — `StateAndCmd` (robot state + joy command) and `PolicyOutput` (actions/kps/kds and extras).
  - `PolicyOutput.actions|kps|kds` are property-backed: assignment always copies into an internally-owned buffer. Do not alias arrays with it. The contract is documented in `FSM/FSMState.py`.
  - Extras on `PolicyOutput`: `ghost_qpos` (reference-motion overlay for viewer), `viz_spheres` (debug geoms), `direct_torque` flag (see PHP below), `debug_target_pos_b` / `debug_target_source` (Score logging hooks).
- `FSM/FSM.py` — instantiates all policies; switching is a two-step CHANGE → NORMAL handshake that calls `exit()` on the outgoing policy and `enter()` on the incoming one before the next `run()`.
- Skill commands are **latched**: the host sets `state_cmd.skill_cmd = FSMCommand.XXX` on button edge, and the current policy's `checkChange()` consumes it by returning a new `FSMStateName`. `PASSIVE` is the one override that always wins — it's set unconditionally on L1/F1.
- Enums live in `common/utils.py` (`FSMStateName`, `FSMCommand`). Adding a new policy means: new enum values in both, new class under `policy/<name>/`, instantiation in `FSM.__init__`, a branch in `get_next_policy`, and a joystick edge in both host loops.

### Policies (`policy/<name>/`)

Each policy subclasses `FSMState` and typically contains `config/*.yaml`, `model/*.onnx` (or `.pt`), and the class file. Active entries:

| Class | State | File | Notes |
|---|---|---|---|
| `PassiveMode` | PASSIVE | damping-only |
| `FixedPose` | FIXEDPOSE | position reset to default pose |
| `LocoMode` | LOCOMODE | TorchScript `.pt` (the only non-ONNX one) |
| `Amp` | SKILL_AMP | ONNX; defaults to run mode |
| `Score` | SKILL_SCORE | ONNX, 325/547-dim obs variants, 5-frame history; **requires ball sensor** |
| `BeyondMimic` | SKILL_BEYONDMIMIC | kept in tree, no default binding |
| `BeyondMimicMJ` | SKILL_BEYONDMIMIC_MJ / SKILL_STANDUP_MJ / SKILL_PINOCCHIO_1_6_MJ | **same class, three instances** with different YAMLs/enum names (see `FSM.__init__`) |
| `PHPParkour` | SKILL_PHP_PARKOUR | perceptive policy, needs depth image + direct-torque output |

Joint ordering is the biggest foot-gun:
- Isaac-Lab-trained policies (`Score`) use BFS order and must be remapped with `MUJOCO_TO_ISAAC` / `ISAAC_TO_MUJOCO` (defined at the top of `Score.py`).
- mjlab-trained policies (`BeyondMimicMJ`, `PHPParkour`) use MuJoCo DFS order, which already matches hardware. No remap needed. The DFS mapping is documented at the top of `BeyondMimicMJ.py` and in policy YAMLs.

### Sensor data paths

- Ball state (`state_cmd.ball_*`):
  - MuJoCo sim: `deploy_mujoco.py` reads it from `MjData` in world frame (`ball_pos_w`/`ball_vel_w`), throttled to `ball_sensor_hz`.
  - Real robot: `BallStateSubscriber` (CycloneDDS topic `rt/ball_state`) delivers it already in **pelvis body frame** (`ball_pos_b` + `ball_valid`). The onboard perception service (lidar or camera) publishes it — see `onboard/README.md`.
  - `Score` branches on `use_body_frame_ball` in its yaml to pick the right source.
- Target state: DDS topic `rt/target_state` via `TargetStateSubscriber` (AprilTag / detector output, pelvis frame).
- PHP depth (`state_cmd.depth_image`): rendered each tick by `deploy_mujoco.py` via `mujoco.Renderer` using the `<camera name="php_depth">` in the XML, with `m.vis.map.znear/zfar` temporarily overridden so the depth buffer has PHP's 0.3–3.0 m range. Only rendered while the PHP policy is active.
- IMU torso quaternion: on the real robot the IMU sits on pelvis, so `deploy_real.py` calls `transform_pelvis_to_torso_complete` using waist yaw/roll/pitch (joints 12/13/14) to produce `torso_quat_w` for BeyondMimic-family policies.

### Direct-torque path

PHPParkour is unique: it outputs actuator-indexed torques, not PD targets. It sets `policy_output.direct_torque = True` and the host writes `actions` straight to `MjData.ctrl` (after `tau_limit` clip) instead of running the outer PD loop. `tau_limit` in `mujoco_php.yaml` must be in **actuator declaration order** (matches `g1_with_terrain.xml`'s actuator block).

### Three deployment targets

1. **`deploy_mujoco/`** — Hydra-driven sim host, pygame joystick, MuJoCo viewer. Handles ghost overlay, viz spheres, ball reset (R1+X), PHP depth render, optional score logging.
2. **`deploy_real/`** — direct Unitree Python SDK (`unitree_sdk2py`). Pub `rt/lowcmd`, sub `rt/lowstate`, reads Unitree remote via `RemoteController`. Not recommended on Orin (see README §1).
3. **`bridge/`** — Orin-friendly split. C++ process (`cpp_bridge_main.cpp`) owns real `rt/lowstate`/`rt/lowcmd` via `unitree_sdk2`, publishes `rt/policy_bridge_state`, subscribes `rt/policy_bridge_cmd`. Python (`bridge/python/deploy_policy.py`) runs `PolicyRuntime` (same FSM) over those bridge topics. DDS schema in `bridge/idl/policy_bridge.idl`. Safety: bridge falls back to damping if cmd stream goes stale or `request_damping`/`exit_requested` flip true. See `bridge/README.md` for the full contract and current build status (uses `idlc -l c` on this machine because the C++ generator isn't installed).

### Config conventions

- `common/path_config.py` exports `PROJECT_ROOT`. Most config loads use absolute paths computed from it, not CWD — you can run scripts from anywhere.
- MuJoCo host uses Hydra; real-robot host uses plain YAML via `deploy_real/config.py`. Real-robot config supports env-var overrides for bridge topics/timeouts (`BRIDGE_*`).
- Per-policy YAMLs sit under `policy/<name>/config/`. Several values (joint names, action_scale, default_joint_pos, kps/kds) are increasingly read from ONNX metadata rather than duplicated in YAML — check the policy's loader to see which.
- MuJoCo robot XMLs live in `g1_description/`. The main scenes are `g1_liao.xml` (default), `scene_with_ball.xml` (Score), and `php_parkour/g1_with_terrain.xml` (PHPParkour).

### Logging

Optional per-state recording (`logging.enabled: true` in the host YAML). Emits paired `.bin` + `.json` under `logs/` with a fixed 114-float32 schema per frame. Replay with `tools/playback_log.py`. Full schema in `docs/logging.md`.
