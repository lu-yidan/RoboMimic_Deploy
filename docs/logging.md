# Recovery 日志 schema 3（2026-09-17）

当前 logger 记录 schema 3，旧 schema 1/2 仍按各自 JSON 的 fields 读取。文件名中的 freekick 仅是配置 tag，不表示运行的策略。

新增信息：

- `executed_fsm_state`：生成本帧指令的 FSM（SMP=18），不是下一状态或已被消费的遥控命令；`smp_activation_id` 区分每次进入 recovery。
- `smp_obs[93]`：送入 ONNX 的实际输入，归一化仍内嵌于模型；`smp_raw_action`、`smp_clipped_action`、`smp_policy_target`、`smp_prelimit_target`、`smp_target_limited`、`smp_warmup_alpha` 保留动作全链。原 `actions` 是最终关节目标，单位 rad。
- `tau_est`：电机固件的力矩估计，仍不是独立力矩传感器测量。`tau_est_valid=0`、NaN 表示缺失；`tau_cmd_est` 仍为主机 PD 估算，不能当实际输出。
- 可选诊断 `imu_accel`、`motor_ddq`、`motor_temperature`（29×2，按 SDK 顺序）。不对加速度计做重力扣除，也不将其解释为外力。
- bridge 路径记录 `state_tick_words`、`command_seq_words`、`remote_raw`、本地接收年龄 `state_age_ms`、循环间隔 `loop_period_ms` 和计算耗时。tick/seq 用两个 uint16 数值保存到 float32，避免单个 float32 丢整数精度；恢复为 low+65536×high。
- JSON 保存代码版本、关节顺序、实际策略 profile、ONNX SHA256、完整策略配置与缺失信息声明。`.events.jsonl` 记录策略执行切换、状态过期后的阻尼请求及恢复。

### 更新部署

Python policy 端必须更新整个仓库代码。仅拷贝 ONNX 或 YAML 不会升级日志。默认 E1 和全部控制参数不变。

电机诊断采用新增的 `<bridge_state_topic>_diagnostics` topic，现有 BridgeState/BridgeCmd 协议保持不变。Python SDK bridge 更新代码后重启；C++ bridge 必须在机器人上重新构建并重启：

```bash
cmake -S bridge -B bridge/build -DUNITREE_SDK2_DIR=/你的/unitree_sdk2
cmake --build bridge/build -j4
```

构建需要 `idlc`，没有自动生成新 IDL 时不能认为诊断通道已升级。新 policy 配旧 bridge 仍可运行，但反馈诊断为 NaN；务必检查 `tau_est_valid` 覆盖率。两个 topic 仅接受相同 firmware tick 的配对，丢包/未配对帧也标缺失，不沿用上一帧力矩。

这些日志记录 policy 端产生的指令，并不证明机器人已接收/执行该条指令。仍没有真实基座位置、双脚接触力、CoP、外部施力或训练时的 SMP/reward。不能据此直接计算实测承重比例或踢击力。没有新增遥控按键；现有 remote 原始值可用于对齐按键事件。

### 审计

```bash
python tools/audit_recovery_log.py logs/xxx.json --out outputs/log_audit.json
```

老 bridge schema 2 的 tau_est 全零是未接通的占位值，不能补回历史反馈。实际字段数以 JSON 为准。每帧 flush 到操作系统，不保证断电时完整落盘。

---

# 旧版 FreeKick 回放说明

## 概述

在执行 `SKILL_FREEKICK` 策略期间，可以将机器人状态、球位置和动作输出实时写入二进制 log 文件。每帧 flush 到操作系统；突然断电仍可能丢失缓存数据。

录制完成后，可在 MuJoCo 中可视化回放：

- **绿色半透明机器人**：还原每帧的关节姿态
- **红色球**：机器人传感器感知的球位置（`ball_pos_b` 变换到世界系）
- **蓝色球**：MuJoCo 仿真 ground truth（仅仿真录制时有效）
- **品红色目标球**：`FreeKick` 实际使用的目标（`debug_target_pos_b` 变换到世界系）
- **紫色小球**：原始 `target_pos_b` 传感器/检测值

主要用途：在真实机器人上录制，然后在 MuJoCo 里回放，直观验证球位置感知是否准确。

---

## 开启录制

在对应部署配置中将 `logging.enabled` 改为 `true`：

**MuJoCo 仿真** — `deploy_mujoco/config/mujoco.yaml`
```yaml
logging:
  enabled: true
  log_dir: "logs"
  tag:     "freekick"
```

**真实机器人** — `deploy_real/config/real.yaml`
```yaml
logging:
  enabled: true
  log_dir: "logs"
  tag:     "freekick"
```

运行程序后，进入 `SKILL_FREEKICK` 状态时自动开始录制，退出时自动关闭并写入元信息。

---

## 文件格式

每次录制生成两个文件：

```
logs/
  20260318_153012_freekick.bin    二进制帧数据
  20260318_153012_freekick.json   元信息（字段定义、帧数、控制频率等）
```

以下为旧 schema 1 的 114 个 float32（456 bytes）字段；当前 schema 3 在此基础上追加字段，以实际 metadata 为准：

| 字段 | 维度 | 说明 |
|---|---|---|
| `step` | 1 | 帧序号 |
| `time_s` | 1 | 时间戳（秒） |
| `q` | 29 | 关节位置 |
| `dq` | 29 | 关节速度 |
| `pelvis_pos_w` | 3 | 浮动基座世界坐标（real 上为零） |
| `pelvis_quat_w` | 4 | 浮动基座姿态 `[w,x,y,z]` |
| `ball_pos_b` | 3 | 球相对 pelvis body frame（传感器值） |
| `ball_pos_w` | 3 | 球世界坐标（仿真 ground truth，real 上为零） |
| `ball_valid` | 1 | 球感知是否有效（0 / 1） |
| `target_pos_b` | 3 | 原始 target 相对 pelvis body frame（来自 `rt/target_state`） |
| `target_valid` | 1 | target 感知是否有效（0 / 1） |
| `vel_cmd` | 3 | 速度命令 |
| `debug_target_pos_b` | 3 | `FreeKick` 实际使用的 target，相对 pelvis body frame |
| `debug_target_source` | 1 | 目标来源编码：0=`none`, 1=`fixed`, 2=`fixed_fallback`, 3=`apriltag`, 4=`imu_hold`, 5=`fixed_sim` |
| `actions` | 29 | policy 输出动作 |

---

## 回放

```bash
# 基本用法
python tools/playback_log.py logs/20260318_153012_freekick.bin

# 指定 xml 和初始速度
python tools/playback_log.py logs/xxx.bin --xml g1_description/scene.xml --speed 0.5
```

**键盘控制：**

| 键 | 功能 |
|---|---|
| `空格` | 暂停 / 继续 |
| `←` / `→` | ±1 帧（暂停时） |
| `[` / `]` | ±10 帧 |
| `f` / `s` | 速度 ×2 / ÷2（范围 0.125× ~ 16×） |
| `r` | 回到第 0 帧 |
| `q` | 退出 |

---

## 在代码中读取 log

```python
from common.logger import Logger

data = Logger.load("logs/20260318_153012_freekick.bin")

data["q"]          # (T, 29) 关节位置
data["ball_pos_b"] # (T, 3)  球 body frame 位置
data["target_pos_b"] # (T, 3) 原始 target body frame 位置
data["debug_target_pos_b"] # (T, 3) FreeKick 实际使用的 target
data["vel_cmd"]    # (T, 3) 速度命令
data["ball_valid"] # (T,)    感知有效性
data["_meta"]      # dict    元信息
```
