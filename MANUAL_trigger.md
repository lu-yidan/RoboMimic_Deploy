# Score 手动 Trigger 与 Anchor Command

本文说明仿真优先的 Score 手动触发与手动 anchor command。当前实现主要面向 `deploy_mujoco/deploy_mujoco.py` 的手柄控制路径，用于在 MuJoCo 中不依赖球进入触发圈，直接用手柄触发 kick，并用摇杆改变 Score policy 观测中的 anchor 位置与相对 yaw。

## 适用范围

- 仿真优先：默认配置在 `policy/score/config/score.yaml` 中启用。
- Score 模式：进入 `FSMCommand.CMD_SCORE` 后由 `policy/score/Score.py` 消费命令。
- 其他策略不受影响：Loco、AMP、BeyondMimic 等仍使用原有 `vel_cmd` 和 skill command。
- 真实机器人路径暂未作为主要目标；如果真实部署需要同样手动命令，应确认 DDS / runtime 输入是否也写入相同 `StateAndCmd` 字段。

## 手柄按键表

| 输入 | 功能 |
| --- | --- |
| R1 | 进入 Score 模式 |
| R3 | Score 内手动 trigger，一次 release/edge 触发 kick |
| 左摇杆前后 | 命令 `anchor_pos_b.x`，正方向为 torso/body frame 前方 |
| 左摇杆左右 | 命令 `anchor_pos_b.y`，正方向为 torso/body frame 左侧 |
| 右摇杆横向 | 命令 anchor relative yaw |
| X | 重置球 |
| L3 | ghost 显示开关 |
| Y | 打印当前关节角 |
| START | 回 FixedPose / POS_RESET |
| B | 进入 Loco |
| A | 进入 AMP |
| D-pad Down | BeyondMimicMJ |
| D-pad Up | StandUpMJ |
| R2 | Pinocchio1.6MJ |
| L2 release | PASSIVE 阻尼保护 |
| SELECT | 退出仿真 viewer loop |

## 配置项

配置位于 `policy/score/config/score.yaml`。

- `manual_trigger: true`
  - 开启后，Score 的等待触发只响应 R3 手动 trigger，不再使用球进入 trigger circle。
  - 关闭后保留原有 ball trigger 行为。

- `manual_anchor_cmd: true`
  - 开启后，只有对应摇杆明显离开 deadzone 时，Score 的 `anchor_pos_b` 或 `anchor_ori_6d` 才来自手柄命令。
  - 摇杆回中或输入为 null / deadzone 内时，会沿用默认 anchor 行为，例如 motion_ref、ball_as_anchor_pos 或 ball_facing_anchor_ori，不会给 policy 写入零 anchor / 零 yaw 覆盖。
  - `anchor_pos_b.z` 仍沿用参考 motion anchor 与 torso 的高度差，避免摇杆直接控制高度。

- `manual_anchor_pos_scale_xy: [1.0, 0.6]`
  - 左摇杆归一化输入到 body-frame anchor x/y 的比例，单位为米。
  - 例：左摇杆前推到 1.0 时，`anchor_pos_b.x` 约为 `+1.0 m`。

- `manual_anchor_yaw_scale: 1.57`
  - 右摇杆横向归一化输入到相对 yaw 的比例，单位为弧度。
  - `1.57` 约等于 90 度。

- `manual_cmd_deadzone: 0.05`
  - 对左摇杆 x/y 与右摇杆 yaw 输入的死区。
  - 小于该绝对值的输入会被视为 inactive，降低摇杆漂移影响。
  - 左摇杆 x/y 和右摇杆 yaw 分开判断 active：只推左摇杆不会把 yaw 强制为 0，只推右摇杆也不会把 anchor x/y 强制为 0。

相关已有项仍然生效：

- `trigger_frame`
- `trigger_play_end_frame`
- `trigger_play_once`
- `wait_for_ball`，仅在 `manual_trigger: false` 时用于原 ball gate

## 工作流与数据流

1. `deploy_mujoco/deploy_mujoco.py` 读取手柄。
2. R3 release 时写入 `state_cmd.score_manual_trigger = True`。
3. 左摇杆与右摇杆横向写入归一化 raw command：
   - `state_cmd.score_anchor_pos_raw_b`
   - `state_cmd.score_anchor_yaw_raw`
4. `policy/score/Score.py` 读取 `score.yaml` 中的 deadzone 与 scale。
5. Score 将 raw command 转成最终命令：
   - `state_cmd.score_anchor_pos_cmd_b`
   - `state_cmd.score_anchor_yaw_cmd`
   - `state_cmd.score_anchor_pos_cmd_active`
   - `state_cmd.score_anchor_yaw_cmd_active`
   - `state_cmd.score_anchor_cmd_active`
6. `_build_obs()` 只在对应 active 标志为 true 时将手动 anchor 写入 ONNX 输入：
   - 左摇杆 active 时，`anchor_pos_b` 使用手柄 x/y，加参考高度差 z。
   - 右摇杆横向 active 时，`anchor_ori_6d` 使用手柄 yaw 构造 body-frame relative yaw quaternion，再转为 6D rotation。
   - inactive 时回落到原本的默认 anchor 计算。
7. ONNX policy 根据新的 obs 输出动作。

## 仿真测试步骤

1. 确认使用包含 Score 的 MuJoCo 配置，并启动仿真。
2. 按 R1 进入 Score。
3. 进入后应看到 Score warmup，然后状态行显示等待 R3，例如 `trig=R3`。
4. 在未按 R3 前，motion 应保持在 frame 0 或 trigger 前等待状态。
5. 推动左摇杆，观察状态行中的 `anchor_cmd=(x,y,z)` 变化；viewer 中橙色 anchor 标记应随 x/y 移动。
6. 横向推动右摇杆，观察状态行中的 `yaw=...` 变化；紫色 anchor 朝向箭头应随 yaw 转动。
7. 松开/点击 R3，Score 应打印 manual trigger 信息，并从 `trigger_frame` 播放到 `trigger_play_end_frame` 或 clip end。
8. 如果配置了有限片段且 `trigger_play_once: false`，片段结束后应回到 frame 0 并再次等待 R3。

## 预期现象

- `manual_trigger: true` 时，球是否进入 trigger circle 不会触发 kick。
- R3 是 edge/release 触发，按下一次只消费一次；Score 消费后会清零，避免重复触发。
- `manual_anchor_cmd: true` 时，摇杆离开 deadzone 后 anchor x/y 和 yaw 由手柄实时控制；摇杆回中时 `active=0`，沿用默认 anchor，不会发送零 anchor 命令。
- Loco/AMP 等模式仍按原逻辑读取 `vel_cmd`，不会被 Score 专用字段改变。

## 注意事项与调参建议

- 如果摇杆轻微漂移导致状态行 `active=1`，先增大 `manual_cmd_deadzone`，例如从 `0.05` 调到 `0.08`；如果正常小幅操作不生效，则适当减小 deadzone。
- 如果 active 后 anchor 移动过大，降低 `manual_anchor_pos_scale_xy`；如果模型反应不明显，可小幅增大。
- 如果 active 后 yaw 转向过强，降低 `manual_anchor_yaw_scale`；默认 `1.57` 是较大的测试范围。
- 手动 anchor 是直接进入 policy obs 的命令，不保证训练分布完全覆盖；调参时建议先在低风险仿真场景中逐步增加幅度。
- 若要回到球触发测试，将 `manual_trigger` 设为 `false`，并按原有 `wait_for_ball`、`trigger_radius`、`trigger_horizon` 配置使用。
