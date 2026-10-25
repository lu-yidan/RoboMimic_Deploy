# 桌下仿真配置

[返回部署手册](../README_zh.md#3-mujoco-场景与参数)

适用分支：`lyd-deploy`。以下命令只启动本地 MuJoCo 仿真：

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_table
```

在 `deploy_mujoco/config/mujoco_table.yaml` 修改桌子完整长宽、总高、板厚、腿宽、内缩距离，以及初始姿态/位置/朝向。单位为米；height是桌面上表面相对table.pos.z的高度。默认长1.6、宽0.6、总高0.4、板厚0.04，净空0.36，腿自动接地。旧分支的top_size/leg_size半尺寸输入在本分支不接受。

原来的普通启动命令不增加桌子。物理步长2ms，控制周期20ms。startup_policy支持passive（默认）、fixed_pose、smp_recovery，仅决定启动时的策略。

按键：

- 手柄START：原有FixedPose默认姿态控制，不重置仿真位置、不启动SMP。
- 手柄Y：进入SMP恢复。
- MuJoCo窗口键盘R：重置仿真位置及初始姿态，退出当前策略并进入PASSIVE；随后按Y才进入SMP。这不影响真机程序。
- L2 按下后松开：进入阻尼。SELECT：退出仿真程序。
- Backspace 或 L1 + START：重置到站立 FixedPose；与 R 恢复 YAML 初态不同。

策略在 `config/deploy.yaml` 的 `smp` 段选择：

```yaml
smp:
  profile: v34_balanced4_hard19000
```

默认 profile 为 v34_balanced4_hard19000。`SMP_RECOVERY_PROFILE`环境变量会覆盖文件设置；启动日志会打印最终profile和模型哈希。更改配置后重启。

默认程序化仰卧，root=[-0.1,0.1,0.12]，yaw=0，双臂为default arms。它不是自然动作切片。reset仅回到这个配置状态，不保证与训练录像相同。

无界面验证已检查Hydra配置合成、桌面总高/净空/腿接地、reset姿态复现、START命令保持POS_RESET、模型输出一致性与加载限幅。没有连接真机；未用实体手柄和图形窗口验证按键交互。

完整参数表、压板组合、自定义 YAML 和命令行覆盖示例见 [README 场景配置](../README_zh.md#3-mujoco-场景与参数)。
