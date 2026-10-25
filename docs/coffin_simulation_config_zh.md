# 箱体 / 棺材部署仿真

[返回部署手册](../README_zh.md#3-mujoco-场景与参数)

适用分支：`lyd-deploy`。

1. 在 `config/deploy.yaml` 的 `smp.profile` 选择 `path_a6_9999`、`a6_no_alpha_9999` 或 `v34_balanced4_hard19000`（默认）。
2. 启动：

```bash
python deploy_mujoco/deploy_mujoco.py --config-name=mujoco_coffin
```

默认PASSIVE。手柄Y进入SMP；START仍是FixedPose，不是SMP。窗口R重置箱体、盖板、机器人并回PASSIVE，然后按Y。自动开始可在场景 YAML 改 `startup_policy: smp_recovery`。

所有场景参数在 `deploy_mujoco/config/mujoco_coffin.yaml`，不需要长命令：

- `length/width` 为箱内完整长宽，默认1.4/0.6 m。外尺寸为1.47×0.67 m。
- `height: 0.4` 为箱底下表面到箱壁顶端的总高，包含4 cm底板，不包含3 cm活动盖板。箱内竖向净空36 cm，合盖外部总高43 cm。
- `free: true` 箱体可移动/倾斜，质量15 kg；改false固定箱体。盖板始终为独立自由刚体，质量3 kg。`lid_enabled: false`移除盖板。
- `pos/yaw_deg` 控制箱体和盖板的位置/朝向；默认中心配合程序化仰卧的身体范围。改变机器人位置或yaw时需同时检查侧壁碰撞。
- `initial_state`控制机器人reset，默认root=[0,0,0.16]、default-arms仰卧。
- `rgba/lid_rgba`最后一个值为透明度；改0.3可看清箱内机器人，纯视觉变化不改变碰撞。
- `camera`控制初始视角，可在窗口拖动。

使用 `g1_description/g1_liao.xml`，物理2 ms、控制20 ms。此场景默认关闭 ghost；ScaleBFM 开关由共享 `config/deploy.yaml` 控制，不会因切换场景而自动关闭。该 XML 未通过当前 ScaleBFM squat 兼容性验证，测试箱内 SMP 时保持共享 `enabled: false`。场景配置不改变真机行为。


本分支保留箱体场景，不再包含 A6 V2@500 实验模型。场景接口测试不代表三个保留策略都能完成箱内起身。

完整参数表及固定箱体、无盖、质量、透明度和相机覆盖命令见 [README 箱体配置](../README_zh.md#33-箱体固定自由盖子质量透明度)。
