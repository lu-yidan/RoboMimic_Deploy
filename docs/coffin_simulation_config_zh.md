# 箱体 / 棺材部署仿真

分支：`lyd/scalebfm-table-config-balanced4`。

1. 在 `policy/smp_recovery/config/smp_recovery.yaml` 设置 `profile: a6_coffin_v2_500`。也可选 `a6_no_alpha_9999`、`path_a6_9999`、`v34_balanced4_hard19000`。环境变量 `SMP_RECOVERY_PROFILE` 优先于 YAML。
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

使用 `g1_description/g1_liao.xml`，物理2 ms、控制20 ms。此独立配置关闭ScaleBFM和ghost，不改变原桌子/普通配置。未增加真机行为。

验证：配置合成、29个机器人执行器索引、质量、初始接触、整箱reset、固定/无盖变体通过；A6 V2@500实际部署控制器23秒headless运行无数值异常。未验证实体手柄或图形窗口交互；沿用既有按键逻辑。策略能力及安全限制见 `coffin_a6_v2_500_deployment_zh.md`。
