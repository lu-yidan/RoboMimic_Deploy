# ACP 实时 MuJoCo：B 行走 → Y ACP

当前正式入口是 `LocoMode`（B 键），切入 ACP 使用原 SMP 的 Y 键。
启用 ACP 的仿真、真机 Python 输入路由都采用此映射；关闭 ACP 时 Y 保留原 SMP 行为。
L1+X 不再请求 ACP。阻尼仍优先于所有模式请求。

## 本机启动

在本机图形桌面的终端运行：

```bash
cd "/home/yujy/Documents/ChatGPT/G1 locomotion/RoboMimic_Deploy_acp"
LYD_DEPLOY_CONFIG="$PWD/config/deploy_acp.yaml" \
__GLX_VENDOR_LIBRARY_NAME=mesa \
/home/yujy/miniconda3/envs/robomimic/bin/python deploy_mujoco/deploy_mujoco.py \
  --config-name mujoco_acp \
  "acp_bundle=$PWD/../data/acp_g1_reports/stage3_flat_trial_v1/bundle_stage3_1000"
```

这会实时运行 MuJoCo 物理和当前策略，初始化即处于 B/LocoMode，键盘速度指令为零。
没有手柄时可直接用键盘。`__GLX_VENDOR_LIBRARY_NAME=mesa` 适用于本机显示服务；
其他正常使用 NVIDIA GLX 的桌面可以省略该变量。bundle 为固定的 1k 候选，
`policy.onnx` SHA256：`2a85ea0312bbdff8b681dabe659c8f9c68b1d55221c4f9ec41284dd674e310e8`。

| 按键 | 操作 |
| --- | --- |
| B | 行走模式 LocoMode；从 ACP 返回行走模式 |
| Y | 从当前 LocoMode 切入 ACP；手柄与键盘一致 |
| W / S | 增减行走前后速度，每次 0.1 m/s |
| A / D | 增减左右速度，每次 0.1 m/s |
| Q / E | 增减转向速度，每次 0.2 rad/s |
| Z | 将键盘行走速度设为零 |
| 1 | 以当前骨盆位置和航向设置前方 1 m 目标 |
| 2 | 设置左侧 1 m、航向左转 90° 的目标 |
| 3 | 将当前位置与航向设为停车目标 |
| 空格 | 暂停 / 继续仿真 |
| F1 | 阻尼 |
| R | **仅仿真**复位并重新初始化到 B/LocoMode |

建议先在 B 模式按 1，再按 Y；到点后可继续按 2 或 3 改目标，或按 B 返回行走。
紫色圆点与箭头表示 ACP 的目标位置和目标航向。可使用 MuJoCo 原生鼠标操作观察、
旋转镜头与施加扰动。手柄保留原摇杆行走输入；键盘设置速度后，明显移动摇杆会接回手柄速度。
F1 阻尼后需要重新准备；在仿真中可用 R 恢复到 B。

## 与历史站姿试验的区别

仿真启动或按 R 时，仅一次性将机器人放到原 Loco 策略自己的默认关节姿态，
按脚底碰撞几何拟合地面高度，清零速度并执行实际 Loco 策略。
从此以后根部完全自由，不固定骨盆、不施加支撑力。
**按 Y 时不复位位置、姿态、速度，不经过 FixedPose，也不插入站姿渐变。**

ACP 入口仍要求有效、同步、未过期的骨盆位姿和地形，以及有效姿态与高度。
Loco 的动态姿态不再要求接近 ACP 默认角；每次切入按导出合同重置 ACP 历史和上一动作。
Python bridge 同样从 B/LocoMode 接受 Y，实时输出沿用原 PD 限幅和阻尼保护。
此仿真采用 MuJoCo 真值定位与射线地形，不代表现场定位链路已经验证。

`tools/acp/evaluate_mujoco.py` 与 `evaluate_recorded_entry.py` 显式使用
`entry_mode: fixed_pose`，只用于重现旧站姿对照，不是本页的实时入口。
历史 36 场景的通过/失败不能替代 B → Y 验收。

## 本轮核查

同一 1k bundle，实际 Loco 运行 2 秒后直接切入，切换前后 qpos 未改变，
全程无根部支撑。三个短检查的完整记录位于本机
`data/acp_g1_reports/live_loco_entry_v1`：

| 场景 | 末段停稳 | 终点距离 | 说明 |
| --- | --- | ---: | --- |
| Loco 零速度 → 当前点停车 | 通过 | 0.011 m | 自由运行 15 秒 |
| Loco 前进指令 → 前方 1 m | 通过 | 0.035 m | 自由运行 15 秒 |
| Loco 转向指令 → 左侧 1 m / 转 90° | 未通过 | 0.188 m | 倾角越界，同周期进入阻尼 |

这是少量入口检查，尚未通过动态转向切入验收。失败记录保留，不通过调高倾角阈值掩盖。
实际 GUI 启动并运行验证已完成，保存了584帧 Loco 日志；窗口在检查结束后关闭。
ACP、FSM、输入及相关回归66项通过；1项虚拟手柄测试因环境跳过。
额外8项旧 PHP bridge 测试因本机缺少 `depth_backbone.onnx` 等外置权重未跑通，
未声明整仓回归通过。没有启动训练或连接真机。
