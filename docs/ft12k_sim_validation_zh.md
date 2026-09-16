# V33-reward FT 12k：RoboMimic 部署仿真验证

分支 `codex/v33-ft12k-sim-validation`，新增显式 profile `ft12k_sim`。默认仍为 `v35_rd_gate5999`。没有执行真机程序、DDS连接或真机动作。

## 模型与接口

来源：早期 L4 model_9999 经 V33 风格奖励微调至 model_12000，93D actor/29D action。checkpoint SHA256 `8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768`。

ONNX：`policy/smp_recovery/model/smp_v33_reward_ft12000_93d.onnx`，嵌入 checkpoint 观测归一化。512个固定随机输入与训练侧PyTorch确定性actor最大绝对误差1.1444e-5；另外400个真实部署仿真观测误差1.9073e-6。接口固定[1,93]→[1,29]，输出有限。

default_joint_pos、action_scale、Kp、Kd、tau_limit、动作clip、入场混合步数与控制周期均与训练冻结合同逐项相同。XML与训练 source.xml SHA256完全一致：`66f252f310befd4f1dc3265d223d5f74849e5b9967fdeb49dac19b5c98afc111`。部署CPU MuJoCo3.2.3，训练评测3.8.1/MJWarp；均2ms物理步、20ms控制步、Euler、Newton、100 iterations/50 line-search iterations。训练组装时移除了远处box等场景物体，部署保留原XML场景。

测试直接使用仓库的 `SmpRecovery.enter/run`、ONNX、PD和每2ms力矩饱和，绕过手柄与完整FSM。入场先保持测得的关节位置，20ms后第一次策略更新，随后50Hz。ORT测试只限制线程数，不改变模型。不是对SDK、通信延迟、真实电机或完整FSM切换的验证。

## 倒地姿态验证

程序化验证库每个方向前16个样本，共64；没有根据结果挑选样本。每个20秒，要求连续10秒满足原严格站立条件：头高1.15、upright .93、膝角<.8、底座速度<.15、角速度<.3、关节RMS<.5、脚速<.1、双脚载荷>20N、非脚触地<20N、站距.12–.45m。

|方向|部署CPU|训练侧相同64个样本|
|---|---:|---:|
|仰卧|16/16|16/16|
|伏卧|15/16|16/16|
|左侧|16/16|16/16|
|右侧|16/16|16/16|

部署63/64。失败伏卧没有到达直立，最大头高约.235m，并非只是稳定计时没达标。关节速度峰值P95=16.40rad/s，关节机械功率峰值P95=842W，头部竖直速度峰值P95=2.19m/s。峰值力矩仍需结合139Nm限值理解，不能把低平均功率当作硬件安全证明。不同规模评测的P95不可当作严格配对效应。

## 坐姿/深蹲零速度重新进入

从一个成功仰卧部署轨迹取t=1.0、1.2、1.38、1.6、1.8、2.0、2.2秒的7个不同姿态。分别保留原速度、清零所有机器人速度后重新调用enter（上一动作清零、入场混合重新开始）。两种条件各7/7通过。初始接触穿透均不超过5mm。

这说明这些具体中间姿态在标称部署仿真中不要求继承之前的动量；不是所有坐姿/摩擦条件都已覆盖，也不等于真机验证。

重要限制：原先选取t=1.4的状态，左脚与骨盆碰撞体存在约7.8mm穿透，虽两种速度条件都成功，不能计作接触有效证据，故改用相邻t=1.38状态。连续仰卧动作本身仍经过这段碰撞。四个代表轨迹前5秒的50Hz检查还发现约7.8–11.2mm的自碰撞穿透（包含脚-骨盆、手-腿），因此不能宣称收腿路径已真实可行，也不能确认它是否利用了接触软性。应进一步检查自碰撞余量和更严格接触条件下的路径。

## 复现与观看

当前机器已验证的部署Python为 `/home/luyd/miniconda3/envs/robomimic/bin/python`。基础conda Python的MuJoCo安装不可用，训练venv没有onnxruntime，勿混用。实际导出在训练venv完成，ONNX在部署环境验证。

```bash
cd /home/luyd/workspace/RoboMimic_Deploy
SMP_RECOVERY_PROFILE=ft12k_sim /home/luyd/miniconda3/envs/robomimic/bin/python deploy_mujoco/deploy_mujoco.py
```

这是已有手柄/FSM入口，Y进入SMP（不同时按L1），模型不会因配置加入而自动替换默认。此次自动化测试未打开该完整交互入口。

无需手柄的已验证自动化入口：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/luyd/miniconda3/envs/robomimic/bin/python tools/validate_ft12k_sim.py \
  --profile ft12k_sim \
  --bank /home/luyd/workspace/smp-master-repro/datasets/reset_banks/procedural_low_v1/validation.npz \
  --per-direction 16 --workers 4 --out outputs/ft12k_sim/repeat
```

输出位于 `outputs/ft12k_sim/`：
- `nominal_verified/`：有效的64姿态原始数据、逐例结果、四个代表轨迹。
- `valid_intermediate_summary.json`：接触筛选后的14个配对测试。
- `self_contact_diagnosis.json`：代表轨迹自碰撞。
- `supine_deploy_slow.mp4`：部署仰卧前6秒，1/4速侧面重放。
- `seated_zero_velocity.mp4`：t=1.38姿态零速度、重新进入后的20秒重放。
- `regression.log`：原部署13个测试全部通过。

早期 `nominal/` 的summary已标记VALID=false：初版诊断错误地把远处box与地面的负载计入机器人非脚接触，导致成功统计错误；已修正并重跑全部64例，以nominal_verified为准。没有改变控制策略或机器人动力学来修正这个计数问题。

本轮结论：接口与名义sim2sim初步通过；具体坐姿零速度恢复得到正面证据；自碰撞余量仍是进入真机验证之前需要解决的具体问题。没有测试延迟、足底摩擦扰动、真实力矩-速度包络或通信链路。
