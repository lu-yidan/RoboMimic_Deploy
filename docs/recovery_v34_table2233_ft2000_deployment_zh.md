# V34 桌子续训2233 @2000 部署候选

## 一行切换

在 `policy/smp_recovery/config/smp_recovery.yaml` 中设置：

```yaml
profile: v34_table2233_ft2000
```

重启部署进程。环境变量 `SMP_RECOVERY_PROFILE` 优先于文件；使用文件选择前应取消该覆盖。仓库默认仍为 `a6_no_alpha_9999`。

## 来源

4090-bj：`/root/workplace/smp-v34-plate-transfer/logs/V34MO_TableRoof2233_6000_20261010/model_2000.pt`。
父策略是 `V34_model_only_deploy_6000_20261010/model_5999.pt`。训练分布为平地20%、上下活动压板20%、固定四腿桌30%、固定无腿顶板30%，本文件不是@5000，也不是父策略@5999。

输入93维，无base线速度；输出29维；原checkpoint观测归一化嵌入ONNX。源环境的robot与actions配置逐项与父策略核对一致。沿用父策略90%标称力矩上限（profile独立tau_limit）、默认姿态、Kp/Kd、action scale；未为该策略放大电机力矩。

## 已完成检查与局限

- 512条ONNX/PyTorch输出核对通过，最大绝对误差1.14441e-5。
- 实际SmpRecovery加载、hash校验、enter/run、限幅检查通过。
- 部署CPU MuJoCo 3.2.3、实际控制器、平地4条初态（仰卧2/伏卧2），各23秒，未出现仿真异常；SR3为0/4。
- 两条仰卧最大头高约0.260/0.392m；两条伏卧约1.035/1.034m。后者能支撑到低位，但未达到1.10m站立门槛。该策略不是已验证可稳定恢复的部署版本。

这4条是接口及部署动力学冒烟检查，不能代替总体成功率。部署包含10步进入过渡、action clip10和目标力矩投影；训练没有这些处理，且物理步长为5ms、此次部署模拟为2ms。因此训练端能支撑起来不保证部署端相同。

SR3检查：头高≥1.10m、base直立度≥.90、双足合力各>20N，20秒内开始连续3秒，观察23秒。部署脚力为合力，训练评测为max-contact。没有连接真机或运行DDS，尚无本checkpoint真机验证。

模型同名manifest记录来源hash、控制匹配、导出核对和冒烟检查结果。该checkpoint仅作为用户指定的对比候选；不能用其地面翻滚行为替代论文中“脱困后稳定站立”的恢复成功。
