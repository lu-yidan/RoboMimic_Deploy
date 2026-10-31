# T10（oracle 观测）实验的准备与阻塞点

## 机制（已确认）

`training/bar_geometry_observation.py`：把 **12 维**杆位偏移（`rung_01/02/05/06` × xyz，归一化 0.02 m）
追加到 136 维观测后面，得到 148 维：

```python
features = offsets.reshape(-1) if self.mode == 'oracle' else np.zeros(12)
return np.r_[observation, features]
```

当前部署的策略是 **148 维、按零特征（blind）训练**的。所以"策略看不见杆位"不是因为信息不存在，
而是因为它从来只见过零。

注入已验证可用：`.dsh-watch/wide/oracle_observation.py`，环境变量 `G1_ORACLE_OBSERVATION=1`，
把 `BarGeometryObservation.__init__` 强制成 `mode='oracle'`（marker 文件确认生效）。

## 阻塞点：gate 的哈希链

训练器有一句硬断言：

```python
assert gate['passed'] and gate['protocol_sha256'] == sha(a.protocol/'protocol.json')
```

所以**不能**靠"复制 gate 目录 + 改 protocol 的步数"来缩短训练——协议一改，`protocol_sha256` 就对不上。
可行的只有两条：

1. **老老实实跑一个 32k 的 gate**（4 个代表工况，约 15 分钟），再训练 32k 步（约 47 分钟），
   再评估 24 工况（约 50 分钟）。合计 **约 2 小时**。
2. 直接在现有 131k 步的 gate 上训（约 2.5 小时训练）。

## 推荐命令序列（方案 1）

```powershell
$env:PYTHONPATH='D:\G1Climbing\.dsh-watch\wide;D:\G1Climbing\external\RoboMimic_Deploy'
$env:G1_ORACLE_OBSERVATION='1'
$env:G1_RESIDUAL_AUTHORITY='0.01875'
$py='D:\G1Climbing\external\RoboMimic_Deploy\.venv\Scripts\python.exe'

# 1) 32k 的 gate（4 工况，约 15 分钟）
& $py -m tools.prepare_geometry_training_gate --steps 32768 --output <新 gate 目录>

# 2) oracle + 1.25× 微调 32k 步（约 47 分钟）
& $py -u -m tools.train_calibrated_landing --protocol <新 gate 目录> --mode average --output <新 run>

# 3) 24 工况评估（约 50 分钟），评估时同样要带 G1_ORACLE_OBSERVATION=1
```

> 注意：评估**必须**也开 oracle，否则 148 维输入与训练时分布不一致。

## 判据（事先定好）

| 观察 | 结论 |
|---|---|
| 被位移工况（`training_*`）继续下降、低闭合工况不再被抬高 | **策略学会了利用几何信息** → RL 有戏，可以排长训练 |
| 32k 步内完全无方向（闭合分布与 blind 一致） | 这条通路即使打开也不够 → 要重新考虑"参考+残差"架构 |

对照基线（blind、1.25×、新参考、24 工况）：**mean 37.905 / P95 38.800 / max 38.844**。

## 当前状态

- 实验**未启动**（两次尝试都被 gate 哈希链挡住），已释放 `cpu4` 锁。
- 注入机制、协议副本、脚本都在位；上面三条命令可直接执行。
